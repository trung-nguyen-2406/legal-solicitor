import time
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
import numpy as np
from qdrant_client import models

from apps.schemas.planner_models import AggregationSweepArgs, GraphHopArgs, RewrittenSearchArgs
from apps.services.post_rerank_service import apply_temporal_dedup_quota_and_pin
from apps.services.react_agent_service import (
    call_ollama_planner_async,
    execute_graph_hop,
    execute_rewritten_search,
    score_chunk_bank_candidates,
)
from apps.services.resources import (
    get_collection_name,
    get_db_service,
    get_embedding_service,
    get_int8_reranker,
    get_qdrant_client,
)
from apps.workflow.nodes_fast import _active_filter, _point_to_dict, _build_inventory_table
from apps.workflow.state import LegalRAGState, traced_node

CIRCUIT_BREAKER_MAX_STEPS = 2
CIRCUIT_BREAKER_TIMEOUT_S = 35.0


def route_escalation_gate(state: LegalRAGState) -> Literal["keep_fast_path", "enter_react"]:
    top_chunks = state.get("top_chunks") or []
    fast_branch = state.get("fast_branch")
    is_foreign = bool(state.get("is_foreign_lang", False))
    is_multistep = bool(state.get("is_multistep", False))
    word_count = int(state.get("word_count", 0))
    max_score = float(state.get("max_score", 0.0))
    has_list_cue = bool(state.get("has_list_cue", False))
    exact_agg_count = int(state.get("exact_agg_count", 0))

    if fast_branch == "direct_point" and len(top_chunks) > 0:
        return "keep_fast_path"

    if fast_branch == "aggregation":
        asks_lineage = bool(state.get("doc_filters")) and any(
            kw in (state.get("query") or "").lower()
            for kw in ("sửa đổi", "bổ sung", "thay thế", "bãi bỏ", "lần sửa đổi", "lịch sử hiệu lực")
        )
        is_grounded_agg = (exact_agg_count > 0 and max_score >= 0.35) or (asks_lineage and len(top_chunks) > 0)
        if is_grounded_agg and not is_foreign:
            return "keep_fast_path"
        return "enter_react"

    if has_list_cue and max_score < 0.50:
        return "enter_react"

    if max_score < 0.42 or is_foreign or is_multistep or word_count > 65:
        return "enter_react"

    return "keep_fast_path"


def route_planner_action(
    state: LegalRAGState,
) -> Literal["abstain", "rewritten_search", "aggregation_sweep", "graph_hop", "finish_agg_direct", "finish_rerank"]:
    decision = state.get("planner_decision") or {}
    act = decision.get("action")
    if act == "ABSTAIN":
        return "abstain"
    if act == "tool_rewritten_search":
        return "rewritten_search"
    if act == "tool_aggregation_sweep":
        return "aggregation_sweep"
    if act == "tool_graph_hop":
        return "graph_hop"
    if state.get("is_aggregation"):
        return "finish_agg_direct"
    return "finish_rerank"


@traced_node("Qwen7BJsonPlannerNode")
async def Qwen7BJsonPlannerNode(state: LegalRAGState) -> Dict[str, Any]:
    step = int(state.get("react_step", 0))
    start_time = float(state.get("start_time") or time.time())
    elapsed = time.time() - start_time

    if step >= CIRCUIT_BREAKER_MAX_STEPS or elapsed >= CIRCUIT_BREAKER_TIMEOUT_S:
        return {
            "planner_decision": {
                "thought": "Circuit breaker reached",
                "action": "FINISH",
                "abstain_reason": None,
                "search_args": None,
                "graph_args": None,
            },
            "execution_mode": "slow_react_timeout",
            "react_step": step + 1,
            "start_time": start_time,
        }

    query = state.get("query", "")
    scratchpad = state.get("pointer_scratchpad") or []

    remaining_budget = max(5.0, CIRCUIT_BREAKER_TIMEOUT_S - elapsed)
    planner_timeout = min(40.0, remaining_budget)

    try:
        decision = await call_ollama_planner_async(
            query=query,
            scratchpad=scratchpad,
            step=step,
            elapsed_s=elapsed,
            timeout=planner_timeout,
        )
    except Exception:
        return {
            "planner_decision": {
                "thought": "Fallback to FINISH due to Planner error",
                "action": "FINISH",
                "abstain_reason": None,
                "search_args": None,
                "graph_args": None,
            },
            "execution_mode": "slow_react",
            "react_step": step + 1,
            "start_time": start_time,
        }

    update: Dict[str, Any] = {
        "planner_decision": decision.model_dump(),
        "react_step": step + 1,
        "execution_mode": "slow_react",
        "start_time": start_time,
    }

    if decision.action == "ABSTAIN":
        update["abstained"] = True
        update["abstain_reason"] = decision.abstain_reason or "out_of_scope"
        update["execution_mode"] = "slow_react_abstain"

    return update


@traced_node("ToolRewrittenSearchNode")
def ToolRewrittenSearchNode(state: LegalRAGState) -> Dict[str, Any]:
    decision = state.get("planner_decision") or {}
    raw_args = decision.get("search_args") or {}
    search_args = RewrittenSearchArgs(**raw_args) if isinstance(raw_args, dict) else raw_args

    new_chunks, new_subqueries, new_pointers = execute_rewritten_search(
        search_args=search_args,
        embed_service=get_embedding_service(),
        qdrant_client=get_qdrant_client(),
        collection_name=get_collection_name(),
        db_service=get_db_service(),
        top_k=25,
    )

    merged_matched = dict(state.get("matched_subqueries") or {})
    merged_matched.update(new_subqueries)

    return {
        "chunk_bank": new_chunks,
        "matched_subqueries": merged_matched,
        "pointer_scratchpad": new_pointers,
    }


@traced_node("ToolGraphHopNode")
def ToolGraphHopNode(state: LegalRAGState) -> Dict[str, Any]:
    decision = state.get("planner_decision") or {}
    raw_args = decision.get("graph_args") or {}
    graph_args = GraphHopArgs(**raw_args) if isinstance(raw_args, dict) else raw_args

    new_chunks, new_pointers = execute_graph_hop(
        graph_args=graph_args,
        db_service=get_db_service(),
        limit=10,
    )

    return {
        "chunk_bank": new_chunks,
        "pointer_scratchpad": new_pointers,
    }


@traced_node("ToolAggregationSweepNode")
def ToolAggregationSweepNode(state: LegalRAGState) -> Dict[str, Any]:
    decision = state.get("planner_decision") or {}
    agg_args = decision.get("agg_args") or {}
    if hasattr(agg_args, "model_dump"):
        agg_args = agg_args.model_dump()
    elif not isinstance(agg_args, dict):
        agg_args = {}

    sub_queries = agg_args.get("sub_queries") or state.get("sub_queries") or [state.get("cleaned_query") or state.get("query") or ""]
    anchor_keywords = agg_args.get("anchor_keywords") or state.get("anchor_keywords") or []
    year_from = agg_args.get("year_from") or state.get("year_from")
    year_to = agg_args.get("year_to") or state.get("year_to")

    qfilter = _active_filter(
        year_from=year_from,
        year_to=year_to,
    )
    client = get_qdrant_client()
    collection = get_collection_name()
    embed_svc = get_embedding_service()
    db = get_db_service()

    sweep_payload_fields = [
        "doc_id", "doc_number", "doc_title", "article_code",
        "clause_code", "issued_year", "legal_field", "breadcrumb", "anchor_tag"
    ]

    all_sweep_candidates: List[Dict[str, Any]] = []

    for sq in sub_queries:
        enc = embed_svc.encode(sq)
        sq_dense = enc["dense"]
        sq_sparse_ind = enc["sparse"]["indices"]
        sq_sparse_val = enc["sparse"]["values"]

        prefetch = [
            models.Prefetch(query=sq_dense, using="dense_vector", filter=qfilter, limit=300),
            models.Prefetch(
                query=models.SparseVector(indices=sq_sparse_ind, values=sq_sparse_val),
                using="sparse_vector",
                filter=qfilter,
                limit=300,
            ),
        ]
        pts = client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=300,
            with_payload=sweep_payload_fields,
        ).points

        if not pts:
            continue

        for idx, pt in enumerate(pts):
            c_dict = _point_to_dict(pt, rank=idx + 1)
            if anchor_keywords:
                title_txt = (c_dict.get("doc_title") or "").lower()
                crumb_txt = (c_dict.get("breadcrumb") or "").lower()
                field_txt = (c_dict.get("legal_field") or "").lower()
                combined_txt = f"{title_txt} {crumb_txt} {field_txt}"
                if not any(ak.lower() in combined_txt for ak in anchor_keywords):
                    continue
            all_sweep_candidates.append(c_dict)

    deduped_by_article: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cand in all_sweep_candidates:
        doc = cand.get("doc_number") or ""
        art = cand.get("article_code") or ""
        key = (doc, art)
        if key not in deduped_by_article or float(cand.get("rrf_score", 0.0)) > float(deduped_by_article[key].get("rrf_score", 0.0)):
            deduped_by_article[key] = cand

    sorted_articles = sorted(deduped_by_article.values(), key=lambda c: float(c.get("rrf_score", 0.0)), reverse=True)
    exact_agg_count = len(sorted_articles)

    top_5_diverse: List[Dict[str, Any]] = []
    seen_docs: Set[str] = set()
    for art in sorted_articles:
        doc = art.get("doc_number") or ""
        if doc and doc not in seen_docs:
            top_5_diverse.append(art)
            seen_docs.add(doc)
            if len(top_5_diverse) == 5:
                break
    if len(top_5_diverse) < 5:
        for art in sorted_articles:
            if art not in top_5_diverse:
                top_5_diverse.append(art)
                if len(top_5_diverse) == 5:
                    break

    hydrated_top_5 = db.hydrate_chunks(top_5_diverse) if top_5_diverse else []

    max_score = 0.0
    if hydrated_top_5:
        top_3 = hydrated_top_5[:3]
        query_for_rerank = state.get("cleaned_query") or state.get("query") or ""
        raw_logits = get_int8_reranker().rerank_logits(query_for_rerank, top_3)
        for c, lg in zip(top_3, raw_logits):
            sc = 1.0 / (1.0 + np.exp(-float(lg)))
            c["rerank_score"] = round(float(sc), 4)
        rerank_scores = [float(c.get("rerank_score", 0.0)) for c in top_3]
        max_score = max(rerank_scores, default=0.0)

    inventory_table = _build_inventory_table(sorted_articles) if sorted_articles else None
    bank = {c["chunk_id"]: c for c in hydrated_top_5}

    return {
        "is_aggregation": True,
        "execution_mode": "react_agg_sweep",
        "inventory_table": inventory_table,
        "top_chunks": hydrated_top_5,
        "chunk_bank": bank,
        "max_score": max_score,
        "exact_agg_count": exact_agg_count,
        "agg_catalog_items": sorted_articles,
        "pointer_scratchpad": hydrated_top_5,
    }


@traced_node("FinalChunkBankInt8RerankNode")
def FinalChunkBankInt8RerankNode(state: LegalRAGState) -> Dict[str, Any]:
    chunk_bank = state.get("chunk_bank") or {}
    cleaned_query = state.get("cleaned_query") or state.get("query") or ""
    matched_subqueries = state.get("matched_subqueries") or {}

    pruned_cands, max_logits = score_chunk_bank_candidates(
        chunk_bank=chunk_bank,
        cleaned_vn_query=cleaned_query,
        matched_subqueries=matched_subqueries,
        reranker=get_int8_reranker(),
        top_candidates_limit=30,
    )

    scored_cands = []
    for c, lg in zip(pruned_cands, max_logits):
        c_copy = dict(c)
        sc = 1.0 / (1.0 + np.exp(-float(lg)))
        c_copy["rerank_score"] = float(sc)
        scored_cands.append(c_copy)
    retrieved_top_25 = sorted(scored_cands, key=lambda x: x["rerank_score"], reverse=True)[:25]

    return {
        "rrf_candidates": pruned_cands,
        "raw_logits": max_logits,
        "retrieved_top_25": retrieved_top_25,
    }


@traced_node("DedupQuotaAndAmendmentPinNode")
def DedupQuotaAndAmendmentPinNode(state: LegalRAGState) -> Dict[str, Any]:
    candidates = state.get("rrf_candidates") or []
    raw_logits = state.get("raw_logits") or []
    doc_filters = state.get("doc_filters") or []

    if not candidates or not raw_logits:
        return {
            "top_chunks": [],
            "retrieved_top_25": [],
            "max_score": 0.0,
        }

    top_chunks, deduped_25 = apply_temporal_dedup_quota_and_pin(
        candidates=candidates,
        raw_logits=raw_logits,
        doc_filters=doc_filters,
        db_service=get_db_service(),
        top_n=5,
        return_all_deduped=True,
    )

    non_pinned = [float(c.get("rerank_score", 0.0)) for c in top_chunks if not c.get("is_pinned_amendment")]
    max_score = max(non_pinned, default=0.0)

    bank_updates: Dict[str, Dict[str, Any]] = {c["chunk_id"]: c for c in top_chunks}
    for c in deduped_25:
        if c["chunk_id"] not in bank_updates:
            bank_updates[c["chunk_id"]] = c

    scratchpad_items = [
        {
            "chunk_id": c["chunk_id"],
            "doc_number": c.get("doc_number", ""),
            "article_code": c.get("article_code", ""),
            "rerank_score": c.get("rerank_score", 0.0),
            "breadcrumb": c.get("breadcrumb", ""),
        }
        for c in top_chunks
    ]

    return {
        "top_chunks": top_chunks,
        "retrieved_top_25": deduped_25,
        "max_score": max_score,
        "chunk_bank": bank_updates,
        "pointer_scratchpad": scratchpad_items,
    }

