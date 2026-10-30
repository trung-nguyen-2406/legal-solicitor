import time
from typing import Any, Dict, Literal

from apps.schemas.planner_models import GraphHopArgs, RewrittenSearchArgs
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

    if fast_branch == "direct_point" and len(top_chunks) > 0:
        return "keep_fast_path"
    if fast_branch == "aggregation" and len(top_chunks) >= 3 and not is_foreign:
        return "keep_fast_path"

    if max_score < 0.42 or is_foreign or is_multistep or word_count > 65:
        return "enter_react"

    return "keep_fast_path"


def route_planner_action(
    state: LegalRAGState,
) -> Literal["abstain", "rewritten_search", "graph_hop", "finish_react"]:
    decision = state.get("planner_decision") or {}
    act = decision.get("action")
    if act == "ABSTAIN":
        return "abstain"
    if act == "tool_rewritten_search":
        return "rewritten_search"
    if act == "tool_graph_hop":
        return "graph_hop"
    return "finish_react"


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
        top_candidates_limit=20,
    )

    return {
        "rrf_candidates": pruned_cands,
        "raw_logits": max_logits,
    }


@traced_node("DedupQuotaAndAmendmentPinNode")
def DedupQuotaAndAmendmentPinNode(state: LegalRAGState) -> Dict[str, Any]:
    candidates = state.get("rrf_candidates") or []
    raw_logits = state.get("raw_logits") or []
    doc_filters = state.get("doc_filters") or []

    if not candidates or not raw_logits:
        return {
            "top_chunks": [],
            "max_score": 0.0,
        }

    top_chunks = apply_temporal_dedup_quota_and_pin(
        candidates=candidates,
        raw_logits=raw_logits,
        doc_filters=doc_filters,
        db_service=get_db_service(),
        top_n=5,
    )

    non_pinned = [float(c.get("rerank_score", 0.0)) for c in top_chunks if not c.get("is_pinned_amendment")]
    max_score = max(non_pinned, default=0.0)

    bank_updates: Dict[str, Dict[str, Any]] = {c["chunk_id"]: c for c in top_chunks}
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
        "max_score": max_score,
        "chunk_bank": bank_updates,
        "pointer_scratchpad": scratchpad_items,
    }

