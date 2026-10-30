import asyncio
import logging
import os
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Set
import uuid
from dotenv import load_dotenv
from langgraph.config import get_stream_writer
import psycopg2

from apps.services.cache_service import set_cache_with_doc_tags, set_negative_cache
from apps.services.llm_service import (
    INVENTORY_TABLE_MAX_TOKENS,
    MAX_CONTEXT_TOKENS,
    build_hierarchical_inventory_table,
    build_irac_context_and_citations,
    get_llm_model_name,
    stream_irac_ollama,
)
from apps.workflow.state import LegalRAGState, traced_node

load_dotenv(".env")
logger = logging.getLogger(__name__)

CONFIDENCE_MIN_SCORE = 0.20
INVENTORY_MIN_BANK_SIZE = 15

RETRIEVAL_LATENCY_NODES = (
    "EncodeQueryAndLinearDomainProbeNode",
    "DirectPointGetNode",
    "BiDirectionalGraphExpandNode",
    "FilteredSemanticOrLineageAggNode",
    "StratifiedMinQuotaSearchNode",
    "ScopedHybridRrfPrefetch50Node",
    "Qwen7BJsonPlannerNode",
    "ToolRewrittenSearchNode",
    "ToolGraphHopNode",
)

RERANK_LATENCY_NODES = (
    "Top10PruneAndInt8RerankNode",
    "DedupQuotaAndTemporalBoostNode",
    "FinalChunkBankInt8RerankNode",
    "DedupQuotaAndAmendmentPinNode",
)

GENERATION_LATENCY_NODES = (
    "GenerateHierarchicalInventoryTableNode",
    "IRACGenerateNode",
    "AbstainWithReasonNode",
    "AbstainFastExitNode",
)

REFUSAL_MESSAGES: Dict[str, str] = {
    "out_of_scope": (
        "Hệ thống từ chối trả lời: Câu hỏi nằm ngoài phạm vi tư vấn pháp luật Việt Nam hiện hành."
    ),
    "harmful_or_fictional": (
        "Hệ thống từ chối trả lời: Yêu cầu đề cập đến văn bản quy phạm pháp luật giả định, chưa ban hành hoặc không hợp lệ."
    ),
    "insufficient_evidence": (
        "Hệ thống từ chối trả lời: Không tìm thấy đủ căn cứ pháp lý xác thực trong cơ sở dữ liệu văn bản hiện hành để kết luận chính xác."
    ),
}

_background_audit_tasks: Set[asyncio.Task[Any]] = set()


def _get_stream_writer_safe() -> Callable[[Any], None]:
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


def _coerce_uuid(raw_id: Optional[str]) -> str:
    if not raw_id:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(str(raw_id)))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, str(raw_id)))


def _is_grounded(state: LegalRAGState) -> bool:
    if state.get("abstained", False):
        return False
    is_agg = bool(state.get("is_aggregation", False))
    top_chunks = state.get("top_chunks") or []
    max_score = float(state.get("max_score", 0.0))
    if is_agg:
        return len(top_chunks) > 0
    return max_score >= CONFIDENCE_MIN_SCORE and len(top_chunks) > 0


def _compute_stage_latencies(latency_ms: Dict[str, float]) -> Dict[str, int]:
    ret_ms = sum(float(latency_ms.get(k, 0.0)) for k in RETRIEVAL_LATENCY_NODES)
    rer_ms = sum(float(latency_ms.get(k, 0.0)) for k in RERANK_LATENCY_NODES)
    gen_ms = sum(float(latency_ms.get(k, 0.0)) for k in GENERATION_LATENCY_NODES)
    return {
        "retrieval_ms": int(round(ret_ms)),
        "rerank_ms": int(round(rer_ms)),
        "generation_ms": int(round(gen_ms)),
    }


@traced_node("ConfidenceGate")
def ConfidenceGate(state: LegalRAGState) -> Dict[str, Any]:
    if _is_grounded(state):
        return {"abstained": False}
    return {
        "abstained": True,
        "abstain_reason": state.get("abstain_reason") or "insufficient_evidence",
    }


def route_after_confidence(
    state: LegalRAGState,
) -> Literal["grounded_inventory", "grounded_direct_irac", "ungrounded_abstain"]:
    if not _is_grounded(state):
        return "ungrounded_abstain"

    is_agg = bool(state.get("is_aggregation", False))
    chunk_bank = state.get("chunk_bank") or {}
    if is_agg and len(chunk_bank) > INVENTORY_MIN_BANK_SIZE:
        return "grounded_inventory"

    return "grounded_direct_irac"


@traced_node("GenerateHierarchicalInventoryTableNode")
def GenerateHierarchicalInventoryTableNode(state: LegalRAGState) -> Dict[str, Any]:
    chunk_bank = state.get("chunk_bank") or {}
    bank_chunks = list(chunk_bank.values()) if chunk_bank else list(state.get("top_chunks") or [])

    table_md, all_citations = build_hierarchical_inventory_table(
        chunks=bank_chunks,
        max_tokens=INVENTORY_TABLE_MAX_TOKENS,
    )

    existing_top = list(state.get("top_chunks") or [])
    if existing_top:
        top_5 = existing_top[:5]
    else:
        top_5 = sorted(
            bank_chunks,
            key=lambda c: (float(c.get("rerank_score") or 0.0), float(c.get("rrf_score") or 0.0)),
            reverse=True,
        )[:5]

    return {
        "inventory_table": table_md,
        "top_chunks": top_5,
        "citations": all_citations,
    }


@traced_node("IRACGenerateNode")
async def IRACGenerateNode(state: LegalRAGState) -> Dict[str, Any]:
    request_id = _coerce_uuid(state.get("request_id"))
    top_chunks = state.get("top_chunks") or []
    inventory_table = state.get("inventory_table")
    existing_citations = state.get("citations")

    context_block, citations = build_irac_context_and_citations(
        top_chunks=top_chunks,
        inventory_table=inventory_table,
        existing_citations=existing_citations,
        max_context_tokens=MAX_CONTEXT_TOKENS,
    )

    writer = _get_stream_writer_safe()
    writer({
        "event": "metadata",
        "data": {
            "request_id": request_id,
            "abstained": False,
            "citations": citations,
        },
    })

    query = state.get("query") or ""
    answer_parts: List[str] = []
    async for delta in stream_irac_ollama(query=query, context_block=context_block):
        answer_parts.append(delta)
        writer({"event": "token", "data": {"delta": delta}})

    return {
        "request_id": request_id,
        "answer": "".join(answer_parts).strip(),
        "citations": citations,
        "abstained": False,
    }


@traced_node("SetCacheWithDocTagsNode")
def SetCacheWithDocTagsNode(state: LegalRAGState) -> Dict[str, Any]:
    sha = state.get("query_sha256") or ""
    if not sha:
        return {}

    citations = state.get("citations") or []
    top_chunks = state.get("top_chunks") or []
    doc_numbers = [c.get("doc_number", "") for c in citations] + [c.get("doc_number", "") for c in top_chunks]

    payload = {
        "request_id": state.get("request_id", ""),
        "query": state.get("query", ""),
        "answer": state.get("answer", ""),
        "citations": citations,
        "abstained": False,
        "top_chunks": top_chunks,
        "inventory_table": state.get("inventory_table"),
    }
    set_cache_with_doc_tags(query_sha256=sha, payload=payload, doc_numbers=doc_numbers)
    return {}


@traced_node("AbstainWithReasonNode")
def AbstainWithReasonNode(state: LegalRAGState) -> Dict[str, Any]:
    request_id = _coerce_uuid(state.get("request_id"))
    reason_key = state.get("abstain_reason") or "insufficient_evidence"

    existing_ans = (state.get("answer") or "").strip()
    if existing_ans.startswith("Hệ thống từ chối trả lời:"):
        refusal_text = existing_ans
    elif state.get("is_future_year"):
        refusal_text = "Hệ thống từ chối trả lời: Văn bản hoặc mốc thời gian yêu cầu vượt quá năm hiện hành (2026)."
    else:
        refusal_text = REFUSAL_MESSAGES.get(reason_key, REFUSAL_MESSAGES["insufficient_evidence"])

    writer = _get_stream_writer_safe()
    writer({
        "event": "metadata",
        "data": {
            "request_id": request_id,
            "abstained": True,
            "citations": [],
        },
    })
    writer({"event": "token", "data": {"delta": refusal_text}})

    return {
        "request_id": request_id,
        "abstained": True,
        "abstain_reason": reason_key,
        "answer": refusal_text,
        "citations": [],
        "top_chunks": [],
    }


@traced_node("SetGlobalNegativeCacheNode")
def SetGlobalNegativeCacheNode(state: LegalRAGState) -> Dict[str, Any]:
    sha = state.get("query_sha256") or ""
    if not sha:
        return {}

    payload = {
        "request_id": state.get("request_id", ""),
        "query": state.get("query", ""),
        "abstained": True,
        "abstain_reason": state.get("abstain_reason") or "insufficient_evidence",
        "answer": state.get("answer", ""),
        "citations": [],
        "top_chunks": [],
    }
    set_negative_cache(query_sha256=sha, payload=payload)
    return {}


def _insert_audit_sync(snapshot: Dict[str, Any]) -> None:
    dsn = os.getenv("DATABASE_DSN")
    if not dsn:
        return

    conn = psycopg2.connect(dsn, connect_timeout=8)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_log (
                        request_id, raw_query, normalized_query, execution_mode, react_steps,
                        extracted_doc_number, extracted_article, cache_hit, abstained,
                        generated_answer, llm_model_name, retrieval_ms, rerank_ms, generation_ms
                    ) VALUES (
                        %s::uuid, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (request_id) DO NOTHING
                    """,
                    (
                        snapshot["request_id"],
                        snapshot["raw_query"],
                        snapshot["normalized_query"],
                        snapshot["execution_mode"],
                        snapshot["react_steps"],
                        snapshot["extracted_doc_number"],
                        snapshot["extracted_article"],
                        snapshot["cache_hit"],
                        snapshot["abstained"],
                        snapshot["generated_answer"],
                        snapshot["llm_model_name"],
                        snapshot["retrieval_ms"],
                        snapshot["rerank_ms"],
                        snapshot["generation_ms"],
                    ),
                )

                for chunk_row in snapshot["retrieved_chunks"]:
                    cur.execute(
                        """
                        INSERT INTO query_retrieved_chunk (
                            request_id, chunk_id, rank_order,
                            dense_score, sparse_score, rrf_score, rerank_score, passed_threshold
                        )
                        SELECT %s::uuid, lc.chunk_id, %s, %s, %s, %s, %s, %s
                        FROM legal_chunks lc
                        WHERE lc.chunk_id = %s::uuid
                        """,
                        (
                            snapshot["request_id"],
                            chunk_row["rank_order"],
                            chunk_row["dense_score"],
                            chunk_row["sparse_score"],
                            chunk_row["rrf_score"],
                            chunk_row["rerank_score"],
                            chunk_row["passed_threshold"],
                            chunk_row["chunk_id"],
                        ),
                    )
    finally:
        conn.close()


async def _persist_audit_log_async(snapshot: Dict[str, Any]) -> None:
    try:
        await asyncio.to_thread(_insert_audit_sync, snapshot)
    except Exception as exc:
        logger.error("Async PostgreSQL audit insert failed (request_id=%s): %s", snapshot["request_id"], exc)


async def flush_pending_audit_tasks() -> None:
    if _background_audit_tasks:
        await asyncio.gather(*list(_background_audit_tasks), return_exceptions=True)


@traced_node("WriteAuditLogToPostgresNode")
async def WriteAuditLogToPostgresNode(state: LegalRAGState) -> Dict[str, Any]:
    request_id = _coerce_uuid(state.get("request_id"))
    doc_filters = state.get("doc_filters") or []
    extracted_doc = ",".join(doc_filters)[:100] if doc_filters else None
    article_filter = state.get("article_filter")
    extracted_art = article_filter[:50] if article_filter else None

    latency_map = state.get("latency_ms") or {}
    stage_ms = _compute_stage_latencies(latency_map)
    abstained = bool(state.get("abstained", False))
    is_agg = bool(state.get("is_aggregation", False))

    chunk_rows: List[Dict[str, Any]] = []
    for idx, c in enumerate(state.get("top_chunks") or [], start=1):
        raw_cid = c.get("chunk_id", "")
        try:
            valid_cid = str(uuid.UUID(str(raw_cid)))
        except ValueError:
            continue
        r_score = float(c.get("rerank_score") or 0.0)
        passed = (not abstained) and (is_agg or r_score >= CONFIDENCE_MIN_SCORE or bool(c.get("is_pinned_amendment")))
        chunk_rows.append({
            "chunk_id": valid_cid,
            "rank_order": int(c.get("rank_order") or idx),
            "dense_score": float(c["dense_score"]) if c.get("dense_score") is not None else None,
            "sparse_score": float(c["sparse_score"]) if c.get("sparse_score") is not None else None,
            "rrf_score": float(c.get("rrf_score") or 0.0),
            "rerank_score": r_score,
            "passed_threshold": passed,
        })

    snapshot: Dict[str, Any] = {
        "request_id": request_id,
        "raw_query": state.get("query") or "",
        "normalized_query": state.get("cleaned_query") or state.get("query") or "",
        "execution_mode": (state.get("execution_mode") or "fast_path")[:50],
        "react_steps": int(state.get("react_step", 0)),
        "extracted_doc_number": extracted_doc,
        "extracted_article": extracted_art,
        "cache_hit": bool(state.get("cache_hit", False)),
        "abstained": abstained,
        "generated_answer": state.get("answer") or "",
        "llm_model_name": get_llm_model_name()[:50],
        "retrieval_ms": stage_ms["retrieval_ms"],
        "rerank_ms": stage_ms["rerank_ms"],
        "generation_ms": stage_ms["generation_ms"],
        "retrieved_chunks": chunk_rows,
    }

    task = asyncio.create_task(_persist_audit_log_async(snapshot))
    _background_audit_tasks.add(task)
    task.add_done_callback(_background_audit_tasks.discard)

    return {"request_id": request_id}


@traced_node("ReturnResponseNode")
def ReturnResponseNode(state: LegalRAGState) -> Dict[str, Any]:
    request_id = _coerce_uuid(state.get("request_id"))
    latency_map = dict(state.get("latency_ms") or {})
    stage_ms = _compute_stage_latencies(latency_map)
    summed_nodes_ms = round(sum(float(v) for v in latency_map.values()), 2)
    start_time = state.get("start_time")
    wall_ms = round((time.time() - float(start_time)) * 1000.0, 2) if start_time else summed_nodes_ms

    metrics = {
        "total_ms": max(wall_ms, summed_nodes_ms),
        "retrieval_ms": stage_ms["retrieval_ms"],
        "rerank_ms": stage_ms["rerank_ms"],
        "generation_ms": stage_ms["generation_ms"],
        "react_steps": int(state.get("react_step", 0)),
        "execution_mode": state.get("execution_mode") or "fast_path",
        "cache_hit": bool(state.get("cache_hit", False)),
        "abstained": bool(state.get("abstained", False)),
        "node_latencies_ms": latency_map,
    }

    writer = _get_stream_writer_safe()
    writer({
        "event": "done",
        "data": {
            "request_id": request_id,
            "metrics": metrics,
        },
    })

    return {"request_id": request_id}

