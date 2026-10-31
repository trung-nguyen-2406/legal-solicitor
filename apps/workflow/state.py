import inspect
import time
from functools import wraps
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict
import mlflow


def merge_latency(left: Optional[Dict[str, float]], right: Optional[Dict[str, float]]) -> Dict[str, float]:
    merged = dict(left or {})
    for k, v in (right or {}).items():
        merged[k] = round(merged.get(k, 0.0) + float(v), 2)
    return merged


def merge_chunk_bank(left: Optional[Dict[str, Dict[str, Any]]], right: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    merged = dict(left or {})
    for cid, chunk in (right or {}).items():
        if cid not in merged:
            merged[cid] = chunk
        else:
            curr_rerank = float(merged[cid].get("rerank_score", 0.0))
            new_rerank = float(chunk.get("rerank_score", 0.0))
            if new_rerank > curr_rerank:
                merged[cid] = chunk
            elif new_rerank == curr_rerank:
                curr_rrf = float(merged[cid].get("rrf_score", 0.0))
                new_rrf = float(chunk.get("rrf_score", 0.0))
                if new_rrf >= curr_rrf:
                    merged[cid] = chunk
    return merged


def reduce_scratchpad(left: Optional[List[Dict[str, Any]]], right: Optional[List[Any]]) -> List[Dict[str, Any]]:
    if not right:
        return list(left or [])
    if right == ["__RESET__"]:
        return []
    base = list(left or [])
    for item in right:
        base.append({
            "chunk_id": item.get("chunk_id", ""),
            "doc_number": item.get("doc_number", ""),
            "article_code": item.get("article_code", ""),
            "score": round(float(item.get("rerank_score") or item.get("score") or 0.0), 4),
            "title_lead_in": (item.get("title_lead_in") or item.get("breadcrumb") or "")[:120],
        })
    return base[-16:]


class LegalRAGState(TypedDict, total=False):
    request_id: str
    query: str
    query_sha256: str
    cache_hit: bool
    execution_mode: str
    doc_filters: List[str]
    article_filter: Optional[str]
    is_aggregation: bool
    is_foreign_lang: bool
    start_time: float
    react_step: int
    pointer_scratchpad: Annotated[List[Dict[str, Any]], reduce_scratchpad]
    chunk_bank: Annotated[Dict[str, Dict[str, Any]], merge_chunk_bank]
    top_chunks: List[Dict[str, Any]]
    inventory_table: Optional[str]
    abstained: bool
    answer: str
    citations: List[Dict[str, Any]]
    latency_ms: Annotated[Dict[str, float], merge_latency]

    cleaned_query: str
    is_multistep: bool
    word_count: int
    is_future_year: bool
    domain_prob: float
    dense_vector: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]
    fast_branch: str
    rrf_candidates: List[Dict[str, Any]]
    raw_logits: List[float]
    max_score: float
    abstain_reason: Optional[str]
    planner_decision: Optional[Dict[str, Any]]
    matched_subqueries: Dict[str, str]

    sub_queries: List[str]
    anchor_keywords: List[str]
    has_list_cue: bool
    year_from: Optional[int]
    year_to: Optional[int]
    exact_agg_count: Optional[int]
    agg_catalog_items: List[Dict[str, Any]]
    retrieved_top_25: List[Dict[str, Any]]
    retrieved_candidates: List[Dict[str, Any]]


def _brief_chunks(chunks: List[Dict[str, Any]], limit: int = 10) -> List[str]:
    items: List[str] = []
    for c in (chunks or [])[:limit]:
        doc = c.get("doc_number") or ""
        art = c.get("article_code") or ""
        score = float(c.get("rerank_score") or c.get("rrf_score") or 0.0)
        pin = " [PIN]" if c.get("is_pinned_amendment") else ""
        items.append(f"{doc}:{art}:{score:.3f}{pin}")
    return items


def _build_span_inputs(node_name: str, state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    q = state.get("cleaned_query") or state.get("query") or ""
    if node_name == "PureRedisCacheCheckNode":
        return {"query": state.get("query", ""), "query_sha256": state.get("query_sha256", "")}
    if node_name == "StructuredQueryDescriptorNode":
        return {"query": state.get("query", "")}
    if node_name == "EncodeQueryAndLinearDomainProbeNode":
        return {
            "cleaned_query": q,
            "doc_filters": state.get("doc_filters", []),
            "article_filter": state.get("article_filter"),
            "is_aggregation": state.get("is_aggregation", False),
            "is_foreign_lang": state.get("is_foreign_lang", False),
            "is_future_year": state.get("is_future_year", False),
        }
    if node_name in (
        "DirectPointGetNode",
        "BiDirectionalGraphExpandNode",
        "FilteredSemanticOrLineageAggNode",
        "StratifiedMinQuotaSearchNode",
        "ScopedHybridRrfPrefetch50Node",
    ):
        return {
            "query": q,
            "doc_filters": state.get("doc_filters", []),
            "article_filter": state.get("article_filter"),
            "domain_prob": round(float(state.get("domain_prob") or 0.0), 4),
            "top_chunks_in": _brief_chunks(state.get("top_chunks") or [], 5),
        }
    if node_name in (
        "Top10PruneAndInt8RerankNode",
        "DedupQuotaAndTemporalBoostNode",
        "FinalChunkBankInt8RerankNode",
        "DedupQuotaAndAmendmentPinNode",
    ):
        return {
            "query": q,
            "rrf_candidates_count": len(state.get("rrf_candidates") or []),
            "chunk_bank_size": len(state.get("chunk_bank") or {}),
            "doc_filters": state.get("doc_filters", []),
        }
    if node_name in ("Qwen7BJsonPlannerNode", "ToolRewrittenSearchNode", "ToolGraphHopNode"):
        return {
            "query": state.get("query", ""),
            "react_step": state.get("react_step", 0),
            "max_score": round(float(state.get("max_score") or 0.0), 4),
            "planner_decision": state.get("planner_decision"),
            "pointer_scratchpad": state.get("pointer_scratchpad", []),
            "chunk_bank_size": len(state.get("chunk_bank") or {}),
        }
    if node_name == "ConfidenceGate":
        return {
            "max_score": round(float(state.get("max_score") or 0.0), 4),
            "is_aggregation": state.get("is_aggregation", False),
            "top_chunks_count": len(state.get("top_chunks") or []),
            "chunk_bank_size": len(state.get("chunk_bank") or {}),
        }
    if node_name == "GenerateHierarchicalInventoryTableNode":
        return {
            "chunk_bank_size": len(state.get("chunk_bank") or {}),
            "top_chunks": _brief_chunks(state.get("top_chunks") or [], 5),
        }
    if node_name == "IRACGenerateNode":
        return {
            "query": state.get("query", ""),
            "top_chunks": _brief_chunks(state.get("top_chunks") or [], 7),
            "has_inventory_table": bool(state.get("inventory_table")),
        }
    return {
        "request_id": state.get("request_id", ""),
        "execution_mode": state.get("execution_mode", ""),
        "abstained": state.get("abstained", False),
        "abstain_reason": state.get("abstain_reason"),
        "citations_count": len(state.get("citations") or []),
        "top_chunks": _brief_chunks(state.get("top_chunks") or [], 5),
    }


def _build_span_outputs(node_name: str, result: Any, elapsed_ms: float) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"result": str(result), "node_latency_ms": elapsed_ms}
    out: Dict[str, Any] = {}
    for k, v in result.items():
        if k == "latency_ms":
            continue
        if k == "dense_vector":
            out["dense_vector_dim"] = len(v) if isinstance(v, list) else 0
        elif k == "sparse_indices":
            out["sparse_nonzero_terms"] = len(v) if isinstance(v, list) else 0
        elif k == "sparse_values":
            continue
        elif k == "rrf_candidates" and isinstance(v, list):
            out["rrf_candidates_count"] = len(v)
            out["rrf_top10"] = _brief_chunks(v, 10)
        elif k == "chunk_bank" and isinstance(v, dict):
            vals = list(v.values())
            out["chunk_bank_updates"] = len(vals)
            out["chunk_bank_preview"] = _brief_chunks(vals, 10)
        elif k == "top_chunks" and isinstance(v, list):
            out["top_chunks_count"] = len(v)
            out["top_chunks"] = _brief_chunks(v, 10)
        elif k == "raw_logits" and isinstance(v, list):
            out["raw_logits"] = [round(float(x), 3) for x in v[:12]]
        elif k == "domain_prob" and isinstance(v, (int, float)):
            out["domain_prob"] = round(float(v), 4)
        elif k == "max_score" and isinstance(v, (int, float)):
            out["max_score"] = round(float(v), 4)
        else:
            out[k] = v
    if not out:
        out["status"] = "ok"
    out["node_latency_ms"] = elapsed_ms
    return out


def traced_node(node_name: str) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                state_arg = args[0] if args else kwargs.get("state")
                with mlflow.start_span(name=node_name, span_type="NODE") as span:
                    try:
                        span.set_inputs(_build_span_inputs(node_name, state_arg))
                    except Exception:
                        pass
                    result = await func(*args, **kwargs)
                    elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
                    try:
                        span.set_outputs(_build_span_outputs(node_name, result, elapsed))
                    except Exception:
                        pass
                result.setdefault("latency_ms", {})[node_name] = elapsed
                return result
            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            state_arg = args[0] if args else kwargs.get("state")
            with mlflow.start_span(name=node_name, span_type="NODE") as span:
                try:
                    span.set_inputs(_build_span_inputs(node_name, state_arg))
                except Exception:
                    pass
                result = func(*args, **kwargs)
                elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
                try:
                    span.set_outputs(_build_span_outputs(node_name, result, elapsed))
                except Exception:
                    pass
            result.setdefault("latency_ms", {})[node_name] = elapsed
            return result
        return sync_wrapper

    return decorator
