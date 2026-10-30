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
        if cid not in merged or float(chunk.get("rerank_score", 0.0)) > float(merged[cid].get("rerank_score", 0.0)):
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
    max_score: float
    abstain_reason: Optional[str]
    planner_decision: Optional[Dict[str, Any]]
    matched_subqueries: Dict[str, str]


def traced_node(node_name: str) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                with mlflow.start_span(name=node_name, span_type="NODE"):
                    result = await func(*args, **kwargs)
                elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
                result.setdefault("latency_ms", {})[node_name] = elapsed
                return result
            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            with mlflow.start_span(name=node_name, span_type="NODE"):
                result = func(*args, **kwargs)
            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            result.setdefault("latency_ms", {})[node_name] = elapsed
            return result
        return sync_wrapper

    return decorator
