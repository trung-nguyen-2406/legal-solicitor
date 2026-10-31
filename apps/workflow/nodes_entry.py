import json
import time
from typing import Any, Callable, Dict, Literal
import uuid
from langgraph.config import get_stream_writer
import numpy as np

from apps.services.resources import (
    get_descriptor_service,
    get_domain_probe,
    get_embedding_service,
    get_redis_client,
)
from apps.workflow.state import LegalRAGState, traced_node


def _get_stream_writer_safe() -> Callable[[Any], None]:
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


def route_after_cache_check(state: LegalRAGState) -> Literal["cache_hit", "cache_miss"]:
    return "cache_hit" if state.get("cache_hit", False) else "cache_miss"


@traced_node("PureRedisCacheCheckNode")
def PureRedisCacheCheckNode(state: LegalRAGState) -> Dict[str, Any]:
    r = get_redis_client()
    sha = state.get("query_sha256")
    if not sha and state.get("query"):
        sha = get_descriptor_service().parse(state["query"])["query_sha256"]
    if not r or not sha:
        return {"cache_hit": False, "query_sha256": sha or ""}
    cached_raw = r.get(f"rag:cache:{sha}")
    if cached_raw:
        data = json.loads(cached_raw)
        return {
            "query_sha256": sha,
            "cache_hit": True,
            "answer": data.get("answer", ""),
            "citations": data.get("citations", []),
            "abstained": data.get("abstained", False),
            "abstain_reason": data.get("abstain_reason"),
            "top_chunks": data.get("top_chunks", []),
            "inventory_table": data.get("inventory_table"),
            "execution_mode": "cache_hit",
        }
    return {"cache_hit": False, "query_sha256": sha}


@traced_node("ReturnCachedResponseNode")
def ReturnCachedResponseNode(state: LegalRAGState) -> Dict[str, Any]:
    req_id = state.get("request_id") or str(uuid.uuid4())
    writer = _get_stream_writer_safe()
    writer({
        "event": "metadata",
        "data": {
            "request_id": req_id,
            "abstained": bool(state.get("abstained", False)),
            "citations": state.get("citations") or [],
        },
    })
    ans = state.get("answer") or ""
    if ans:
        writer({"event": "token", "data": {"delta": ans}})
    latency_map = dict(state.get("latency_ms") or {})
    start_time = state.get("start_time")
    total_ms = round((time.time() - float(start_time)) * 1000.0, 2) if start_time else round(sum(latency_map.values()), 2)
    writer({
        "event": "done",
        "data": {
            "request_id": req_id,
            "metrics": {
                "total_ms": total_ms,
                "retrieval_ms": 0,
                "rerank_ms": 0,
                "generation_ms": 0,
                "react_steps": 0,
                "execution_mode": "cache_hit",
                "cache_hit": True,
                "abstained": bool(state.get("abstained", False)),
                "node_latencies_ms": latency_map,
            },
        },
    })
    return {"request_id": req_id, "execution_mode": "cache_hit"}


@traced_node("StructuredQueryDescriptorNode")
def StructuredQueryDescriptorNode(state: LegalRAGState) -> Dict[str, Any]:
    descriptor = get_descriptor_service()
    return descriptor.parse(state["query"])


@traced_node("EncodeQueryAndLinearDomainProbeNode")
def EncodeQueryAndLinearDomainProbeNode(state: LegalRAGState) -> Dict[str, Any]:
    q = state.get("cleaned_query") or state.get("query", "")
    embed = get_embedding_service()
    enc = embed.encode(q)
    dense_vec = enc["dense"]
    sparse_ind = enc["sparse"]["indices"]
    sparse_val = enc["sparse"]["values"]

    probe = get_domain_probe()
    w, b = probe["w"], probe["b"]
    logit = float(np.dot(w, np.array(dense_vec, dtype=np.float32)) + b)
    domain_prob = float(1.0 / (1.0 + np.exp(-logit)))

    return {
        "dense_vector": dense_vec,
        "sparse_indices": sparse_ind,
        "sparse_values": sparse_val,
        "domain_prob": domain_prob
    }


def route_after_domain_probe(state: LegalRAGState) -> Literal["fast_abstain", "fast_retrieval"]:
    if state.get("is_future_year", False):
        return "fast_abstain"
    if (
        state.get("domain_prob", 1.0) < 0.15
        and len(state.get("doc_filters", [])) == 0
        and not state.get("is_foreign_lang", False)
    ):
        return "fast_abstain"
    return "fast_retrieval"


@traced_node("AbstainFastExitNode")
def AbstainFastExitNode(state: LegalRAGState) -> Dict[str, Any]:
    req_id = state.get("request_id") or str(uuid.uuid4())
    if state.get("is_future_year"):
        reason = "Văn bản hoặc mốc thời gian yêu cầu vượt quá năm hiện hành (2026)."
    else:
        reason = "Yêu cầu nằm ngoài phạm vi pháp luật Việt Nam hoặc không thuộc danh mục văn bản được hỗ trợ."
    answer = f"Hệ thống từ chối trả lời: {reason}"

    writer = _get_stream_writer_safe()
    writer({
        "event": "metadata",
        "data": {
            "request_id": req_id,
            "abstained": True,
            "citations": [],
        },
    })
    writer({"event": "token", "data": {"delta": answer}})

    return {
        "request_id": req_id,
        "abstained": True,
        "abstain_reason": reason,
        "answer": answer,
        "top_chunks": [],
        "citations": [],
        "execution_mode": "fast_abstain"
    }


@traced_node("WriteShortTtlNegativeCacheNode")
def WriteShortTtlNegativeCacheNode(state: LegalRAGState) -> Dict[str, Any]:
    r = get_redis_client()
    sha = state.get("query_sha256")
    if r and sha:
        payload = {
            "query": state.get("query"),
            "abstained": state.get("abstained", True),
            "abstain_reason": state.get("abstain_reason"),
            "answer": state.get("answer", ""),
            "citations": [],
            "top_chunks": []
        }
        cache_key = f"rag:cache:{sha}"
        r.set(cache_key, json.dumps(payload, ensure_ascii=False), ex=600)
        r.sadd("rag:tag:negative_cache", cache_key)
    return {}

