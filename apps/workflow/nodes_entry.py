import json
from typing import Any, Dict, Literal
import numpy as np

from apps.services.resources import (
    get_descriptor_service,
    get_domain_probe,
    get_embedding_service,
    get_redis_client,
)
from apps.workflow.state import LegalRAGState, traced_node


@traced_node("PureRedisCacheCheckNode")
def PureRedisCacheCheckNode(state: LegalRAGState) -> Dict[str, Any]:
    r = get_redis_client()
    if not r:
        return {"cache_hit": False}
    sha = state.get("query_sha256")
    cached_raw = r.get(f"rag:cache:{sha}") if sha else None
    if cached_raw:
        data = json.loads(cached_raw)
        return {
            "cache_hit": True,
            "answer": data.get("answer", ""),
            "citations": data.get("citations", []),
            "abstained": data.get("abstained", False),
            "top_chunks": data.get("top_chunks", []),
            "execution_mode": "cache_hit"
        }
    return {"cache_hit": False}


@traced_node("ReturnCachedResponseNode")
def ReturnCachedResponseNode(state: LegalRAGState) -> Dict[str, Any]:
    return {"execution_mode": "cache_hit"}


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
    if state.get("is_future_year"):
        reason = "Văn bản hoặc mốc thời gian yêu cầu vượt quá năm hiện hành (2026)."
    else:
        reason = "Yêu cầu nằm ngoài phạm vi pháp luật Việt Nam hoặc không thuộc danh mục văn bản được hỗ trợ."
    answer = f"Hệ thống từ chối trả lời: {reason}"
    return {
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

