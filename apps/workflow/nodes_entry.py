import json
import os
from typing import Any, Dict, Literal, Optional
import numpy as np
import redis

from apps.services.descriptor_service import QueryDescriptorService
from apps.services.embedding_service import EmbeddingService
from apps.workflow.state import LegalRAGState, traced_node

_redis_client: Optional[redis.Redis] = None
_descriptor_service: Optional[QueryDescriptorService] = None
_embedding_service: Optional[EmbeddingService] = None
_domain_probe: Optional[Dict[str, Any]] = None


def get_redis_client() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client is None:
        dsn = os.getenv("REDIS_DSN") or os.getenv("UPSTASH_REDIS_REST_URL")
        if dsn:
            _redis_client = redis.from_url(dsn, decode_responses=True)
    return _redis_client


def get_descriptor_service() -> QueryDescriptorService:
    global _descriptor_service
    if _descriptor_service is None:
        _descriptor_service = QueryDescriptorService()
    return _descriptor_service


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        model_path = "models/bge-m3" if os.path.isdir("models/bge-m3") else "BAAI/bge-m3"
        _embedding_service = EmbeddingService(model_name_or_path=model_path, device="cpu", use_fp16=False)
    return _embedding_service


def get_domain_probe() -> Dict[str, Any]:
    global _domain_probe
    if _domain_probe is None:
        path = "models/linear_domain_probe.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required linear domain probe model not found at {path}")
        data = np.load(path)
        _domain_probe = {"w": data["w"].astype(np.float32), "b": float(data["b"])}
    return _domain_probe


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

