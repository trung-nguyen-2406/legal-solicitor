import json
from typing import Any, Dict, Iterable, List, Optional
import redis

from apps.services.resources import get_redis_client

POSITIVE_CACHE_TTL_S = 86400
NEGATIVE_CACHE_TTL_S = 600
CACHE_KEY_PREFIX = "rag:cache:"
DOC_TAG_PREFIX = "rag:tag:doc:"
NEGATIVE_CACHE_TAG_KEY = "rag:tag:negative_cache"


def cache_key_for_sha(query_sha256: str) -> str:
    return f"{CACHE_KEY_PREFIX}{query_sha256}"


def doc_tag_key(doc_number: str) -> str:
    return f"{DOC_TAG_PREFIX}{doc_number.strip()}"


def set_cache_with_doc_tags(
    query_sha256: str,
    payload: Dict[str, Any],
    doc_numbers: Iterable[str],
    ttl_seconds: int = POSITIVE_CACHE_TTL_S,
    redis_client: Optional[redis.Redis] = None,
) -> bool:
    if not query_sha256:
        return False
    r = redis_client or get_redis_client()
    if r is None:
        return False

    unique_docs: List[str] = []
    seen = set()
    for d in doc_numbers:
        clean = (d or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            unique_docs.append(clean)

    serialized = json.dumps(payload, ensure_ascii=False)
    pipe = r.pipeline()
    pipe.setex(cache_key_for_sha(query_sha256), ttl_seconds, serialized)
    for d in unique_docs:
        tkey = doc_tag_key(d)
        pipe.sadd(tkey, query_sha256)
        pipe.expire(tkey, ttl_seconds)
    pipe.execute()
    return True


def set_negative_cache(
    query_sha256: str,
    payload: Dict[str, Any],
    ttl_seconds: int = NEGATIVE_CACHE_TTL_S,
    redis_client: Optional[redis.Redis] = None,
) -> bool:
    if not query_sha256:
        return False
    r = redis_client or get_redis_client()
    if r is None:
        return False

    serialized = json.dumps(payload, ensure_ascii=False)
    pipe = r.pipeline()
    pipe.setex(cache_key_for_sha(query_sha256), ttl_seconds, serialized)
    pipe.sadd(NEGATIVE_CACHE_TAG_KEY, query_sha256)
    pipe.execute()
    return True


def invalidate_by_doc(
    doc_number: str,
    redis_client: Optional[redis.Redis] = None,
) -> int:
    clean = (doc_number or "").strip()
    if not clean:
        return 0
    r = redis_client or get_redis_client()
    if r is None:
        return 0

    tag_keys = [doc_tag_key(clean)]
    if clean.upper() != clean:
        tag_keys.append(doc_tag_key(clean.upper()))

    cache_keys_to_delete = set()
    for tkey in tag_keys:
        members = r.smembers(tkey) or set()
        for member in members:
            m_str = str(member).strip()
            if not m_str:
                continue
            ckey = m_str if m_str.startswith(CACHE_KEY_PREFIX) else cache_key_for_sha(m_str)
            cache_keys_to_delete.add(ckey)

    all_keys = list(cache_keys_to_delete) + tag_keys
    if all_keys:
        r.delete(*all_keys)
    return len(cache_keys_to_delete)

