import os
from typing import Any, Dict, Optional, Set
from dotenv import load_dotenv
import numpy as np
from qdrant_client import QdrantClient
import redis

from apps.services.db_service import DatabaseService
from apps.services.descriptor_service import QueryDescriptorService
from apps.services.embedding_service import EmbeddingService, Int8Reranker

load_dotenv(".env")

_qdrant_client: Optional[QdrantClient] = None
_qdrant_indexed_fields: Optional[Set[str]] = None
_db_service: Optional[DatabaseService] = None
_embedding_service: Optional[EmbeddingService] = None
_int8_reranker: Optional[Int8Reranker] = None
_redis_client: Optional[redis.Redis] = None
_descriptor_service: Optional[QueryDescriptorService] = None
_domain_probe: Optional[Dict[str, Any]] = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        host = os.getenv("QDRANT_HOST", "").rstrip("/")
        api_key = os.getenv("QDRANT_API_KEY")
        _qdrant_client = QdrantClient(url=host, api_key=api_key, timeout=30.0)
    return _qdrant_client


def get_collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", "legal_chunks_v1")


def get_qdrant_indexed_fields() -> Set[str]:
    global _qdrant_indexed_fields
    if _qdrant_indexed_fields is None:
        info = get_qdrant_client().get_collection(get_collection_name())
        _qdrant_indexed_fields = set((info.payload_schema or {}).keys())
    return _qdrant_indexed_fields


def get_db_service() -> DatabaseService:
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        model_path = "models/bge-m3" if os.path.isdir("models/bge-m3") else "BAAI/bge-m3"
        _embedding_service = EmbeddingService(model_name_or_path=model_path, device="cpu", use_fp16=False)
    return _embedding_service


def get_int8_reranker() -> Int8Reranker:
    global _int8_reranker
    if _int8_reranker is None:
        env_path = os.getenv("RERANKER_MODEL_PATH", "")
        if env_path and os.path.exists(env_path):
            model_path = env_path
        elif os.path.isdir("models/bge-reranker-v2-m3"):
            model_path = "models/bge-reranker-v2-m3"
        else:
            model_path = "BAAI/bge-reranker-v2-m3"
        _int8_reranker = Int8Reranker(model_path=model_path)
    return _int8_reranker


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


def get_domain_probe() -> Dict[str, Any]:
    global _domain_probe
    if _domain_probe is None:
        path = "models/linear_domain_probe.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required linear domain probe model not found at {path}")
        data = np.load(path)
        _domain_probe = {"w": data["w"].astype(np.float32), "b": float(data["b"])}
    return _domain_probe

