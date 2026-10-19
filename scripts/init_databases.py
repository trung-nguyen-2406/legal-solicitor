import os
import subprocess
from dotenv import load_dotenv
import httpx

load_dotenv()

def init_postgres():
    dsn = os.getenv("DATABASE_DSN")
    if not dsn:
        raise ValueError("DATABASE_DSN not set")
    res = subprocess.run(["psql", dsn, "-f", "scripts/schema.sql"], capture_output=True, text=True)

def init_qdrant():
    host = os.getenv("QDRANT_HOST", "").rstrip("/")
    api_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "legal_chunks_v1")

    headers = {"api-key": api_key, "Content-Type": "application/json"}
    collection_config = {
        "vectors": {
            "dense_vector": {
                "size": 1024,
                "distance": "Cosine",
                "hnsw_config": {"m": 16, "ef_construct": 200},
                "quantization_config": {
                    "scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}
                }
            }
        },
        "sparse_vectors": {"sparse_vector": {"index": {"on_disk": False}}}
    }

    with httpx.Client(base_url=host, headers=headers, timeout=30.0) as client:
        client.delete(f"/collections/{collection}")
        resp = client.put(f"/collections/{collection}", json=collection_config)
        resp.raise_for_status()

        indexes = [
            ("doc_id", "uuid"),
            ("doc_number", "keyword"),
            ("doc_type", "keyword"),
            ("issued_year", "integer"),
            ("effective_status", "keyword"),
            ("article_code", "keyword"),
            ("clause_code", "keyword"),
            ("chunk_sha256", "keyword"),
        ]
        for field, schema in indexes:
            client.put(f"/collections/{collection}/index", json={"field_name": field, "field_schema": schema}).raise_for_status()

if __name__ == "__main__":
    init_postgres()
    init_qdrant()
