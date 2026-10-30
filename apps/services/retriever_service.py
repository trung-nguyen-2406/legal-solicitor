import re
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from apps.services.embedding_service import EmbeddingService, Int8Reranker

RE_DOC_NUMBER = re.compile(r"\b(\d+/\d{4}/[A-ZĐ0-9\-]+)\b")
RE_ARTICLE = re.compile(r"\bĐiều\s+(\d+[a-zđ]?)\b", re.IGNORECASE)

class RetrieverService:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_service: EmbeddingService,
        reranker_service: Int8Reranker,
        collection_name: str = "legal_chunks_v1"
    ):
        self.client = qdrant_client
        self.embed_service = embedding_service
        self.reranker_service = reranker_service
        self.collection_name = collection_name

    def extract_query_metadata(self, query: str) -> Dict[str, Optional[str]]:
        doc_num_match = RE_DOC_NUMBER.search(query)
        article_match = RE_ARTICLE.search(query)
        return {
            "doc_number": doc_num_match.group(1) if doc_num_match else None,
            "article_id": f"Điều {article_match.group(1)}" if article_match else None
        }

    def _build_filter(self, metadata: Dict[str, Optional[str]]) -> models.Filter:
        must_conditions = []
        if metadata.get("doc_number"):
            must_conditions.append(
                models.FieldCondition(
                    key="doc_number",
                    match=models.MatchValue(value=metadata["doc_number"])
                )
            )
        if metadata.get("article_id"):
            must_conditions.append(
                models.FieldCondition(
                    key="article_code",
                    match=models.MatchValue(value=metadata["article_id"])
                )
            )

        must_not_conditions = [
            models.FieldCondition(
                key="effective_status",
                match=models.MatchValue(value="expired")
            )
        ]

        return models.Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions
        )

    def hybrid_search_rrf(
        self,
        query: str,
        top_k: int = 25,
        k_rrf: int = 60
    ) -> List[Dict[str, Any]]:
        meta = self.extract_query_metadata(query)
        qdrant_filter = self._build_filter(meta)
        encoded = self.embed_service.encode(query)

        dense_vector = encoded["dense"]
        sparse_indices = encoded["sparse"]["indices"]
        sparse_values = encoded["sparse"]["values"]

        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using="dense_vector",
                filter=qdrant_filter,
                limit=top_k
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse_vector",
                filter=qdrant_filter,
                limit=top_k
            )
        ]

        points = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True
        ).points

        candidates = []
        for rank, pt in enumerate(points):
            payload = pt.payload or {}
            candidates.append({
                "chunk_id": str(pt.id),
                "doc_id": payload.get("doc_id"),
                "doc_number": payload.get("doc_number"),
                "doc_type": payload.get("doc_type"),
                "issued_year": payload.get("issued_year"),
                "effective_status": payload.get("effective_status"),
                "article_code": payload.get("article_code"),
                "clause_code": payload.get("clause_code"),
                "breadcrumb": payload.get("breadcrumb"),
                "raw_content": payload.get("raw_content", payload.get("breadcrumb", "")),
                "rrf_score": pt.score,
                "rank_order": rank + 1
            })

        return candidates

    def retrieve_and_rerank(
        self,
        query: str,
        top_k_retrieval: int = 25,
        top_n_rerank: int = 5,
        threshold: float = 0.35
    ) -> Dict[str, Any]:
        candidates = self.hybrid_search_rrf(query, top_k=top_k_retrieval)
        reranked = self.reranker_service.rerank(
            query=query,
            candidates=candidates,
            top_n=top_n_rerank,
            threshold=threshold
        )
        return {
            "query": query,
            "metadata_filters": self.extract_query_metadata(query),
            "candidates_count": len(candidates),
            "top_candidates": candidates,
            "reranked_chunks": reranked
        }