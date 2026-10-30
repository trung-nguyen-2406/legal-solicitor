from collections import defaultdict
import os
import re
from typing import Any, Dict, List, Literal, Optional, Set
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from apps.services.db_service import DatabaseService
from apps.services.embedding_service import Int8Reranker
from apps.services.post_rerank_service import (
    apply_temporal_dedup_quota_and_pin,
    pre_rerank_diversify,
)
from apps.workflow.state import LegalRAGState, traced_node

PREFETCH_LIMIT_K = 50
STRATIFIED_MIN_QUOTA_PER_DOC = 5
AGG_SCORE_THRESHOLD = 0.52
AGG_LIMIT = 200
AGG_VERBATIM_LIMIT = 15

RE_NON_LATIN = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u0e00-\u0e7f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
RE_YEAR_FILTER = re.compile(r"\bnăm\s+(19\d{2}|20\d{2})\b", re.IGNORECASE)
LINEAGE_KEYWORDS = ("sửa đổi", "bổ sung", "thay thế", "bãi bỏ", "lần sửa đổi", "lịch sử hiệu lực")
LEGAL_FIELD_KEYWORDS = (
    "xây dựng", "đô thị", "thương mại", "doanh nghiệp", "đầu tư", "chứng khoán",
    "bảo hiểm", "tiền tệ", "ngân hàng", "thuế", "tài chính", "lao động", "giáo dục",
    "y tế", "giao thông", "tài nguyên", "môi trường", "hình sự", "dân sự", "hành chính",
)

_qdrant_client: Optional[QdrantClient] = None
_qdrant_indexed_fields: Optional[Set[str]] = None
_db_service: Optional[DatabaseService] = None
_int8_reranker: Optional[Int8Reranker] = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        load_dotenv(".env")
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


def get_int8_reranker() -> Int8Reranker:
    global _int8_reranker
    if _int8_reranker is None:
        load_dotenv(".env")
        env_path = os.getenv("RERANKER_MODEL_PATH", "")
        if env_path and os.path.exists(env_path):
            model_path = env_path
        elif os.path.isdir("models/bge-reranker-v2-m3"):
            model_path = "models/bge-reranker-v2-m3"
        else:
            model_path = "BAAI/bge-reranker-v2-m3"
        _int8_reranker = Int8Reranker(model_path=model_path)
    return _int8_reranker


def _point_to_dict(
    pt: Any,
    rank: int = 1,
    rrf_override: Optional[float] = None,
    rerank_override: Optional[float] = None,
) -> Dict[str, Any]:
    payload = pt.payload or {}
    doc_number = payload.get("doc_number") or ""
    art_code = payload.get("article_code") or ""
    raw_score = float(getattr(pt, "score", 0.0) or 0.0)
    rrf_score = rrf_override if rrf_override is not None else raw_score
    rerank_score = rerank_override if rerank_override is not None else 0.0
    return {
        "chunk_id": str(pt.id),
        "doc_id": payload.get("doc_id") or "",
        "doc_number": doc_number,
        "doc_title": payload.get("doc_title") or "",
        "doc_type": payload.get("doc_type") or "",
        "issued_year": payload.get("issued_year"),
        "effective_status": payload.get("effective_status") or "active",
        "legal_field": payload.get("legal_field") or "",
        "article_code": art_code,
        "clause_code": payload.get("clause_code") or "ALL",
        "breadcrumb": payload.get("breadcrumb") or "",
        "raw_content": payload.get("raw_content") or "",
        "anchor_tag": payload.get("anchor_tag") or f"[{doc_number}:{art_code}]",
        "rrf_score": rrf_score,
        "rerank_score": rerank_score,
        "rank_order": rank,
    }


def _active_filter(
    doc_numbers: Optional[List[str]] = None,
    article_code: Optional[str] = None,
    issued_year: Optional[int] = None,
    legal_field: Optional[str] = None,
) -> models.Filter:
    must: List[Any] = []
    if doc_numbers:
        if len(doc_numbers) == 1:
            must.append(models.FieldCondition(key="doc_number", match=models.MatchValue(value=doc_numbers[0])))
        else:
            must.append(models.FieldCondition(key="doc_number", match=models.MatchAny(any=doc_numbers)))
    if article_code:
        must.append(models.FieldCondition(key="article_code", match=models.MatchValue(value=article_code)))
    if issued_year is not None:
        must.append(models.FieldCondition(key="issued_year", match=models.MatchValue(value=int(issued_year))))
    if legal_field and "legal_field" in get_qdrant_indexed_fields():
        must.append(models.FieldCondition(key="legal_field", match=models.MatchValue(value=legal_field)))

    must_not = [models.FieldCondition(key="effective_status", match=models.MatchValue(value="expired"))]
    return models.Filter(must=must or None, must_not=must_not)


def _build_inventory_table(chunks: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    grouped: Dict[str, List[str]] = defaultdict(list)
    meta_by_doc: Dict[str, str] = {}
    for c in chunks:
        doc = c.get("doc_number") or "N/A"
        art = c.get("article_code") or c.get("clause_code") or "Điều ?"
        if art not in grouped[doc]:
            grouped[doc].append(art)
        if doc not in meta_by_doc:
            yr = c.get("issued_year") or ""
            st = c.get("effective_status") or "active"
            meta_by_doc[doc] = f"{yr} | {st}"

    lines = [
        f"Tổng số đoạn trích thỏa mãn: {len(chunks)} trên {len(grouped)} văn bản.",
        "| Số hiệu văn bản | Năm / Trạng thái | Các điều khoản liên quan |",
        "|---|---|---|",
    ]
    total_len = sum(len(x) for x in lines)
    for doc, arts in grouped.items():
        art_summary = ", ".join(arts[:8])
        if len(arts) > 8:
            art_summary += f" (+{len(arts) - 8} điều)"
        row = f"| {doc} | {meta_by_doc[doc]} | {art_summary} |"
        if total_len + len(row) > max_chars:
            lines.append("| ... | ... | (Đã rút gọn để giữ dưới 600 tokens) |")
            break
        lines.append(row)
        total_len += len(row)
    return "\n".join(lines)


def route_fast_path_branch(
    state: LegalRAGState,
) -> Literal["direct_point", "aggregation", "stratified_multi", "standard_hybrid"]:
    doc_filters = state.get("doc_filters") or []
    article_filter = state.get("article_filter")
    is_aggregation = bool(state.get("is_aggregation", False))

    if len(doc_filters) == 1 and article_filter is not None and not is_aggregation:
        return "direct_point"
    if is_aggregation:
        return "aggregation"
    if len(doc_filters) >= 2:
        return "stratified_multi"
    return "standard_hybrid"


@traced_node("DirectPointGetNode")
def DirectPointGetNode(state: LegalRAGState) -> Dict[str, Any]:
    doc_filters = state.get("doc_filters") or []
    article_filter = state.get("article_filter")
    if not doc_filters or not article_filter:
        return {
            "fast_branch": "direct_point",
            "execution_mode": "fast_path",
            "top_chunks": [],
            "max_score": 0.0,
        }

    qfilter = _active_filter(doc_numbers=[doc_filters[0]], article_code=article_filter)
    pts, _ = get_qdrant_client().scroll(
        collection_name=get_collection_name(),
        scroll_filter=qfilter,
        limit=20,
        with_payload=True,
    )

    raw_chunks = [_point_to_dict(pt, rank=idx + 1, rrf_override=1.0, rerank_override=1.0) for idx, pt in enumerate(pts)]
    hydrated = get_db_service().hydrate_chunks(raw_chunks)
    bank = {c["chunk_id"]: c for c in hydrated}
    max_score = 1.0 if hydrated else 0.0

    return {
        "fast_branch": "direct_point",
        "execution_mode": "fast_path",
        "top_chunks": hydrated,
        "chunk_bank": bank,
        "max_score": max_score,
    }


@traced_node("BiDirectionalGraphExpandNode")
def BiDirectionalGraphExpandNode(state: LegalRAGState) -> Dict[str, Any]:
    base_chunks = list(state.get("top_chunks") or [])
    if not base_chunks:
        return {
            "top_chunks": [],
            "max_score": 0.0,
        }

    expanded = get_db_service().expand_bidirectional(base_chunks, limit=3)
    for item in expanded:
        item["rerank_score"] = 0.95
        item["is_graph_expanded"] = True

    combined = base_chunks + expanded
    bank = {c["chunk_id"]: c for c in combined}
    max_score = 1.0 if state.get("fast_branch") == "direct_point" else max(
        (float(c.get("rerank_score", 0.0)) for c in combined), default=0.0
    )

    return {
        "top_chunks": combined,
        "chunk_bank": bank,
        "max_score": max_score,
        "pointer_scratchpad": combined,
    }


@traced_node("FilteredSemanticOrLineageAggNode")
def FilteredSemanticOrLineageAggNode(state: LegalRAGState) -> Dict[str, Any]:
    query = state.get("query") or ""
    q_lower = query.lower()
    doc_filters = state.get("doc_filters") or []
    db = get_db_service()

    asks_lineage = bool(doc_filters) and any(kw in q_lower for kw in LINEAGE_KEYWORDS)
    agg_chunks: List[Dict[str, Any]] = []
    if asks_lineage:
        agg_chunks = db.fetch_amendment_lineage_cte(doc_filters[0], limit=AGG_LIMIT)

    if not agg_chunks:
        yr_m = RE_YEAR_FILTER.search(query)
        issued_year = int(yr_m.group(1)) if yr_m else None
        detected_field = next((kw for kw in LEGAL_FIELD_KEYWORDS if kw in q_lower), None)

        qfilter = _active_filter(
            doc_numbers=doc_filters or None,
            issued_year=issued_year,
            legal_field=detected_field,
        )
        pts = get_qdrant_client().query_points(
            collection_name=get_collection_name(),
            query=state["dense_vector"],
            using="dense_vector",
            query_filter=qfilter,
            score_threshold=AGG_SCORE_THRESHOLD,
            limit=AGG_LIMIT,
            with_payload=True,
        ).points

        raw_chunks = [
            _point_to_dict(pt, rank=idx + 1, rerank_override=float(getattr(pt, "score", 0.0) or 0.0))
            for idx, pt in enumerate(pts)
        ]
        hydrated = db.hydrate_chunks(raw_chunks)
        if detected_field and "legal_field" not in get_qdrant_indexed_fields():
            field_filtered = [c for c in hydrated if detected_field in (c.get("legal_field") or "").lower()]
            if field_filtered:
                hydrated = field_filtered
        agg_chunks = hydrated

    bank = {c["chunk_id"]: c for c in agg_chunks}
    if len(agg_chunks) > AGG_VERBATIM_LIMIT:
        inventory_table: Optional[str] = _build_inventory_table(agg_chunks)
        top_chunks = agg_chunks[:5]
    else:
        inventory_table = None
        top_chunks = agg_chunks

    max_score = max((float(c.get("rerank_score", 0.0)) for c in agg_chunks), default=0.0)
    return {
        "fast_branch": "aggregation",
        "execution_mode": "fast_path",
        "inventory_table": inventory_table,
        "top_chunks": top_chunks,
        "chunk_bank": bank,
        "max_score": max_score,
        "pointer_scratchpad": top_chunks,
    }


def _hybrid_or_dense_query(
    state: LegalRAGState,
    qfilter: models.Filter,
    limit: int,
) -> List[Any]:
    client = get_qdrant_client()
    collection = get_collection_name()
    dense_vec = state["dense_vector"]
    sparse_ind = state.get("sparse_indices") or []
    sparse_val = state.get("sparse_values") or []
    query_str = state.get("query") or ""

    non_latin = bool(state.get("is_foreign_lang")) and bool(RE_NON_LATIN.search(query_str))
    if non_latin or not sparse_ind:
        return client.query_points(
            collection_name=collection,
            query=dense_vec,
            using="dense_vector",
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
        ).points

    prefetch = [
        models.Prefetch(query=dense_vec, using="dense_vector", filter=qfilter, limit=limit),
        models.Prefetch(
            query=models.SparseVector(indices=sparse_ind, values=sparse_val),
            using="sparse_vector",
            filter=qfilter,
            limit=limit,
        ),
    ]
    return client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points


@traced_node("StratifiedMinQuotaSearchNode")
def StratifiedMinQuotaSearchNode(state: LegalRAGState) -> Dict[str, Any]:
    doc_filters = state.get("doc_filters") or []
    seen_ids: Set[str] = set()
    merged: List[Dict[str, Any]] = []

    per_doc_limit = max(STRATIFIED_MIN_QUOTA_PER_DOC * 2, 10)
    for doc_num in doc_filters:
        d_filter = _active_filter(doc_numbers=[doc_num])
        pts = _hybrid_or_dense_query(state, d_filter, limit=per_doc_limit)
        added_for_doc = 0
        for pt in pts:
            cid = str(pt.id)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            boost_rrf = 1.0 + float(getattr(pt, "score", 0.0) or 0.0)
            merged.append(_point_to_dict(pt, rank=len(merged) + 1, rrf_override=boost_rrf))
            added_for_doc += 1
            if added_for_doc >= STRATIFIED_MIN_QUOTA_PER_DOC:
                break

    global_pts = _hybrid_or_dense_query(state, _active_filter(), limit=PREFETCH_LIMIT_K)
    for pt in global_pts:
        cid = str(pt.id)
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        merged.append(_point_to_dict(pt, rank=len(merged) + 1))
        if len(merged) >= PREFETCH_LIMIT_K + len(doc_filters) * STRATIFIED_MIN_QUOTA_PER_DOC:
            break

    bank = {c["chunk_id"]: c for c in merged}
    return {
        "fast_branch": "stratified_multi",
        "execution_mode": "fast_path",
        "rrf_candidates": merged,
        "chunk_bank": bank,
    }


@traced_node("ScopedHybridRrfPrefetch50Node")
def ScopedHybridRrfPrefetch50Node(state: LegalRAGState) -> Dict[str, Any]:
    doc_filters = state.get("doc_filters") or []
    article_filter = state.get("article_filter")
    qfilter = _active_filter(
        doc_numbers=doc_filters if len(doc_filters) == 1 else None,
        article_code=article_filter,
    )

    pts = _hybrid_or_dense_query(state, qfilter, limit=PREFETCH_LIMIT_K)
    candidates = [_point_to_dict(pt, rank=idx + 1) for idx, pt in enumerate(pts)]
    bank = {c["chunk_id"]: c for c in candidates}

    return {
        "fast_branch": "standard_hybrid",
        "execution_mode": "fast_path",
        "rrf_candidates": candidates,
        "chunk_bank": bank,
    }


@traced_node("Top10PruneAndInt8RerankNode")
def Top10PruneAndInt8RerankNode(state: LegalRAGState) -> Dict[str, Any]:
    rrf_candidates = state.get("rrf_candidates") or list((state.get("chunk_bank") or {}).values())
    if not rrf_candidates:
        return {"rrf_candidates": [], "raw_logits": []}

    diversified = pre_rerank_diversify(
        rrf_candidates=rrf_candidates,
        target_k=12,
        soft_doc_cap=3,
        doc_filters=state.get("doc_filters"),
    )
    hydrated = get_db_service().hydrate_chunks(diversified)
    query_for_rerank = state.get("cleaned_query") or state.get("query") or ""
    raw_logits = get_int8_reranker().rerank_logits(query_for_rerank, hydrated)

    return {
        "rrf_candidates": hydrated,
        "raw_logits": raw_logits,
    }


@traced_node("DedupQuotaAndTemporalBoostNode")
def DedupQuotaAndTemporalBoostNode(state: LegalRAGState) -> Dict[str, Any]:
    candidates = state.get("rrf_candidates") or []
    raw_logits = state.get("raw_logits") or []
    if not candidates or not raw_logits:
        return {
            "top_chunks": [],
            "max_score": 0.0,
        }

    top_chunks = apply_temporal_dedup_quota_and_pin(
        candidates=candidates,
        raw_logits=raw_logits,
        doc_filters=state.get("doc_filters") or [],
        db_service=get_db_service(),
        top_n=5,
    )

    non_pinned_scores = [float(c.get("rerank_score", 0.0)) for c in top_chunks if not c.get("is_pinned_amendment")]
    max_score = max(non_pinned_scores, default=0.0)

    bank_updates: Dict[str, Dict[str, Any]] = {c["chunk_id"]: c for c in top_chunks}
    return {
        "top_chunks": top_chunks,
        "max_score": max_score,
        "chunk_bank": bank_updates,
        "pointer_scratchpad": [
            {
                "chunk_id": c["chunk_id"],
                "doc_number": c.get("doc_number", ""),
                "article_code": c.get("article_code", ""),
                "rerank_score": c.get("rerank_score", 0.0),
                "breadcrumb": c.get("breadcrumb", ""),
            }
            for c in top_chunks
        ],
    }
