from collections import defaultdict
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
import numpy as np
from qdrant_client import models

from apps.services.post_rerank_service import (
    apply_temporal_dedup_quota_and_pin,
    pre_rerank_diversify,
)
from apps.services.resources import (
    get_collection_name,
    get_db_service,
    get_embedding_service,
    get_int8_reranker,
    get_qdrant_client,
    get_qdrant_indexed_fields,
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
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
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
    if year_from is not None and year_to is not None:
        if year_from == year_to:
            must.append(models.FieldCondition(key="issued_year", match=models.MatchValue(value=int(year_from))))
        else:
            must.append(models.FieldCondition(key="issued_year", range=models.Range(gte=int(year_from), lte=int(year_to))))
    elif issued_year is not None:
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
    if asks_lineage:
        agg_chunks = db.fetch_amendment_lineage_cte(doc_filters[0], limit=AGG_LIMIT)
        bank = {c["chunk_id"]: c for c in agg_chunks}
        top_chunks = agg_chunks[:5] if len(agg_chunks) > 5 else agg_chunks
        max_score = max((float(c.get("rerank_score", 0.0)) for c in agg_chunks), default=0.0)
        return {
            "fast_branch": "aggregation",
            "execution_mode": "fast_path",
            "inventory_table": _build_inventory_table(agg_chunks),
            "top_chunks": top_chunks,
            "chunk_bank": bank,
            "max_score": max_score,
            "exact_agg_count": len(agg_chunks),
            "agg_catalog_items": agg_chunks,
            "pointer_scratchpad": top_chunks,
        }

    sub_queries = state.get("sub_queries") or [state.get("cleaned_query") or query]
    anchor_keywords = state.get("anchor_keywords") or []
    year_from = state.get("year_from")
    year_to = state.get("year_to")
    detected_field = next((kw for kw in LEGAL_FIELD_KEYWORDS if kw in q_lower), None)

    qfilter = _active_filter(
        doc_numbers=doc_filters or None,
        year_from=year_from,
        year_to=year_to,
        legal_field=detected_field,
    )

    client = get_qdrant_client()
    collection = get_collection_name()
    embed_svc = get_embedding_service()

    sweep_payload_fields = [
        "doc_id", "doc_number", "doc_title", "article_code",
        "clause_code", "issued_year", "legal_field", "breadcrumb", "anchor_tag"
    ]

    all_sweep_candidates: List[Dict[str, Any]] = []

    for sq in sub_queries:
        enc = embed_svc.encode(sq)
        sq_dense = enc["dense"]
        sq_sparse_ind = enc["sparse"]["indices"]
        sq_sparse_val = enc["sparse"]["values"]

        prefetch = [
            models.Prefetch(query=sq_dense, using="dense_vector", filter=qfilter, limit=300),
            models.Prefetch(
                query=models.SparseVector(indices=sq_sparse_ind, values=sq_sparse_val),
                using="sparse_vector",
                filter=qfilter,
                limit=300,
            ),
        ]
        pts = client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=300,
            with_payload=sweep_payload_fields,
        ).points

        if not pts:
            continue

        top1_rrf_score = float(getattr(pts[0], "score", 0.0) or 0.0)
        elbow_threshold = 0.72 * top1_rrf_score

        retained = [pt for pt in pts if float(getattr(pt, "score", 0.0) or 0.0) >= elbow_threshold]

        for idx, pt in enumerate(retained):
            c_dict = _point_to_dict(pt, rank=idx + 1)
            if anchor_keywords:
                title_txt = (c_dict.get("doc_title") or "").lower()
                crumb_txt = (c_dict.get("breadcrumb") or "").lower()
                combined_txt = f"{title_txt} {crumb_txt}"
                if not any(ak in combined_txt for ak in anchor_keywords):
                    continue
            all_sweep_candidates.append(c_dict)

    deduped_by_article: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cand in all_sweep_candidates:
        doc = cand.get("doc_number") or ""
        art = cand.get("article_code") or ""
        key = (doc, art)
        if key not in deduped_by_article or float(cand.get("rrf_score", 0.0)) > float(deduped_by_article[key].get("rrf_score", 0.0)):
            deduped_by_article[key] = cand

    sorted_articles = sorted(deduped_by_article.values(), key=lambda c: float(c.get("rrf_score", 0.0)), reverse=True)
    exact_agg_count = len(sorted_articles)

    top_5_diverse: List[Dict[str, Any]] = []
    seen_docs: Set[str] = set()
    for art in sorted_articles:
        doc = art.get("doc_number") or ""
        if doc and doc not in seen_docs:
            top_5_diverse.append(art)
            seen_docs.add(doc)
            if len(top_5_diverse) == 5:
                break
    if len(top_5_diverse) < 5:
        for art in sorted_articles:
            if art not in top_5_diverse:
                top_5_diverse.append(art)
                if len(top_5_diverse) == 5:
                    break

    hydrated_top_5 = db.hydrate_chunks(top_5_diverse) if top_5_diverse else []

    max_score = 0.0
    if hydrated_top_5:
        top_3 = hydrated_top_5[:3]
        query_for_rerank = state.get("cleaned_query") or query
        raw_logits = get_int8_reranker().rerank_logits(query_for_rerank, top_3)
        for c, lg in zip(top_3, raw_logits):
            sc = 1.0 / (1.0 + np.exp(-float(lg)))
            c["rerank_score"] = round(float(sc), 4)
        rerank_scores = [float(c.get("rerank_score", 0.0)) for c in top_3]
        max_score = max(rerank_scores, default=0.0)

    inventory_table = _build_inventory_table(sorted_articles) if sorted_articles else None
    bank = {c["chunk_id"]: c for c in hydrated_top_5}

    return {
        "fast_branch": "aggregation",
        "execution_mode": "fast_path",
        "inventory_table": inventory_table,
        "top_chunks": hydrated_top_5,
        "chunk_bank": bank,
        "max_score": max_score,
        "exact_agg_count": exact_agg_count,
        "agg_catalog_items": sorted_articles,
        "pointer_scratchpad": hydrated_top_5,
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
        target_k=25,
        soft_doc_cap=5,
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
            "retrieved_top_25": [],
            "max_score": 0.0,
        }

    top_chunks, deduped_25 = apply_temporal_dedup_quota_and_pin(
        candidates=candidates,
        raw_logits=raw_logits,
        doc_filters=state.get("doc_filters") or [],
        db_service=get_db_service(),
        top_n=5,
        return_all_deduped=True,
    )

    non_pinned_scores = [float(c.get("rerank_score", 0.0)) for c in top_chunks if not c.get("is_pinned_amendment")]
    max_score = max(non_pinned_scores, default=0.0)

    bank_updates: Dict[str, Dict[str, Any]] = {c["chunk_id"]: c for c in top_chunks}
    for c in deduped_25:
        if c["chunk_id"] not in bank_updates:
            bank_updates[c["chunk_id"]] = c

    return {
        "top_chunks": top_chunks,
        "retrieved_top_25": deduped_25,
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
