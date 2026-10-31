from collections import defaultdict
import math
import re
from typing import Any, Dict, List, Optional

PER_DOC_QUOTA_CAP = 2
TEMPORAL_ALPHA = 0.08
TEMPORAL_BASE_YEAR = 2015
RE_YEAR_IN_DOC = re.compile(r"/(\d{4})/")


def _extract_issued_year(candidate: Dict[str, Any]) -> int:
    yr = candidate.get("issued_year")
    if yr is not None:
        return int(yr)
    m = RE_YEAR_IN_DOC.search(candidate.get("doc_number") or "")
    return int(m.group(1)) if m else TEMPORAL_BASE_YEAR


def _article_key(candidate: Dict[str, Any]) -> Any:
    doc = candidate.get("doc_number") or ""
    art = candidate.get("article_code") or ""
    if doc and art:
        return (doc, art)
    return candidate.get("chunk_id") or id(candidate)


def pre_rerank_diversify(
    rrf_candidates: List[Dict[str, Any]],
    target_k: int = 12,
    soft_doc_cap: int = 3,
    doc_filters: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not rrf_candidates:
        return []

    ordered = sorted(
        enumerate(rrf_candidates),
        key=lambda item: (-float(item[1].get("rrf_score", 0.0)), item[0]),
    )

    best_by_article: Dict[Any, Dict[str, Any]] = {}
    article_order: List[Any] = []
    for _, cand in ordered:
        key = _article_key(cand)
        if key not in best_by_article:
            best_by_article[key] = cand
            article_order.append(key)
        elif float(cand.get("rrf_score", 0.0)) > float(best_by_article[key].get("rrf_score", 0.0)):
            best_by_article[key] = cand

    unique_articles = [best_by_article[k] for k in article_order]
    if doc_filters and len(doc_filters) == 1:
        return unique_articles[:target_k]

    selected: List[Dict[str, Any]] = []
    overflow: List[Dict[str, Any]] = []
    doc_counts: Dict[str, int] = defaultdict(int)

    for cand in unique_articles:
        doc = cand.get("doc_number") or ""
        if not doc or doc_counts[doc] < soft_doc_cap:
            selected.append(cand)
            if doc:
                doc_counts[doc] += 1
            if len(selected) == target_k:
                return selected
        else:
            overflow.append(cand)

    if len(selected) < target_k and overflow:
        selected.extend(overflow[: target_k - len(selected)])

    return selected[:target_k]


def apply_temporal_dedup_quota_and_pin(
    candidates: List[Dict[str, Any]],
    raw_logits: List[float],
    doc_filters: Optional[List[str]],
    db_service: Optional[Any],
    top_n: int = 5,
    return_all_deduped: bool = False,
) -> Any:
    if not candidates or not raw_logits:
        return ([], []) if return_all_deduped else []

    scored: List[Dict[str, Any]] = []
    for cand, raw_logit in zip(candidates, raw_logits):
        item = dict(cand)
        year = _extract_issued_year(item)
        logit_adj = float(raw_logit) + TEMPORAL_ALPHA * max(0, year - TEMPORAL_BASE_YEAR)
        clamped = max(-50.0, min(50.0, logit_adj))
        score = 1.0 / (1.0 + math.exp(-clamped))
        item["issued_year"] = year
        item["raw_logit"] = float(raw_logit)
        item["logit_adj"] = logit_adj
        item["rerank_score"] = score
        scored.append(item)

    best_by_article: Dict[Any, Dict[str, Any]] = {}
    for item in scored:
        key = _article_key(item)
        if key not in best_by_article or item["rerank_score"] > best_by_article[key]["rerank_score"]:
            best_by_article[key] = item

    deduped = sorted(best_by_article.values(), key=lambda x: x["rerank_score"], reverse=True)

    is_single_doc = bool(doc_filters and len(doc_filters) == 1)
    if is_single_doc:
        top_selected = deduped[:top_n]
    else:
        top_selected = []
        doc_counts: Dict[str, int] = defaultdict(int)
        for item in deduped:
            doc = item.get("doc_number") or ""
            if not doc or doc_counts[doc] < PER_DOC_QUOTA_CAP:
                top_selected.append(item)
                if doc:
                    doc_counts[doc] += 1
                if len(top_selected) == top_n:
                    break

    if db_service is not None and top_selected:
        doc_ids = [str(c["doc_id"]) for c in top_selected if c.get("doc_id")]
        existing_cids = {str(c.get("chunk_id", "")) for c in top_selected}
        art_codes = [str(c["article_code"]) for c in top_selected if c.get("article_code")]
        pinned = db_service.fetch_amending_provisions(
            doc_ids=doc_ids,
            existing_chunk_ids=existing_cids,
            article_codes=art_codes,
            limit=2,
        )
        base_pin_score = top_selected[0]["rerank_score"] * 0.99
        for p in pinned:
            p["rerank_score"] = base_pin_score
            top_selected.append(p)

    if return_all_deduped:
        return top_selected, deduped[:25]
    return top_selected

