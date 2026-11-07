import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

EXACT_51_IDS = [
    "EVAL_001", "EVAL_002", "EVAL_003", "EVAL_004", "EVAL_059", "EVAL_060", "EVAL_061", "EVAL_062",
    "EVAL_089", "EVAL_090", "EVAL_091", "EVAL_092", "EVAL_101", "EVAL_102", "EVAL_103", "EVAL_104",
    "EVAL_141", "EVAL_142", "EVAL_143", "EVAL_144", "EVAL_171", "EVAL_172", "EVAL_173", "EVAL_174",
    "EVAL_191", "EVAL_192", "EVAL_193", "EVAL_194", "EVAL_196", "EVAL_197", "EVAL_198", "EVAL_199",
    "EVAL_204", "EVAL_205", "EVAL_206", "EVAL_207", "EVAL_208", "EVAL_209", "EVAL_210", "EVAL_211",
    "EVAL_212", "EVAL_213", "EVAL_214", "EVAL_226", "EVAL_227", "EVAL_228", "EVAL_229", "EVAL_236",
    "EVAL_237", "EVAL_238", "EVAL_239",
]

RE_DOC_NUMBER = re.compile(r"\b\d+[a-zA-Z]?/\d{4}/[\w\-]+|\b\d+-CP\b", re.IGNORECASE)
RE_ARTICLE_MENTION = re.compile(r"\bĐiều\s+\d+[a-zA-ZđĐ]?\b", re.IGNORECASE)
RE_CITATION = re.compile(
    r"(Điều\s+\d+[a-zA-ZđĐ]?)[^\n.;\]\)]{0,60}?(\b\d+[a-zA-Z]?/\d{4}/[\w\-]+|\b\d+-CP\b)",
    re.IGNORECASE,
)
RE_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
RE_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
RE_COMPLETE_END = re.compile(r"[.!?:;)\]|*`…\"”']\s*$")
REFUSAL_MARKERS = ("không tìm thấy", "từ chối trả lời", "không có quy định", "không đủ căn cứ")


def normalize_article(article: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (article or "").strip()).casefold()


def normalize_doc(doc: Optional[str]) -> str:
    return (doc or "").strip().upper()


def match_target(cand: Dict[str, Any], target: Dict[str, Any]) -> bool:
    """Exact document and article match. Substring matching made "Điều 1" also match "Điều 18"."""
    cand_doc = cand.get("doc_number") or cand.get("document_number", "")
    if normalize_doc(cand_doc) != normalize_doc(target.get("doc_number")):
        return False
    target_art = target.get("article_code")
    if not target_art:
        return True
    return normalize_article(cand.get("article_code")) == normalize_article(target_art)


def match_doc(cand: Dict[str, Any], target: Dict[str, Any]) -> bool:
    cand_doc = cand.get("doc_number") or cand.get("document_number", "")
    return normalize_doc(cand_doc) == normalize_doc(target.get("doc_number"))


def _article_key(chunk: Dict[str, Any]) -> Tuple[str, str]:
    return normalize_doc(chunk.get("doc_number")), normalize_article(chunk.get("article_code"))


def ranked_pool(top_chunks: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One ranked list per query: reranked chunks first, then the remaining candidates.

    Pinned amendments are excluded, so @5 is always a prefix of @25 and Hit@5 can never exceed Hit@25.
    """
    pool: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for chunk in list(top_chunks or []) + list(candidates or []):
        if chunk.get("is_pinned_amendment"):
            continue
        key = _article_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        pool.append(chunk)
    return pool


def score_retrieval(
    top_chunks: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pool = ranked_pool(top_chunks, candidates)
    at5, at25 = pool[:5], pool[:25]
    n = max(len(targets), 1)

    def matched(items: List[Dict[str, Any]]) -> int:
        return sum(1 for t in targets if any(match_target(c, t) for c in items))

    m5, m25, m_all = matched(at5), matched(at25), matched(pool)
    mrr = 0.0
    for rank, chunk in enumerate(at5, 1):
        if any(match_target(chunk, t) for t in targets):
            mrr = 1.0 / rank
            break

    return {
        "hit_rate_at_5": 1.0 if m5 else 0.0,
        "hit_rate_at_25": 1.0 if m25 else 0.0,
        "proportional_recall_at_5": m5 / n,
        "proportional_recall_at_25": m25 / n,
        "mrr_at_5": mrr,
        "doc_hit_at_5": 1.0 if any(match_doc(c, t) for t in targets for c in at5) else 0.0,
        "doc_hit_at_25": 1.0 if any(match_doc(c, t) for t in targets for c in at25) else 0.0,
        "pool_recall": m_all / n,
        "pool_size": len(pool),
    }


def wilson_interval(successes: float, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def extract_citations(answer: str) -> List[Tuple[str, str]]:
    return [(normalize_doc(doc), normalize_article(art)) for art, doc in RE_CITATION.findall(answer or "")]


def citation_precision(answer: str, context_chunks: Iterable[Dict[str, Any]]) -> Optional[float]:
    """Share of "Điều X ... Số hiệu" citations naming an article that was supplied. None if no citations.

    Matches any wrapper the model uses: [Điều 5, Thông tư 1/2020/TT-BYT], (Điều 5, ...) or plain text.
    """
    cited = extract_citations(answer)
    if not cited:
        return None
    supplied = {_article_key(c) for c in context_chunks}
    return sum(1 for doc, art in cited if (doc, art) in supplied) / len(cited)


def unsupported_numbers(answer: str, context_text: str) -> List[str]:
    """Figures in the answer that never appear in the context, ignoring article and document references."""
    stripped = RE_DOC_NUMBER.sub(" ", RE_ARTICLE_MENTION.sub(" ", RE_CITATION.sub(" ", answer or "")))
    context_numbers = set(RE_NUMBER.findall(context_text or ""))
    found: List[str] = []
    for number in RE_NUMBER.findall(stripped):
        if len(number) == 1 and number.isdigit():
            continue  # list markers and small ordinals
        if number not in context_numbers and number not in found:
            found.append(number)
    return found


def has_language_drift(answer: str, query: str) -> bool:
    return bool(RE_CJK.search(answer or "")) and not RE_CJK.search(query or "")


def is_complete_answer(answer: str) -> bool:
    return bool(RE_COMPLETE_END.search(answer or ""))


def is_refusal(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)
