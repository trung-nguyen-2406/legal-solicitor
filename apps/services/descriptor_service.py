import hashlib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

try:
    import ahocorasick # type: ignore[import-untyped, import-not-found]
except ImportError:
    ahocorasick = None

MAX_VALID_ISSUED_YEAR = 2026

LEGAL_REPLACEMENTS = {
    "hồi tố": "hiệu lực trở về trước",
    "lex posterior": "áp dụng văn bản ban hành sau khi có quy định khác nhau",
    "lex specialis": "áp dụng luật chuyên ngành",
    "lex superior": "áp dụng văn bản có hiệu lực pháp lý cao hơn",
    "de jure": "về mặt pháp lý",
    "de facto": "trên thực tế",
    "chạy án": "hối lộ can thiệp xét xử",
    "lách luật": "trốn tránh quy định pháp luật",
    "doanh nghiệp ma": "doanh nghiệp không hoạt động tại trụ sở",
    "hóa đơn khống": "hóa đơn không có hàng hóa dịch vụ thực tế",
}

STRUCTURAL_NOUNS = [
    "văn bản", "luật", "nghị định", "thông tư", "điều", "khoản",
    "quy định mới", "lần sửa đổi", "điều luật", "bộ luật", "quyết định"
]

SUBSTANTIVE_NOUNS = [
    "khung giờ", "cấp độ", "trường hợp", "thủ tục", "hồ sơ",
    "loại giấy phép", "mức phạt", "hành vi", "tiêu chí", "điều kiện"
]

COUNTING_PREFIXES = [
    "có bao nhiêu", "tổng số", "thống kê", "liệt kê các",
    "danh sách các", "bao nhiêu"
]

ADMIN_STOPWORDS = {
    "quản lý", "doanh nghiệp", "nhà nước", "quy định", "ban hành",
    "thực hiện", "hướng dẫn", "áp dụng", "cơ quan", "tổ chức", "cá nhân",
    "thủ tục", "hồ sơ", "về", "các", "những", "cho", "của", "tại", "trong",
    "theo", "đối với", "mới", "năm", "pháp luật", "văn bản"
}

BOILERPLATE_PATTERNS = [
    r'^\s*(?:tổng\s+số|có\s+bao\s+nhiêu|thống\s+kê|liệt\s+kê|danh\s+sách|bao\s+nhiêu)\s+(?:các\s+)?(?:điều\s+khoản|điều\s+luật|quy\s+định|văn\s+bản|nghị\s+định|thông\s+tư)?\s*(?:quy\s+định)?\s*(?:về|liên\s+quan\s+đến)?\s*',
    r'^\s*(?:các|những)\s+quy\s+định\s+về\s*',
    r'^\s*quy\s+định\s+về\s*',
]

TEMPORAL_BOILERPLATE_PATTERNS = [
    r'\b(?:mới\s+)?ban\s+hành\s+năm\s+\d{4}(?:\s*[-–—/]\s*\d{4})?\b',
    r'\bnăm\s+\d{4}\s*[-–—/]\s*\d{4}\b',
    r'\b(?:theo\s+)?quy\s+định\s+mới\s+nhất\b',
    r'\bmới\s+nhất\b',
]


class QueryDescriptorService:
    def __init__(self, max_valid_year: int = MAX_VALID_ISSUED_YEAR):
        self.max_valid_year = max_valid_year
        self.doc_pattern = re.compile(r'\b\d+[a-zA-Z]?/\d{4}/[\w\-]+|\b\d+-CP\b', re.IGNORECASE)
        self.art_pattern = re.compile(r'\bĐiều\s+(\d+[a-zA-ZđĐ]?)\b', re.IGNORECASE)
        self.statutory_year_pattern = re.compile(
            r'(?:luật|nghị định|thông tư|pháp lệnh|nghị quyết)[^.,;?!]{0,40}\bnăm\s+(\d{4})',
            re.IGNORECASE
        )
        self.cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
        self.foreign_words = {
            "what", "which", "who", "when", "where", "how", "why", "decree",
            "circular", "article", "clause", "quelle", "quelles", "droit", "loi"
        }

        self.automaton = None
        if ahocorasick is not None:
            self.automaton = ahocorasick.Automaton()
            for key, val in LEGAL_REPLACEMENTS.items():
                self.automaton.add_word(key, (key, val))
            self.automaton.make_automaton()

    def _replace_slang(self, text: str) -> str:
        if self.automaton is not None:
            matches: List[Tuple[int, int, str]] = []
            for end_idx, (orig, repl) in self.automaton.iter(text.lower()):
                start_idx = end_idx - len(orig) + 1
                matches.append((start_idx, end_idx + 1, repl))
            if not matches:
                return text
            matches.sort(key=lambda x: x[0])
            out = []
            curr = 0
            for start, end, repl in matches:
                if start >= curr:
                    out.append(text[curr:start])
                    out.append(repl)
                    curr = end
            out.append(text[curr:])
            return "".join(out)

        for src, dst in LEGAL_REPLACEMENTS.items():
            if src in text.lower():
                text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)
        return text

    def parse(self, raw_query: str) -> Dict[str, Any]:
        normalized = unicodedata.normalize('NFC', raw_query.strip())
        normalized = self._replace_slang(normalized)
        q_lower = normalized.lower()

        is_foreign = bool(self.cjk_pattern.search(normalized)) or any(
            re.search(rf'\b{w}\b', q_lower) for w in self.foreign_words
        )

        raw_docs = self.doc_pattern.findall(normalized)
        doc_filters: List[str] = []
        for d in raw_docs:
            norm_d = d.upper().strip()
            if norm_d not in doc_filters:
                doc_filters.append(norm_d)

        art_m = self.art_pattern.search(normalized)
        article_filter = f"Điều {art_m.group(1)}" if art_m else None

        is_future_year = False
        for doc in doc_filters:
            y_m = re.search(r'/(\d{4})/', doc)
            if y_m and int(y_m.group(1)) > self.max_valid_year:
                is_future_year = True
                break

        if not is_future_year:
            for y_str in self.statutory_year_pattern.findall(normalized):
                if int(y_str) > self.max_valid_year:
                    is_future_year = True
                    break

        has_list_cue = bool(
            re.search(r'\bnhững\b.*\bnào\b', q_lower)
            or re.search(r'\bcác\s+quy\s+định\s+về\b', q_lower)
            or re.search(r'\bdanh\s+sách\b', q_lower)
            or re.search(r'\bmấy\s+điều\b', q_lower)
            or re.search(r'\bbao\s+gồm\s+những\b', q_lower)
        )

        year_from: Optional[int] = None
        year_to: Optional[int] = None
        y_range_m = re.search(r'\b(?:năm\s+)?(\d{4})\s*[-–—/]\s*(\d{4})\b', normalized)
        if y_range_m:
            year_from = int(y_range_m.group(1))
            year_to = int(y_range_m.group(2))
        else:
            y_single_m = re.search(r'\bnăm\s+(\d{4})\b', normalized, re.IGNORECASE)
            if y_single_m:
                year_from = int(y_single_m.group(1))
                year_to = int(y_single_m.group(1))

        has_counting = any(k in q_lower for k in COUNTING_PREFIXES)
        has_structural = any(k in q_lower for k in STRUCTURAL_NOUNS)
        has_substantive = any(k in q_lower for k in SUBSTANTIVE_NOUNS)

        if (has_counting or has_list_cue) and has_structural and not has_substantive:
            is_aggregation = True
        elif has_counting and has_substantive:
            is_aggregation = False
        else:
            is_aggregation = bool((has_counting or has_list_cue) and has_structural)

        cleaned = normalized
        for pat in BOILERPLATE_PATTERNS:
            cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)
        for pat in TEMPORAL_BOILERPLATE_PATTERNS:
            cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)
        cleaned = self.doc_pattern.sub('', cleaned)
        cleaned = self.art_pattern.sub('', cleaned)
        cleaned = re.sub(r'[?!\.]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned) < 5:
            cleaned = normalized

        sub_queries: List[str] = []
        if is_aggregation or ("," in cleaned or " và " in cleaned.lower() or " hoặc " in cleaned.lower()):
            parts = re.split(r'[,;]|\s+(?:và|hoặc)\s+', cleaned, flags=re.IGNORECASE)
            sub_queries = [p.strip() for p in parts if len(p.strip()) > 2][:4]
        if not sub_queries:
            sub_queries = [cleaned]

        anchor_keywords: List[str] = []
        for sq in sub_queries:
            temp = sq.lower()
            for stop in sorted(ADMIN_STOPWORDS, key=len, reverse=True):
                temp = re.sub(rf'\b{re.escape(stop)}\b', ' ', temp)
            temp = re.sub(r'\s+', ' ', temp).strip()
            if temp and temp not in anchor_keywords:
                anchor_keywords.append(temp)
            words = temp.split()
            if len(words) > 1:
                for i in range(len(words) - 1):
                    subphrase = f"{words[i]} {words[i+1]}"
                    if subphrase not in anchor_keywords and subphrase not in ADMIN_STOPWORDS:
                        anchor_keywords.append(subphrase)
            for w in words:
                if len(w) >= 3 and w not in anchor_keywords and w not in ADMIN_STOPWORDS:
                    anchor_keywords.append(w)

        is_multistep = len(doc_filters) >= 2 or any(
            k in q_lower for k in ["so sánh", "sửa đổi", "bãi bỏ", "thay thế", "quan hệ", "xâu chuỗi", "xung đột"]
        )

        hash_payload = f"{cleaned.lower()}|{'|'.join(sorted(doc_filters))}|{article_filter or ''}"
        query_sha256 = hashlib.sha256(hash_payload.encode('utf-8')).hexdigest()

        return {
            "query": normalized,
            "cleaned_query": cleaned,
            "query_sha256": query_sha256,
            "doc_filters": doc_filters,
            "article_filter": article_filter,
            "is_aggregation": is_aggregation,
            "is_foreign_lang": is_foreign,
            "is_future_year": is_future_year,
            "word_count": len(normalized.split()),
            "is_multistep": is_multistep,
            "sub_queries": sub_queries,
            "anchor_keywords": anchor_keywords,
            "has_list_cue": has_list_cue,
            "year_from": year_from,
            "year_to": year_to,
        }

