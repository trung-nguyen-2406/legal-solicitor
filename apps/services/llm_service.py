from collections import OrderedDict
import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from dotenv import load_dotenv
import httpx

load_dotenv(".env")

INVENTORY_TABLE_MAX_TOKENS = 600
MAX_CONTEXT_TOKENS = 2500

IRAC_SYSTEM_PROMPT = """Bạn là Chuyên gia Tư vấn Pháp luật Việt Nam. Hãy phân tích và trả lời câu hỏi của người dùng theo đúng cấu trúc pháp lý IRAC dựa TUYỆT ĐỐI vào các đoạn trích văn bản quy phạm pháp luật được cung cấp trong ngữ cảnh.

Cấu trúc câu trả lời bắt buộc:
1. **Vấn đề pháp lý (Issue)**: Xác định ngắn gọn vấn đề pháp lý cốt lõi cần giải quyết.
2. **Quy định pháp luật (Rule)**: Viện dẫn chính xác nội dung quy định áp dụng kèm trích dẫn nguyên văn định dạng `[Nguồn i: Điều X Văn bản Y]`. Nếu ngữ cảnh có văn bản sửa đổi, bổ sung, ưu tiên áp dụng quy định mới nhất đang có hiệu lực.
3. **Phân tích & Áp dụng (Application)**: Đối chiếu quy định pháp luật đã nêu vào câu hỏi/tình huống cụ thể (nếu là câu hỏi thống kê/tổng hợp, tổng hợp đầy đủ từ Bảng tổng hợp và các đoạn trích).
4. **Kết luận (Conclusion)**: Đưa ra kết luận pháp lý trực tiếp, rõ ràng, ngắn gọn.

Nguyên tắc bắt buộc:
- Chỉ sử dụng thông tin có trong ngữ cảnh được cung cấp.
- Tuyệt đối không tự suy diễn hoặc bịa đặt số hiệu văn bản, số điều khoản, mức phạt hay tỷ lệ ngoài ngữ cảnh.
- Luôn gắn nhãn trích dẫn `[Nguồn i: Điều X Văn bản Y]` ngay sau nhận định pháp lý tương ứng."""


def get_llm_model_name() -> str:
    return os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b-instruct-q4_K_M")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(int(len(text.split()) * 1.5), len(text) // 3)


def _truncate_to_tokens(text: str, budget_tokens: int) -> str:
    if budget_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= budget_tokens:
        return text
    max_chars = max(30, budget_tokens * 3 - 12)
    clipped = text[:max_chars].rsplit(" ", 1)[0]
    while clipped and estimate_tokens(clipped + " ...") > budget_tokens:
        clipped = clipped[: int(len(clipped) * 0.85)].rsplit(" ", 1)[0]
    return (clipped + " ...") if clipped else text[:24]


def _chunk_to_citation(chunk: Dict[str, Any], source_index: Optional[int] = None) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "chunk_id": chunk.get("chunk_id", ""),
        "doc_number": chunk.get("doc_number", ""),
        "doc_title": chunk.get("doc_title", ""),
        "article_code": chunk.get("article_code", ""),
        "clause_code": chunk.get("clause_code", "ALL"),
        "issued_year": chunk.get("issued_year"),
        "rerank_score": round(float(chunk.get("rerank_score") or chunk.get("rrf_score") or 0.0), 4),
        "is_pinned_amendment": bool(chunk.get("is_pinned_amendment", False)),
    }
    if source_index is not None:
        item["source_index"] = source_index
    return item


def build_hierarchical_inventory_table(
    chunks: List[Dict[str, Any]],
    max_tokens: int = INVENTORY_TABLE_MAX_TOKENS,
) -> Tuple[str, List[Dict[str, Any]]]:
    if not chunks:
        return "", []

    ordered = sorted(
        chunks,
        key=lambda c: (
            float(c.get("rerank_score") or 0.0),
            float(c.get("rrf_score") or 0.0),
            int(c.get("issued_year") or 0),
        ),
        reverse=True,
    )

    grouped: "OrderedDict[Tuple[str, str, Any], List[str]]" = OrderedDict()
    all_citations: List[Dict[str, Any]] = []
    seen_pairs = set()

    for c in ordered:
        doc_num = (c.get("doc_number") or "N/A").strip()
        doc_title = (c.get("doc_title") or "").strip()
        yr = c.get("issued_year") or ""
        art = (c.get("article_code") or c.get("clause_code") or "Điều ?").strip()

        gkey = (doc_num, doc_title, yr)
        if gkey not in grouped:
            grouped[gkey] = []
        if art not in grouped[gkey]:
            grouped[gkey].append(art)

        pair = (doc_num, art)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            all_citations.append(_chunk_to_citation(c))

    lines = [
        f"Tổng hợp hệ thống: {len(seen_pairs)} điều khoản trên {len(grouped)} văn bản.",
        "| Số hiệu | Tên văn bản | Năm | Điều khoản liên quan |",
        "|---|---|---|---|",
    ]

    omitted_docs = 0
    items = list(grouped.items())
    for idx, ((doc_num, doc_title, yr), arts) in enumerate(items):
        short_title = (doc_title[:55] + "...") if len(doc_title) > 58 else (doc_title or "-")
        short_title = short_title.replace("|", "/")
        art_str = ", ".join(arts[:8])
        if len(arts) > 8:
            art_str += f" (+{len(arts) - 8})"
        row = f"| {doc_num} | {short_title} | {yr} | {art_str} |"

        footer_reserve = 28 if idx < len(items) - 1 else 0
        candidate = "\n".join(lines + [row])
        if estimate_tokens(candidate) + footer_reserve > max_tokens:
            omitted_docs = len(items) - idx
            break
        lines.append(row)

    if omitted_docs > 0:
        footer = f"| ... | (Còn {omitted_docs} văn bản khác đã rút gọn) | ... | ... |"
        if estimate_tokens("\n".join(lines + [footer])) <= max_tokens:
            lines.append(footer)

    table_md = "\n".join(lines)
    if estimate_tokens(table_md) > max_tokens:
        table_md = _truncate_to_tokens(table_md, max_tokens)

    return table_md, all_citations


def build_irac_context_and_citations(
    top_chunks: List[Dict[str, Any]],
    inventory_table: Optional[str] = None,
    existing_citations: Optional[List[Dict[str, Any]]] = None,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> Tuple[str, List[Dict[str, Any]]]:
    sections: List[str] = []
    used_tokens = 0

    if inventory_table:
        inv_header = "=== BẢNG TỔNG HỢP ĐIỀU KHOẢN ===\n"
        inv_block = inv_header + _truncate_to_tokens(inventory_table, INVENTORY_TABLE_MAX_TOKENS)
        sections.append(inv_block)
        used_tokens += estimate_tokens(inv_block)

    citations: List[Dict[str, Any]] = []
    seen_pairs = set()

    for idx, c in enumerate(top_chunks, start=1):
        remaining = max_context_tokens - used_tokens
        if remaining <= 45:
            break

        doc_num = c.get("doc_number") or "N/A"
        art_code = c.get("article_code") or "Điều ?"
        doc_title = c.get("doc_title") or ""
        yr = c.get("issued_year") or ""
        amend_note = " [Văn bản sửa đổi, bổ sung]" if c.get("is_pinned_amendment") else ""
        header = f"[Nguồn {idx}: {art_code} Văn bản {doc_num}] ({doc_title}, năm {yr}){amend_note}"
        header_tokens = estimate_tokens(header) + 4
        body_budget = remaining - header_tokens
        if body_budget <= 20:
            break

        raw_body = (c.get("raw_content") or c.get("breadcrumb") or "").strip()
        trimmed_body = _truncate_to_tokens(raw_body, body_budget)
        chunk_block = f"{header}\n{trimmed_body}"

        sections.append(chunk_block)
        used_tokens += estimate_tokens(chunk_block)

        citations.append(_chunk_to_citation(c, source_index=idx))
        seen_pairs.add((doc_num, art_code))

    if existing_citations:
        for extra in existing_citations:
            pair = (extra.get("doc_number", ""), extra.get("article_code", ""))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                citations.append(dict(extra))

    return "\n\n".join(sections), citations


async def stream_irac_ollama(
    query: str,
    context_block: str,
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 90.0,
) -> AsyncIterator[str]:
    ollama_host = host or os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    if not ollama_host.startswith("http"):
        ollama_host = f"http://{ollama_host}"
    model_name = model or get_llm_model_name()

    user_prompt = (
        f"NGỮ CẢNH PHÁP LÝ:\n{context_block}\n\n"
        f"CÂU HỎI CỦA NGƯỜI DÙNG:\n{query}\n\n"
        "Hãy trả lời theo đúng cấu trúc IRAC và trích dẫn chính xác `[Nguồn i: Điều X Văn bản Y]`."
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": IRAC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "options": {
            "temperature": 0.1,
            "top_p": 0.90,
            "num_predict": 1024,
        },
    }

    url = f"{ollama_host.rstrip('/')}/api/chat"
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                packet = json.loads(line)
                delta = (packet.get("message") or {}).get("content") or ""
                if delta:
                    yield delta
                if packet.get("done"):
                    break

