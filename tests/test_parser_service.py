from apps.services.parser_service import LegalParserService

def test_parser_normal_article():
    sample_text = """
Chương I QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
Luật này quy định về trật tự giao thông.
1. Quy tắc tham gia giao thông.
2. Xử phạt theo quy định tại Điều 15.
"""
    doc_meta = {
        "doc_id": "00000000-0000-0000-0000-000000000001",
        "doc_number": "100/2024/QH15",
        "doc_title": "Luật Trật tự ATGT",
        "doc_type": "Luật",
        "issued_year": 2024,
    }

    parser = LegalParserService()
    chunks = parser.parse_document(sample_text, doc_meta)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.doc_number == "100/2024/QH15"
    assert chunk.clause_id == "ALL"
    assert chunk.article_id == "Điều 1"
    assert "Chương I" in chunk.breadcrumb
    assert "Điều 15" in chunk.referenced_provisions

def test_parser_oversized_article_splits_clauses():
    doc_meta = {
        "doc_id": "00000000-0000-0000-0000-000000000002",
        "doc_number": "01/2024/NĐ-CP",
        "doc_title": "Nghị định hướng dẫn",
        "doc_type": "Nghị định",
        "issued_year": 2024,
    }

    long_text = "Nội dung quy định chi tiết. " * 500
    sample_text = f"""
Điều 2. Quy định chi tiết
Phần mở đầu dẫn nhập cho các khoản dưới đây:
1. {long_text}
2. Khoản ngắn thứ hai.
"""

    parser = LegalParserService()
    chunks = parser.parse_document(sample_text, doc_meta)

    assert len(chunks) == 2
    assert chunks[0].clause_id == "Khoản 1"
    assert chunks[1].clause_id == "Khoản 2"
    assert "Phần mở đầu dẫn nhập cho các khoản dưới đây:" in chunks[0].raw_content
    assert "Phần mở đầu dẫn nhập cho các khoản dưới đây:" in chunks[1].raw_content

if __name__ == "__main__":
    test_parser_normal_article()
    test_parser_oversized_article_splits_clauses()
