import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_common import (
    citation_precision,
    has_language_drift,
    is_complete_answer,
    is_refusal,
    match_target,
    ranked_pool,
    score_retrieval,
    unsupported_numbers,
    wilson_interval,
)

TARGET = {"doc_number": "11/2025/TT-BYT", "article_code": "Điều 1"}


def chunk(doc, art, **extra):
    return {"doc_number": doc, "article_code": art, **extra}


def test_match_is_exact_not_substring():
    assert match_target(chunk("11/2025/TT-BYT", "Điều 1"), TARGET)
    assert not match_target(chunk("11/2025/TT-BYT", "Điều 18"), TARGET)
    assert not match_target(chunk("11/2025/TT-BYT", "Điều 18", breadcrumb="Điều 1. Sửa đổi"), TARGET)
    assert not match_target(chunk("12/2025/TT-BYT", "Điều 1"), TARGET)


def test_match_normalizes_case_and_spacing():
    assert match_target(chunk("11/2025/tt-byt", "điều  1"), TARGET)


def test_target_without_article_matches_on_document():
    assert match_target(chunk("11/2025/TT-BYT", "Điều 7"), {"doc_number": "11/2025/TT-BYT"})


def test_ranked_pool_drops_pins_and_duplicates():
    top = [chunk("A", "Điều 1"), chunk("B", "Điều 2", is_pinned_amendment=True)]
    pool = ranked_pool(top, [chunk("A", "Điều 1"), chunk("C", "Điều 3")])
    assert [(c["doc_number"], c["article_code"]) for c in pool] == [("A", "Điều 1"), ("C", "Điều 3")]


def test_hit_at_5_can_never_exceed_hit_at_25():
    top = [chunk("11/2025/TT-BYT", "Điều 1")]
    catalog = [chunk("X", f"Điều {i}") for i in range(40)]
    scores = score_retrieval(top, catalog, [TARGET])
    assert scores["hit_rate_at_5"] == 1.0
    assert scores["hit_rate_at_25"] == 1.0


def test_rank_cut_applies_to_large_catalogs():
    catalog = [chunk("X", f"Điều {i}") for i in range(30)] + [chunk("11/2025/TT-BYT", "Điều 1")]
    scores = score_retrieval([], catalog, [TARGET])
    assert scores["hit_rate_at_25"] == 0.0
    assert scores["pool_recall"] == 1.0


def test_pinned_amendment_does_not_count_as_a_hit():
    scores = score_retrieval([chunk("11/2025/TT-BYT", "Điều 1", is_pinned_amendment=True)], [], [TARGET])
    assert scores["hit_rate_at_5"] == 0.0


def test_mrr_uses_rank():
    top = [chunk("X", "Điều 9"), chunk("11/2025/TT-BYT", "Điều 1")]
    assert score_retrieval(top, [], [TARGET])["mrr_at_5"] == 0.5


def test_wilson_interval_bounds():
    low, high = wilson_interval(10, 47)
    assert 0.1 < low < 0.2128 < high < 0.36
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_citation_precision():
    context = [chunk("50/2014/QH13", "Điều 21")]
    good = "Hồ sơ được lưu trữ [Điều 21, 50/2014/QH13]."
    bad = "Hồ sơ được lưu trữ [Điều 21, 50/2014/QH13] và [Điều 99, 50/2014/QH13]."
    assert citation_precision(good, context) == 1.0
    assert citation_precision(bad, context) == 0.5
    assert citation_precision("Không có trích dẫn.", context) is None


def test_citation_with_document_type_and_other_wrappers():
    context = [chunk("04/2022/TT-BXD", "Điều 38")]
    assert citation_precision("Theo [Điều 38, Thông tư 04/2022/TT-BXD].", context) == 1.0
    assert citation_precision("Như vậy (Điều 38, Thông tư 04/2022/TT-BXD).", context) == 1.0
    assert citation_precision("Theo Điều 38, Thông tư 04/2022/TT-BXD.", context) == 1.0
    assert citation_precision("Theo [Điều 39, Thông tư 04/2022/TT-BXD].", context) == 0.0


def test_citation_with_clause_detail():
    assert citation_precision("[Điều 21 Khoản 2, 50/2014/QH13]", [chunk("50/2014/QH13", "Điều 21")]) == 1.0


def test_unsupported_numbers_ignore_references():
    context = "Mức phạt từ 5.000.000 đồng trong 30 ngày."
    assert unsupported_numbers("Phạt 5.000.000 đồng trong 30 ngày [Điều 21, 50/2014/QH13].", context) == []
    assert unsupported_numbers("Phạt 7.000.000 đồng theo Điều 21 và 50/2014/QH13.", context) == ["7.000.000"]


def test_language_drift():
    assert has_language_drift("Scratchpad已提供", "Thủ tục là gì?")
    assert not has_language_drift("根据法律", "法定流程是什么？")
    assert not has_language_drift("Theo quy định.", "Thủ tục là gì?")


def test_answer_completeness_and_refusal():
    assert is_complete_answer("Hồ sơ gồm hai phần.")
    assert not is_complete_answer("Hồ sơ gồm hai phần và")
    assert is_complete_answer('trừ Hiến pháp."')
    assert is_refusal("Không tìm thấy quy định phù hợp trong cơ sở dữ liệu.")
    assert not is_refusal("Hồ sơ gồm hai phần.")
