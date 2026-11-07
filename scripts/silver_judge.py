"""Builds the judging tasks: is each question answerable, and which other retrieved articles also answer it."""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from dotenv import load_dotenv

from scripts.eval_common import normalize_article, normalize_doc
from scripts.silver_sample import article_of, body_of

POOL_SIZE = 8

ANSWERABLE_SYSTEM = """Bạn kiểm tra câu hỏi kiểm thử pháp luật. Chỉ dựa vào đoạn luật được cung cấp.
answerable: true nếu đoạn luật trả lời ĐẦY ĐỦ câu hỏi, không cần thêm văn bản khác.
specific: true nếu câu hỏi đứng một mình vẫn đủ rõ để biết đang hỏi về lĩnh vực và đối tượng nào.
Trả về JSON: {"reason": "tối đa 15 từ", "answerable": true/false, "specific": true/false}"""

GRADE_SYSTEM = """Bạn chấm mức liên quan của một đoạn luật với câu hỏi.
2 = đoạn luật trực tiếp trả lời câu hỏi, đúng đối tượng và đúng tình huống.
1 = cùng chủ đề nhưng chỉ trả lời một phần, hoặc khác đối tượng.
0 = không trả lời câu hỏi.
Trả về JSON: {"reason": "tối đa 15 từ", "grade": 0}"""


SPECIFIC_SYSTEM = """Bạn chỉ thấy một câu hỏi pháp luật, không thấy văn bản nào.
specific = true nếu một luật sư đọc xong biết ngay cần tra quy định về đối tượng và vấn đề gì.
specific = false nếu thiếu đối tượng hoặc vấn đề, phải hỏi lại mới tra được.
Ví dụ true: "Tôi muốn cải tạo nhà đang thuê thì cần ai đồng ý?"; "điều kiện cấp giấy phép phòng khám dinh dưỡng".
Ví dụ false: "đầu tư xây dựng đường bộ cần gì"; "ngân sách nhà nước tạm ứng"; "Thời gian thực hiện dự án là bao lâu?".
Trả về JSON: {"specific": true}"""


def key(doc: str, article: str) -> Tuple[str, str]:
    return normalize_doc(doc), normalize_article(article)


def judge_task(task_id: str, system: str, question: str, passage: str) -> Dict[str, Any]:
    return {
        "id": task_id,
        "json": True,
        "temperature": 0.0,
        "max_tokens": 80,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Câu hỏi: {question}\n\nĐoạn luật:\n{passage}"},
        ],
    }


def gold_text_lookup(targets: Set[Tuple[str, str]]) -> Dict[Tuple[str, str], str]:
    """Article text for Set B's existing gold labels, which the pool may not contain."""
    load_dotenv(".env")
    conn = psycopg2.connect(os.environ["DATABASE_DSN"])
    conn.set_session(readonly=True)
    docs = sorted({doc for doc, _ in targets})
    found: Dict[Tuple[str, str], List[str]] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ld.doc_number, lc.breadcrumb, lc.raw_content
            FROM legal_chunks lc JOIN legal_documents ld ON ld.doc_id = lc.doc_id
            WHERE upper(ld.doc_number) = ANY(%s)
            """,
            (docs,),
        )
        for doc, breadcrumb, raw in cur:
            article = article_of(breadcrumb)
            if article and key(doc, article[0]) in targets:
                found.setdefault(key(doc, article[0]), []).append(body_of(raw, breadcrumb))
    conn.close()
    return {k: "\n".join(v)[:1800] for k, v in found.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/v5_1_phase2")
    parser.add_argument("--specificity-only", action="store_true", help="Standalone check: the first pass echoed its prompt")
    args = parser.parse_args()

    def load(name: str) -> Any:
        with open(os.path.join(args.dir, name), "r", encoding="utf-8") as f:
            return json.load(f)

    generated, set_b = load("filtered.json"), load("setb.json")
    with open(os.path.join(args.dir, "pools.jsonl"), "r", encoding="utf-8") as f:
        pools = {row["id"]: row["pool"] for row in map(json.loads, f)}

    if args.specificity_only:
        tasks = [
            {"id": f"SP|{item['id']}", "json": True, "temperature": 0.0, "max_tokens": 15,
             "messages": [{"role": "system", "content": SPECIFIC_SYSTEM}, {"role": "user", "content": item["query"]}]}
            for item in generated
        ]
        with open(os.path.join(args.dir, "specific_tasks.json"), "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False)
        print(len(tasks), "tasks")
        return

    tasks: List[Dict[str, Any]] = []
    for item in generated:
        if item["set"] == "A":
            tasks.append(judge_task(f"J|{item['id']}", ANSWERABLE_SYSTEM, item["query"], item["sources"][0][:2600]))
        else:
            tasks.append(judge_task(f"J|{item['id']}", ANSWERABLE_SYSTEM, item["query"], "\n\n".join(s[:1500] for s in item["sources"])))
            tasks.append(judge_task(f"JA|{item['id']}", ANSWERABLE_SYSTEM, item["query"], item["sources"][0][:2600]))
            tasks.append(judge_task(f"JB|{item['id']}", ANSWERABLE_SYSTEM, item["query"], item["sources"][1][:2600]))

    for item in generated + set_b:
        gold = {key(t["doc_number"], t.get("article_code")) for t in item["targets"]}
        graded = 0
        for candidate in pools.get(item["id"], []):
            candidate_key = key(candidate["doc_number"], candidate["article_code"])
            if item["id"].startswith("B_") or candidate_key not in gold:
                task_id = f"G|{item['id']}|{candidate['doc_number']}|{candidate['article_code']}"
                tasks.append(judge_task(task_id, GRADE_SYSTEM, item["query"], f"{candidate['breadcrumb']}\n{candidate['text']}"))
                graded += 1
            if graded == POOL_SIZE:
                break

    # Old labels get graded too, so a wrong label is dropped instead of trusted.
    wanted = {key(t["doc_number"], t.get("article_code")) for b in set_b for t in b["targets"] if t.get("article_code")}
    texts = gold_text_lookup(wanted)
    for item in set_b:
        for target in item["targets"]:
            text = texts.get(key(target["doc_number"], target.get("article_code")))
            task_id = f"G|{item['id']}|{target['doc_number']}|{target.get('article_code')}"
            if text and not any(t["id"] == task_id for t in tasks):
                tasks.append(judge_task(task_id, GRADE_SYSTEM, item["query"], text))

    with open(os.path.join(args.dir, "judge_tasks.json"), "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False)
    print(len(tasks), "tasks; old gold texts found", len(texts), "of", len(wanted))


if __name__ == "__main__":
    main()
