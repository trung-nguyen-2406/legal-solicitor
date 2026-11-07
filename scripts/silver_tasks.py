"""Builds the LLM tasks for the silver benchmark from seeds.json."""
import argparse
import json
import os
import random
from typing import Any, Dict, List, Set

STYLES = {
    "direct": "một câu hỏi trực tiếp về nội dung quy định",
    "situational": "một tình huống thực tế ngắn (2-3 câu) của người dân hoặc doanh nghiệp rồi hỏi cách xử lý",
    "procedure": "một câu hỏi về thủ tục, điều kiện, thời hạn hoặc mức cụ thể",
    "keyword": "một truy vấn ngắn 5-9 từ kiểu gõ vào ô tìm kiếm, không cần thành câu, nhưng phải nêu rõ đối tượng và vấn đề cụ thể (đặt vào trường question)",
}

SINGLE_SYSTEM = """Bạn soạn câu hỏi kiểm thử cho hệ thống tra cứu pháp luật Việt Nam.
Quy tắc:
1. Câu hỏi phải trả lời được chỉ bằng điều luật được cung cấp.
2. Viết như người dân bình thường hỏi: diễn đạt lại bằng từ ngữ đời thường, không chép cụm dài từ điều luật hay tiêu đề điều.
3. Không nêu số hiệu văn bản, số Điều, tên văn bản, và không dùng "điều này", "văn bản này", "theo quy định trên".
4. Câu hỏi phải tự đủ nghĩa khi đứng một mình, nêu rõ lĩnh vực hoặc đối tượng.
5. evidence_quote: chép NGUYÊN VĂN một đoạn 8-40 từ trong điều luật chứa câu trả lời.
6. Hỏi về nội dung cụ thể có trong điều luật, không hỏi về nội dung chỉ được dẫn chiếu sang văn bản khác.
Ví dụ về rule 2:
Điều luật: "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước ít nhất 45 ngày".
Sai: "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động phải báo trước bao nhiêu ngày?"
Đúng: "Tôi muốn tự nghỉ việc thì phải nói với công ty trước bao lâu?"
Trả về JSON: {"question": "...", "evidence_quote": "...", "answer": "câu trả lời ngắn"}"""

PAIR_SYSTEM = """Bạn soạn câu hỏi kiểm thử cho hệ thống tra cứu pháp luật Việt Nam.
Có hai điều luật thuộc hai văn bản khác nhau, điều A dẫn chiếu tới điều B.
Quy tắc:
1. Câu hỏi phải cần CẢ HAI điều mới trả lời đầy đủ: thiếu một điều thì không trả lời được.
2. Viết như người dân bình thường hỏi, không chép cụm dài từ điều luật.
3. Không nêu số hiệu văn bản, số Điều hay tên văn bản.
4. Câu hỏi phải tự đủ nghĩa khi đứng một mình.
5. evidence_a, evidence_b: chép NGUYÊN VĂN một đoạn 8-40 từ từ mỗi điều.
6. Nếu hai điều không thể kết hợp thành một câu hỏi tự nhiên, trả về {"question": ""}.
Trả về JSON: {"question": "...", "evidence_a": "...", "evidence_b": "...", "answer": "câu trả lời ngắn"}"""


def article_block(item: Dict[str, Any]) -> str:
    return f"{item['heading']}\n{item['text']}"


def build(seeds: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    styles = list(STYLES)
    for index, item in enumerate(seeds["articles"]):
        style = styles[index % len(styles)]
        tasks.append({
            "id": f"A{index:03d}",
            "json": True,
            "temperature": 0.4,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": SINGLE_SYSTEM},
                {"role": "user", "content": f"Điều luật:\n{article_block(item)}\n\nHãy soạn {STYLES[style]}."},
            ],
        })
    for index, pair in enumerate(seeds["pairs"]):
        tasks.append({
            "id": f"M{index:03d}",
            "json": True,
            "temperature": 0.4,
            "max_tokens": 350,
            "messages": [
                {"role": "system", "content": PAIR_SYSTEM},
                {"role": "user", "content": f"Điều A:\n{article_block(pair['source'])}\n\nĐiều B:\n{article_block(pair['target'])}"},
            ],
        })
    return tasks


def build_extras(seeds: Dict[str, Any], filtered: List[Dict[str, Any]], done: Set[str]) -> List[Dict[str, Any]]:
    """Second generation pass: pairs not tried yet, English versions and foreign-law near-misses."""
    tasks = [t for t in build(seeds) if t["id"] not in done]
    full_sentences = [q for q in filtered if q["set"] == "A" and q["style"] != "keyword"]
    random.Random(seeds["seed"]).shuffle(full_sentences)
    for item in full_sentences[:30]:
        tasks.append({
            "id": f"E{item['id']}",
            "max_tokens": 150,
            "messages": [
                {"role": "system", "content": "Translate the user's Vietnamese legal question into natural English. Output only the translation."},
                {"role": "user", "content": item["query"]},
            ],
        })
    for item in full_sentences[30:50]:
        tasks.append({
            "id": f"F{item['id']}",
            "max_tokens": 150,
            "temperature": 0.5,
            "messages": [
                {"role": "system", "content": "Viết lại câu hỏi để hỏi về đúng vấn đề đó nhưng theo pháp luật của một nước khác (ví dụ Thái Lan, Pháp, Nhật Bản, Hàn Quốc, Đức, Singapore), nêu rõ tên nước. Chỉ trả về câu hỏi."},
                {"role": "user", "content": item["query"]},
            ],
        })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="results/v5_1_phase2/seeds.json")
    parser.add_argument("--out", default="results/v5_1_phase2/gen_tasks.json")
    parser.add_argument("--extras-after", default=None, help="Earlier output JSONL: build the second pass instead")
    parser.add_argument("--filtered", default="results/v5_1_phase2/filtered.json")
    parser.add_argument("--limit", type=int, default=None, help="Take every Nth task to pilot a representative slice")
    args = parser.parse_args()
    with open(args.seeds, "r", encoding="utf-8") as f:
        seeds = json.load(f)
    if args.extras_after:
        with open(args.extras_after, "r", encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
        with open(args.filtered, "r", encoding="utf-8") as f:
            tasks = build_extras(seeds, json.load(f), done)
    else:
        tasks = build(seeds)
    if args.limit:
        tasks = tasks[:: max(1, len(tasks) // args.limit)][: args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False)
    print(len(tasks), "tasks")


if __name__ == "__main__":
    main()
