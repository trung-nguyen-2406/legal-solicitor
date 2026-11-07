"""Model bake-off around scripts/eval_generation.py. All models answer from identical saved contexts.

  contexts  picks a fixed slice of the dev retrieval records
  tasks     builds faithfulness and blind A/B judging tasks for scripts/silver_colab.py
  score     applies the switch rule, which was fixed before any model ran
"""
import argparse
import collections
import json
import os
import random
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.silver_filter import parse

DIR = "results/v5_1_phase3"
BASELINE = "qwen2.5-7b"
CANDIDATES = ("qwen2.5-3b", "qwen3-4b", "gemma3-4b")
SLICE = {"A": 24, "A2": 8, "M": 10, "C": 18}
MAX_CONTEXT_CHARS = 5000

FAITHFUL_SYSTEM = """Bạn kiểm tra một câu trả lời pháp luật có bám sát tài liệu không.
faithful = true nếu mọi nhận định trong câu trả lời đều có trong tài liệu, hoặc câu trả lời nói không tìm thấy thông tin.
faithful = false nếu có nhận định, con số hay trích dẫn không có trong tài liệu.
Trả về JSON: {"faithful": true}"""

PREFERENCE_SYSTEM = """Hai câu trả lời A và B cho cùng một câu hỏi, dựa trên cùng tài liệu.
Chọn câu đúng với tài liệu và hữu ích hơn. Độ dài không phải tiêu chí.
Trả về JSON: {"better": "A"} hoặc {"better": "B"} hoặc {"better": "tie"}"""


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def answers(model: str) -> Dict[str, Dict[str, Any]]:
    return {row["query_id"]: row for row in load_json(f"{DIR}/generation/{model}.json")["rows"]}


def pick_contexts() -> None:
    records = load_json(f"{DIR}/retrieval_dev/retrieval.json")["records"]
    sets = {q["query_id"]: q["set"] for q in load_json("tests/fixtures/silver_v1_dev.json")}
    rng = random.Random(51)
    picked: List[Dict[str, Any]] = []
    for name, count in SLICE.items():
        members = [r for r in records if sets[r["query_id"]] == name and r["top_chunks"]]
        picked.extend(rng.sample(members, min(count, len(members))))
    with open(f"{DIR}/contexts.json", "w", encoding="utf-8") as f:
        json.dump({"records": picked}, f, ensure_ascii=False)
    print(len(picked), "contexts")


def judge_task(task_id: str, system: str, body: str) -> Dict[str, Any]:
    return {"id": task_id, "json": True, "temperature": 0.0, "max_tokens": 20,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": body}]}


def build_tasks() -> None:
    records = load_json(f"{DIR}/contexts.json")["records"]
    all_answers = {m: answers(m) for m in (BASELINE, *CANDIDATES)}
    rng = random.Random(51)
    tasks: List[Dict[str, Any]] = []
    for record in records:
        query_id = record["query_id"]
        context = "\n\n".join(c["raw_content"] or "" for c in record["top_chunks"])[:MAX_CONTEXT_CHARS]
        for model, rows in all_answers.items():
            body = f"Tài liệu:\n{context}\n\nCâu trả lời:\n{rows[query_id]['answer']}"
            tasks.append(judge_task(f"F|{model}|{query_id}", FAITHFUL_SYSTEM, body))
        for model in CANDIDATES:
            swap = rng.random() < 0.5  # hides which answer is the baseline
            first, second = (model, BASELINE) if swap else (BASELINE, model)
            body = (f"Câu hỏi: {record['query']}\n\nTài liệu:\n{context[:3500]}\n\n"
                    f"Câu trả lời A:\n{all_answers[first][query_id]['answer']}\n\nCâu trả lời B:\n{all_answers[second][query_id]['answer']}")
            tasks.append(judge_task(f"P|{model}|{query_id}|{first}", PREFERENCE_SYSTEM, body))
    with open(f"{DIR}/judge_tasks.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False)
    print(len(tasks), "tasks")


def score() -> None:
    with open(f"{DIR}/judge_out.jsonl", "r", encoding="utf-8") as f:
        judged = [json.loads(line) for line in f]
    faithful: Dict[str, List[bool]] = collections.defaultdict(list)
    preference: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in judged:
        verdict = parse(row["text"]) or {}
        parts = row["id"].split("|")
        if parts[0] == "F":
            faithful[parts[1]].append(bool(verdict.get("faithful")))
        else:
            model, first = parts[1], parts[3]
            better = verdict.get("better")
            winner = {"A": first, "B": BASELINE if first == model else model}.get(better, "tie")
            preference[model]["candidate" if winner == model else "baseline" if winner == BASELINE else "tie"] += 1

    summaries = {m: load_json(f"{DIR}/generation/{m}.json")["summary"] for m in (BASELINE, *CANDIDATES)}
    base = summaries[BASELINE]
    base_faithful = sum(faithful[BASELINE]) / len(faithful[BASELINE])
    print(f"{'model':<13}{'faith':>7}{'cite':>7}{'cited%':>8}{'nums%':>7}{'drift':>7}{'done%':>7}{'refuse':>8}{'ttft':>7}{'p50 s':>7}  preferred (cand/base/tie)  switch?")
    for model, s in summaries.items():
        rate = sum(faithful[model]) / len(faithful[model])
        checks = [
            s["citation_precision"] >= base["citation_precision"],
            rate >= base_faithful - 0.03,
            s["language_drift"] == 0,
            s["total_p50_s"] * 2 <= base["total_p50_s"],
        ]
        pref = preference[model]
        decision = "-" if model == BASELINE else "YES" if all(checks) else "no: " + ",".join(n for n, ok in zip(("cite", "faith", "drift", "speed"), checks) if not ok)
        print(f"{model:<13}{rate:>7.2f}{s['citation_precision']:>7.2f}{s['answers_with_citations'] * 100:>8.0f}"
              f"{s['answers_with_unsupported_numbers'] * 100:>7.0f}{s['language_drift']:>7.2f}{s['complete_answers'] * 100:>7.0f}"
              f"{s['refusal_on_negatives']:>8.2f}{s['first_token_p50_s']:>7.1f}{s['total_p50_s']:>7.1f}"
              f"  {pref['candidate']}/{pref['baseline']}/{pref['tie']}  {decision}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["contexts", "tasks", "score"])
    {"contexts": pick_contexts, "tasks": build_tasks, "score": score}[parser.parse_args().step]()
