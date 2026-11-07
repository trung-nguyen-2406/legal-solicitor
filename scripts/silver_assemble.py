"""Combines generated questions, judge verdicts and by-construction items into the silver benchmark.

Output is a dataset for scripts/eval_retrieval.py --dataset, plus a CSV for the human spot check.
"""
import argparse
import collections
import csv
import json
import os
import random
import sys
import unicodedata
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.silver_filter import parse

FAKE_NUMBERS = ["77/2029/TT-BYT", "412/2031/NĐ-CP", "93/2030/TT-BTC", "88/2032/TT-BGDĐT", "501/2029/NĐ-CP",
                "66/2031/TT-BCT", "71/2030/TT-BTNMT", "59/2032/TT-BXD", "318/2030/NĐ-CP", "84/2029/TT-BLĐTBXH"]
TEMPORAL_TEMPLATES = ["Văn bản nào đã {verb} {doc}?", "{doc} còn hiệu lực không, văn bản nào {verb} nó?"]
STRUCTURAL_TEMPLATE = "{doc_type} {doc_number} có tổng cộng bao nhiêu điều?"
SPOT_CHECK = {"A": 18, "M": 9, "T": 2, "S": 1}


def strip_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def add_typos(text: str, rng: random.Random) -> str:
    """Swaps two adjacent letters in about one word in four, as fast typing does."""
    words = text.split()
    for index in rng.sample(range(len(words)), max(1, len(words) // 4)):
        word = words[index]
        if len(word) >= 4:
            at = rng.randrange(1, len(word) - 2)
            words[index] = word[:at] + word[at + 1] + word[at] + word[at + 2:]
    return " ".join(words)


def entry(query_id: str, set_name: str, kind: str, query: str, targets: List[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    return {
        "query_id": query_id,
        "set": set_name,
        "type": kind,
        "query": query,
        "expected_targets": targets,
        "should_abstain": False,
        "is_broad_aggregation": False,
        **extra,
    }


def verdict(rows: Dict[str, Dict[str, Any]], task_id: str) -> Optional[Dict[str, Any]]:
    row = rows.get(task_id)
    return parse(row["text"]) if row else None


def alternatives(grades: Dict[str, List[Dict[str, Any]]], query_id: str) -> List[Dict[str, Any]]:
    return [{"doc_number": g["doc_number"], "article_code": g["article_code"]} for g in grades.get(query_id, []) if g["grade"] == 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/v5_1_phase2")
    parser.add_argument("--out", default="tests/fixtures/silver_v1.json")
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    def load(name: str) -> Any:
        with open(os.path.join(args.dir, name), "r", encoding="utf-8") as f:
            return json.load(f)

    def load_rows(name: str) -> Dict[str, Dict[str, Any]]:
        with open(os.path.join(args.dir, name), "r", encoding="utf-8") as f:
            return {r["id"]: r for r in map(json.loads, f)}

    seeds, generated, set_b = load("seeds.json"), load("filtered.json"), load("setb.json")
    judged, extras = load_rows("judge_out.jsonl"), load_rows("gen2_out.jsonl")
    specific = load_rows("specific_out.jsonl")

    grades: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for task_id, row in judged.items():
        if task_id.startswith("G|"):
            _, query_id, doc, article = task_id.split("|")
            value = (parse(row["text"]) or {}).get("grade")
            if value in (0, 1, 2):
                grades[query_id].append({"doc_number": doc, "article_code": article, "grade": value})

    items: List[Dict[str, Any]] = []
    dropped: collections.Counter = collections.Counter()
    by_source: Dict[str, Dict[str, Any]] = {}
    for item in generated:
        check = verdict(judged, f"J|{item['id']}") or {}
        if not check.get("answerable"):
            dropped[f"{item['set']}_not_answerable"] += 1
            continue
        if not (verdict(specific, f"SP|{item['id']}") or {}).get("specific"):
            dropped[f"{item['set']}_not_specific"] += 1
            continue
        provenance = {"seed_style": item["style"], "evidence": item["evidence"], "answer": item["answer"],
                      "doc_type": item["doc_type"], "era": item["era"], "legal_field": item["legal_field"]}
        if item["set"] == "M":
            # The judge calls one article "enough" too readily to gate on, so this is recorded, not enforced.
            one_is_enough = any((verdict(judged, f"{side}|{item['id']}") or {}).get("answerable") for side in ("JA", "JB"))
            items.append(entry(item["id"], "M", "multihop", item["query"], item["targets"],
                               requires_all_targets=True, judge_says_one_article_enough=one_is_enough, **provenance))
        else:
            new = entry(item["id"], "A", item["style"], item["query"], item["targets"] + alternatives(grades, item["id"]),
                        original_target=item["targets"][0], **provenance)
            items.append(new)
            by_source[item["id"]] = new

    # Variants inherit the gold of a question that survived judging.
    sources = sorted(by_source)
    rng.shuffle(sources)
    full = [s for s in sources if by_source[s]["type"] != "keyword"]
    english = [s for s in sources if f"E{s}" in extras]
    for source in english[:20]:
        items.append(entry(f"{source}_en", "A2", "english", extras[f"E{source}"]["text"].strip(), by_source[source]["expected_targets"], variant_of=source))
    plain = [s for s in full if s not in english[:20]]
    for source in plain[:20]:
        items.append(entry(f"{source}_nd", "A2", "no_diacritics", strip_diacritics(by_source[source]["query"]), by_source[source]["expected_targets"], variant_of=source))
    for source in plain[20:40]:
        items.append(entry(f"{source}_ty", "A2", "typos", add_typos(by_source[source]["query"], rng), by_source[source]["expected_targets"], variant_of=source))

    for index, row in enumerate(seeds["temporal"]):
        query = TEMPORAL_TEMPLATES[index % 2].format(verb=row["verb"], doc=row["old_doc"])
        items.append(entry(f"T{index:03d}", "T", "temporal", query, [{"doc_number": row["new_doc"]}], old_doc=row["old_doc"]))

    for index, row in enumerate(seeds["structural"]):
        items.append(entry(f"S{index:03d}", "S", "structural_count", STRUCTURAL_TEMPLATE.format(**row),
                           [{"doc_number": row["doc_number"]}], exact_answer=row["article_count"]))

    # Set B is reported, not included: too few old queries had any article graded as answering them.
    valid_b = [b for b in set_b if alternatives(grades, b["id"])]
    dropped["B_no_valid_article"] = len(set_b) - len(valid_b)

    with open("tests/fixtures/eval_245_queries.json", "r", encoding="utf-8") as f:
        for row in json.load(f):
            if row["type"] == "negative_abstention":
                items.append({**row, "set": "C", "type": "negative_existing", "expected_targets": [], "should_abstain": True})

    foreign = [r for k, r in sorted(extras.items()) if k.startswith("F") and "Việt Nam" not in r["text"]]
    for row in foreign[:10]:
        items.append({**entry(row["id"], "C", "negative_foreign", row["text"].strip(), []), "should_abstain": True})
    for index, source in enumerate(plain[40:50]):
        number = FAKE_NUMBERS[index]
        query = f"Theo văn bản số {number}, {by_source[source]['query'][0].lower()}{by_source[source]['query'][1:]}"
        items.append({**entry(f"CF{index:02d}", "C", "negative_fictional_doc", query, []), "should_abstain": True})

    # Variants share a split with their source, so a question and its rewrite cannot straddle dev and test.
    splits: Dict[str, str] = {}
    groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for item in items:
        if "variant_of" not in item:
            groups[item["set"]].append(item)
    for rows in groups.values():
        rng.shuffle(rows)
        for index, item in enumerate(rows):
            splits[item["query_id"]] = "dev" if index % 2 == 0 else "test"
    for item in items:
        item["split"] = splits[item.get("variant_of", item["query_id"])]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    for split in ("dev", "test"):
        with open(args.out.replace(".json", f"_{split}.json"), "w", encoding="utf-8") as f:
            json.dump([i for i in items if i["split"] == split], f, ensure_ascii=False, indent=1)

    text_of = {i["id"]: i["sources"] for i in generated}
    with open(os.path.join(args.dir, "spot_check.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "set", "question", "gold_articles", "gold_text", "other_accepted_articles", "valid (y/n)", "note"])
        for set_name, count in SPOT_CHECK.items():
            pool = [i for i in items if i["set"] == set_name]
            for item in rng.sample(pool, min(count, len(pool))):
                gold = [item.get("original_target")] if item.get("original_target") else item["expected_targets"]
                others = [t for t in item["expected_targets"] if t not in gold]
                label = lambda t: f"{t.get('article_code') or ''} {t['doc_number']}".strip()
                source = "\n\n".join(text_of.get(item["query_id"], [])) or item.get("exact_answer", "")
                writer.writerow([item["query_id"], set_name, item["query"], "; ".join(map(label, gold)), source,
                                 "; ".join(map(label, others)), "", ""])

    print("items", len(items), dict(collections.Counter(i["set"] for i in items)))
    print("dev/test", dict(collections.Counter(i["split"] for i in items)))
    print("types", dict(collections.Counter(i["type"] for i in items)))
    print("dropped", dict(dropped))
    set_a = [i for i in items if i["set"] == "A"]
    print("A with alternatives", sum(len(i["expected_targets"]) > 1 for i in set_a), "of", len(set_a))
    print("M judged one-article-enough", sum(bool(i.get("judge_says_one_article_enough")) for i in items if i["set"] == "M"))


if __name__ == "__main__":
    main()
