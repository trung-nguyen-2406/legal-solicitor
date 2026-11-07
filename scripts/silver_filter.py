"""Applies the automatic rejection rules to generated questions. No LLM, no network."""
import argparse
import collections
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_common import RE_DOC_NUMBER

RE_WORD = re.compile(r"\w+", re.UNICODE)
RE_LEAK = re.compile(r"\bđiều\s+\d+|\bkhoản\s+\d+|điều này|văn bản này|thông tư này|nghị định này|luật này|quy định trên", re.IGNORECASE)
RE_POINTER_ANSWER = re.compile(r"theo quy định tại|thực hiện theo|được quy định tại", re.IGNORECASE)
MAX_SHARED_5GRAMS = 0.25
MAX_NEAR_DUPLICATE = 0.7
MIN_WORDS = {"keyword": 4, "other": 7}


def words(text: str) -> List[str]:
    return RE_WORD.findall(unicodedata.normalize("NFC", text or "").lower())


def grams(tokens: List[str], size: int) -> Set[Tuple[str, ...]]:
    return {tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def copied_share(question: str, source: str, size: int = 5) -> float:
    """Share of the question's word 5-grams found verbatim in the source article."""
    own = grams(words(question), size)
    return len(own & grams(words(source), size)) / len(own) if own else 0.0


def quote_is_verbatim(quote: str, source: str) -> bool:
    needle = " ".join(words(quote))
    return len(needle.split()) >= 5 and needle in " ".join(words(source))


def parse(text: str) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def reject_reason(data: Optional[Dict[str, Any]], sources: List[str], quotes: List[str], style: str) -> Optional[str]:
    if not data or not isinstance(data.get("question"), str) or not data["question"].strip():
        return "no_question"
    question = data["question"].strip()
    if len(words(question)) < MIN_WORDS["keyword" if style == "keyword" else "other"]:
        return "too_short"
    if RE_DOC_NUMBER.search(question) or RE_LEAK.search(question):
        return "leaks_reference"
    for key, source in zip(quotes, sources):
        if not quote_is_verbatim(str(data.get(key) or ""), source):
            return "quote_not_verbatim"
    if style != "multihop" and RE_POINTER_ANSWER.search(str(data.get("answer") or "")):
        return "answer_is_a_pointer"
    if copied_share(question, "\n".join(sources)) > MAX_SHARED_5GRAMS:
        return "copies_article"
    return None


def main() -> None:
    from scripts.silver_tasks import STYLES

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="results/v5_1_phase2/seeds.json")
    parser.add_argument("--generated", default="results/v5_1_phase2/gen_out.jsonl")
    parser.add_argument("--out", default="results/v5_1_phase2/filtered.json")
    args = parser.parse_args()

    with open(args.seeds, "r", encoding="utf-8") as f:
        seeds = json.load(f)
    with open(args.generated, "r", encoding="utf-8") as f:
        rows = sorted((json.loads(line) for line in f), key=lambda r: r["id"])

    styles = list(STYLES)
    kept: List[Dict[str, Any]] = []
    reasons: collections.Counter = collections.Counter()
    seen: List[Set[str]] = []
    for row in rows:
        index = int(row["id"][1:])
        data = parse(row["text"])
        if row["id"].startswith("A"):
            item = seeds["articles"][index]
            style, items, quotes = styles[index % len(styles)], [item], ["evidence_quote"]
        else:
            pair = seeds["pairs"][index]
            style, items, quotes = "multihop", [pair["source"], pair["target"]], ["evidence_a", "evidence_b"]
        sources = [f"{i['heading']}\n{i['text']}" for i in items]
        reason = reject_reason(data, sources, quotes, style)
        bag = set(words((data or {}).get("question", "") if isinstance((data or {}).get("question"), str) else ""))
        if not reason and any(len(bag & other) / max(len(bag | other), 1) > MAX_NEAR_DUPLICATE for other in seen):
            reason = "near_duplicate"
        if reason:
            reasons[reason] += 1
            continue
        seen.append(bag)
        kept.append({
            "id": row["id"],
            "set": "A" if row["id"].startswith("A") else "M",
            "style": style,
            "query": data["question"].strip(),
            "answer": str(data.get("answer") or ""),
            "evidence": [str(data[k]) for k in quotes],
            "targets": [{"doc_number": i["doc_number"], "article_code": i["article_code"]} for i in items],
            "sources": sources,
            "doc_type": items[0]["doc_type"],
            "era": items[0]["era"],
            "legal_field": items[0]["legal_field"],
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    print("generated", len(rows), "kept", len(kept), dict(reasons))
    print(collections.Counter((k["set"], k["style"]) for k in kept))


if __name__ == "__main__":
    main()
