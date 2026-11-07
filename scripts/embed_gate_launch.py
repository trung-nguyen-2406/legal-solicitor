"""Starts scripts/colab_embed_gate.py on a one-shot T4 and scores the result.

The Postgres DSN is read from .env and passed with --env at runtime, so it never reaches a file or a log.
"""
import argparse
import base64
import collections
import gzip
import json
import os
import subprocess
import tempfile

from dotenv import load_dotenv

COLAB = os.path.expanduser("~/.local/bin/colab")
HERE = os.path.dirname(os.path.abspath(__file__))
PHASE2 = "results/v5_1_phase2"
OUT_DIR = "results/v5_1_phase4"
RETRIEVAL_SETS = ("A", "A2", "M")
# Fixed before the run: the new text must gain 3 points of Hit@25 and lose nothing at Hit@5.
MIN_HIT25_GAIN, MIN_HIT5_GAIN = 0.03, 0.0


def gate_input() -> str:
    with open("tests/fixtures/silver_v1_dev.json", "r", encoding="utf-8") as f:
        queries = [q for q in json.load(f) if q["set"] in RETRIEVAL_SETS]
    with open(os.path.join(PHASE2, "pools.jsonl"), "r", encoding="utf-8") as f:
        pools = {row["id"]: row["pool"] for row in map(json.loads, f)}
    documents = {t["doc_number"] for q in queries for t in q["expected_targets"]}
    for query in queries:
        documents.update(c["doc_number"] for c in pools.get(query.get("variant_of", query["query_id"]), []))
    payload = {"queries": queries, "priority_docs": sorted(documents)}
    return base64.b64encode(gzip.compress(json.dumps(payload, ensure_ascii=False).encode())).decode()


def hit_rate(rows, name: str, depth: int) -> float:
    return sum(1 for r in rows if r[name] and r[name] <= depth) / len(rows)


def score_gate(rows) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "gate.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["set"]].append(row)
    for name, members in [("all", rows)] + sorted(groups.items()):
        print(f"{name:<4} n={len(members):<4} Hit@5 {hit_rate(members, 'old', 5):.3f} -> {hit_rate(members, 'new', 5):.3f}"
              f"   Hit@25 {hit_rate(members, 'old', 25):.3f} -> {hit_rate(members, 'new', 25):.3f}")
    gained = sum(1 for r in rows if (r["new"] or 999) <= 25 < (r["old"] or 999))
    lost = sum(1 for r in rows if (r["old"] or 999) <= 25 < (r["new"] or 999))
    gain25 = hit_rate(rows, "new", 25) - hit_rate(rows, "old", 25)
    gain5 = hit_rate(rows, "new", 5) - hit_rate(rows, "old", 5)
    print(f"Hit@25 gained {gained}, lost {lost}. Gate:", "PASS" if gain25 >= MIN_HIT25_GAIN and gain5 >= MIN_HIT5_GAIN else "FAIL")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=75)
    args = parser.parse_args()

    load_dotenv(".env")
    with open(os.path.join(HERE, "colab_embed_gate.py"), "r", encoding="utf-8") as f:
        script = f.read().replace("__GATE_INPUT__", gate_input())

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "job.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        command = [COLAB, "run", "--gpu", "T4", "--session", "v51-gate", "--timeout", str(args.minutes * 60),
                   "--env", f"DATABASE_DSN={os.environ['DATABASE_DSN']}", path]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        gate_rows = None
        for line in process.stdout:
            if line.startswith("@@GATE "):
                gate_rows = json.loads(line[len("@@GATE "):])
            elif "@@END" not in line:
                print(line.rstrip()[:300], flush=True)
        print("exit", process.wait(), flush=True)
    if gate_rows:
        score_gate(gate_rows)


if __name__ == "__main__":
    main()
