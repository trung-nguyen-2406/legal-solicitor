"""Collects each query's top articles from the live retriever, for grading other valid answers.

RRF order only, no reranker and no LLM. Resumes from an existing output file.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from typing import Any, Dict, List

from scripts.eval_common import ranked_pool
from scripts.eval_retrieval import run_with_retry

POOL_SIZE = 8
SNIPPET_CHARS = 1400


def pool_for(query: str) -> Dict[str, Any]:
    state = run_with_retry(query, rerank=False)
    chunks = ranked_pool(state.get("top_chunks") or [], state.get("retrieved_top_25") or state.get("rrf_candidates") or [])
    return {
        "branch": state.get("branch"),
        "pool": [
            {
                "doc_number": c.get("doc_number"),
                "article_code": c.get("article_code"),
                "breadcrumb": c.get("breadcrumb"),
                "text": (c.get("raw_content") or "")[:SNIPPET_CHARS],
            }
            for c in chunks[: POOL_SIZE + 4]
            if c.get("doc_number") and c.get("article_code")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="JSON list of {id, query}")
    parser.add_argument("--out", default="results/v5_1_phase2/pools.jsonl")
    args = parser.parse_args()

    with open(args.queries, "r", encoding="utf-8") as f:
        queries: List[Dict[str, Any]] = json.load(f)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}

    started = time.time()
    with open(args.out, "a", encoding="utf-8") as out:
        for index, item in enumerate(q for q in queries if q["id"] not in done):
            out.write(json.dumps({"id": item["id"], **pool_for(item["query"])}, ensure_ascii=False) + "\n")
            out.flush()
            if index % 25 == 24:
                print(f"{index + 1} done, {(time.time() - started) / (index + 1):.1f}s each", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
