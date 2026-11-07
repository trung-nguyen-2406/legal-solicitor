"""Scores answers on the fixed contexts saved by scripts/eval_retrieval.py.

Contexts are identical across runs, so the model or prompt is the only variable.
"""
import os
import sys

os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import argparse
import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np

from apps.services.llm_service import (
    MAX_CONTEXT_TOKENS,
    build_irac_context_and_citations,
    get_llm_model_name,
    stream_natural_ollama,
)
from scripts.eval_common import (
    citation_precision,
    has_language_drift,
    is_complete_answer,
    is_refusal,
    unsupported_numbers,
)


async def answer_one(record: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
    chunks = record["top_chunks"]
    context, _ = build_irac_context_and_citations(
        top_chunks=chunks,
        inventory_table=record.get("inventory_table"),
        max_context_tokens=MAX_CONTEXT_TOKENS,
    )
    started = time.perf_counter()
    first_token: Optional[float] = None
    parts: List[str] = []
    async for delta in stream_natural_ollama(query=record["query"], context_block=context, model=model, timeout=240.0):
        if first_token is None:
            first_token = time.perf_counter() - started
        parts.append(delta)
    answer = "".join(parts).strip()

    return {
        "query_id": record["query_id"],
        "type": record["type"],
        "should_abstain": record["should_abstain"],
        "answer": answer,
        "first_token_s": round(first_token or 0.0, 2),
        "total_s": round(time.perf_counter() - started, 2),
        "citation_precision": citation_precision(answer, chunks),
        "unsupported_numbers": unsupported_numbers(answer, context),
        "language_drift": has_language_drift(answer, record["query"]),
        "complete": is_complete_answer(answer),
        "refused": is_refusal(answer),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cited = [r["citation_precision"] for r in rows if r["citation_precision"] is not None]
    total = [r["total_s"] for r in rows]
    first = [r["first_token_s"] for r in rows]
    negatives = [r for r in rows if r["should_abstain"]]
    return {
        "n": len(rows),
        "citation_precision": float(np.mean(cited)) if cited else None,
        "answers_with_citations": len(cited) / len(rows),
        "answers_with_unsupported_numbers": float(np.mean([bool(r["unsupported_numbers"]) for r in rows])),
        "language_drift": float(np.mean([r["language_drift"] for r in rows])),
        "complete_answers": float(np.mean([r["complete"] for r in rows])),
        "refusal_on_negatives": float(np.mean([r["refused"] for r in negatives])) if negatives else None,
        "first_token_p50_s": float(np.percentile(first, 50)),
        "first_token_p95_s": float(np.percentile(first, 95)),
        "total_p50_s": float(np.percentile(total, 50)),
        "total_p95_s": float(np.percentile(total, 95)),
    }


async def run(args: argparse.Namespace) -> None:
    with open(args.contexts, "r", encoding="utf-8") as f:
        records = [r for r in json.load(f)["records"] if r["top_chunks"]]
    if args.limit:
        records = records[: args.limit]

    model = args.model or get_llm_model_name()
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, 1):
        row = await answer_one(record, args.model)
        rows.append(row)
        print(f"[{idx}/{len(records)}] {row['query_id']} {row['total_s']:.1f}s cite={row['citation_precision']}", flush=True)

    summary = summarize(rows)
    print(json.dumps({"model": model, **summary}, ensure_ascii=False, indent=2), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": model, "summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)


def rescore(path: str, contexts_path: str) -> None:
    """Recomputes the automatic checks from saved answers without calling the LLM."""
    with open(contexts_path, "r", encoding="utf-8") as f:
        records = {r["query_id"]: r for r in json.load(f)["records"]}
    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    for row in saved["rows"]:
        record = records[row["query_id"]]
        context, _ = build_irac_context_and_citations(
            top_chunks=record["top_chunks"],
            inventory_table=record.get("inventory_table"),
            max_context_tokens=MAX_CONTEXT_TOKENS,
        )
        row["citation_precision"] = citation_precision(row["answer"], record["top_chunks"])
        row["unsupported_numbers"] = unsupported_numbers(row["answer"], context)
        row["complete"] = is_complete_answer(row["answer"])
        row["refused"] = is_refusal(row["answer"])
    saved["summary"] = summarize(saved["rows"])
    print(json.dumps({"model": saved["model"], **saved["summary"]}, ensure_ascii=False, indent=2), flush=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="results/retrieval/retrieval.json")
    parser.add_argument("--out", default="results/generation/generation.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rescore", action="store_true", help="Recompute checks for --out without calling the LLM")
    cli_args = parser.parse_args()
    if cli_args.rescore:
        rescore(cli_args.out, cli_args.contexts)
    else:
        asyncio.run(run(cli_args))
