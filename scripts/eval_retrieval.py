"""Retrieval benchmark that runs the real fast-path nodes with no LLM call.

Writes <out-dir>/retrieval.json, whose contexts scripts/eval_generation.py reuses as fixed inputs.
"""
import os
import sys

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import argparse
from collections import defaultdict
import json
import time
from typing import Any, Callable, Dict, List

import mlflow
import numpy as np

mlflow.tracing.disable()

from apps.services.post_rerank_service import pre_rerank_diversify
from apps.services.resources import reset_network_clients
from apps.workflow.langgraph_rag import route_domain_or_fast_branch
from apps.workflow.nodes_entry import EncodeQueryAndLinearDomainProbeNode, StructuredQueryDescriptorNode
from apps.workflow.nodes_fast import (
    BiDirectionalGraphExpandNode,
    DedupQuotaAndTemporalBoostNode,
    DirectPointGetNode,
    FilteredSemanticOrLineageAggNode,
    ScopedHybridRrfPrefetch50Node,
    StratifiedMinQuotaSearchNode,
    Top10PruneAndInt8RerankNode,
)
from apps.workflow.nodes_output import _is_grounded
from apps.workflow.state import merge_chunk_bank, merge_latency
from scripts.eval_common import EXACT_51_IDS, score_retrieval, wilson_interval

METRICS = (
    "hit_rate_at_5", "hit_rate_at_25", "proportional_recall_at_5", "proportional_recall_at_25",
    "mrr_at_5", "doc_hit_at_5", "doc_hit_at_25",
)
CONTEXT_FIELDS = (
    "chunk_id", "doc_number", "doc_title", "article_code", "issued_year",
    "raw_content", "breadcrumb", "rerank_score", "is_pinned_amendment",
)


def apply_node(state: Dict[str, Any], node: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
    update = node(state)
    for key, value in update.items():
        if key == "chunk_bank":
            state[key] = merge_chunk_bank(state.get(key), value)
        elif key == "latency_ms":
            state[key] = merge_latency(state.get(key), value)
        elif key != "pointer_scratchpad":
            state[key] = value


def run_fast_path(query: str, rerank: bool) -> Dict[str, Any]:
    state: Dict[str, Any] = {"query": query, "chunk_bank": {}, "latency_ms": {}}
    apply_node(state, StructuredQueryDescriptorNode)
    apply_node(state, EncodeQueryAndLinearDomainProbeNode)
    branch = route_domain_or_fast_branch(state)
    state["branch"] = branch

    if branch == "fast_abstain":
        return state
    if branch == "direct_point":
        apply_node(state, DirectPointGetNode)
        apply_node(state, BiDirectionalGraphExpandNode)
        return state
    if branch == "aggregation":
        apply_node(state, FilteredSemanticOrLineageAggNode)
        return state

    apply_node(state, StratifiedMinQuotaSearchNode if branch == "stratified_multi" else ScopedHybridRrfPrefetch50Node)
    if rerank:
        apply_node(state, Top10PruneAndInt8RerankNode)
        apply_node(state, DedupQuotaAndTemporalBoostNode)
    else:
        ranked = pre_rerank_diversify(state.get("rrf_candidates") or [], target_k=25)
        state["retrieved_top_25"] = ranked
        state["top_chunks"] = ranked[:5]
    return state


def run_with_retry(query: str, rerank: bool, attempts: int = 3) -> Dict[str, Any]:
    for attempt in range(1, attempts + 1):
        try:
            return run_fast_path(query, rerank)
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"  [WARN] attempt {attempt}: {exc}", flush=True)
            reset_network_clients()
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [r for r in records if not r["should_abstain"]]
    negatives = [r for r in records if r["should_abstain"]]
    summary: Dict[str, Any] = {"answerable": len(answerable), "negatives": len(negatives)}
    for metric in METRICS:
        summary[metric] = float(np.mean([r[metric] for r in answerable])) if answerable else 0.0
    for metric in ("hit_rate_at_5", "hit_rate_at_25"):
        hits = sum(r[metric] for r in answerable)
        summary[f"{metric}_ci95"] = wilson_interval(hits, len(answerable))
    aggregates = [r for r in answerable if r["type"] == "aggregation"]
    if aggregates:
        summary["aggregation_pool_recall"] = float(np.mean([r["pool_recall"] for r in aggregates]))
    if negatives:
        summary["negative_rejected"] = float(np.mean([r["rejected"] for r in negatives]))
    if answerable:
        summary["answerable_rejected"] = float(np.mean([r["rejected"] for r in answerable]))
    latencies = [r["retrieval_ms"] for r in records]
    summary["retrieval_p50_ms"] = float(np.percentile(latencies, 50))
    summary["retrieval_p95_ms"] = float(np.percentile(latencies, 95))
    branches: Dict[str, int] = defaultdict(int)
    for r in records:
        branches[r["branch"]] += 1
    summary["branches"] = dict(branches)
    return summary


def write_report(path: str, summary: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    lines = [
        f"Answerable {summary['answerable']} | Negatives {summary['negatives']} | Branches {summary['branches']}",
        "Hit@5   {:.3f}  95% CI {:.3f}-{:.3f}".format(summary["hit_rate_at_5"], *summary["hit_rate_at_5_ci95"]),
        "Hit@25  {:.3f}  95% CI {:.3f}-{:.3f}".format(summary["hit_rate_at_25"], *summary["hit_rate_at_25_ci95"]),
        f"MRR@5   {summary['mrr_at_5']:.3f}",
        f"Doc@5   {summary['doc_hit_at_5']:.3f} | Doc@25 {summary['doc_hit_at_25']:.3f}",
        f"Rejected: negatives {summary.get('negative_rejected', 0):.3f} | answerable {summary.get('answerable_rejected', 0):.3f}",
        f"Retrieval p50 {summary['retrieval_p50_ms']:.0f} ms | p95 {summary['retrieval_p95_ms']:.0f} ms",
        "",
        f"{'category':<28}{'n':>4}{'Hit@5':>8}{'Hit@25':>8}{'Doc@25':>8}",
    ]
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if not r["should_abstain"]:
            by_type[r["type"]].append(r)
    for name, rows in sorted(by_type.items()):
        mean = lambda m: float(np.mean([r[m] for r in rows]))
        lines.append(f"{name:<28}{len(rows):>4}{mean('hit_rate_at_5'):>8.2f}{mean('hit_rate_at_25'):>8.2f}{mean('doc_hit_at_25'):>8.2f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/fixtures/eval_245_queries.json")
    parser.add_argument("--out-dir", default="results/retrieval")
    parser.add_argument("--all", action="store_true", help="Run every query instead of the 51-query subset")
    parser.add_argument("--query-ids", nargs="+", default=None)
    parser.add_argument("--no-rerank", action="store_true", help="Score RRF order only")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    wanted = args.query_ids or (None if args.all else EXACT_51_IDS)
    if wanted:
        by_id = {item["query_id"]: item for item in dataset}
        dataset = [by_id[q] for q in wanted if q in by_id]

    os.makedirs(args.out_dir, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(dataset, 1):
        t0 = time.perf_counter()
        state = run_with_retry(item["query"], rerank=not args.no_rerank)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        top_chunks = state.get("top_chunks") or []
        aggregated = bool(state.get("is_aggregation"))
        candidates = (state.get("agg_catalog_items") if aggregated else state.get("retrieved_top_25")) or []
        scores = score_retrieval(top_chunks, candidates, item.get("expected_targets") or [])
        records.append({
            "query_id": item["query_id"],
            "type": item.get("type", "unknown"),
            "query": item["query"],
            "should_abstain": bool(item.get("should_abstain")),
            "branch": state["branch"],
            "is_aggregation": aggregated,
            "max_score": float(state.get("max_score") or 0.0),
            "rejected": 0.0 if _is_grounded(state) else 1.0,
            "retrieval_ms": round(elapsed_ms, 1),
            "node_latency_ms": state.get("latency_ms") or {},
            "inventory_table": state.get("inventory_table"),
            "top_chunks": [{k: c.get(k) for k in CONTEXT_FIELDS} for c in top_chunks],
            **scores,
        })
        print(
            f"[{idx}/{len(dataset)}] {item['query_id']} {state['branch']} {elapsed_ms / 1000:.1f}s "
            f"hit5={scores['hit_rate_at_5']:.0f} hit25={scores['hit_rate_at_25']:.0f}",
            flush=True,
        )

    summary = summarize(records)
    with open(os.path.join(args.out_dir, "retrieval.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
    write_report(os.path.join(args.out_dir, "retrieval_report.txt"), summary, records)


if __name__ == "__main__":
    main()
