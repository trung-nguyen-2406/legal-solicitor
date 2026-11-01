import os
import sys

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

import argparse
import asyncio
import ctypes
import json
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import numpy as np

load_dotenv(".env")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "legal_rag_v4_production_benchmark")

import mlflow
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

from apps.services.resources import (
    get_descriptor_service,
    get_redis_client,
    reset_network_clients,
)
from apps.workflow.langgraph_rag import build_initial_state, compiled_graph
from apps.workflow.nodes_output import flush_pending_audit_tasks
from apps.workflow.state import _brief_chunks

EXACT_51_IDS = [
    "EVAL_001", "EVAL_002", "EVAL_003", "EVAL_004", "EVAL_059", "EVAL_060", "EVAL_061", "EVAL_062",
    "EVAL_089", "EVAL_090", "EVAL_091", "EVAL_092", "EVAL_101", "EVAL_102", "EVAL_103", "EVAL_104",
    "EVAL_141", "EVAL_142", "EVAL_143", "EVAL_144", "EVAL_171", "EVAL_172", "EVAL_173", "EVAL_174",
    "EVAL_191", "EVAL_192", "EVAL_193", "EVAL_194", "EVAL_196", "EVAL_197", "EVAL_198", "EVAL_199",
    "EVAL_204", "EVAL_205", "EVAL_206", "EVAL_207", "EVAL_208", "EVAL_209", "EVAL_210", "EVAL_211",
    "EVAL_212", "EVAL_213", "EVAL_214", "EVAL_226", "EVAL_227", "EVAL_228", "EVAL_229", "EVAL_236",
    "EVAL_237", "EVAL_238", "EVAL_239",
]


def _set_prevent_idle_sleep(enable: bool) -> None:
    if os.name != "nt":
        return
    try:
        es_continuous = 0x80000000
        es_system_required = 0x00000001
        flags = (es_continuous | es_system_required) if enable else es_continuous
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def match_target(cand: Dict[str, Any], target: Dict[str, Any]) -> bool:
    cand_doc = cand.get("doc_number") or cand.get("document_number", "")
    target_doc = target.get("doc_number", "")
    if cand_doc != target_doc:
        return False

    target_art = target.get("article_code")
    if not target_art:
        return True

    cand_art = cand.get("article_code", "")
    cand_crumb = cand.get("breadcrumb", "")
    return cand_art == target_art or target_art in cand_crumb


def compute_running_metrics(
    query_results: List[Dict[str, Any]],
    latencies: List[float],
    path_counts: Dict[str, int],
    total_expected: int,
) -> Dict[str, Any]:
    non_abstain = [r for r in query_results if not r["should_abstain"]]
    abstain_recs = [r for r in query_results if r["should_abstain"]]

    hit25 = float(np.mean([r["hit_rate_at_25"] for r in non_abstain])) if non_abstain else 0.0
    prop25 = float(np.mean([r["proportional_recall_at_25"] for r in non_abstain])) if non_abstain else 0.0
    hit5 = float(np.mean([r["hit_rate_at_5"] for r in non_abstain])) if non_abstain else 0.0
    prop5 = float(np.mean([r["proportional_recall_at_5"] for r in non_abstain])) if non_abstain else 0.0
    mrr5 = float(np.mean([r["mrr_at_5"] for r in non_abstain])) if non_abstain else 0.0
    abs_acc = float(np.mean([r["abstain_success"] for r in abstain_recs])) if abstain_recs else 1.0

    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    avg_lat = float(np.mean(latencies)) if latencies else 0.0

    categories = sorted(list(set(r["type"] for r in query_results)))
    cat_metrics: Dict[str, float] = {}
    for cat in categories:
        cat_recs = [r for r in query_results if r["type"] == cat and not r["should_abstain"]]
        if cat_recs:
            cat_metrics[f"{cat}_hit_rate_at_5"] = float(np.mean([r["hit_rate_at_5"] for r in cat_recs]))
            cat_metrics[f"{cat}_hit_rate_at_25"] = float(np.mean([r["hit_rate_at_25"] for r in cat_recs]))

    mlflow_metrics: Dict[str, float] = {
        "completed_queries": float(len(query_results)),
        "progress_pct": round(100.0 * len(query_results) / max(total_expected, 1), 2),
        "overall_hit_rate_at_25": hit25,
        "overall_proportional_recall_at_25": prop25,
        "overall_hit_rate_at_5": hit5,
        "overall_proportional_recall_at_5": prop5,
        "overall_mrr_at_5": mrr5,
        "abstention_accuracy": abs_acc,
        "latency_p95_ms": p95_lat,
        "latency_p50_ms": p50_lat,
        "latency_avg_ms": avg_lat,
        "path_fast_count": float(path_counts.get("fast_path", 0)),
        "path_slow_react_count": float(path_counts.get("slow_system2_react", 0)),
        "path_abstain_count": float(path_counts.get("abstain", 0)),
        "path_cache_hit_count": float(path_counts.get("cache_hit", 0)),
        **cat_metrics,
    }
    return {
        "mlflow_metrics": mlflow_metrics,
        "categories": categories,
        "cat_metrics": cat_metrics,
        "non_abstain_count": len(non_abstain),
        "abstain_count": len(abstain_recs),
    }


def write_live_reports(
    details_file: str,
    analytics_file: str,
    incorrect_file: str,
    query_results: List[Dict[str, Any]],
    stats: Dict[str, Any],
    path_counts: Dict[str, int],
    total_expected: int,
    elapsed_sec: float,
) -> None:
    m = stats["mlflow_metrics"]
    cat_metrics = stats["cat_metrics"]
    categories = stats["categories"]

    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "progress": f"{len(query_results)}/{total_expected}",
                "summary": {
                    "completed_queries": len(query_results),
                    "total_queries": total_expected,
                    "hit_rate_at_25": m["overall_hit_rate_at_25"],
                    "proportional_recall_at_25": m["overall_proportional_recall_at_25"],
                    "hit_rate_at_5": m["overall_hit_rate_at_5"],
                    "proportional_recall_at_5": m["overall_proportional_recall_at_5"],
                    "mrr_at_5": m["overall_mrr_at_5"],
                    "abstention_accuracy": m["abstention_accuracy"],
                    "latency_p95_ms": m["latency_p95_ms"],
                    "latency_avg_ms": m["latency_avg_ms"],
                    "path_counts": path_counts,
                    "elapsed_sec": round(elapsed_sec, 1),
                },
                "category_metrics": cat_metrics,
                "query_details": query_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "=" * 80,
        f"PRODUCTION LANGGRAPH RAG BENCHMARK REPORT ({len(query_results)}/{total_expected} COMPLETED)",
        "=" * 80,
        f"Execution Paths:                {path_counts}",
        f"HitRate@25 (Binary Hit):        {m['overall_hit_rate_at_25']*100:6.2f}%",
        f"Proportional Recall@25 (Ratio): {m['overall_proportional_recall_at_25']*100:6.2f}%",
        f"HitRate@5 (Top 5 Reranked):     {m['overall_hit_rate_at_5']*100:6.2f}%",
        f"Proportional Recall@5 (Ratio):  {m['overall_proportional_recall_at_5']*100:6.2f}%",
        f"MRR@5:                          {m['overall_mrr_at_5']:8.4f}",
        f"Abstention Accuracy:            {m['abstention_accuracy']*100:6.2f}%",
        f"Latency P95:                    {m['latency_p95_ms']:8.1f} ms",
        f"Latency Avg:                    {m['latency_avg_ms']:8.1f} ms",
        f"Elapsed Duration:               {elapsed_sec:8.1f} s",
        "-" * 80,
        f"{'Category':<28} | {'Hit@25':<9} | {'Hit@5':<9}",
        "-" * 50,
    ]
    for cat in categories:
        if cat != "negative_abstention":
            h25 = cat_metrics.get(f"{cat}_hit_rate_at_25", 0.0) * 100
            h5 = cat_metrics.get(f"{cat}_hit_rate_at_5", 0.0) * 100
            lines.append(f"{cat:<28} | {h25:8.2f}% | {h5:8.2f}%")
    if stats["abstain_count"] > 0:
        lines.append(f"{'negative_abstention':<28} | {'N/A':<9} | {m['abstention_accuracy']*100:5.1f}% Acc")
    lines.append("=" * 80)

    with open(analytics_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    failures = [r for r in query_results if not r["should_abstain"] and r["hit_rate_at_5"] == 0.0]
    lines_fail = [
        "=" * 80,
        f"PRODUCTION BENCHMARK - INCORRECT RETRIEVALS ({len(failures)} failures so far)",
        "=" * 80,
    ]
    for f_idx, f_item in enumerate(failures, 1):
        lines_fail.append(
            f"[{f_idx}] {f_item['query_id']} ({f_item['type']}) | Path: {f_item['path']} | Latency: {f_item['latency_ms']:.0f}ms"
        )
        lines_fail.append(f"    Query: {f_item['query']}")
        lines_fail.append(f"    Hit@25: {f_item['hit_rate_at_25']} | Top Cands: {f_item['candidates_count']}")
        lines_fail.append(f"    Answer preview: {f_item.get('answer_preview', '')}")
        lines_fail.append("-" * 80)

    with open(incorrect_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_fail) + "\n")


async def _run_single_case_with_retry(
    item: Dict[str, Any],
    exp_id: str,
    mlflow_client: mlflow.tracking.MlflowClient,
    max_retries: int = 3,
) -> Dict[str, Any]:
    q_id = item["query_id"]
    q_type = item.get("type", "unknown")
    query_text = item["query"]
    expected_targets = item.get("expected_targets", [])
    should_abstain = bool(item.get("should_abstain", False))
    is_broad = bool(item.get("is_broad_aggregation", False))

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        initial_state = build_initial_state(query_text)
        t_q0 = time.time()
        trace_id: Optional[str] = None
        try:
            with mlflow.start_span(name="legal_rag_request", span_type="CHAIN") as root_span:
                trace_id = root_span.trace_id
                root_span.set_inputs(
                    {
                        "query_id": q_id,
                        "category": q_type,
                        "query": query_text,
                        "should_abstain": should_abstain,
                        "expected_targets": expected_targets[:5],
                    }
                )
                res = await compiled_graph.ainvoke(initial_state)
                wall_ms = (time.time() - t_q0) * 1000.0
                node_sum_ms = sum(
                    float(v) for k, v in (res.get("latency_ms") or {}).items() if k != "total_ms" and float(v) < 600_000.0
                )
                latency_ms = node_sum_ms if (wall_ms > 600_000.0 and node_sum_ms > 0) else wall_ms

                exec_mode = res.get("execution_mode", "fast_path")
                is_abstained = bool(res.get("abstained", False))
                if is_abstained and not res.get("top_chunks"):
                    exec_path = "abstain"
                elif res.get("cache_hit"):
                    exec_path = "cache_hit"
                elif "react" in exec_mode:
                    exec_path = "slow_system2_react"
                else:
                    exec_path = "fast_path"

                top_candidates = list((res.get("chunk_bank") or {}).values()) or res.get("top_chunks", [])
                reranked_chunks = res.get("top_chunks", [])
                answer = res.get("answer", "")
                citations = res.get("citations", [])

                record: Dict[str, Any] = {
                    "query_id": q_id,
                    "type": q_type,
                    "query": query_text,
                    "path": exec_path,
                    "execution_mode": exec_mode,
                    "should_abstain": should_abstain,
                    "is_abstained": is_abstained,
                    "is_broad_aggregation": is_broad,
                    "latency_ms": round(latency_ms, 1),
                    "candidates_count": len(top_candidates),
                    "reranked_count": len(reranked_chunks),
                    "hit_rate_at_25": 0.0,
                    "proportional_recall_at_25": 0.0,
                    "hit_rate_at_5": 0.0,
                    "proportional_recall_at_5": 0.0,
                    "mrr_at_5": 0.0,
                    "abstain_success": None,
                    "answer_preview": (answer or "")[:200],
                    "citations_count": len(citations),
                    "trace_id": trace_id,
                }

                if should_abstain:
                    is_success = is_abstained or len(reranked_chunks) == 0
                    record["abstain_success"] = 1.0 if is_success else 0.0
                    passed_case = is_success
                else:
                    matched_25 = [t for t in expected_targets if any(match_target(c, t) for c in top_candidates)]
                    matched_5 = [t for t in expected_targets if any(match_target(c, t) for c in reranked_chunks)]
                    num_exp = max(len(expected_targets), 1)

                    record["hit_rate_at_25"] = 1.0 if len(matched_25) > 0 else 0.0
                    record["proportional_recall_at_25"] = len(matched_25) / num_exp
                    record["hit_rate_at_5"] = 1.0 if len(matched_5) > 0 else 0.0
                    record["proportional_recall_at_5"] = len(matched_5) / num_exp

                    mrr = 0.0
                    for r_idx, c in enumerate(reranked_chunks, 1):
                        if any(match_target(c, t) for t in expected_targets):
                            mrr = 1.0 / r_idx
                            break
                    record["mrr_at_5"] = mrr
                    passed_case = record["hit_rate_at_5"] == 1.0

                root_span.set_outputs(
                    {
                        "query_id": q_id,
                        "path": exec_path,
                        "abstained": is_abstained,
                        "hit_rate_at_5": record["hit_rate_at_5"],
                        "hit_rate_at_25": record["hit_rate_at_25"],
                        "mrr_at_5": round(record["mrr_at_5"], 4),
                        "abstain_success": record["abstain_success"],
                        "top_chunks": _brief_chunks(reranked_chunks, 7),
                        "citations": citations,
                        "answer": answer,
                        "node_timings_ms": res.get("latency_ms", {}),
                        "total_latency_ms": round(latency_ms, 1),
                    }
                )

            try:
                mlflow.flush_trace_async_logging()
            except Exception:
                pass

            if trace_id:
                try:
                    for tk, tv in {
                        "query_id": q_id,
                        "category": q_type,
                        "path": exec_path,
                        "passed": str(passed_case),
                        "hit_at_5": str(record["hit_rate_at_5"]),
                        "hit_at_25": str(record["hit_rate_at_25"]),
                    }.items():
                        mlflow_client.set_trace_tag(trace_id, tk, tv)

                    mlflow.log_feedback(
                        trace_id=trace_id,
                        name="pass_eval",
                        value=bool(passed_case),
                        rationale=f"{q_id} ({q_type}) via {exec_path}",
                    )
                    if should_abstain:
                        mlflow.log_feedback(
                            trace_id=trace_id,
                            name="abstain_correct",
                            value=bool(record["abstain_success"] == 1.0),
                        )
                    else:
                        mlflow.log_feedback(
                            trace_id=trace_id,
                            name="hit_at_5",
                            value=bool(record["hit_rate_at_5"] == 1.0),
                        )
                        mlflow.log_feedback(
                            trace_id=trace_id,
                            name="hit_at_25",
                            value=bool(record["hit_rate_at_25"] == 1.0),
                        )
                        mlflow.log_feedback(
                            trace_id=trace_id,
                            name="mrr_at_5",
                            value=round(float(record["mrr_at_5"]), 4),
                        )
                except Exception:
                    pass

            return record
        except Exception as exc:
            last_err = exc
            print(f"  [WARN] {q_id} attempt {attempt}/{max_retries} encountered transient error: {exc}", flush=True)
            try:
                mlflow.flush_trace_async_logging()
                if trace_id:
                    mlflow_client.delete_traces(experiment_id=exp_id, trace_ids=[trace_id])
            except Exception:
                pass
            reset_network_clients()
            await asyncio.sleep(3.0 * attempt)

    raise RuntimeError(f"Failed {q_id} after {max_retries} attempts: {last_err}") from last_err


async def evaluate_async(
    dataset_path: str = "tests/fixtures/eval_245_queries.json",
    limit: Optional[int] = None,
    start_idx: int = 0,
    experiment_name: str = "legal_rag_v4_production_benchmark",
    run_name: str = "production_langgraph_eval_51q",
    fresh_cache: bool = True,
    resume_run_id: Optional[str] = None,
):
    _set_prevent_idle_sleep(True)
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        exp = mlflow.set_experiment(experiment_name)
        mlflow_client = mlflow.tracking.MlflowClient()

        with open(dataset_path, "r", encoding="utf-8") as f:
            all_dataset = json.load(f)

        dataset_map = {item["query_id"]: item for item in all_dataset}
        dataset = [dataset_map[qid] for qid in EXACT_51_IDS if qid in dataset_map]
        if start_idx > 0:
            dataset = dataset[start_idx:]
        if limit and limit > 0:
            dataset = dataset[:limit]

        os.makedirs("results", exist_ok=True)
        details_file = "results/eval_production_details.json"
        analytics_file = "results/benchmark_analytics_production.txt"
        incorrect_file = "results/incorrect_retrievals_production.txt"

        query_results: List[Dict[str, Any]] = []
        latencies: List[float] = []
        path_counts = {"cache_hit": 0, "fast_path": 0, "slow_system2_react": 0, "abstain": 0}
        completed_ids = set()

        if resume_run_id and os.path.exists(details_file):
            saved = json.load(open(details_file, "r", encoding="utf-8"))
            for r in saved.get("query_details", []):
                query_results.append(r)
                latencies.append(float(r["latency_ms"]))
                p = r.get("path", "fast_path")
                path_counts[p] = path_counts.get(p, 0) + 1
                completed_ids.add(r["query_id"])
            print(f"Resuming with {len(completed_ids)} completed queries: {sorted(completed_ids)}", flush=True)

        remaining_items = [item for item in dataset if item["query_id"] not in completed_ids]

        if fresh_cache and remaining_items:
            r_client = get_redis_client()
            desc_svc = get_descriptor_service()
            if r_client:
                keys_to_clear = [
                    f"rag:cache:{desc_svc.parse(item['query'])['query_sha256']}" for item in remaining_items
                ]
                if keys_to_clear:
                    deleted = r_client.delete(*keys_to_clear)
                    print(f"Cleared {deleted} stale eval cache keys from Redis for remaining {len(remaining_items)} queries.", flush=True)

        print(f"--- Running Production LangGraph Evaluation ({len(dataset)} total, {len(remaining_items)} remaining) ---", flush=True)
        start_total_time = time.time()

        run_kwargs: Dict[str, Any] = {"run_id": resume_run_id} if resume_run_id else {"run_name": run_name}
        with mlflow.start_run(**run_kwargs) as active_run:
            if not resume_run_id:
                mlflow.log_params(
                    {
                        "pipeline": "LangGraph Production Vietnamese Legal RAG (SDD v4.0)",
                        "llm_model": os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b-instruct-q4_K_M"),
                        "dense_model": "BAAI/bge-m3",
                        "reranker_model": "BAAI/bge-reranker-v2-m3 (INT8)",
                        "total_queries": len(dataset),
                        "fresh_cache_on_start": fresh_cache,
                        "max_react_steps": 2,
                        "confidence_threshold": 0.20,
                    }
                )

            for idx, item in enumerate(dataset, 1):
                if item["query_id"] in completed_ids:
                    continue

                record = await _run_single_case_with_retry(
                    item=item,
                    exp_id=exp.experiment_id,
                    mlflow_client=mlflow_client,
                    max_retries=3,
                )

                exec_path = record["path"]
                path_counts[exec_path] = path_counts.get(exec_path, 0) + 1
                latencies.append(float(record["latency_ms"]))
                query_results.append(record)

                elapsed_so_far = time.time() - start_total_time
                stats = compute_running_metrics(query_results, latencies, path_counts, len(dataset))

                mlflow.log_metrics(stats["mlflow_metrics"], step=len(query_results))
                write_live_reports(
                    details_file=details_file,
                    analytics_file=analytics_file,
                    incorrect_file=incorrect_file,
                    query_results=query_results,
                    stats=stats,
                    path_counts=path_counts,
                    total_expected=len(dataset),
                    elapsed_sec=elapsed_so_far,
                )

                m = stats["mlflow_metrics"]
                print(
                    f"[{len(query_results)}/{len(dataset)}] {record['query_id']} ({record['type']}) | Path: {exec_path} | "
                    f"{record['latency_ms']/1000.0:.1f}s | Case Hit@5: {record['hit_rate_at_5']} | "
                    f"LIVE Hit@25: {m['overall_hit_rate_at_25']*100:.1f}% Hit@5: {m['overall_hit_rate_at_5']*100:.1f}% "
                    f"MRR@5: {m['overall_mrr_at_5']:.3f} AbstainAcc: {m['abstention_accuracy']*100:.0f}%",
                    flush=True,
                )

            await flush_pending_audit_tasks()
            total_duration = time.time() - start_total_time
            final_stats = compute_running_metrics(query_results, latencies, path_counts, len(dataset))
            write_live_reports(
                details_file=details_file,
                analytics_file=analytics_file,
                incorrect_file=incorrect_file,
                query_results=query_results,
                stats=final_stats,
                path_counts=path_counts,
                total_expected=len(dataset),
                elapsed_sec=total_duration,
            )

            mlflow.log_artifact(details_file)
            mlflow.log_artifact(analytics_file)
            mlflow.log_artifact(incorrect_file)
            try:
                mlflow.flush_trace_async_logging()
            except Exception:
                pass

            fm = final_stats["mlflow_metrics"]
            print("\n================ PRODUCTION PIPELINE BENCHMARK RESULTS ================", flush=True)
            print(f"Total Queries: {len(dataset)} | Paths: {path_counts}", flush=True)
            print(f"HitRate@25: {fm['overall_hit_rate_at_25']*100:.2f}% | PropRecall@25: {fm['overall_proportional_recall_at_25']*100:.2f}%", flush=True)
            print(f"HitRate@5:  {fm['overall_hit_rate_at_5']*100:.2f}% | PropRecall@5:  {fm['overall_proportional_recall_at_5']*100:.2f}%", flush=True)
            print(f"MRR@5:      {fm['overall_mrr_at_5']:.4f} | Abstention Acc: {fm['abstention_accuracy']*100:.2f}%", flush=True)
            print(f"Latency P95: {fm['latency_p95_ms']:.1f}ms | Avg: {fm['latency_avg_ms']:.1f}ms | Total: {total_duration:.1f}s", flush=True)
            print(f"MLflow Run ID: {active_run.info.run_id}", flush=True)
            print("=======================================================================\n", flush=True)
    finally:
        _set_prevent_idle_sleep(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--experiment-name", type=str, default="legal_rag_v4_production_benchmark")
    parser.add_argument("--run-name", type=str, default="production_langgraph_eval_51q")
    parser.add_argument("--resume-run-id", type=str, default=None)
    parser.add_argument("--keep-cache", action="store_true", help="Do not clear pre-existing eval cache keys")
    args = parser.parse_args()

    asyncio.run(
        evaluate_async(
            limit=args.limit,
            start_idx=args.start_idx,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            fresh_cache=not args.keep_cache,
            resume_run_id=args.resume_run_id,
        )
    )
