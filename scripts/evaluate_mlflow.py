import os
import sys

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import argparse
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import mlflow

load_dotenv("c:/Users/quytr/Documents/code/VCC-project/.env")

from apps.services.embedding_service import EmbeddingService, Int8Reranker
from apps.services.retriever_service import RetrieverService

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

def evaluate(
    dataset_path: str = "tests/fixtures/eval_245_queries.json",
    limit: int = None,
    experiment_name: str = "legal_rag_retrieval_benchmark",
    run_name: str = "bge_m3_rrf_hybrid_benchmark"
):
    print(f"Loading benchmark dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if limit and limit > 0:
        dataset = dataset[:limit]
        print(f"Running evaluation on first {len(dataset)} queries.")
    else:
        print(f"Running evaluation on all {len(dataset)} queries.")

    qdrant_host = os.getenv("QDRANT_HOST", "").rstrip("/")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_host, api_key=qdrant_api_key)

    print("Initializing EmbeddingService (BGE-M3)...")
    embed_service = EmbeddingService(model_name_or_path="models/bge-m3", use_fp16=False, device="cpu")

    print("Initializing Int8Reranker (BGE-Reranker-v2-m3 INT8)...")
    reranker_service = Int8Reranker(model_path="models/bge-reranker-v2-m3")

    print("Initializing RetrieverService...")
    retriever = RetrieverService(
        qdrant_client=client,
        embedding_service=embed_service,
        reranker_service=reranker_service,
        collection_name="legal_chunks_v1"
    )

    query_results = []
    latencies = []

    print("\n--- Starting Evaluation Run ---")
    start_total_time = time.time()

    for idx, item in enumerate(dataset, 1):
        q_id = item["query_id"]
        q_type = item.get("type", "unknown")
        query_text = item["query"]
        should_abstain = item.get("should_abstain", False)
        is_broad = item.get("is_broad_aggregation", False)
        expected_targets = item.get("expected_targets", [])

        t0 = time.time()
        res = retriever.retrieve_and_rerank(
            query=query_text,
            top_k_retrieval=25,
            top_n_rerank=5,
            threshold=0.35
        )
        latency_ms = (time.time() - t0) * 1000.0
        latencies.append(latency_ms)

        top_candidates = res["top_candidates"]
        reranked_chunks = res["reranked_chunks"]

        record = {
            "query_id": q_id,
            "type": q_type,
            "query": query_text,
            "should_abstain": should_abstain,
            "is_broad_aggregation": is_broad,
            "latency_ms": latency_ms,
            "candidates_count": len(top_candidates),
            "reranked_count": len(reranked_chunks),
            "recall_at_25": 0.0,
            "hit_rate_at_5": 0.0,
            "mrr_at_5": 0.0,
            "abstain_success": None
        }

        if should_abstain:
            is_success = len(reranked_chunks) == 0
            record["abstain_success"] = 1.0 if is_success else 0.0
        else:
            matched_targets_25 = [
                t for t in expected_targets
                if any(match_target(cand, t) for cand in top_candidates)
            ]
            matched_targets_5 = [
                t for t in expected_targets
                if any(match_target(cand, t) for cand in reranked_chunks)
            ]
            num_expected = max(len(expected_targets), 1)

            record["hit_rate_at_25"] = 1.0 if len(matched_targets_25) > 0 else 0.0
            record["proportional_recall_at_25"] = len(matched_targets_25) / num_expected
            record["recall_at_25"] = record["hit_rate_at_25"]  # standard binary hit benchmark

            record["hit_rate_at_5"] = 1.0 if len(matched_targets_5) > 0 else 0.0
            record["proportional_recall_at_5"] = len(matched_targets_5) / num_expected

            mrr = 0.0
            for rank_idx, item_rerank in enumerate(reranked_chunks, 1):
                cand = item_rerank.get("candidate", item_rerank)
                if any(match_target(cand, target) for target in expected_targets):
                    mrr = 1.0 / rank_idx
                    break
            record["mrr_at_5"] = mrr

        query_results.append(record)

        print(f"[{idx}/{len(dataset)}] Evaluated {q_id} ({q_type}) in {latency_ms:.1f}ms", flush=True)

    total_duration = time.time() - start_total_time

    # Aggregate Metrics
    non_abstain_records = [r for r in query_results if not r["should_abstain"]]
    abstain_records = [r for r in query_results if r["should_abstain"]]

    overall_hit_rate_25 = float(np.mean([r["hit_rate_at_25"] for r in non_abstain_records])) if non_abstain_records else 0.0
    overall_prop_recall_25 = float(np.mean([r["proportional_recall_at_25"] for r in non_abstain_records])) if non_abstain_records else 0.0
    overall_hit_rate_5 = float(np.mean([r["hit_rate_at_5"] for r in non_abstain_records])) if non_abstain_records else 0.0
    overall_prop_recall_5 = float(np.mean([r["proportional_recall_at_5"] for r in non_abstain_records])) if non_abstain_records else 0.0
    overall_mrr = float(np.mean([r["mrr_at_5"] for r in non_abstain_records])) if non_abstain_records else 0.0
    abstention_acc = float(np.mean([r["abstain_success"] for r in abstain_records])) if abstain_records else 0.0

    p95_latency = float(np.percentile(latencies, 95))
    p50_latency = float(np.percentile(latencies, 50))
    avg_latency = float(np.mean(latencies))

    # Category breakdowns
    categories = sorted(list(set(r["type"] for r in query_results)))
    category_metrics = {}
    for cat in categories:
        cat_records = [r for r in query_results if r["type"] == cat and not r["should_abstain"]]
        if cat_records:
            category_metrics[f"{cat}_hit_rate_at_25"] = float(np.mean([r["hit_rate_at_25"] for r in cat_records]))
            category_metrics[f"{cat}_prop_recall_at_25"] = float(np.mean([r["proportional_recall_at_25"] for r in cat_records]))
            category_metrics[f"{cat}_hit_rate_at_5"] = float(np.mean([r["hit_rate_at_5"] for r in cat_records]))
            category_metrics[f"{cat}_prop_recall_at_5"] = float(np.mean([r["proportional_recall_at_5"] for r in cat_records]))
            category_metrics[f"{cat}_mrr_at_5"] = float(np.mean([r["mrr_at_5"] for r in cat_records]))

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Total Queries: {len(dataset)} (Standard/Challenging: {len(non_abstain_records)}, Abstention: {len(abstain_records)})")
    print(f"HitRate@25 (Binary Hit):        {overall_hit_rate_25*100:.2f}% (Target: >= 92.0%)")
    print(f"Proportional Recall@25 (Ratio): {overall_prop_recall_25*100:.2f}%")
    print(f"HitRate@5 (Top 5 Reranked):     {overall_hit_rate_5*100:.2f}% (Target: >= 85.0%)")
    print(f"Proportional Recall@5 (Ratio):  {overall_prop_recall_5*100:.2f}%")
    print(f"MRR@5:                          {overall_mrr:.4f}")
    print(f"Abstention Accuracy:            {abstention_acc*100:.2f}%")
    print(f"Latency P95: {p95_latency:.1f}ms | P50: {p50_latency:.1f}ms | Avg: {avg_latency:.1f}ms")
    print(f"Total Run Time: {total_duration:.1f}s")
    print("===================================================\n")

    # MLflow Logging
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "dense_model": "BAAI/bge-m3",
            "sparse_model": "BAAI/bge-m3",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "top_k_retrieval": 25,
            "top_n_rerank": 5,
            "rerank_threshold": 0.35,
            "total_queries": len(dataset),
            "non_abstain_queries": len(non_abstain_records),
            "abstain_queries": len(abstain_records)
        })

        mlflow.log_metrics({
            "overall_hit_rate_at_25": overall_hit_rate_25,
            "overall_proportional_recall_at_25": overall_prop_recall_25,
            "overall_recall_at_25": overall_hit_rate_25,  # legacy compatibility
            "overall_hit_rate_at_5": overall_hit_rate_5,
            "overall_proportional_recall_at_5": overall_prop_recall_5,
            "overall_mrr_at_5": overall_mrr,
            "abstention_accuracy": abstention_acc,
            "latency_p95_ms": p95_latency,
            "latency_p50_ms": p50_latency,
            "latency_avg_ms": avg_latency,
            **category_metrics
        })

        os.makedirs("results", exist_ok=True)
        results_file = "results/eval_benchmark_details.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "overall_hit_rate_at_25": overall_hit_rate_25,
                    "overall_proportional_recall_at_25": overall_prop_recall_25,
                    "overall_recall_at_25": overall_hit_rate_25,
                    "overall_hit_rate_at_5": overall_hit_rate_5,
                    "overall_proportional_recall_at_5": overall_prop_recall_5,
                    "overall_mrr_at_5": overall_mrr,
                    "abstention_accuracy": abstention_acc,
                    "latency_p95_ms": p95_latency,
                    "total_queries": len(dataset)
                },
                "category_metrics": category_metrics,
                "query_details": query_results
            }, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(results_file)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Hybrid Retrieval & Reranking MLflow Benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument("--run-name", type=str, default="bge_m3_rrf_hybrid_benchmark", help="MLflow run name")
    args = parser.parse_args()

    evaluate(limit=args.limit, run_name=args.run_name)
