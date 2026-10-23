import math
from typing import List, Dict, Any, Union
from FlagEmbedding import FlagReranker

class RerankerService:
    def __init__(self, model_path: str = "models/bge-reranker-v2-m3", device: str = "cpu", use_fp16: bool = False):
        self.model = FlagReranker(model_path, use_fp16=use_fp16, device=device)

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def compute_score(self, pairs: List[List[str]], batch_size: int = 16) -> List[float]:
        if not pairs:
            return []
        raw_scores = self.model.compute_score(pairs, batch_size=batch_size, normalize=False)
        if isinstance(raw_scores, (int, float)):
            return [self._sigmoid(float(raw_scores))]
        return [self._sigmoid(float(s)) for s in raw_scores]

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
        threshold: float = 0.35,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pairs = [[query, c.get("raw_content", "")] for c in candidates]
        scores = self.compute_score(pairs, batch_size=batch_size)

        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = score
            item["passed_threshold"] = score >= threshold
            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        filtered = [c for c in scored_candidates if c["passed_threshold"]]
        return filtered[:top_n]