import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel

try:
    import onnxruntime as ort
except ImportError:
    ort = None

RERANK_MAX_TOKENS = 512


def _find_onnx_file(model_path: str) -> Optional[str]:
    if os.path.isfile(model_path) and model_path.endswith(".onnx"):
        return model_path
    if not os.path.isdir(model_path):
        return None
    for candidate in ("model_int8.onnx", "model_quantized.onnx", "onnx/model_int8.onnx", "onnx/model_quantized.onnx"):
        full = os.path.join(model_path, candidate)
        if os.path.isfile(full):
            return full
    plain = os.path.join(model_path, "model.onnx")
    if os.path.isfile(plain) and os.path.getsize(plain) > 5 * 1024 * 1024:
        return plain
    return None


class EmbeddingService:
    def __init__(self, model_name_or_path: str = "models/bge-m3", use_fp16: bool = False, device: Optional[str] = None):
        self.onnx_path = _find_onnx_file(model_name_or_path)
        self.ort_session = None
        self.tokenizer = None
        self.model = None

        if self.onnx_path and ort is not None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            tok_dir = os.path.dirname(self.onnx_path) if os.path.isfile(model_name_or_path) else model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        else:
            self.model = BGEM3FlagModel(model_name_or_path, use_fp16=use_fp16, device=device)

    def encode(self, texts: Union[str, List[str]], batch_size: int = 12) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        is_single = isinstance(texts, str)
        input_texts = [texts] if is_single else texts

        if self.ort_session is not None and self.tokenizer is not None:
            results: List[Dict[str, Any]] = []
            input_names = {inp.name for inp in self.ort_session.get_inputs()}
            for start in range(0, len(input_texts), batch_size):
                batch = input_texts[start : start + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding="longest",
                    truncation=True,
                    max_length=1024,
                    return_tensors="np",
                )
                ort_inputs = {k: v for k, v in enc.items() if k in input_names}
                ort_outs = self.ort_session.run(None, ort_inputs)
                hidden = ort_outs[0]
                cls_vecs = hidden[:, 0, :] if hidden.ndim == 3 else hidden
                norms = np.linalg.norm(cls_vecs, axis=1, keepdims=True)
                dense_vecs = cls_vecs / np.clip(norms, 1e-12, None)
                for idx in range(len(batch)):
                    ids_row = enc["input_ids"][idx]
                    mask_row = enc["attention_mask"][idx]
                    token_counts: Dict[int, float] = {}
                    for tid, m in zip(ids_row.tolist(), mask_row.tolist()):
                        if m == 1 and tid not in (0, 1, 2):
                            token_counts[int(tid)] = max(token_counts.get(int(tid), 0.0), 1.0)
                    results.append({
                        "dense": dense_vecs[idx].astype(np.float32).tolist(),
                        "sparse": {
                            "indices": list(token_counts.keys()),
                            "values": list(token_counts.values()),
                        },
                    })
            return results[0] if is_single else results

        outputs = self.model.encode(
            input_texts,
            batch_size=batch_size,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        results = []
        dense_vecs = outputs["dense_vecs"]
        lexical_weights = outputs["lexical_weights"]

        for idx in range(len(input_texts)):
            sparse_dict = lexical_weights[idx]
            indices = [int(k) for k in sparse_dict.keys()]
            values = [float(v) for v in sparse_dict.values()]
            results.append({
                "dense": dense_vecs[idx].tolist(),
                "sparse": {
                    "indices": indices,
                    "values": values,
                },
            })

        return results[0] if is_single else results


class Int8Reranker:
    def __init__(self, model_path: str = "models/bge-reranker-v2-m3", max_tokens: int = RERANK_MAX_TOKENS):
        self.max_tokens = max_tokens
        self.onnx_path = _find_onnx_file(model_path)
        tok_dir = os.path.dirname(self.onnx_path) if os.path.isfile(model_path) else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        self.ort_session = None
        self.torch_model = None

        if self.onnx_path and ort is not None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            base_model.eval()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.torch_model = torch.ao.quantization.quantize_dynamic(
                    base_model, {torch.nn.Linear}, dtype=torch.qint8
                )

    @staticmethod
    def format_chunk_text(chunk: Dict[str, Any]) -> str:
        anchor_tag = chunk.get("anchor_tag") or f"[{chunk.get('doc_number', '')}:{chunk.get('article_code', '')}]"
        raw_content = chunk.get("raw_content") or chunk.get("breadcrumb") or ""
        return f"{anchor_tag} {raw_content}".strip()

    def score_pairs(self, pairs: List[Tuple[str, str]], batch_size: int = 8) -> List[float]:
        if not pairs:
            return []

        indexed_pairs = sorted(enumerate(pairs), key=lambda item: len(item[1][0]) + len(item[1][1]))
        raw_logits = [0.0] * len(pairs)

        if self.ort_session is not None:
            input_names = {inp.name for inp in self.ort_session.get_inputs()}
            for start in range(0, len(indexed_pairs), batch_size):
                sub = indexed_pairs[start : start + batch_size]
                batch_texts = [p for _, p in sub]
                enc = self.tokenizer(
                    batch_texts,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="np",
                )
                ort_inputs = {k: v for k, v in enc.items() if k in input_names}
                logits = np.asarray(self.ort_session.run(None, ort_inputs)[0]).reshape(-1)
                for (orig_idx, _), val in zip(sub, logits.tolist()):
                    raw_logits[orig_idx] = float(val)
            return raw_logits

        with torch.inference_mode():
            for start in range(0, len(indexed_pairs), batch_size):
                sub = indexed_pairs[start : start + batch_size]
                batch_texts = [p for _, p in sub]
                enc = self.tokenizer(
                    batch_texts,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                )
                logits = self.torch_model(**enc).logits.view(-1).tolist()
                for (orig_idx, _), val in zip(sub, logits):
                    raw_logits[orig_idx] = float(val)

        return raw_logits

    def rerank_logits(self, query: str, chunks: List[Dict[str, Any]], batch_size: int = 8) -> List[float]:
        if not chunks:
            return []
        pairs = [(query, self.format_chunk_text(c)) for c in chunks]
        return self.score_pairs(pairs, batch_size=batch_size)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
        threshold: float = 0.35,
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        logits = self.rerank_logits(query, candidates, batch_size=batch_size)
        scored = []
        for c, logit in zip(candidates, logits):
            item = dict(c)
            prob = float(1.0 / (1.0 + np.exp(-logit)))
            item["rerank_score"] = prob
            item["passed_threshold"] = bool(prob >= threshold)
            scored.append(item)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return [c for c in scored if c["passed_threshold"]][:top_n]
