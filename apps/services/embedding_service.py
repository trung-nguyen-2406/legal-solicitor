from typing import List, Dict, Any, Union
from FlagEmbedding import BGEM3FlagModel

class EmbeddingService:
    def __init__(self, model_name_or_path: str = "BAAI/bge-m3", use_fp16: bool = True, device: str = None):
        self.model = BGEM3FlagModel(model_name_or_path, use_fp16=use_fp16, device=device)

    def encode(self, texts: Union[str, List[str]], batch_size: int = 12) -> List[Dict[str, Any]]:
        is_single = isinstance(texts, str)
        input_texts = [texts] if is_single else texts

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
                }
            })

        return results[0] if is_single else results
