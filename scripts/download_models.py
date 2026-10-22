import os
from huggingface_hub import snapshot_download

def download_models():
    target_dir = os.path.abspath("models")
    
    # 1. Download BAAI/bge-m3 ONNX weights
    print("[1/2] Downloading BAAI/bge-m3 ONNX model artifacts...")
    bge_m3_path = os.path.join(target_dir, "bge-m3")
    snapshot_download(
        repo_id="BAAI/bge-m3",
        local_dir=bge_m3_path,
        allow_patterns=["*.onnx", "*.json", "*.txt", "tokenizer*", "*.model"],
        ignore_patterns=["*.pt", "*.bin", "*.safetensors"]
    )
    print(f"bge-m3 ONNX ready at {bge_m3_path}")

    # 2. Download BAAI/bge-reranker-v2-m3
    print("[2/2] Downloading BAAI/bge-reranker-v2-m3 artifacts...")
    reranker_path = os.path.join(target_dir, "bge-reranker-v2-m3")
    snapshot_download(
        repo_id="BAAI/bge-reranker-v2-m3",
        local_dir=reranker_path,
        allow_patterns=["*.onnx", "*.json", "*.txt", "tokenizer*", "*.safetensors", "*.model"]
    )
    print(f"bge-reranker-v2-m3 ready at {reranker_path}")

if __name__ == "__main__":
    download_models()
