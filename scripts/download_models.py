import os
from huggingface_hub import snapshot_download

def download_models():
    target_dir = os.path.abspath("models")
    
    # Download BAAI/bge-m3 
    bge_m3_path = os.path.join(target_dir, "bge-m3")
    snapshot_download(
        repo_id="BAAI/bge-m3",
        local_dir=bge_m3_path,
        allow_patterns=["*.onnx", "*.json", "*.txt", "tokenizer*", "*.model"],
        ignore_patterns=["*.pt", "*.bin", "*.safetensors"]
    )

    # Download BAAI/bge-reranker-v2-m3
    reranker_path = os.path.join(target_dir, "bge-reranker-v2-m3")
    snapshot_download(
        repo_id="BAAI/bge-reranker-v2-m3",
        local_dir=reranker_path,
        allow_patterns=["*.onnx", "*.json", "*.txt", "tokenizer*", "*.safetensors", "*.model"]
    )

if __name__ == "__main__":
    download_models()
