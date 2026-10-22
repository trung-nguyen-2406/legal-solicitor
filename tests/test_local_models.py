import httpx
from apps.services.reranker_service import RerankerService

def test_ollama_local_qwen():
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "prompt": "Xin chào, phản hồi bằng 1 từ: OK",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 10
        }
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert len(data["response"].strip()) > 0

def test_local_reranker():
    service = RerankerService(model_path="models/bge-reranker-v2-m3", device="cpu")
    query = "Quy định về an toàn giao thông đường bộ"
    candidates = [
        {"chunk_id": "c1", "raw_content": "Luật này quy định về quy tắc giao thông đường bộ, phương tiện tham gia giao thông."},
        {"chunk_id": "c2", "raw_content": "Khai hoang đất rừng phòng hộ phục vụ sản xuất lâm nghiệp nông thôn."}
    ]
    results = service.rerank(query, candidates, top_n=2, threshold=0.35)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rerank_score"] > results[1]["rerank_score"]
    assert results[0]["passed_threshold"] is True
    assert results[1]["passed_threshold"] is False

if __name__ == "__main__":
    test_ollama_local_qwen()
    test_local_reranker()
