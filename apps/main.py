import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import numpy as np
from pydantic import BaseModel, Field

from apps.services.cache_service import invalidate_by_doc
from apps.services.llm_service import get_llm_model_name
from apps.services.resources import (
    get_descriptor_service,
    get_domain_probe,
    get_embedding_service,
    get_int8_reranker,
)
from apps.workflow.langgraph_rag import build_initial_state, compiled_graph
from apps.workflow.nodes_output import flush_pending_audit_tasks
import mlflow

load_dotenv(".env")
logger = logging.getLogger(__name__)

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "vietnamese_legal_rag"))


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    stream: bool = Field(default=True)
    request_id: Optional[str] = Field(default=None)


class CacheInvalidateRequest(BaseModel):
    doc_number: str = Field(..., min_length=1, max_length=100)


async def _warmup_ollama_keep_alive() -> bool:
    ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    if not ollama_host.startswith("http"):
        ollama_host = f"http://{ollama_host}"
    payload = {
        "model": get_llm_model_name(),
        "messages": [{"role": "user", "content": "OK"}],
        "stream": False,
        "keep_alive": "30m",
        "options": {"num_predict": 1, "temperature": 0.0},
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(f"{ollama_host.rstrip('/')}/api/chat", json=payload)
            resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Ollama keep-alive warmup skipped/failed: %s", exc)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_descriptor_service()

    embed_svc = get_embedding_service()
    dummy_enc = await asyncio.to_thread(
        embed_svc.encode,
        "Quy định pháp luật Việt Nam về xử phạt vi phạm hành chính.",
    )

    reranker = get_int8_reranker()
    await asyncio.to_thread(
        reranker.score_pairs,
        [
            ("Quy định xử phạt giao thông đường bộ", "Điều 6. Xử phạt người điều khiển xe ô tô"),
            ("Quy định lưu trữ hồ sơ quy hoạch", "Điều 21. Lưu trữ hồ sơ đồ án quy hoạch xây dựng"),
        ],
    )

    probe = get_domain_probe()
    _ = float(np.dot(probe["w"], np.array(dummy_enc["dense"], dtype=np.float32)) + probe["b"])

    ollama_ready = await _warmup_ollama_keep_alive()
    app.state.warmup_status = {
        "bge_m3_ready": True,
        "bge_reranker_int8_ready": True,
        "domain_probe_ready": True,
        "ollama_keep_alive_30m": ollama_ready,
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
    }

    try:
        yield
    finally:
        await flush_pending_audit_tasks()


app = FastAPI(
    title="Vietnamese Legal RAG System (LangGraph SDD v4.0)",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_sse_frame(event_name: str, data: Dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_graph_sse(initial_state: Dict[str, Any]) -> AsyncIterator[str]:
    with mlflow.start_span(name="legal_rag_request", span_type="CHAIN") as root_span:
        root_span.set_inputs({"request_id": initial_state.get("request_id"), "query": initial_state.get("query"), "stream": True})
        tokens: list[str] = []
        done_meta: Dict[str, Any] = {}
        abstained_flag = False
        citations_out: Any = []
        async for item in compiled_graph.astream(initial_state, stream_mode="custom"):
            if isinstance(item, dict) and "event" in item:
                ev = str(item["event"])
                data = item.get("data") or {}
                if ev == "metadata":
                    abstained_flag = bool(data.get("abstained", False))
                    citations_out = data.get("citations", [])
                elif ev == "token":
                    tokens.append(str(data.get("delta", "")))
                elif ev == "done":
                    done_meta = data
                yield format_sse_frame(ev, data)
        root_span.set_outputs({
            "abstained": abstained_flag,
            "citations": citations_out,
            "answer": "".join(tokens).strip(),
            "metrics": done_meta.get("metrics", {}),
        })


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "warmup": getattr(app.state, "warmup_status", {}),
    }


@app.post("/api/v1/query")
async def query_legal_rag(req: QueryRequest):
    initial_state = build_initial_state(query=req.query, request_id=req.request_id)

    if req.stream:
        return StreamingResponse(
            stream_graph_sse(initial_state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    with mlflow.start_span(name="legal_rag_request", span_type="CHAIN") as root_span:
        root_span.set_inputs({"request_id": initial_state.get("request_id"), "query": req.query, "stream": False})
        final_state = await compiled_graph.ainvoke(initial_state)
        response_payload = {
            "request_id": final_state.get("request_id", initial_state["request_id"]),
            "query": final_state.get("query", req.query),
            "abstained": bool(final_state.get("abstained", False)),
            "abstain_reason": final_state.get("abstain_reason"),
            "answer": final_state.get("answer", ""),
            "citations": final_state.get("citations", []),
            "inventory_table": final_state.get("inventory_table"),
            "execution_mode": final_state.get("execution_mode", "fast_path"),
            "cache_hit": bool(final_state.get("cache_hit", False)),
            "react_steps": int(final_state.get("react_step", 0)),
            "latency_ms": final_state.get("latency_ms", {}),
        }
        root_span.set_outputs(response_payload)
    return JSONResponse(content=response_payload)


@app.post("/api/v1/cache/invalidate")
async def invalidate_document_cache(req: CacheInvalidateRequest) -> Dict[str, Any]:
    deleted = invalidate_by_doc(req.doc_number)
    return {"doc_number": req.doc_number, "invalidated_keys": deleted}

