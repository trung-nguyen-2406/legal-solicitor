import os
from dataclasses import dataclass
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv(".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    OLLAMA_NUM_CTX: int = _env_int("OLLAMA_NUM_CTX", 4096)
    # Layers on the GPU. Ollama's auto split uses ~2.3 GB of the 4 GB card; 22 measured 2.6x faster
    # end to end, and above 24 weights spill into shared memory and prefill collapses. Negative = auto.
    OLLAMA_NUM_GPU: int = _env_int("OLLAMA_NUM_GPU", 22)
    OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    # 300 truncated about 45% of answers mid-sentence; 450 leaves about 10% cut off.
    ANSWER_MAX_TOKENS: int = _env_int("ANSWER_MAX_TOKENS", 450)


settings = Settings()


def ollama_request_fields(**options: Any) -> Dict[str, Any]:
    """Fields shared by every /api/chat call.

    All callers must send identical num_ctx and num_gpu, otherwise Ollama reloads the model.
    """
    merged: Dict[str, Any] = {"num_ctx": settings.OLLAMA_NUM_CTX, **options}
    if settings.OLLAMA_NUM_GPU >= 0:
        merged["num_gpu"] = settings.OLLAMA_NUM_GPU
    return {"keep_alive": settings.OLLAMA_KEEP_ALIVE, "options": merged}
