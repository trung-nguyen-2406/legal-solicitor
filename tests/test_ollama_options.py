import asyncio
import json
import os
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps import main as main_module
from apps.config import ollama_request_fields, settings
from apps.services import llm_service, react_agent_service


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> Dict[str, Any]:
        content = json.dumps({"thought": "ok", "action": "FINISH"})
        return {"message": {"content": content}}

    async def aiter_lines(self):
        yield json.dumps({"message": {"content": "OK."}, "done": True})


class FakeStream:
    async def __aenter__(self) -> FakeResponse:
        return FakeResponse()

    async def __aexit__(self, *exc: Any) -> None:
        pass


@pytest.fixture
def captured(monkeypatch) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            pass

        async def post(self, url: str, json: Dict[str, Any]) -> FakeResponse:
            payloads.append(json)
            return FakeResponse()

        def stream(self, method: str, url: str, json: Dict[str, Any]) -> FakeStream:
            payloads.append(json)
            return FakeStream()

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)
    return payloads


def assert_stable_load_options(payload: Dict[str, Any]) -> None:
    assert payload["options"]["num_ctx"] == settings.OLLAMA_NUM_CTX
    expected_gpu = settings.OLLAMA_NUM_GPU if settings.OLLAMA_NUM_GPU >= 0 else None
    assert payload["options"].get("num_gpu") == expected_gpu
    assert payload["keep_alive"] == settings.OLLAMA_KEEP_ALIVE


async def drain(stream) -> str:
    return "".join([part async for part in stream])


def test_every_ollama_call_shares_load_options(captured):
    asyncio.run(drain(llm_service.stream_natural_ollama("q", "ctx")))
    asyncio.run(drain(llm_service.stream_irac_ollama("q", "ctx")))
    asyncio.run(react_agent_service.call_ollama_planner_async("q", [], 0, 0.0))
    asyncio.run(main_module._warmup_ollama_keep_alive())

    assert len(captured) == 4
    for payload in captured:
        assert_stable_load_options(payload)


def test_natural_answer_uses_token_cap(captured):
    asyncio.run(drain(llm_service.stream_natural_ollama("q", "ctx")))
    assert captured[0]["options"]["num_predict"] == settings.ANSWER_MAX_TOKENS


def test_request_fields_keep_caller_options():
    fields = ollama_request_fields(temperature=0.2, num_predict=5)
    assert fields["options"]["temperature"] == 0.2
    assert fields["options"]["num_predict"] == 5
    assert fields["options"]["num_ctx"] == settings.OLLAMA_NUM_CTX
