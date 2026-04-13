from __future__ import annotations

import httpx
import pytest

from agent_harness_eval.config.providers import ModelSpec, ProviderConfig
from agent_harness_eval.graders.judge_client import (
    HttpJudgeLLM,
    RetryableJudgeError,
)


class DummyResponse:
    def __init__(self, status_code: int, payload: dict, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self) -> dict:
        return self._payload


class DummyAsyncClient:
    def __init__(self, *, timeout: httpx.Timeout, response: DummyResponse | Exception, recorder: list[dict]):
        self.timeout = timeout
        self._response = response
        self._recorder = recorder

    async def __aenter__(self) -> DummyAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, json: dict, headers: dict) -> DummyResponse:
        self._recorder.append({"url": url, "json": json, "headers": headers, "timeout": self.timeout})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.mark.asyncio
async def test_http_judge_llm_builds_anthropic_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    response = DummyResponse(200, {"content": [{"text": "anthropic ok"}]})

    monkeypatch.setattr(
        "agent_harness_eval.graders.judge_client.httpx.AsyncClient",
        lambda timeout: DummyAsyncClient(timeout=timeout, response=response, recorder=calls),
    )

    client = HttpJudgeLLM(
        ModelSpec(provider="anthropic", model="claude-sonnet-4-6"),
        ProviderConfig(base_url="https://api.anthropic.com", api_key="ant-key", api_format="anthropic"),
    )

    text = await client._do_request("judge me")

    assert text == "anthropic ok"
    assert calls[0]["url"] == "https://api.anthropic.com/v1/messages"
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": "judge me"}]
    assert calls[0]["headers"]["x-api-key"] == "ant-key"


@pytest.mark.asyncio
async def test_http_judge_llm_builds_openai_responses_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    response = DummyResponse(
        200,
        {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "responses ok"}],
                }
            ]
        },
    )

    monkeypatch.setattr(
        "agent_harness_eval.graders.judge_client.httpx.AsyncClient",
        lambda timeout: DummyAsyncClient(timeout=timeout, response=response, recorder=calls),
    )

    client = HttpJudgeLLM(
        ModelSpec(provider="relay", model="gpt-5.4"),
        ProviderConfig(base_url="https://relay.example", api_key="sk-test", api_format="openai-responses"),
    )

    text = await client._do_request("judge me")

    assert text == "responses ok"
    assert calls[0]["url"] == "https://relay.example/v1/responses"
    assert calls[0]["json"] == {"model": "gpt-5.4", "temperature": 0, "input": "judge me"}
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_http_judge_llm_retries_generate_on_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = HttpJudgeLLM(
        ModelSpec(provider="openai", model="gpt-5.4"),
        ProviderConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            api_format="openai-chat-completions",
        ),
        max_attempts=3,
    )
    attempts = {"count": 0}

    async def flaky(prompt: str) -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RetryableJudgeError("rate limited", 429)
        return "ok"

    monkeypatch.setattr(client, "_do_request", flaky)

    text = await client.generate("judge me")

    assert text == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_http_judge_llm_raises_on_non_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    response = DummyResponse(400, {}, text="bad request")

    monkeypatch.setattr(
        "agent_harness_eval.graders.judge_client.httpx.AsyncClient",
        lambda timeout: DummyAsyncClient(timeout=timeout, response=response, recorder=calls),
    )

    client = HttpJudgeLLM(
        ModelSpec(provider="openai", model="gpt-5.4"),
        ProviderConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            api_format="openai-chat-completions",
        ),
    )

    with pytest.raises(RuntimeError, match="Judge API error 400"):
        await client._do_request("judge me")


@pytest.mark.asyncio
async def test_http_judge_llm_wraps_timeout_as_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    response = httpx.ReadTimeout("timed out")

    monkeypatch.setattr(
        "agent_harness_eval.graders.judge_client.httpx.AsyncClient",
        lambda timeout: DummyAsyncClient(timeout=timeout, response=response, recorder=calls),
    )

    client = HttpJudgeLLM(
        ModelSpec(provider="openai", model="gpt-5.4"),
        ProviderConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            api_format="openai-chat-completions",
        ),
        timeout_ms=1234,
    )

    with pytest.raises(RetryableJudgeError, match="timed out after 1234ms"):
        await client._do_request("judge me")


def test_provider_config_endpoint_url_builds_correct_urls() -> None:
    anthropic = ProviderConfig(base_url="https://api.anthropic.com", api_key="k", api_format="anthropic")
    assert anthropic.endpoint_url() == "https://api.anthropic.com/v1/messages"

    openai = ProviderConfig(
        base_url="https://api.openai.com/v1",
        api_key="k",
        api_format="openai-chat-completions",
    )
    assert openai.endpoint_url() == "https://api.openai.com/v1/chat/completions"

    openai_root = ProviderConfig(
        base_url="https://api.openai.com",
        api_key="k",
        api_format="openai-chat-completions",
    )
    assert openai_root.endpoint_url() == "https://api.openai.com/v1/chat/completions"

    relay = ProviderConfig(base_url="https://relay.example", api_key="k", api_format="openai-responses")
    assert relay.endpoint_url() == "https://relay.example/v1/responses"

    # Trailing slash stripped
    relay_slash = ProviderConfig(base_url="https://relay.example/", api_key="k", api_format="openai-responses")
    assert relay_slash.endpoint_url() == "https://relay.example/v1/responses"
