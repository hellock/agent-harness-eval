"""Judge LLM client construction and HTTP transport."""

from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..config.providers import ModelSpec, ProviderConfig
from ..config.runtime import RuntimeConfig
from .interface import JudgeLLM

RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


class RetryableJudgeError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    max_attempts = retry_state.retry_object.stop.max_attempt_number
    if isinstance(exc, RetryableJudgeError):
        reason = f"HTTP {exc.status}" if exc.status else "transient error"
    else:
        reason = "transient error"
    print(f"  Judge {reason} (attempt {attempt}/{max_attempts}), retrying in {retry_state.next_action.sleep:.0f}ms...")


class HttpJudgeLLM:
    """Judge LLM client using httpx."""

    def __init__(
        self,
        spec: ModelSpec,
        provider: ProviderConfig,
        *,
        timeout_ms: int = 60_000,
        max_attempts: int = 4,
    ):
        self.spec = spec
        self.provider = provider
        self.timeout_ms = timeout_ms
        self.max_attempts = max_attempts

    async def generate(self, prompt: str) -> str:
        @retry(
            retry=retry_if_exception_type(RetryableJudgeError),
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(initial=1, max=30, jitter=1),
            before_sleep=_log_retry,
            reraise=True,
        )
        async def _call() -> str:
            return await self._do_request(prompt)

        return await _call()

    async def _do_request(self, prompt: str) -> str:
        timeout = httpx.Timeout(self.timeout_ms / 1000)

        api_format = self.provider.api_format

        if api_format == "anthropic":
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.provider.api_key,
                "anthropic-version": "2023-06-01",
                **(self.provider.extra_headers or {}),
            }
            url = self.provider.endpoint_url()
            body = {
                "model": self.spec.model,
                "max_tokens": 1024,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        elif api_format == "openai-responses":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.provider.api_key}",
                **(self.provider.extra_headers or {}),
            }
            url = self.provider.endpoint_url()
            body = {
                "model": self.spec.model,
                "temperature": 0,
                "input": prompt,
            }
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.provider.api_key}",
                **(self.provider.extra_headers or {}),
            }
            url = self.provider.endpoint_url()
            body = {
                "model": self.spec.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(url, json=body, headers=headers)
            except httpx.TimeoutException as exc:
                raise RetryableJudgeError(f"Judge request timed out after {self.timeout_ms}ms") from exc
            except httpx.HTTPError as exc:
                raise RetryableJudgeError(str(exc)) from exc

            if resp.status_code >= 400:
                text = resp.text[:200]
                message = f"Judge API error {resp.status_code}: {text}"
                if resp.status_code in RETRYABLE_HTTP_STATUSES:
                    raise RetryableJudgeError(message, resp.status_code)
                raise RuntimeError(message)

            data = resp.json()
            if api_format == "anthropic":
                return data.get("content", [{}])[0].get("text", "")
            if api_format == "openai-responses":
                # Responses API: output is a list of items with type "message"
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for block in item.get("content", []):
                            if block.get("type") == "output_text":
                                return block.get("text", "")
                return ""
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def create_judge_llm(
    spec: ModelSpec,
    provider: ProviderConfig,
    rc: RuntimeConfig | None = None,
) -> JudgeLLM:
    return HttpJudgeLLM(
        spec,
        provider,
        timeout_ms=rc.judge_timeout_ms if rc else 60_000,
        max_attempts=rc.judge_max_attempts if rc else 4,
    )
