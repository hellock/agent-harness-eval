"""Tests for per-harness provider override and supported_api_formats validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness_eval.adapters.interface import HarnessAdapter, PreparedRun
from agent_harness_eval.config.providers import ModelSpec, ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.executor import Executor
from agent_harness_eval.task import Task
from agent_harness_eval.types import RunResult


class _StubAdapter(HarnessAdapter):
    name = "stub"

    def prepare(self, task: Task, run_id: str) -> PreparedRun:
        raise NotImplementedError

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        raise NotImplementedError

    def cleanup(self, prepared: PreparedRun) -> None:
        pass


class _StubExecutor(Executor):
    name = "stub"

    async def execute(self, harness, policy, inner_command, inner_args, inner_env, timeout_ms=0):
        raise NotImplementedError


def _make_rc(**overrides) -> RuntimeConfig:
    defaults = dict(
        project_root=Path.cwd(),
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com",
                api_key="sk-ant-test",
                api_format="anthropic",
            ),
            "relay_openai": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="sk-relay-test",
                api_format="openai-responses",
            ),
        },
    )
    defaults.update(overrides)
    return RuntimeConfig(**defaults)


def test_resolve_provider_uses_model_spec_provider_by_default():
    rc = _make_rc()
    adapter = _StubAdapter(rc, _StubExecutor(rc))
    spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")

    provider = adapter.resolve_provider(spec)
    assert provider.api_format == "anthropic"
    assert provider.api_key == "sk-ant-test"


def test_resolve_provider_uses_harness_override():
    rc = _make_rc(harness_config={"stub": {"provider": "relay_openai"}})
    adapter = _StubAdapter(rc, _StubExecutor(rc))
    spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")

    provider = adapter.resolve_provider(spec)
    assert provider.api_format == "openai-responses"
    assert provider.api_key == "sk-relay-test"


def test_resolve_provider_raises_if_provider_missing():
    rc = _make_rc(harness_config={"stub": {"provider": "nonexistent"}})
    adapter = _StubAdapter(rc, _StubExecutor(rc))
    spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")

    with pytest.raises(ValueError, match="nonexistent"):
        adapter.resolve_provider(spec)


def test_resolve_provider_raises_if_api_format_mismatch():
    rc = _make_rc()
    adapter = _StubAdapter(rc, _StubExecutor(rc))
    adapter.supported_api_formats = ["openai-responses"]
    spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")

    with pytest.raises(ValueError, match="openai-responses"):
        adapter.resolve_provider(spec)


def test_resolve_provider_passes_if_api_format_matches():
    rc = _make_rc(harness_config={"stub": {"provider": "relay_openai"}})
    adapter = _StubAdapter(rc, _StubExecutor(rc))
    adapter.supported_api_formats = ["openai-responses"]
    spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")

    provider = adapter.resolve_provider(spec)
    assert provider.api_format == "openai-responses"


def test_resolve_provider_no_format_constraint():
    """Adapters without supported_api_formats accept any provider."""
    rc = _make_rc()
    adapter = _StubAdapter(rc, _StubExecutor(rc))
    adapter.supported_api_formats = None
    spec = ModelSpec(provider="relay_openai", model="claude-sonnet-4-6")

    provider = adapter.resolve_provider(spec)
    assert provider.api_format == "openai-responses"


def test_resolve_provider_handles_dash_to_underscore_config_key():
    """Adapter name 'claude-code' maps to YAML key 'claude_code'."""
    rc = _make_rc(harness_config={"claude_code": {"provider": "relay_openai"}})
    adapter = _StubAdapter(rc, _StubExecutor(rc))
    adapter.name = "claude-code"
    spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")

    provider = adapter.resolve_provider(spec)
    assert provider.api_format == "openai-responses"
