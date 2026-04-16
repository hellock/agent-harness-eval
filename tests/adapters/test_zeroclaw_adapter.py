from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.adapters.zeroclaw import (
    ZeroClawAdapter,
    _normalize_runtime_trace_ts,
    _patch_zeroclaw_autonomy_config,
    _read_zeroclaw_runtime_trace,
    _recover_zeroclaw_trace_from_logs,
)
from agent_harness_eval.config.providers import ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.task import Task
from agent_harness_eval.types import CanonicalTraceEvent
from agent_harness_eval.utils.subprocess import SubprocessResult

from ._regression_helpers import SequentialRecordingExecutor, arg_value


@pytest.mark.asyncio
async def test_zeroclaw_run_uses_onboard_then_agent_in_authoritative_workspace(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
    )
    zc_workspace_holder: dict[str, Path] = {}
    executor = SequentialRecordingExecutor(
        runtime_config,
        [
            SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
            SubprocessResult(stdout="ZERO_OK", stderr="", exit_code=0, timed_out=False),
        ],
        side_effects=[
            None,
            lambda: (
                zc_workspace_holder["path"].joinpath("README.md").write_text("updated by zeroclaw\n", encoding="utf-8"),
                zc_workspace_holder["path"].joinpath("created.txt").write_text("new file\n", encoding="utf-8"),
            ),
        ],
    )
    adapter = ZeroClawAdapter(runtime_config, executor)
    task = Task(
        id="zeroclaw.regression.01",
        category="coding",
        description="zeroclaw command contract",
        user_query="Reply with ZERO_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "zeroclaw-regression")
    config_dir = Path(prepared.env["_EVAL_CONFIG_DIR"])
    workspace_dir = Path(prepared.workspace_dir)
    zc_workspace_holder["path"] = workspace_dir

    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._read_zeroclaw_runtime_trace",
        lambda trace_path: {
            "trace": [
                CanonicalTraceEvent(type="message", role="assistant", text="ZERO_OK", ts="2026-01-01T00:00:00+00:00")
            ],
            "usage": {
                "input": 11,
                "output": 4,
                "cache_read": 0,
                "cache_write": 0,
                "total": 15,
                "turns": 1,
            },
            "tool_calls": 1,
        },
    )
    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._patch_zeroclaw_autonomy_config",
        lambda config_dir_arg, workspace_dir_arg, **kwargs: (config_dir / "config.toml").write_text(
            '[security]\nlevel = "full"\n', encoding="utf-8"
        ),
    )
    try:
        result = await adapter.run(prepared, "relay:claude-sonnet-4-6")

        assert result.status == "completed"
        assert result.final_text == "ZERO_OK"
        assert len(executor.calls) == 2

        onboard_call = executor.calls[0]
        onboard_args = onboard_call["inner_args"]
        assert isinstance(onboard_args, list)
        assert onboard_args[0] == "onboard"
        assert "--config-dir" not in onboard_args
        assert "--quick" in onboard_args
        assert "--force" in onboard_args
        assert arg_value(onboard_args, "--provider") == "anthropic-custom:https://relay.example.com/v1"
        assert arg_value(onboard_args, "--model") == "claude-sonnet-4-6"

        agent_call = executor.calls[1]
        agent_args = agent_call["inner_args"]
        assert isinstance(agent_args, list)
        assert agent_args[0] == "agent"
        assert "--config-dir" not in agent_args
        assert arg_value(agent_args, "--provider") == "anthropic-custom:https://relay.example.com/v1"
        assert arg_value(agent_args, "--message")

        env = agent_call["inner_env"]
        assert isinstance(env, dict)
        assert env["API_KEY"] == "anthropic-key"
        assert env["HOME"] == prepared.env["_EVAL_RUNTIME_DIR"]
        assert env["ZEROCLAW_WORKSPACE"] == prepared.workspace_dir
        assert config_dir == workspace_dir.parent / ".zeroclaw"

        assert workspace_dir.joinpath("README.md").read_text(encoding="utf-8") == "updated by zeroclaw\n"
        assert workspace_dir.joinpath("created.txt").read_text(encoding="utf-8") == "new file\n"
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_zeroclaw_run_returns_failed_result_when_trace_parse_fails(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
    )
    executor = SequentialRecordingExecutor(
        runtime_config,
        [
            SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
            SubprocessResult(stdout="ZERO_OK", stderr="trace parse failed", exit_code=0, timed_out=False),
        ],
    )
    adapter = ZeroClawAdapter(runtime_config, executor)
    task = Task(
        id="zeroclaw.regression.parse-failure",
        category="coding",
        description="zeroclaw parse failure",
        user_query="Reply with ZERO_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "zeroclaw-parse-failure")
    config_dir = Path(prepared.env["_EVAL_CONFIG_DIR"])

    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._patch_zeroclaw_autonomy_config",
        lambda config_dir_arg, workspace_dir_arg, **kwargs: (config_dir / "config.toml").write_text(
            '[security]\nlevel = "full"\n', encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._read_zeroclaw_runtime_trace",
        lambda trace_path: (_ for _ in ()).throw(ValueError("bad runtime trace")),
    )

    try:
        result = await adapter.run(prepared, "relay:claude-sonnet-4-6")

        assert result.status == "failed"
        assert result.final_text == "ZERO_OK"
        assert result.failure_origin == "adapter"
        assert result.infra_error_code == "adapter_output_error"
        assert len(executor.calls) == 2
        assert result.trace and result.trace[0].type == "task_failed"
        assert "ZeroClaw output parse error" in (result.trace[0].error or "")
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_zeroclaw_run_uses_custom_provider_for_openai_compatible_relays(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="relay-key",
                api_format="openai-chat-completions",
            )
        },
    )
    executor = SequentialRecordingExecutor(
        runtime_config,
        [
            SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
            SubprocessResult(stdout="MINIMAX_OK", stderr="", exit_code=0, timed_out=False),
        ],
    )
    adapter = ZeroClawAdapter(runtime_config, executor)
    task = Task(
        id="zeroclaw.regression.openai-relay",
        category="coding",
        description="zeroclaw openai relay contract",
        user_query="Reply with MINIMAX_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "zeroclaw-openai-relay")
    config_dir = Path(prepared.env["_EVAL_CONFIG_DIR"])

    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._read_zeroclaw_runtime_trace",
        lambda trace_path: {
            "trace": [
                CanonicalTraceEvent(type="message", role="assistant", text="MINIMAX_OK", ts="2026-01-01T00:00:00+00:00")
            ],
            "usage": {
                "input": 7,
                "output": 3,
                "cache_read": 0,
                "cache_write": 0,
                "total": 10,
                "turns": 1,
            },
            "tool_calls": 0,
        },
    )
    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._patch_zeroclaw_autonomy_config",
        lambda config_dir_arg, workspace_dir_arg, **kwargs: (config_dir / "config.toml").write_text(
            '[security]\nlevel = "full"\n', encoding="utf-8"
        ),
    )

    try:
        result = await adapter.run(prepared, "relay:MiniMax-M2.7")

        assert result.status == "completed"
        assert len(executor.calls) == 2

        onboard_args = executor.calls[0]["inner_args"]
        agent_args = executor.calls[1]["inner_args"]
        assert isinstance(onboard_args, list)
        assert isinstance(agent_args, list)
        assert arg_value(onboard_args, "--provider") == "custom:https://relay.example.com/v1"
        assert arg_value(agent_args, "--provider") == "custom:https://relay.example.com/v1"
        assert executor.calls[1]["inner_env"]["API_KEY"] == "relay-key"
        assert 'api_url = "https://relay.example.com/v1"' not in config_dir.joinpath("config.toml").read_text(
            encoding="utf-8"
        )
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_zeroclaw_run_fails_fast_when_onboard_exits_non_zero(
    isolated_run_dir: Path,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
    )
    executor = SequentialRecordingExecutor(
        runtime_config,
        [
            SubprocessResult(stdout="", stderr="bad config", exit_code=2, timed_out=False),
        ],
    )
    adapter = ZeroClawAdapter(runtime_config, executor)
    task = Task(
        id="zeroclaw.regression.onboard-failure",
        category="coding",
        description="zeroclaw onboard failure",
        user_query="Reply with ZERO_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "zeroclaw-onboard-failure")

    try:
        result = await adapter.run(prepared, "relay:claude-sonnet-4-6")

        assert result.status == "failed"
        assert len(executor.calls) == 1
        assert result.trace and result.trace[0].type == "task_failed"
        assert "ZeroClaw onboard exited with code 2" in (result.trace[0].error or "")
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_zeroclaw_run_syncs_workspace_back_even_when_agent_fails(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
    )
    zc_workspace_holder: dict[str, Path] = {}
    executor = SequentialRecordingExecutor(
        runtime_config,
        [
            SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
            SubprocessResult(stdout="", stderr="agent failed", exit_code=3, timed_out=False),
        ],
        side_effects=[
            None,
            lambda: (
                zc_workspace_holder["path"]
                .joinpath("README.md")
                .write_text("modified before failure\n", encoding="utf-8")
            ),
        ],
    )
    adapter = ZeroClawAdapter(runtime_config, executor)
    task = Task(
        id="zeroclaw.regression.sync-on-failure",
        category="coding",
        description="zeroclaw workspace sync on failure",
        user_query="Reply with ZERO_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "zeroclaw-sync-failure")
    config_dir = Path(prepared.env["_EVAL_CONFIG_DIR"])
    zc_workspace_holder["path"] = Path(prepared.workspace_dir)

    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._patch_zeroclaw_autonomy_config",
        lambda config_dir_arg, workspace_dir_arg, **kwargs: (config_dir / "config.toml").write_text(
            '[security]\nlevel = "full"\n', encoding="utf-8"
        ),
    )
    try:
        result = await adapter.run(prepared, "relay:claude-sonnet-4-6")

        assert result.status == "failed"
        assert result.trace and result.trace[0].type == "task_failed"
        assert Path(prepared.workspace_dir, "README.md").read_text(encoding="utf-8") == "modified before failure\n"
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_zeroclaw_run_recovers_tool_calls_from_stdout_logs_on_provider_failure(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
    )
    stdout = (
        "\x1b[2m2026-04-16T08:55:21.287537Z\x1b[0m \x1b[32m INFO\x1b[0m "
        "\x1b[2mzeroclaw::tools::web_search_tool\x1b[0m\x1b[2m:\x1b[0m "
        "Searching web for: 东京樱花2026年花期预测 4月中下旬\n"
        "\x1b[2m2026-04-16T08:55:21.289887Z\x1b[0m \x1b[32m INFO\x1b[0m "
        "\x1b[2mzeroclaw::tools::web_search_tool\x1b[0m\x1b[2m:\x1b[0m "
        "Searching web for: 北京到东京机票价格2026年4月\n"
    )
    executor = SequentialRecordingExecutor(
        runtime_config,
        [
            SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
            SubprocessResult(
                stdout=stdout,
                stderr=(
                    "Error: All providers/models failed. Attempts:\n"
                    "provider=anthropic-custom:https://relay.example.com/v1 model=claude-sonnet-4-6 "
                    "attempt 1/3: non_retryable; error=Anthropic API error (400 Bad Request)"
                ),
                exit_code=1,
                timed_out=False,
            ),
        ],
    )
    adapter = ZeroClawAdapter(runtime_config, executor)
    task = Task(
        id="zeroclaw.regression.provider-failure-recovery",
        category="skills",
        description="zeroclaw provider failure retains tool trace",
        user_query="Reply with ZERO_OK.",
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "zeroclaw-provider-failure-recovery")
    config_dir = Path(prepared.env["_EVAL_CONFIG_DIR"])

    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._patch_zeroclaw_autonomy_config",
        lambda config_dir_arg, workspace_dir_arg, **kwargs: (config_dir / "config.toml").write_text(
            '[security]\nlevel = "full"\n', encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._read_zeroclaw_runtime_trace",
        lambda trace_path: {
            "trace": [],
            "usage": {
                "input": 0,
                "output": 0,
                "cache_read": 0,
                "cache_write": 0,
                "total": 0,
                "turns": 0,
            },
            "tool_calls": 0,
        },
    )

    try:
        result = await adapter.run(prepared, "relay:claude-sonnet-4-6")

        assert result.status == "failed"
        assert result.failure_origin == "provider"
        assert result.infra_error_code == "provider_api_error"
        assert [event.type for event in result.trace] == [
            "tool_call_started",
            "tool_call_started",
            "tool_call_completed",
            "tool_call_completed",
            "task_failed",
        ]
        assert result.trace[0].tool_name == "web_search"
        assert result.trace[0].input == {"query": "东京樱花2026年花期预测 4月中下旬"}
        assert result.trace[2].success is False
        assert result.trace[2].output == "<no result recorded>"
        assert result.metrics.tool_calls == 2
    finally:
        adapter.cleanup(prepared)


def test_patch_zeroclaw_autonomy_config_relaxes_shell_policy_for_docker(tmp_path: Path) -> None:
    config_dir = tmp_path / ".zeroclaw"
    config_dir.mkdir()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    config_path = config_dir / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[security]",
                'level = "supervised"',
                'forbidden_paths = ["/tmp", "/etc"]',
                "allowed_roots = []",
                'allowed_commands = ["git", "npm"]',
                "require_approval_for_medium_risk = true",
                "block_high_risk_commands = true",
                "max_actions_per_hour = 30",
                "workspace_only = true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _patch_zeroclaw_autonomy_config(
        str(config_dir),
        str(workspace_dir),
        relax_shell_commands=True,
    )

    content = config_path.read_text(encoding="utf-8")
    assert 'level = "full"' in content
    assert f'allowed_roots = ["{workspace_dir}"]' in content
    assert 'allowed_commands = ["*"]' in content
    assert "block_high_risk_commands = false" in content
    assert "require_approval_for_medium_risk = false" in content
    assert "workspace_only = true" in content


@pytest.mark.asyncio
async def test_zeroclaw_run_recovers_trace_and_usage_on_timeout(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
    )
    executor = SequentialRecordingExecutor(
        runtime_config,
        [
            SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
            SubprocessResult(stdout="partial output\n", stderr="", exit_code=0, timed_out=True),
        ],
    )
    adapter = ZeroClawAdapter(runtime_config, executor)
    task = Task(
        id="zeroclaw.regression.timeout-recovery",
        category="coding",
        description="zeroclaw timeout recovery",
        user_query="Reply with ZERO_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "zeroclaw-timeout-recovery")
    config_dir = Path(prepared.env["_EVAL_CONFIG_DIR"])

    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._patch_zeroclaw_autonomy_config",
        lambda config_dir_arg, workspace_dir_arg, **kwargs: (config_dir / "config.toml").write_text(
            '[security]\nlevel = "full"\n', encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._read_zeroclaw_runtime_trace",
        lambda trace_path: {
            "trace": [
                CanonicalTraceEvent(type="tool_call_started", tool_name="read_file", ts="2026-01-01T00:00:00+00:00")
            ],
            "usage": {
                "input": 11,
                "output": 4,
                "cache_read": 1,
                "cache_write": 0,
                "total": 16,
                "turns": 1,
            },
            "tool_calls": 1,
        },
    )

    try:
        result = await adapter.run(prepared, "relay:claude-sonnet-4-6")

        assert result.status == "timed_out"
        assert result.final_text == "partial output"
        assert len(result.trace) == 1
        assert result.trace[0].type == "tool_call_started"
        assert result.metrics.input_tokens == 11
        assert result.metrics.output_tokens == 4
        assert result.metrics.cache_read_tokens == 1
        assert result.metrics.total_tokens == 16
        assert result.metrics.tool_calls == 1
        assert result.metrics.turns == 1
    finally:
        adapter.cleanup(prepared)


def test_recover_zeroclaw_trace_from_logs_extracts_tool_inputs() -> None:
    stdout = (
        "\x1b[2m2026-04-16T08:55:21.287537Z\x1b[0m \x1b[32m INFO\x1b[0m "
        "\x1b[2mzeroclaw::tools::web_search_tool\x1b[0m\x1b[2m:\x1b[0m "
        "Searching web for: Tokyo cherry blossom 2026 forecast late April\n"
    )

    recovered = _recover_zeroclaw_trace_from_logs(stdout)

    assert recovered["tool_calls"] == 1
    assert [event.type for event in recovered["trace"]] == [
        "tool_call_started",
        "tool_call_completed",
    ]
    assert recovered["trace"][0].tool_name == "web_search"
    assert recovered["trace"][0].input == {"query": "Tokyo cherry blossom 2026 forecast late April"}
    assert recovered["trace"][1].success is False


def test_normalize_runtime_trace_ts_truncates_nanoseconds_to_milliseconds() -> None:
    # Real zeroclaw samples emit 9-digit fractional seconds; the canonical
    # contract is millisecond precision (utils.timestamps.to_canonical_ts),
    # so anything sub-ms must be silently truncated.
    from datetime import datetime as _dt

    ns_precision = "2026-04-15T08:53:57.840372668+00:00"
    result = _normalize_runtime_trace_ts(ns_precision)
    assert result == "2026-04-15T08:53:57.840+00:00"
    reparsed = _dt.fromisoformat(result)
    assert reparsed.tzinfo is not None
    # ms precision → microsecond is a multiple of 1000.
    assert reparsed.microsecond % 1000 == 0
    assert reparsed.microsecond == 840_000


def test_normalize_runtime_trace_ts_forces_utc_on_naive_input() -> None:
    from datetime import datetime as _dt

    result = _normalize_runtime_trace_ts("2026-04-12T10:00:00")
    reparsed = _dt.fromisoformat(result)
    assert reparsed.tzinfo is not None
    assert reparsed.utcoffset().total_seconds() == 0


def test_normalize_runtime_trace_ts_accepts_numeric_epoch() -> None:
    from datetime import datetime as _dt

    # Seconds since epoch (≤ 1e12).
    secs = _normalize_runtime_trace_ts(1_700_000_000)
    assert _dt.fromisoformat(secs).year == 2023

    # Millis since epoch (> 1e12) — same instant, scaled.
    millis = _normalize_runtime_trace_ts(1_700_000_000_000)
    assert _dt.fromisoformat(millis).year == 2023


def test_normalize_runtime_trace_ts_falls_back_to_now_on_garbage() -> None:
    from datetime import datetime as _dt

    for bad in (None, "", "not-a-timestamp", "2026-13-40"):
        result = _normalize_runtime_trace_ts(bad)
        reparsed = _dt.fromisoformat(result)
        assert reparsed.tzinfo is not None


def test_read_zeroclaw_runtime_trace_missing_file() -> None:
    result = _read_zeroclaw_runtime_trace("/nonexistent/path/trace.jsonl")
    assert result["trace"] == []
    assert result["usage"]["input"] == 0
    assert result["usage"]["output"] == 0
    assert result["tool_calls"] == 0
    assert result["usage"]["turns"] == 0


def test_read_zeroclaw_runtime_trace_parses_events() -> None:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        trace_path = root / "runtime_trace.jsonl"
        events = [
            {
                "event_type": "llm_response",
                "timestamp": "2026-04-12T10:00:00Z",
                "payload": {
                    "input_tokens": 500,
                    "output_tokens": 200,
                    "cached_input_tokens": 50,
                    # NOTE: zeroclaw's runtime_trace.jsonl does not expose
                    # cache-creation (cache-write) tokens — only
                    # ``cached_input_tokens`` (= cache_read). The adapter
                    # therefore reports cache_write as 0 for this harness.
                    "raw_response": "I will read the file.",
                },
            },
            {
                "event_type": "tool_call_start",
                "timestamp": "2026-04-12T10:00:01Z",
                "payload": {"tool": "read_file", "arguments": '{"path": "src/cache.ts"}'},
            },
            {
                "event_type": "tool_call_result",
                "timestamp": "2026-04-12T10:00:02Z",
                "payload": {"tool": "read_file", "output": "export class CacheManager { ... }"},
            },
            {
                "event_type": "llm_response",
                "timestamp": "2026-04-12T10:00:03Z",
                "payload": {"input_tokens": 800, "output_tokens": 400},
            },
            {
                "event_type": "tool_call_start",
                "timestamp": "2026-04-12T10:00:04Z",
                "payload": {
                    "tool": "write_file",
                    "arguments": json.dumps({"path": "src/cache.ts", "content": "new code"}),
                },
            },
            {
                "event_type": "tool_call_result",
                "timestamp": "2026-04-12T10:00:05Z",
                "payload": {"tool": "write_file", "output": "File written."},
            },
        ]
        trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

        result = _read_zeroclaw_runtime_trace(str(trace_path))
        usage = result["usage"]
        trace = result["trace"]

        # Canonical shape: ``total`` (not ``total_tokens``), ``tool_calls``
        # top-level alongside usage.
        assert usage["input"] == 1300
        assert usage["output"] == 600
        assert usage["cache_read"] == 50
        # zeroclaw does not emit cache_creation_input_tokens in its runtime
        # trace, so the adapter reports cache_write as 0 unconditionally.
        assert usage["cache_write"] == 0
        # total = input (1300) + output (600) + cache_read (50) = 1950.
        # cache_write is not added because zeroclaw does not expose it.
        assert usage["total"] == 1950
        assert result["tool_calls"] == 2
        assert usage["turns"] == 2

        assert [e.type for e in trace] == [
            "message",
            "tool_call_started",
            "tool_call_completed",
            "tool_call_started",
            "tool_call_completed",
        ]
        assert trace[1].tool_name == "read_file"
        assert trace[2].tool_name == "read_file"
        assert trace[2].success is True
        assert trace[3].tool_name == "write_file"
        assert trace[3].input == {"path": "src/cache.ts", "content": "new code"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_read_zeroclaw_runtime_trace_dedupes_turn_final_response() -> None:
    """Regression: v2 showed 4/4 zeroclaw completed runs with two adjacent
    identical assistant messages 2-4ms apart — llm_response.raw_response
    followed by turn_final_response.text carrying the same text. After the
    fix, only the llm_response branch emits a trace message; turn_final_
    response is consumed for usage/metadata only (today: ignored for trace).
    """
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        trace_path = root / "runtime_trace.jsonl"
        events = [
            {
                "event_type": "llm_response",
                "timestamp": "2026-04-15T10:00:00.100+00:00",
                "payload": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "raw_response": "the final answer",
                },
            },
            {
                "event_type": "turn_final_response",
                "timestamp": "2026-04-15T10:00:00.103+00:00",
                "payload": {"text": "the final answer"},
            },
        ]
        trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

        result = _read_zeroclaw_runtime_trace(str(trace_path))
        trace = result["trace"]

        assistant_msgs = [e for e in trace if e.type == "message" and e.role == "assistant"]
        assert len(assistant_msgs) == 1, (
            f"expected exactly 1 assistant message, got {len(assistant_msgs)}: "
            f"{[(e.ts, e.text) for e in assistant_msgs]}"
        )
        assert assistant_msgs[0].text == "the final answer"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_read_zeroclaw_runtime_trace_handles_malformed_lines() -> None:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        trace_path = root / "runtime_trace.jsonl"
        lines = [
            "not valid json",
            "",
            json.dumps(
                {
                    "event_type": "llm_response",
                    "timestamp": "2026-04-12T10:00:00Z",
                    "payload": {"input_tokens": 100, "output_tokens": 50},
                }
            ),
        ]
        trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = _read_zeroclaw_runtime_trace(str(trace_path))
        assert result["usage"]["input"] == 100
        assert result["usage"]["output"] == 50
        assert result["usage"]["turns"] == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_read_zeroclaw_runtime_trace_honors_explicit_success_false() -> None:
    """Regression: zeroclaw runtime_trace puts ``success`` at event top-level
    (NOT payload — see zeroclaw src/observability/runtime_trace.rs:47 and
    src/agent/loop_.rs:3187-3200). Pre-fix, the adapter hardcoded
    ``success=True`` for every tool_call_result, silently masking every
    real tool failure. After the fix, explicit ``success: false`` at top
    level is reflected in the canonical trace.
    """
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        trace_path = root / "runtime_trace.jsonl"
        events = [
            {
                "event_type": "tool_call_start",
                "timestamp": "2026-04-12T10:00:01Z",
                "payload": {"tool": "exec", "arguments": '{"command": "false"}'},
            },
            {
                "event_type": "tool_call_result",
                "timestamp": "2026-04-12T10:00:02Z",
                "success": False,
                "message": "exit code 1",
                "payload": {"tool": "exec", "output": "", "duration_ms": 42},
            },
        ]
        trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

        result = _read_zeroclaw_runtime_trace(str(trace_path))
        trace = result["trace"]

        # Exactly 2 events, balanced start/completed; no orphan.
        assert len(trace) == 2
        assert trace[1].type == "tool_call_completed"
        assert trace[1].success is False
        # When payload.output is empty on failure, surface the top-level
        # ``message`` (zeroclaw's error_reason) so graders see why it failed.
        assert trace[1].output == "exit code 1"

    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_read_zeroclaw_runtime_trace_defaults_success_to_true_when_absent() -> None:
    """Back-compat: legacy/incomplete traces with no top-level ``success``
    field (or success=null) should continue to be treated as successful —
    matches pre-fix behavior for the common-case happy path.
    """
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        trace_path = root / "runtime_trace.jsonl"
        events = [
            {
                "event_type": "tool_call_start",
                "timestamp": "2026-04-12T10:00:01Z",
                "payload": {"tool": "read_file"},
            },
            {
                "event_type": "tool_call_result",
                "timestamp": "2026-04-12T10:00:02Z",
                # No "success" field at all — legacy shape
                "payload": {"tool": "read_file", "output": "file contents here"},
            },
            {
                "event_type": "tool_call_start",
                "timestamp": "2026-04-12T10:00:03Z",
                "payload": {"tool": "read_file"},
            },
            {
                "event_type": "tool_call_result",
                "timestamp": "2026-04-12T10:00:04Z",
                # Explicit None — should also default to True
                "success": None,
                "payload": {"tool": "read_file", "output": "more contents"},
            },
        ]
        trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

        result = _read_zeroclaw_runtime_trace(str(trace_path))
        completed = [e for e in result["trace"] if e.type == "tool_call_completed"]
        assert len(completed) == 2
        assert all(e.success is True for e in completed)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_read_zeroclaw_runtime_trace_pending_tools_closed() -> None:
    """Regression: a tool_call_start without a matching tool_call_result
    (agent crashed mid-tool / stream truncated) must emit a synthetic
    ``tool_call_completed(success=False)`` at the trace tail. Pre-fix, the
    pending_tools tracker was a dict[str, str] that was never populated, so
    the orphan-close loop silently no-op'd and left unpaired ``started``
    events at the tail. That made zeroclaw the only adapter where a mid-tool
    crash could leave the trace structurally unbalanced, and it violated
    the invariant every other adapter (claude_code, openclaw) holds.
    """
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        trace_path = root / "runtime_trace.jsonl"
        events = [
            {
                "event_type": "tool_call_start",
                "timestamp": "2026-04-12T10:00:01Z",
                "payload": {"tool": "exec", "arguments": '{"command": "npm test"}'},
            },
        ]
        trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

        result = _read_zeroclaw_runtime_trace(str(trace_path))
        trace = result["trace"]
        assert len(trace) == 2, f"expected tool_call_started + synthetic orphan closure, got {len(trace)}"
        assert trace[0].type == "tool_call_started"
        assert trace[0].tool_name == "exec"
        assert trace[1].type == "tool_call_completed"
        assert trace[1].tool_name == "exec"
        assert trace[1].success is False
        assert trace[1].output == "<no result recorded>"
    finally:
        shutil.rmtree(root, ignore_errors=True)
