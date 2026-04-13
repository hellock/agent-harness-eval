from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.adapters.zeroclaw import ZeroClawAdapter, _read_zeroclaw_runtime_trace, _sync_back
from agent_harness_eval.config.providers import ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.task import Task
from agent_harness_eval.types import CanonicalTraceEvent
from agent_harness_eval.utils.subprocess import SubprocessResult

from ._regression_helpers import SequentialRecordingExecutor, arg_value


@pytest.mark.asyncio
async def test_zeroclaw_run_uses_onboard_then_agent_and_syncs_private_workspace(
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
    zc_workspace = config_dir / "workspace"
    zc_workspace_holder["path"] = zc_workspace

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
                "total_tokens": 15,
                "tool_calls": 1,
                "turns": 1,
            },
        },
    )
    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._patch_zeroclaw_autonomy_config",
        lambda config_dir_arg, workspace_dir_arg: (config_dir / "config.toml").write_text(
            '[security]\nlevel = "full"\n', encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._inject_zeroclaw_api_key",
        lambda config_dir_arg, api_key: (config_dir / "config.toml").write_text(
            f'api_key = "{api_key}"\n', encoding="utf-8"
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
        assert arg_value(onboard_args, "--config-dir") == prepared.env["_EVAL_CONFIG_DIR"]
        assert "--quick" in onboard_args
        assert "--force" in onboard_args
        assert arg_value(onboard_args, "--provider") == "anthropic-custom:https://relay.example.com/v1"
        assert arg_value(onboard_args, "--model") == "claude-sonnet-4-6"

        agent_call = executor.calls[1]
        agent_args = agent_call["inner_args"]
        assert isinstance(agent_args, list)
        assert agent_args[0] == "agent"
        assert arg_value(agent_args, "--config-dir") == prepared.env["_EVAL_CONFIG_DIR"]
        assert arg_value(agent_args, "--provider") == "anthropic-custom:https://relay.example.com/v1"
        assert arg_value(agent_args, "--message")

        env = agent_call["inner_env"]
        assert isinstance(env, dict)
        assert env["ANTHROPIC_API_KEY"] == "anthropic-key"
        assert env["HOME"] == prepared.env["_EVAL_RUNTIME_DIR"]

        assert (zc_workspace / "README.md").read_text(encoding="utf-8") == "updated by zeroclaw\n"
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
        lambda config_dir_arg, workspace_dir_arg: (config_dir / "config.toml").write_text(
            '[security]\nlevel = "full"\n', encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        "agent_harness_eval.adapters.zeroclaw._inject_zeroclaw_api_key",
        lambda config_dir_arg, api_key: (config_dir / "config.toml").write_text(
            f'api_key = "{api_key}"\n', encoding="utf-8"
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


def test_sync_back_mirrors_workspace_state() -> None:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        src_dir = root / "src"
        dst_dir = root / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        (src_dir / "keep.txt").write_text("new contents\n")
        (src_dir / "nested").mkdir()
        (src_dir / "nested" / "created.txt").write_text("created\n")
        (src_dir / "dir-target").mkdir()
        (src_dir / "dir-target" / "inside.txt").write_text("dir now\n")
        (src_dir / "file-target.txt").write_text("file now\n")

        (dst_dir / "keep.txt").write_text("old contents\n")
        (dst_dir / "stale.txt").write_text("remove me\n")
        (dst_dir / "nested").mkdir()
        (dst_dir / "nested" / "stale.txt").write_text("remove nested\n")
        (dst_dir / "dir-target").write_text("was a file\n")
        (dst_dir / "file-target.txt").mkdir()
        (dst_dir / "file-target.txt" / "old.txt").write_text("was a dir\n")

        _sync_back(str(src_dir), str(dst_dir))

        assert (dst_dir / "keep.txt").read_text() == "new contents\n"
        assert not (dst_dir / "stale.txt").exists()
        assert not (dst_dir / "nested" / "stale.txt").exists()
        assert (dst_dir / "nested" / "created.txt").read_text() == "created\n"
        assert (dst_dir / "dir-target").is_dir()
        assert (dst_dir / "dir-target" / "inside.txt").read_text() == "dir now\n"
        assert (dst_dir / "file-target.txt").is_file()
        assert (dst_dir / "file-target.txt").read_text() == "file now\n"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_read_zeroclaw_runtime_trace_missing_file() -> None:
    result = _read_zeroclaw_runtime_trace("/nonexistent/path/trace.jsonl")
    assert result["trace"] == []
    assert result["usage"]["input"] == 0
    assert result["usage"]["output"] == 0
    assert result["usage"]["tool_calls"] == 0
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
                    "cache_creation_input_tokens": 30,
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

        assert usage["input"] == 1300
        assert usage["output"] == 600
        assert usage["cache_read"] == 50
        assert usage["cache_write"] == 30
        assert usage["total_tokens"] == 1980
        assert usage["tool_calls"] == 2
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


def test_read_zeroclaw_runtime_trace_pending_tools_closed() -> None:
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
        assert len(trace) == 1
        assert trace[0].type == "tool_call_started"
        assert trace[0].tool_name == "exec"
    finally:
        shutil.rmtree(root, ignore_errors=True)
