from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent_harness_eval.adapters.openclaw import (
    OpenClawAdapter,
    _finalize_openclaw_stream_tasks,
    _read_openclaw_session_final_text,
    _read_openclaw_session_terminal_text,
)
from agent_harness_eval.config.providers import ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.executor import VolumeMount, create_executor
from agent_harness_eval.task import Task, ToolBoundary
from agent_harness_eval.types import CanonicalTraceEvent
from agent_harness_eval.utils.subprocess import SubprocessResult

from ._regression_helpers import RecordingExecutor


@pytest.mark.asyncio
async def test_openclaw_prepare_uses_run_private_state_layout() -> None:
    rc = RuntimeConfig(project_root=Path.cwd())
    adapter = OpenClawAdapter(rc, create_executor(rc))
    task = Task(
        id="openclaw.prepare.01",
        category="coding",
        description="prepare should isolate workspace and state per run",
        user_query="Reply with OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        tool_boundary=ToolBoundary(
            internet="disabled",
            shell="enabled",
            file_write="disabled",
        ),
        timeout_sec=30,
    )

    prepared = adapter.prepare(task, "openclaw-prepare")

    try:
        assert prepared.workspace_dir == prepared.layout.workspace_dir
        assert prepared.env["_OPENCLAW_STATE_DIR"] == prepared.layout.state_dir

        config_path = Path(prepared.layout.state_dir) / "openclaw.json"
        workspace_seed_path = Path(prepared.layout.workspace_seed_dir) / "README.md"
        runtime_workspace_path = Path(prepared.layout.workspace_dir) / "README.md"

        assert config_path.is_file()
        assert workspace_seed_path.read_text() == "seed\n"
        assert runtime_workspace_path.read_text() == "seed\n"

        mounts = set(prepared.execution_policy.extra_mounts)
        assert (
            VolumeMount(
                source=prepared.layout.input_dir,
                target=prepared.layout.input_dir,
                mode="ro",
            )
            in mounts
        )
        assert (
            VolumeMount(
                source=prepared.layout.state_dir,
                target=prepared.layout.state_dir,
                mode="rw",
            )
            in mounts
        )
        assert (
            VolumeMount(
                source=prepared.layout.output_dir,
                target=prepared.layout.output_dir,
                mode="rw",
            )
            in mounts
        )
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_openclaw_run_builds_register_and_agent_commands_with_private_state(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
        _process_env={"ANTHROPIC_API_KEY": "anthropic-key"},
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
    )
    adapter = OpenClawAdapter(runtime_config, executor)
    task = Task(
        id="openclaw.regression.01",
        category="coding",
        description="openclaw command contract",
        user_query="Reply with OPENCLAW_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "openclaw-regression")

    monkeypatch.setattr(
        "agent_harness_eval.adapters.openclaw._read_openclaw_session_with_usage",
        lambda state_dir, agent_id, session_id: {
            "trace": [
                CanonicalTraceEvent(
                    type="message", role="assistant", text="OPENCLAW_OK", ts="2026-01-01T00:00:00+00:00"
                )
            ],
            "usage": {"input": 20, "output": 6, "cache_read": 0, "cache_write": 0, "total": 26, "turns": 1},
            "tool_calls": 0,
        },
    )

    async def fake_run_until_json(*args, **kwargs):
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
            "session_completed": True,
        }

    monkeypatch.setattr(
        "agent_harness_eval.adapters.openclaw._run_openclaw_subprocess_until_json",
        fake_run_until_json,
    )

    try:
        result = await adapter.run(prepared, "anthropic:claude-sonnet-4-6")

        assert result.status == "completed"
        assert result.final_text == "OPENCLAW_OK"
        assert len(executor.wrapped_calls) == 1
        call = executor.wrapped_calls[0]
        assert call["inner_command"] == "sh"
        args = call["inner_args"]
        assert isinstance(args, list)
        assert "-c" in args
        script = args[1]
        assert "agents add" in script
        assert "agent --local --json" in script
        assert "--session-id" in script
        assert "--timeout 30" in script
        assert "anthropic/claude-sonnet-4-6" in script
        env = call["inner_env"]
        assert isinstance(env, dict)
        assert env["OPENCLAW_STATE_DIR"] == prepared.env["_OPENCLAW_STATE_DIR"]
        assert env["OPENCLAW_CONFIG_PATH"].endswith("openclaw.json")
        assert env["HOME"] == prepared.env["_OPENCLAW_STATE_DIR"]
        assert env["ANTHROPIC_API_KEY"] == "anthropic-key"
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_openclaw_run_returns_failed_result_on_nonzero_exit(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
        _process_env={"ANTHROPIC_API_KEY": "anthropic-key"},
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(
            stdout="partial agent output",
            stderr="permission denied by sandbox",
            exit_code=2,
            timed_out=False,
        ),
    )
    adapter = OpenClawAdapter(runtime_config, executor)
    task = Task(
        id="openclaw.regression.nonzero-exit",
        category="coding",
        description="openclaw non-zero exit",
        user_query="Reply with OPENCLAW_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "openclaw-nonzero-exit")

    async def fake_run_until_json(*args, **kwargs):
        return {
            "stdout": "partial agent output",
            "stderr": "permission denied by sandbox",
            "exit_code": 2,
            "timed_out": False,
            "session_completed": False,
        }

    monkeypatch.setattr(
        "agent_harness_eval.adapters.openclaw._run_openclaw_subprocess_until_json",
        fake_run_until_json,
    )

    try:
        result = await adapter.run(prepared, "anthropic:claude-sonnet-4-6")

        assert result.status == "failed"
        assert result.final_text == "partial agent output"
        assert result.failure_origin == "sandbox"
        assert result.infra_error_code == "sandbox_permission_error"
        assert result.trace and result.trace[0].type == "task_failed"
        assert "OpenClaw exited with code 2" in (result.trace[0].error or "")
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_openclaw_timeout_recovers_partial_session_trace(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
        _process_env={"ANTHROPIC_API_KEY": "anthropic-key"},
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=True),
    )
    adapter = OpenClawAdapter(runtime_config, executor)
    task = Task(
        id="openclaw.regression.timeout-recovery",
        category="coding",
        description="openclaw timeout recovery",
        user_query="Reply with OPENCLAW_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "openclaw-timeout-recovery")

    monkeypatch.setattr(
        "agent_harness_eval.adapters.openclaw._read_openclaw_session_with_usage",
        lambda state_dir, agent_id, session_id: {
            "trace": [
                CanonicalTraceEvent(
                    type="message",
                    role="assistant",
                    text="partial answer",
                    ts="2026-01-01T00:00:00+00:00",
                ),
                CanonicalTraceEvent(
                    type="tool_call_started",
                    tool_name="read_file",
                    ts="2026-01-01T00:00:01+00:00",
                ),
            ],
            "usage": {
                "input": 10,
                "output": 5,
                "cache_read": 2,
                "cache_write": 1,
                "total": 18,
                "turns": 3,
            },
            "tool_calls": 1,
        },
    )

    async def fake_run_until_json(*args, **kwargs):
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "timed_out": True,
            "session_completed": False,
        }

    monkeypatch.setattr(
        "agent_harness_eval.adapters.openclaw._run_openclaw_subprocess_until_json",
        fake_run_until_json,
    )

    try:
        result = await adapter.run(prepared, "anthropic:claude-sonnet-4-6")

        assert result.status == "timed_out"
        assert result.final_text == "partial answer"
        assert [event.type for event in result.trace] == ["message", "tool_call_started", "task_failed"]
        assert result.trace[-1].error == "OpenClaw timed out"
        assert result.metrics.input_tokens == 10
        assert result.metrics.output_tokens == 5
        assert result.metrics.cache_read_tokens == 2
        assert result.metrics.cache_write_tokens == 1
        assert result.metrics.total_tokens == 18
        assert result.metrics.tool_calls == 1
        assert result.metrics.turns == 3
        assert result.metrics.latency_sec == 30
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_openclaw_run_classifies_empty_subprocess_output_as_failed(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a clean-exit subprocess that produces no content must not be
    labeled "completed".

    Observed in v1 rerun on skills.02: subprocess ran 79s, exited 0, stdout had
    structurally-valid JSON but ``payloads=[]`` and no ``sessionId`` in
    ``agentMeta`` → the previous code path returned ``status="completed"`` with
    empty ``final_text`` and all-zero metrics. Downstream graders saw "completed
    with 0/9 passed" which is indistinguishable from a real bad-answer failure.
    The fix reclassifies as ``failed`` with ``infra_error_code="adapter_empty_output"``
    so the silent crash is visible in reports.
    """
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
        _process_env={"ANTHROPIC_API_KEY": "anthropic-key"},
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
    )
    adapter = OpenClawAdapter(runtime_config, executor)
    task = Task(
        id="openclaw.regression.empty-output",
        category="skills",
        description="openclaw silent failure",
        user_query="Reply with OPENCLAW_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "openclaw-empty-output")

    async def fake_run_until_json(*args, **kwargs):
        # Valid JSON, but payloads=[] and no sessionId → every content-producing
        # codepath yields nothing, which is the silent-failure signature.
        return {
            "stdout": '{"payloads": [], "meta": {"agentMeta": {}}}',
            "stderr": "agent exited before first turn",
            "exit_code": 0,
            "timed_out": False,
            "session_completed": False,
        }

    monkeypatch.setattr(
        "agent_harness_eval.adapters.openclaw._run_openclaw_subprocess_until_json",
        fake_run_until_json,
    )
    # No session log should be consulted since agentMeta has no sessionId, but
    # stub it anyway so an accidental call returns an empty result rather than
    # hitting disk.
    monkeypatch.setattr(
        "agent_harness_eval.adapters.openclaw._read_openclaw_session_with_usage",
        lambda state_dir, agent_id, session_id: {
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
        result = await adapter.run(prepared, "anthropic:claude-sonnet-4-6")

        assert result.status == "failed"
        assert result.failure_origin == "adapter"
        assert result.infra_error_code == "adapter_empty_output"
        assert result.final_text != ""  # final_text holds the raw stdout for post-mortem
        assert result.trace and result.trace[0].type == "task_failed"
        assert "produced no output" in (result.trace[0].error or "")
        # stderr tail must be preserved so operators can see what the subprocess said
        assert "agent exited before first turn" in (result.trace[0].error or "")
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_openclaw_run_fails_fast_when_provider_is_missing(
    isolated_run_dir: Path,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={},
        _process_env={"ANTHROPIC_API_KEY": "anthropic-key"},
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
    )
    adapter = OpenClawAdapter(runtime_config, executor)
    task = Task(
        id="openclaw.regression.missing-provider",
        category="coding",
        description="openclaw missing provider",
        user_query="Reply with OPENCLAW_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "openclaw-missing-provider")

    try:
        with pytest.raises(ValueError, match='openclaw: no provider "anthropic" configured'):
            await adapter.run(prepared, "anthropic:claude-sonnet-4-6")
        assert executor.calls == []
    finally:
        adapter.cleanup(prepared)


def test_read_openclaw_session_final_text_falls_back_to_latest_session_file(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    latest_path = session_dir / "actual.jsonl"
    latest_path.write_text(
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "FINAL_ANSWER"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    final_text = _read_openclaw_session_final_text(str(session_dir), str(session_dir / "missing.jsonl"))

    assert final_text == "FINAL_ANSWER"


def test_read_openclaw_session_terminal_text_requires_non_tooluse_terminal_message(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    latest_path = session_dir / "actual.jsonl"
    latest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "INTERMEDIATE"},
                                {"type": "toolCall", "name": "exec", "arguments": {"command": "python -V"}},
                            ],
                            "stopReason": "toolUse",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "FINAL_ANSWER"}],
                            "stopReason": "stop",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    final_text = _read_openclaw_session_terminal_text(str(session_dir), str(session_dir / "missing.jsonl"))

    assert final_text == "FINAL_ANSWER"


def test_read_openclaw_session_terminal_text_ignores_intermediate_assistant_text(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    latest_path = session_dir / "actual.jsonl"
    latest_path.write_text(
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "INTERMEDIATE"},
                        {"type": "toolCall", "name": "exec", "arguments": {"command": "python -V"}},
                    ],
                    "stopReason": "toolUse",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    final_text = _read_openclaw_session_terminal_text(str(session_dir), str(session_dir / "missing.jsonl"))

    assert final_text == ""


@pytest.mark.asyncio
async def test_finalize_openclaw_stream_tasks_cancels_hanging_readers() -> None:
    async def hanging_reader() -> None:
        await asyncio.Future()

    task = asyncio.create_task(hanging_reader())

    await _finalize_openclaw_stream_tasks(task)

    assert task.done()
    assert task.cancelled()
