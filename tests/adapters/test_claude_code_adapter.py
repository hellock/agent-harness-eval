from __future__ import annotations

from pathlib import Path

import pytest

import agent_harness_eval.adapters.claude_code as claude_code_module
from agent_harness_eval.adapters.claude_code import (
    ClaudeCodeAdapter,
    _events_to_trace,
    _extract_final_text,
    _parse_stream_json,
    _run_claude_streaming,
)
from agent_harness_eval.config.providers import ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.task import Task
from agent_harness_eval.utils.subprocess import SubprocessResult

from ._regression_helpers import RecordingExecutor, arg_value


@pytest.mark.asyncio
async def test_claude_code_run_projects_provider_env_and_cli_contract(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "anthropic": ProviderConfig(
                base_url="https://relay.example.com",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(
            stdout='{"type":"result","result":"CLAUDE_OK","usage":{"input_tokens":12,"output_tokens":3}}\n',
            stderr="",
            exit_code=0,
            timed_out=False,
        ),
    )
    adapter = ClaudeCodeAdapter(runtime_config, executor)
    task = Task(
        id="claude.regression.01",
        category="coding",
        description="claude cli contract",
        user_query="Reply with CLAUDE_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "claude-regression")

    try:

        async def fake_run_claude_streaming(
            command: str,
            args: list[str],
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_ms: int,
        ) -> dict[str, object]:
            return {
                "stdout": '{"type":"result","result":"CLAUDE_OK","usage":{"input_tokens":12,"output_tokens":3}}\n',
                "stderr": "",
                "exit_code": 0,
                "timed_out": False,
            }

        monkeypatch.setattr(claude_code_module, "_run_claude_streaming", fake_run_claude_streaming)
        result = await adapter.run(prepared, "anthropic:claude-sonnet-4-6")

        assert result.status == "completed"
        assert result.final_text == "CLAUDE_OK"
        assert len(executor.calls) == 0
        assert len(executor.wrapped_calls) == 1
        call = executor.wrapped_calls[0]
        assert call["inner_command"] == "claude"
        args = call["inner_args"]
        assert isinstance(args, list)
        assert "--output-format" in args and arg_value(args, "--output-format") == "stream-json"
        assert "--dangerously-skip-permissions" in args
        assert "--no-session-persistence" in args
        assert "--session-id" in args
        assert arg_value(args, "--session-id") == prepared.env["_EVAL_SESSION_ID"]
        env = call["inner_env"]
        assert isinstance(env, dict)
        assert env["ANTHROPIC_API_KEY"] == "anthropic-key"
        assert env["ANTHROPIC_BASE_URL"] == "https://relay.example.com"
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
        assert env["CLAUDE_CONFIG_DIR"] == prepared.env["CLAUDE_CONFIG_DIR"]
        assert env["HOME"] == prepared.env["_HOME_DIR"]
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_claude_code_run_returns_failed_result_on_nonzero_exit(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "anthropic": ProviderConfig(
                base_url="https://relay.example.com",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(
            stdout="partial assistant output",
            stderr="permission denied by sandbox",
            exit_code=2,
            timed_out=False,
        ),
    )
    adapter = ClaudeCodeAdapter(runtime_config, executor)
    task = Task(
        id="claude.regression.nonzero-exit",
        category="coding",
        description="claude non-zero exit",
        user_query="Reply with CLAUDE_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "claude-nonzero-exit")

    try:

        async def fake_run_claude_streaming(
            command: str,
            args: list[str],
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_ms: int,
        ) -> dict[str, object]:
            return {
                "stdout": "partial assistant output",
                "stderr": "permission denied by sandbox",
                "exit_code": 2,
                "timed_out": False,
            }

        monkeypatch.setattr(claude_code_module, "_run_claude_streaming", fake_run_claude_streaming)
        result = await adapter.run(prepared, "anthropic:claude-sonnet-4-6")

        assert result.status == "failed"
        assert result.final_text == "partial assistant output"
        assert result.failure_origin == "sandbox"
        assert result.infra_error_code == "sandbox_permission_error"
        assert result.trace and result.trace[0].type == "task_failed"
        assert "Claude Code exited with code 2" in (result.trace[0].error or "")
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_claude_code_timeout_preserves_partial_stream_trace(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "anthropic": ProviderConfig(
                base_url="https://relay.example.com",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
    )
    adapter = ClaudeCodeAdapter(runtime_config, executor)
    task = Task(
        id="claude.regression.timeout-partial-stream",
        category="coding",
        description="claude timeout partial stream",
        user_query="Reply with CLAUDE_OK after reading a file.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "claude-timeout-partial-stream")

    try:

        async def fake_run_claude_streaming(
            command: str,
            args: list[str],
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_ms: int,
        ) -> dict[str, object]:
            return {
                "stdout": "\n".join(
                    [
                        '{"type":"assistant","message":{"usage":{"input_tokens":11,"output_tokens":0},"content":[{"type":"tool_use","name":"Read","input":{"file_path":"README.md"},"id":"tool-1"}]}}',
                        '{"type":"user","message":{"content":[{"type":"tool_result","content":"seed\\n","tool_use_id":"tool-1"}]}}',
                        '{"type":"assistant","message":{"usage":{"input_tokens":11,"output_tokens":4},"content":[{"type":"text","text":"Working on it"}]}}',
                    ]
                )
                + "\n",
                "stderr": "",
                "exit_code": None,
                "timed_out": True,
            }

        monkeypatch.setattr(claude_code_module, "_run_claude_streaming", fake_run_claude_streaming)

        result = await adapter.run(prepared, "anthropic:claude-sonnet-4-6")

        assert result.status == "timed_out"
        assert result.final_text == "Working on it"
        assert [event.type for event in result.trace] == [
            "tool_call_started",
            "tool_call_completed",
            "message",
        ]
        assert result.trace[0].tool_name == "Read"
        assert result.trace[1].success is True
        assert result.metrics.input_tokens == 22
        assert result.metrics.output_tokens == 4
        assert result.metrics.tool_calls == 1
        assert result.metrics.turns == 2
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_claude_code_run_reclassifies_tool_only_completion_as_failed(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "anthropic": ProviderConfig(
                base_url="https://relay.example.com",
                api_key="anthropic-key",
                api_format="anthropic",
            )
        },
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
    )
    adapter = ClaudeCodeAdapter(runtime_config, executor)
    task = Task(
        id="claude.regression.tool-only-completion",
        category="skills",
        description="claude tool-only completion",
        user_query="Use a tool, then reply.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "claude-tool-only-completion")

    try:

        async def fake_run_claude_streaming(
            command: str,
            args: list[str],
            *,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_ms: int,
        ) -> dict[str, object]:
            return {
                "stdout": "\n".join(
                    [
                        '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"WebSearch","input":{"query":"tokyo sakura"},"id":"tool-1"}]}}',
                        '{"type":"user","message":{"content":[{"type":"tool_result","content":"peak already passed","tool_use_id":"tool-1"}]}}',
                    ]
                )
                + "\n",
                "stderr": "stream ended before final result",
                "exit_code": 0,
                "timed_out": False,
            }

        monkeypatch.setattr(claude_code_module, "_run_claude_streaming", fake_run_claude_streaming)
        result = await adapter.run(prepared, "anthropic:claude-sonnet-4-6")

        assert result.status == "failed"
        assert result.failure_origin == "adapter"
        assert result.infra_error_code == "adapter_empty_output"
        assert result.trace and result.trace[0].type == "task_failed"
        assert "produced no output" in (result.trace[0].error or "")
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_claude_code_run_fails_fast_on_api_format_mismatch(
    isolated_run_dir: Path,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="openai-key",
                api_format="openai-responses",
            )
        },
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
    )
    adapter = ClaudeCodeAdapter(runtime_config, executor)
    task = Task(
        id="claude.regression.api-format-mismatch",
        category="coding",
        description="claude provider api format mismatch",
        user_query="Reply with CLAUDE_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "claude-api-format-mismatch")

    try:
        with pytest.raises(ValueError, match='claude-code: provider "relay" has api_format "openai-responses"'):
            await adapter.run(prepared, "relay:gpt-5.4")
        assert executor.calls == []
    finally:
        adapter.cleanup(prepared)


def test_parse_stream_json_and_events_to_trace_pair_tool_results() -> None:
    """The ``result`` event must NOT emit a message — its text is the same as
    the preceding ``assistant`` text block (claude-code's stream duplicates
    the final answer across both event types). The dedicated extractor
    ``_extract_final_text`` still reads the ``result`` event for the answer.
    """
    events = _parse_stream_json(
        "\n".join(
            [
                '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"read_file","input":{"path":"a.txt"},"id":"tool-1"}]}}',
                '{"type":"user","message":{"content":[{"type":"tool_result","content":"file contents","tool_use_id":"tool-1"}]}}',
                '{"type":"result","result":"Final answer"}',
            ]
        )
    )
    trace = _events_to_trace(events)

    # result event must not emit a message (v2 regression fix): previously
    # the trace ended with a duplicate assistant message 1ms apart from the
    # preceding assistant text block.
    assert [event.type for event in trace] == ["tool_call_started", "tool_call_completed"]
    assert trace[1].success is True
    assert trace[1].output == "file contents"
    # Final-text extraction still pulls from the result event.
    assert _extract_final_text(events) == "Final answer"


def test_events_to_trace_does_not_duplicate_final_assistant_text() -> None:
    """Regression: v2 showed 4/4 claude-code completed runs with two adjacent
    identical assistant messages 1ms apart — assistant.text block + result
    event carrying the same text. After the fix, only the text block emits a
    trace message; the result event is consumed by ``_extract_final_text``.
    """
    events = [
        {
            "type": "assistant",
            "timestamp": "2026-04-15T10:00:00.000+00:00",
            "message": {"content": [{"type": "text", "text": "the final answer"}]},
        },
        {
            "type": "result",
            "timestamp": "2026-04-15T10:00:00.010+00:00",
            "result": "the final answer",
        },
    ]
    trace = _events_to_trace(events)

    # Exactly one message — the assistant text block. No dup from result.
    assistant_msgs = [e for e in trace if e.type == "message" and e.role == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].text == "the final answer"
    # But the extractor still recovers the final answer.
    assert _extract_final_text(events) == "the final answer"


def test_events_to_trace_marks_orphaned_tool_as_failed() -> None:
    trace = _events_to_trace(
        [
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "name": "read_file", "input": {"path": "a.txt"}, "id": "tool-1"}]
                },
            }
        ]
    )

    assert [event.type for event in trace] == ["tool_call_started", "tool_call_completed"]
    assert trace[1].success is False
    assert trace[1].output == "<no result recorded>"


def test_events_to_trace_normalizes_out_of_order_timestamps() -> None:
    trace = _events_to_trace(
        [
            {
                "type": "assistant",
                "timestamp": "2026-04-15T07:31:38+00:00",
                "message": {
                    "content": [{"type": "tool_use", "name": "bash", "input": {"command": "ls"}, "id": "tool-1"}]
                },
            },
            {
                "type": "user",
                "timestamp": "2026-04-15T07:30:28+00:00",
                "message": {"content": [{"type": "tool_result", "content": "ok", "tool_use_id": "tool-1"}]},
            },
        ]
    )

    assert [event.type for event in trace] == ["tool_call_started", "tool_call_completed"]
    assert trace[0].ts is not None
    assert trace[1].ts is not None
    assert trace[0].ts < trace[1].ts


def test_events_to_trace_preserves_real_timestamps_when_forward() -> None:
    """Real timestamps that are naturally forward-moving must survive unchanged.

    Previously the no-ts fallback was ``datetime.now(UTC)`` at parse time, which
    set a baseline far in the future and caused all subsequent real ts to fail
    the monotonic check and get bumped to synthetic epsilon-deltas. After the
    fix, the no-ts fallback inherits ``last_ts``, leaving real ts intact when
    they are already moving forward.
    """
    trace = _events_to_trace(
        [
            {
                "type": "user",
                "timestamp": "2026-04-15T07:30:28.500+00:00",
                "message": {"content": [{"type": "tool_result", "content": "seed", "tool_use_id": "seed"}]},
            },
            {
                # no timestamp — must inherit from previous event, not jump to now()
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "name": "bash", "input": {"command": "ls"}, "id": "tool-2"}]
                },
            },
            {
                "type": "user",
                "timestamp": "2026-04-15T07:30:35.000+00:00",
                "message": {"content": [{"type": "tool_result", "content": "ok", "tool_use_id": "tool-2"}]},
            },
        ]
    )

    # Real ms-aligned timestamps on user events survive at canonical
    # granularity (ms precision is the contract — see utils/timestamps.py).
    assert trace[0].ts == "2026-04-15T07:30:28.500+00:00"
    assert trace[2].ts == "2026-04-15T07:30:35.000+00:00"
    # The middle assistant event inherits last_ts and gets bumped by the
    # canonical monotonicity floor (1 ms), so it stays anchored near its
    # neighbors instead of drifting to parse-time ``now()``.
    assert trace[0].ts < trace[1].ts < trace[2].ts
    assert trace[1].ts == "2026-04-15T07:30:28.501+00:00"


@pytest.mark.asyncio
async def test_run_claude_streaming_uses_shared_subprocess_cleanup_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class DummyStream:
        async def readline(self) -> bytes:
            return b""

    class DummyProc:
        pid = 4321
        returncode = 0
        stdout = DummyStream()
        stderr = DummyStream()

        async def wait(self) -> int:
            return 0

    async def fake_create_subprocess_exec(command, *args, **kwargs):
        calls.append(("preexec_fn", kwargs.get("preexec_fn")))
        return DummyProc()

    monkeypatch.setattr(
        claude_code_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        claude_code_module,
        "_install_subprocess_cleanup_handlers",
        lambda: calls.append(("install_handlers", None)),
    )
    monkeypatch.setattr(
        claude_code_module,
        "_register_active_subprocess",
        lambda proc, use_pgroup: calls.append(("register", (proc.pid, use_pgroup))),
    )
    monkeypatch.setattr(
        claude_code_module,
        "_unregister_active_subprocess",
        lambda proc: calls.append(("unregister", proc.pid)),
    )

    result = await _run_claude_streaming(
        "claude",
        ["--version"],
        timeout_ms=1000,
    )

    assert result["timed_out"] is False
    assert ("install_handlers", None) in calls
    assert any(name == "register" and payload == (4321, True) for name, payload in calls)
    assert ("unregister", 4321) in calls
    preexec_call = next(payload for name, payload in calls if name == "preexec_fn")
    if claude_code_module.sys.platform == "win32":
        assert preexec_call is None
    else:
        assert preexec_call is claude_code_module._preexec_new_pgroup
