from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness_eval.adapters.claude_code import (
    ClaudeCodeAdapter,
    _events_to_trace,
    _extract_final_text,
    _parse_stream_json,
)
from agent_harness_eval.config.providers import ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.task import Task
from agent_harness_eval.utils.subprocess import SubprocessResult

from ._regression_helpers import RecordingExecutor, arg_value


@pytest.mark.asyncio
async def test_claude_code_run_projects_provider_env_and_cli_contract(
    isolated_run_dir: Path,
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
        result = await adapter.run(prepared, "anthropic:claude-sonnet-4-6")

        assert result.status == "completed"
        assert result.final_text == "CLAUDE_OK"
        assert len(executor.calls) == 1
        call = executor.calls[0]
        assert call["inner_command"] == "claude"
        assert call["timeout_ms"] == 30_000
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

    assert [event.type for event in trace] == ["tool_call_started", "tool_call_completed", "message"]
    assert trace[1].success is True
    assert trace[1].output == "file contents"
    assert _extract_final_text(events) == "Final answer"


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
