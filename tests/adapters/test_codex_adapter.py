from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness_eval.adapters.codex import CodexAdapter, _parse_codex_jsonl
from agent_harness_eval.config.providers import ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.task import Task, ToolBoundary
from agent_harness_eval.utils.subprocess import SubprocessResult

from ._regression_helpers import RecordingExecutor, arg_value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("file_write", "expected_sandbox"),
    [
        ("enabled", "workspace-write"),
        ("disabled", "read-only"),
    ],
)
async def test_codex_run_uses_cli_supported_sandbox_modes(
    isolated_run_dir: Path,
    file_write: str,
    expected_sandbox: str,
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
        SubprocessResult(
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}\n',
            stderr="",
            exit_code=0,
            timed_out=False,
        ),
    )
    adapter = CodexAdapter(runtime_config, executor)
    task = Task(
        id=f"codex.sandbox.{file_write}",
        category="coding",
        description="verify codex sandbox selection",
        user_query="Reply with OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        tool_boundary=ToolBoundary(internet="disabled", shell="enabled", file_write=file_write),
        timeout_sec=30,
    )

    prepared = adapter.prepare(task, f"codex-sandbox-{file_write}")

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "completed"
        inner_args = executor.calls[0]["inner_args"]
        assert isinstance(inner_args, list)
        sandbox_index = inner_args.index("--sandbox")
        assert inner_args[sandbox_index + 1] == expected_sandbox
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_codex_run_bypasses_inner_sandbox_for_docker_shell_tasks(
    isolated_run_dir: Path,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        executor_backend="docker",
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
        SubprocessResult(
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}\n',
            stderr="",
            exit_code=0,
            timed_out=False,
        ),
    )
    adapter = CodexAdapter(runtime_config, executor)
    task = Task(
        id="codex.docker.bypass",
        category="coding",
        description="docker codex should rely on outer container isolation",
        user_query="Reply with OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        tool_boundary=ToolBoundary(internet="disabled", shell="enabled", file_write="enabled"),
        timeout_sec=30,
    )

    prepared = adapter.prepare(task, "codex-docker-bypass")

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "completed"
        args = executor.calls[0]["inner_args"]
        assert isinstance(args, list)
        assert "--dangerously-bypass-approvals-and-sandbox" in args
        assert "--sandbox" not in args
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_codex_run_writes_config_and_exec_contract(
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
        SubprocessResult(
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"CODEX_OK"}}\n',
            stderr="",
            exit_code=0,
            timed_out=False,
        ),
    )
    adapter = CodexAdapter(runtime_config, executor)
    task = Task(
        id="codex.regression.01",
        category="coding",
        description="codex command contract",
        user_query="Reply with CODEX_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        tool_boundary=ToolBoundary(internet="disabled", shell="enabled", file_write="enabled"),
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "codex-regression")

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "completed"
        assert result.final_text == "CODEX_OK"
        assert len(executor.calls) == 1
        call = executor.calls[0]
        assert call["inner_command"] == "codex"
        args = call["inner_args"]
        assert isinstance(args, list)
        assert "exec" in args
        assert "--json" in args
        assert "--skip-git-repo-check" in args
        assert "--cd" in args and arg_value(args, "--cd") == str(prepared.workspace_dir)
        assert arg_value(args, "--sandbox") == "workspace-write"
        assert arg_value(args, "--model") == "gpt-5.4"
        env = call["inner_env"]
        assert isinstance(env, dict)
        assert env["OPENAI_API_KEY"] == "openai-key"
        assert env["CODEX_HOME"] == prepared.env["_EVAL_RUNTIME_DIR"]

        config_toml = (Path(prepared.env["_EVAL_RUNTIME_DIR"]) / "config.toml").read_text(encoding="utf-8")
        assert 'model = "gpt-5.4"' in config_toml
        assert 'model_provider = "relay"' in config_toml
        assert 'base_url = "https://relay.example.com/v1"' in config_toml
        assert 'env_key = "OPENAI_API_KEY"' in config_toml
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_codex_run_uses_read_only_sandbox_when_file_write_disabled(
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
        SubprocessResult(
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"CODEX_OK"}}\n',
            stderr="",
            exit_code=0,
            timed_out=False,
        ),
    )
    adapter = CodexAdapter(runtime_config, executor)
    task = Task(
        id="codex.regression.read-only-sandbox",
        category="coding",
        description="codex read-only sandbox contract",
        user_query="Reply with CODEX_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        tool_boundary=ToolBoundary(internet="disabled", shell="enabled", file_write="disabled"),
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "codex-read-only")

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "completed"
        assert len(executor.calls) == 1
        args = executor.calls[0]["inner_args"]
        assert isinstance(args, list)
        assert arg_value(args, "--sandbox") == "read-only"
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_codex_run_returns_failed_result_on_nonzero_exit(
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
        SubprocessResult(
            stdout="partial output",
            stderr="permission denied by sandbox",
            exit_code=2,
            timed_out=False,
        ),
    )
    adapter = CodexAdapter(runtime_config, executor)
    task = Task(
        id="codex.regression.nonzero-exit",
        category="coding",
        description="codex non-zero exit",
        user_query="Reply with CODEX_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "codex-nonzero-exit")

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "failed"
        assert result.final_text == "partial output"
        assert result.failure_origin == "sandbox"
        assert result.infra_error_code == "sandbox_permission_error"
        assert result.trace and result.trace[0].type == "task_failed"
        assert "Codex exited with code 2" in (result.trace[0].error or "")
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_codex_run_fails_fast_when_provider_is_missing(
    isolated_run_dir: Path,
) -> None:
    runtime_config = RuntimeConfig(project_root=isolated_run_dir, providers={})
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False),
    )
    adapter = CodexAdapter(runtime_config, executor)
    task = Task(
        id="codex.regression.missing-provider",
        category="coding",
        description="codex missing provider",
        user_query="Reply with CODEX_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "codex-missing-provider")

    try:
        with pytest.raises(ValueError, match='codex: no provider "relay" configured'):
            await adapter.run(prepared, "relay:gpt-5.4")
        assert executor.calls == []
    finally:
        adapter.cleanup(prepared)


def test_parse_codex_jsonl_extracts_final_text_and_tool_events() -> None:
    """Matches the real schema emitted by `codex exec --json` (verified
    empirically against the codex binary, April 2026):

      - Tool events arrive as ``item.started`` / ``item.completed`` with
        ``item.type`` naming the tool kind (``command_execution`` for shell,
        ``file_change`` / ``web_search`` / ``mcp_tool_call`` for others).
      - Assistant replies arrive as ``item.completed`` with
        ``item.type == "agent_message"`` and ``item.text``.
      - Per-turn token usage arrives on ``turn.completed.usage`` with
        OpenAI-cache semantics (``input_tokens`` / ``cached_input_tokens`` /
        ``output_tokens`` — no cache-creation counter).

    Pre-fix, the adapter matched top-level ``tool_use`` / ``tool_result`` —
    schemas codex never emits. Tool events silently dropped; metrics all zero.
    """
    parsed = _parse_codex_jsonl(
        "\n".join(
            [
                '{"type":"thread.started","thread_id":"th-1"}',
                '{"type":"turn.started"}',
                '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"Reading the file."}}',
                '{"type":"item.started","item":{"id":"item_1","type":"command_execution","command":"cat README.md","status":"in_progress"}}',
                '{"type":"item.completed","item":{"id":"item_1","type":"command_execution","command":"cat README.md","aggregated_output":"hello","exit_code":0,"status":"completed"}}',
                '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"Done"}}',
                '{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":40,"output_tokens":15}}',
            ]
        )
    )

    # Final text = last agent_message
    assert parsed["final_text"] == "Done"

    # Trace: message, tool_call_started (command_execution), tool_call_completed, message
    types = [e.type for e in parsed["trace"]]
    assert types == ["message", "tool_call_started", "tool_call_completed", "message"]

    assert parsed["trace"][0].role == "assistant"
    assert parsed["trace"][0].text == "Reading the file."

    assert parsed["trace"][1].tool_name == "command_execution"
    assert parsed["trace"][1].input == "cat README.md"

    assert parsed["trace"][2].tool_name == "command_execution"
    assert parsed["trace"][2].success is True
    assert parsed["trace"][2].output == "hello"

    assert parsed["trace"][3].text == "Done"

    # Token usage accumulated from turn.completed
    assert parsed["usage"]["input"] == 100
    assert parsed["usage"]["output"] == 15
    assert parsed["usage"]["cache_read"] == 40
    # No cache_write in codex — OpenAI prompt-caching only exposes reads.
    assert parsed["usage"]["cache_write"] == 0
    assert parsed["usage"]["total"] == 155  # 100 + 15 + 40
    assert parsed["usage"]["turns"] == 1
    assert parsed["tool_calls"] == 1


def test_parse_codex_jsonl_filters_reasoning_items() -> None:
    """Codex's ``reasoning`` items are internal chain-of-thought — not a tool
    call, not a user-visible message. Must not appear in the canonical trace.
    """
    parsed = _parse_codex_jsonl(
        "\n".join(
            [
                '{"type":"item.started","item":{"id":"r1","type":"reasoning"}}',
                '{"type":"item.completed","item":{"id":"r1","type":"reasoning","summary":"thinking about it"}}',
                '{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"Answer"}}',
                '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}',
            ]
        )
    )

    # Only the agent_message becomes a trace event; reasoning is dropped.
    assert [e.type for e in parsed["trace"]] == ["message"]
    assert parsed["final_text"] == "Answer"


def test_parse_codex_jsonl_failed_tool_call_recorded_as_failure() -> None:
    """Codex sets ``status="failed"`` + non-zero ``exit_code`` when a
    command_execution fails. The canonical trace must reflect success=False.
    """
    parsed = _parse_codex_jsonl(
        "\n".join(
            [
                '{"type":"item.started","item":{"id":"i1","type":"command_execution","command":"cat /nonexistent","status":"in_progress"}}',
                '{"type":"item.completed","item":{"id":"i1","type":"command_execution","command":"cat /nonexistent","aggregated_output":"cat: /nonexistent: No such file or directory","exit_code":1,"status":"failed"}}',
                '{"type":"turn.completed","usage":{"input_tokens":20,"output_tokens":10}}',
            ]
        )
    )

    completed = [e for e in parsed["trace"] if e.type == "tool_call_completed"]
    assert len(completed) == 1
    assert completed[0].success is False
    assert "No such file" in (completed[0].output or "")


def test_parse_codex_jsonl_accumulates_usage_across_turns() -> None:
    """Multi-turn runs: usage from every ``turn.completed`` must accumulate."""
    parsed = _parse_codex_jsonl(
        "\n".join(
            [
                '{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":20,"output_tokens":10}}',
                '{"type":"turn.completed","usage":{"input_tokens":50,"cached_input_tokens":40,"output_tokens":5}}',
            ]
        )
    )

    assert parsed["usage"]["input"] == 150
    assert parsed["usage"]["output"] == 15
    assert parsed["usage"]["cache_read"] == 60
    assert parsed["usage"]["turns"] == 2


def test_parse_codex_jsonl_orphan_tool_call_synthesizes_completion() -> None:
    """If an ``item.started`` has no matching ``item.completed`` (stream
    truncated mid-tool), emit a synthetic failed tool_call_completed at tail
    — matches claude_code / zeroclaw orphan-handling parity.
    """
    parsed = _parse_codex_jsonl(
        "\n".join(
            [
                '{"type":"item.started","item":{"id":"i1","type":"command_execution","command":"sleep 100","status":"in_progress"}}',
            ]
        )
    )

    types = [e.type for e in parsed["trace"]]
    assert types == ["tool_call_started", "tool_call_completed"]
    assert parsed["trace"][1].success is False
    assert parsed["trace"][1].output == "<no result recorded>"


def test_parse_codex_jsonl_ignores_invalid_lines() -> None:
    parsed = _parse_codex_jsonl("not json\n\n")

    assert parsed["final_text"] == ""
    assert parsed["trace"] == []
    assert parsed["usage"]["turns"] == 0
    assert parsed["tool_calls"] == 0
