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
    parsed = _parse_codex_jsonl(
        "\n".join(
            [
                '{"type":"tool_use","name":"read_file","input":{"path":"README.md"},"timestamp":1704067200000}',
                '{"type":"tool_result","name":"read_file","content":"hello","status":"ok"}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}',
            ]
        )
    )

    assert parsed["final_text"] == "Done"
    assert parsed["trace"][0].type == "tool_call_started"
    assert parsed["trace"][0].tool_name == "read_file"
    assert parsed["trace"][1].type == "tool_call_completed"
    assert parsed["trace"][1].output == "hello"


def test_parse_codex_jsonl_ignores_invalid_lines() -> None:
    parsed = _parse_codex_jsonl("not json\n\n")

    assert parsed == {"final_text": "", "trace": []}
