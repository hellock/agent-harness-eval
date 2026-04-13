from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.executor import ExecutionPolicy
from agent_harness_eval.executor.host import HostExecutor
from agent_harness_eval.utils.subprocess import SubprocessResult


def test_host_wrap_injects_boundary_env_and_cwd(tmp_path: Path) -> None:
    executor = HostExecutor(RuntimeConfig(project_root=tmp_path))
    workspace_dir = str(tmp_path / "workspace")
    Path(workspace_dir).mkdir()
    policy = ExecutionPolicy(
        network=False,
        shell=False,
        file_write=True,
        workspace_dir=workspace_dir,
        timeout_sec=45,
    )

    wrapped = executor._wrap(
        policy,
        "python",
        ["-V"],
        {"HOME": "/tmp/home", "CUSTOM": "value"},
    )

    assert wrapped.command == "python"
    assert wrapped.args == ["-V"]
    assert wrapped.cwd == workspace_dir
    assert wrapped.env["HOME"] == "/tmp/home"
    assert wrapped.env["CUSTOM"] == "value"
    assert wrapped.env["EVAL_BOUNDARY_INTERNET"] == "disabled"
    assert wrapped.env["EVAL_BOUNDARY_SHELL"] == "disabled"
    assert "EVAL_BOUNDARY_FILE_WRITE" not in wrapped.env


def test_host_wrap_uses_explicit_cwd_when_provided(tmp_path: Path) -> None:
    executor = HostExecutor(RuntimeConfig(project_root=tmp_path))
    workspace_dir = str(tmp_path / "workspace")
    repo_dir = str(tmp_path / "workspace" / "repo")
    Path(repo_dir).mkdir(parents=True)
    policy = ExecutionPolicy(
        workspace_dir=workspace_dir,
        cwd=repo_dir,
    )

    wrapped = executor._wrap(
        policy,
        "python",
        ["-V"],
        {"HOME": "/tmp/home"},
    )

    assert wrapped.cwd == repo_dir


def test_host_wrap_marks_workspace_readonly_when_file_write_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = HostExecutor(RuntimeConfig(project_root=tmp_path))
    workspace_dir = str(tmp_path / "workspace")
    Path(workspace_dir).mkdir()
    policy = ExecutionPolicy(
        network=True,
        shell=True,
        file_write=False,
        workspace_dir=workspace_dir,
    )
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> object:
        calls.append(args)
        return object()

    monkeypatch.setattr("agent_harness_eval.executor.host.subprocess.run", fake_run)

    wrapped = executor._wrap(policy, "bash", ["-lc", "echo hi"], {"HOME": "/tmp/home"})

    assert calls == [["chmod", "-R", "a-w", workspace_dir]]
    assert wrapped.env["EVAL_BOUNDARY_FILE_WRITE"] == "disabled"


def test_restore_workspace_restores_write_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = HostExecutor(RuntimeConfig(project_root=tmp_path))
    workspace_dir = str(tmp_path / "workspace")
    Path(workspace_dir).mkdir()
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> object:
        calls.append(args)
        return object()

    monkeypatch.setattr("agent_harness_eval.executor.host.subprocess.run", fake_run)

    executor.restore_workspace(workspace_dir)

    assert calls == [["chmod", "-R", "u+w", workspace_dir]]


@pytest.mark.asyncio
async def test_execute_delegates_to_run_subprocess_with_wrapped_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = HostExecutor(RuntimeConfig(project_root=tmp_path))
    workspace_dir = str(tmp_path / "workspace")
    Path(workspace_dir).mkdir()
    policy = ExecutionPolicy(
        network=False,
        shell=True,
        file_write=True,
        workspace_dir=workspace_dir,
        timeout_sec=30,
    )
    captured: dict[str, object] = {}

    async def fake_run_subprocess(
        command: str,
        args: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_ms: int,
    ) -> SubprocessResult:
        captured["command"] = command
        captured["args"] = args
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout_ms"] = timeout_ms
        return SubprocessResult(stdout="ok", stderr="", exit_code=0, timed_out=False)

    monkeypatch.setattr("agent_harness_eval.executor.host.run_subprocess", fake_run_subprocess)

    result = await executor.execute(
        "codex",
        policy,
        "python",
        ["-c", "print('ok')"],
        {"HOME": "/tmp/home"},
        timeout_ms=1234,
    )

    assert result.stdout == "ok"
    assert captured["command"] == "python"
    assert captured["args"] == ["-c", "print('ok')"]
    assert captured["cwd"] == workspace_dir
    assert captured["timeout_ms"] == 1234
    assert captured["env"] == {
        "HOME": "/tmp/home",
        "EVAL_BOUNDARY_INTERNET": "disabled",
    }
