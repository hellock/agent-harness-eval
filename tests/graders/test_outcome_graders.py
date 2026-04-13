from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.executor import ExecutionPolicy, Executor
from agent_harness_eval.graders import specs as grader_specs
from agent_harness_eval.graders.outcome import (
    run_file_exists_grader,
    run_json_schema_grader,
    run_regex_grader,
    run_test_pass_grader,
)
from agent_harness_eval.types import RunResult
from agent_harness_eval.utils.subprocess import SubprocessResult


@pytest.fixture
def isolated_temp_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _make_result(final_text: str = "") -> RunResult:
    return RunResult(
        task_id="task.outcome",
        harness="codex",
        run_id="run-1",
        run_index=1,
        model="openai:gpt-5.4",
        status="completed",
        final_text=final_text,
    )


class _RecordingExecutor(Executor):
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
        timeout_ms: int,
    ) -> SubprocessResult:
        self.calls.append(
            {
                "harness": harness,
                "policy": policy,
                "inner_command": inner_command,
                "inner_args": list(inner_args),
                "inner_env": dict(inner_env),
                "timeout_ms": timeout_ms,
            }
        )
        return SubprocessResult(stdout="ok\n", stderr="", exit_code=0, timed_out=False)


@pytest.mark.asyncio
async def test_run_test_pass_grader_executes_in_workspace_dir(
    monkeypatch: pytest.MonkeyPatch,
    isolated_temp_dir: Path,
) -> None:
    observed: dict[str, str | int | None] = {}

    async def fake_run_subprocess(
        command: str,
        args: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_ms: int,
        stdin: str | None = None,
        filtered_env: bool = True,
        inherit_env: bool = True,
    ) -> SubprocessResult:
        observed["command"] = command
        observed["cwd"] = cwd
        observed["timeout_ms"] = timeout_ms
        return SubprocessResult(stdout="ok\n", stderr="", exit_code=0, timed_out=False)

    monkeypatch.setattr("agent_harness_eval.graders.outcome.run_subprocess", fake_run_subprocess)

    grader = grader_specs.TestPassGrader(command="pwd")
    result = await run_test_pass_grader(grader, _make_result(), str(isolated_temp_dir))

    assert result.passed is True
    assert observed["command"] == "bash"
    assert observed["cwd"] == str(isolated_temp_dir)
    assert "stdout: ok" in (result.details or "")


@pytest.mark.asyncio
async def test_run_test_pass_grader_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
    isolated_temp_dir: Path,
) -> None:
    async def fake_run_subprocess(
        command: str,
        args: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_ms: int,
        stdin: str | None = None,
        filtered_env: bool = True,
        inherit_env: bool = True,
    ) -> SubprocessResult:
        return SubprocessResult(stdout="", stderr="", exit_code=None, timed_out=True)

    monkeypatch.setattr("agent_harness_eval.graders.outcome.run_subprocess", fake_run_subprocess)

    result = await run_test_pass_grader(
        grader_specs.TestPassGrader(command="sleep 10"),
        _make_result(),
        str(isolated_temp_dir),
    )

    assert result.passed is False
    assert "Command timed out" in (result.details or "")


@pytest.mark.asyncio
async def test_run_test_pass_grader_uses_verifier_policy_with_executor(
    isolated_temp_dir: Path,
) -> None:
    executor = _RecordingExecutor()
    workspace_dir = str(isolated_temp_dir)
    repo_dir = str(isolated_temp_dir / "repo")
    Path(repo_dir).mkdir()

    result = await run_test_pass_grader(
        grader_specs.TestPassGrader(command="pwd", cwd=repo_dir),
        _make_result(),
        workspace_dir,
        executor=executor,
        execution_policy=ExecutionPolicy(
            network=True,
            shell=False,
            file_write=False,
            workspace_dir=workspace_dir,
            cwd=workspace_dir,
            timeout_sec=30,
        ),
        harness_name="codex",
        grader_env={"HOME": "/tmp/home"},
    )

    assert result.passed is True
    assert len(executor.calls) == 1
    call = executor.calls[0]
    assert call["inner_command"] == "bash"
    assert call["inner_env"] == {"HOME": "/tmp/home"}
    policy = call["policy"]
    assert isinstance(policy, ExecutionPolicy)
    assert policy.workspace_dir == workspace_dir
    assert policy.cwd == repo_dir
    assert policy.network is True
    assert policy.strict_network is False
    assert policy.shell is True
    assert policy.file_write is True


def test_run_file_exists_and_regex_graders_handle_nested_artifacts(
    isolated_temp_dir: Path,
) -> None:
    workspace = isolated_temp_dir / "workspace-eval"
    workspace.mkdir()
    artifact = workspace / "report.txt"
    artifact.write_text("Release READY\n")

    file_result = run_file_exists_grader(
        grader_specs.FileExistsGrader(paths=["report.txt"]),
        _make_result(),
        str(isolated_temp_dir),
    )
    regex_result = run_regex_grader(
        grader_specs.RegexGrader(target="artifact", artifact_path="report.txt", pattern="READY"),
        _make_result(),
        str(isolated_temp_dir),
    )

    assert file_result.passed is True
    assert regex_result.passed is True
    assert "found=True" in (regex_result.details or "")


def test_run_regex_grader_reports_invalid_pattern() -> None:
    result = run_regex_grader(
        grader_specs.RegexGrader(target="final_text", pattern="["),
        _make_result("hello"),
        None,
    )

    assert result.passed is False
    assert "Invalid regex" in (result.details or "")


def test_run_json_schema_grader_validates_artifact_payload(
    isolated_temp_dir: Path,
) -> None:
    artifact = isolated_temp_dir / "payload.json"
    artifact.write_text('{"name":"demo","count":2}')

    result = run_json_schema_grader(
        grader_specs.JsonSchemaGrader(
            target="artifact",
            artifact_path=str(artifact),
            schema={
                "type": "object",
                "required": ["name", "count"],
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "number"},
                },
            },
        ),
        _make_result(),
        str(isolated_temp_dir),
    )

    assert result.passed is True
    assert result.details == "Schema validated"
