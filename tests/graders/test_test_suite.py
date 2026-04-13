from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.executor import ExecutionPolicy, Executor
from agent_harness_eval.graders.specs import TestSuiteCase, TestSuiteGrader, parse_grader_spec
from agent_harness_eval.graders.test_suite import run_test_suite_grader
from agent_harness_eval.utils.subprocess import SubprocessResult


@pytest.fixture
def isolated_temp_dir():
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _make_fake_run_subprocess(results: dict[str, SubprocessResult]):
    """Return a fake run_subprocess that maps command strings to results."""

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
        # args[1] is the command string passed to bash -c
        cmd_str = args[1] if len(args) > 1 else ""
        if cmd_str in results:
            return results[cmd_str]
        return SubprocessResult(stdout="", stderr="not found", exit_code=127, timed_out=False)

    return fake_run_subprocess


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
        script = inner_args[1]
        match = re.search(r"RESULTS_DIR='([^']+)'", script)
        assert match is not None
        results_dir = Path(match.group(1))
        results_dir.mkdir(parents=True, exist_ok=True)

        (results_dir / "setup-0.label").write_text("prep", encoding="utf-8")
        (results_dir / "setup-0.exit_code").write_text("0", encoding="utf-8")
        (results_dir / "setup-0.timed_out").write_text("0", encoding="utf-8")
        (results_dir / "setup-0.stdout").write_text("", encoding="utf-8")
        (results_dir / "setup-0.stderr").write_text("", encoding="utf-8")

        (results_dir / "case-0.label").write_text("case_a", encoding="utf-8")
        (results_dir / "case-0.exit_code").write_text("0", encoding="utf-8")
        (results_dir / "case-0.timed_out").write_text("0", encoding="utf-8")
        (results_dir / "case-0.stdout").write_text("", encoding="utf-8")
        (results_dir / "case-0.stderr").write_text("", encoding="utf-8")

        (results_dir / "case-1.label").write_text("case_b", encoding="utf-8")
        (results_dir / "case-1.exit_code").write_text("1", encoding="utf-8")
        (results_dir / "case-1.timed_out").write_text("0", encoding="utf-8")
        (results_dir / "case-1.stdout").write_text("", encoding="utf-8")
        (results_dir / "case-1.stderr").write_text("boom", encoding="utf-8")

        return SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False)


@pytest.mark.asyncio
async def test_all_passing(monkeypatch: pytest.MonkeyPatch, isolated_temp_dir: Path) -> None:
    fake = _make_fake_run_subprocess(
        {
            "echo a": SubprocessResult(stdout="a\n", stderr="", exit_code=0, timed_out=False),
            "echo b": SubprocessResult(stdout="b\n", stderr="", exit_code=0, timed_out=False),
        }
    )
    monkeypatch.setattr("agent_harness_eval.graders.test_suite.run_subprocess", fake)

    spec = TestSuiteGrader(
        cases=[
            TestSuiteCase(name="case_a", command="echo a"),
            TestSuiteCase(name="case_b", command="echo b"),
        ],
    )
    result = await run_test_suite_grader(spec, str(isolated_temp_dir))

    assert result.passed is True
    assert result.score == 1.0
    assert "2/2 cases passed" in (result.details or "")
    assert "FAILED" not in (result.details or "")


@pytest.mark.asyncio
async def test_one_failing(monkeypatch: pytest.MonkeyPatch, isolated_temp_dir: Path) -> None:
    fake = _make_fake_run_subprocess(
        {
            "echo a": SubprocessResult(stdout="a\n", stderr="", exit_code=0, timed_out=False),
            "false": SubprocessResult(stdout="", stderr="err", exit_code=1, timed_out=False),
            "echo c": SubprocessResult(stdout="c\n", stderr="", exit_code=0, timed_out=False),
        }
    )
    monkeypatch.setattr("agent_harness_eval.graders.test_suite.run_subprocess", fake)

    spec = TestSuiteGrader(
        cases=[
            TestSuiteCase(name="case_a", command="echo a"),
            TestSuiteCase(name="case_fail", command="false"),
            TestSuiteCase(name="case_c", command="echo c"),
        ],
    )
    result = await run_test_suite_grader(spec, str(isolated_temp_dir))

    assert result.passed is False
    assert result.score == pytest.approx(2 / 3)
    assert "2/3 cases passed" in (result.details or "")
    assert "case_fail" in (result.details or "")


@pytest.mark.asyncio
async def test_pass_threshold_allows_partial_failure(monkeypatch: pytest.MonkeyPatch, isolated_temp_dir: Path) -> None:
    fake = _make_fake_run_subprocess(
        {
            "echo a": SubprocessResult(stdout="a\n", stderr="", exit_code=0, timed_out=False),
            "false": SubprocessResult(stdout="", stderr="", exit_code=1, timed_out=False),
        }
    )
    monkeypatch.setattr("agent_harness_eval.graders.test_suite.run_subprocess", fake)

    spec = TestSuiteGrader(
        cases=[
            TestSuiteCase(name="case_a", command="echo a"),
            TestSuiteCase(name="case_fail", command="false"),
        ],
        pass_threshold=0.5,
    )
    result = await run_test_suite_grader(spec, str(isolated_temp_dir))

    assert result.passed is True
    assert result.score == 0.5


@pytest.mark.asyncio
async def test_pytest_runner_shorthand() -> None:
    data = {
        "type": "test_suite",
        "runner": "pytest",
        "working_dir": "repo",
        "cases": [
            "tests/test_foo.py::test_fix",
            "tests/test_foo.py::test_existing",
        ],
        "pass_threshold": 1.0,
    }
    spec = parse_grader_spec(data)

    assert isinstance(spec, TestSuiteGrader)
    assert spec.runner == "pytest"
    assert spec.working_dir == "repo"
    assert len(spec.cases) == 2
    assert spec.cases[0].name == "tests/test_foo.py::test_fix"
    assert spec.cases[0].command == "uv run python -m pytest tests/test_foo.py::test_fix -x --tb=short"
    assert spec.cases[1].name == "tests/test_foo.py::test_existing"
    assert spec.cases[1].command == "uv run python -m pytest tests/test_foo.py::test_existing -x --tb=short"


@pytest.mark.asyncio
async def test_no_cases() -> None:
    spec = TestSuiteGrader(cases=[])
    result = await run_test_suite_grader(spec, None)
    assert result.passed is False
    assert result.score == 0.0
    assert "No cases specified" in (result.details or "")


@pytest.mark.asyncio
async def test_executor_path_batches_setup_and_cases_with_cwd_override(isolated_temp_dir: Path) -> None:
    executor = _RecordingExecutor()
    policy = ExecutionPolicy(
        workspace_dir=str(isolated_temp_dir),
        cwd=str(isolated_temp_dir),
        timeout_sec=60,
    )
    spec = TestSuiteGrader(
        setup_commands=["prep"],
        cases=[
            TestSuiteCase(name="case_a", command="echo a"),
            TestSuiteCase(name="case_b", command="echo b"),
        ],
        working_dir="repo",
    )
    (isolated_temp_dir / "repo").mkdir()

    result = await run_test_suite_grader(
        spec,
        str(isolated_temp_dir),
        executor=executor,
        execution_policy=policy,
        harness_name="codex",
        grader_env={"HOME": "/tmp/home"},
    )

    assert result.passed is False
    assert result.score == 0.5
    assert "1/2 cases passed" in (result.details or "")
    assert "case_b (exit 1: boom)" in (result.details or "")
    assert len(executor.calls) == 1
    call = executor.calls[0]
    assert call["inner_command"] == "bash"
    assert call["timeout_ms"] == 185000
    assert call["inner_env"] == {"HOME": "/tmp/home"}
    assert "python - <<'PY'" not in call["inner_args"][1]
    call_policy = call["policy"]
    assert isinstance(call_policy, ExecutionPolicy)
    assert call_policy.workspace_dir == str(isolated_temp_dir)
    assert call_policy.cwd == str(isolated_temp_dir / "repo")
    assert call_policy.network is True
    assert call_policy.strict_network is False
    assert call_policy.shell is True
    assert call_policy.file_write is True
