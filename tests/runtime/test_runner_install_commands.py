from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.adapters.interface import PreparedRun
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.executor import ExecutionPolicy, Executor, VolumeMount
from agent_harness_eval.runner import _run_prepare_commands
from agent_harness_eval.task import Task
from agent_harness_eval.utils.subprocess import SubprocessResult
from agent_harness_eval.utils.workspace import RunLayout


@pytest.fixture
def isolated_run_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _make_prepared_run(root: Path) -> PreparedRun:
    layout = RunLayout(
        root_dir=str(root / "run"),
        input_dir=str(root / "run" / "input"),
        workspace_seed_dir=str(root / "run" / "input" / "workspace"),
        workspace_dir=str(root / "run" / "runtime" / "workspace"),
        state_dir=str(root / "run" / "runtime" / "state"),
        output_dir=str(root / "run" / "output"),
    )
    for path in (
        layout.input_dir,
        layout.workspace_seed_dir,
        layout.workspace_dir,
        layout.state_dir,
        layout.output_dir,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)

    return PreparedRun(
        task=Task(
            id="runner.install.01",
            category="coding",
            description="install commands",
            user_query="Run setup.",
            timeout_sec=30,
        ),
        layout=layout,
        env={"CUSTOM_FLAG": "1"},
        execution_policy=ExecutionPolicy(
            network=False,
            file_write=False,
            shell=False,
            workspace_dir=layout.workspace_dir,
            extra_mounts=[
                VolumeMount(
                    source=layout.state_dir,
                    target=layout.state_dir,
                    mode="rw",
                )
            ],
        ),
    )


class FakeExecutor(Executor):
    """Test executor that records calls instead of running them."""

    def __init__(self, runtime_config: RuntimeConfig):
        super().__init__(runtime_config)
        self.calls: list[dict] = []

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
                "inner_args": inner_args,
                "inner_env": dict(inner_env),
                "timeout_ms": timeout_ms,
            }
        )
        return SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False)


@pytest.mark.asyncio
async def test_prepare_commands_use_run_private_state_env(
    isolated_run_dir: Path,
) -> None:
    prepared = _make_prepared_run(isolated_run_dir)
    runtime_config = RuntimeConfig(project_root=isolated_run_dir, executor_backend="host")
    executor = FakeExecutor(runtime_config)

    await _run_prepare_commands(["npm install"], prepared, "nanobot", executor)

    assert len(executor.calls) == 1
    call = executor.calls[0]

    # Policy: prepare commands get full access regardless of task boundary
    policy = call["policy"]
    assert isinstance(policy, ExecutionPolicy)
    assert policy.network is True
    assert policy.file_write is True
    assert policy.shell is True
    assert policy.extra_mounts == prepared.execution_policy.extra_mounts

    # Env: run-private state dirs are injected
    env = call["inner_env"]
    assert env["HOME"] == prepared.layout.state_dir
    assert env["USERPROFILE"] == prepared.layout.state_dir
    assert env["XDG_CONFIG_HOME"] == str(Path(prepared.layout.state_dir) / ".config")
    assert env["XDG_DATA_HOME"] == str(Path(prepared.layout.state_dir) / ".local" / "share")
    assert env["XDG_CACHE_HOME"] == str(Path(prepared.layout.state_dir) / ".cache")
    assert env["EVAL_RUN_ROOT"] == prepared.layout.root_dir
    assert env["EVAL_INPUT_DIR"] == prepared.layout.input_dir
    assert env["EVAL_WORKSPACE_DIR"] == prepared.layout.workspace_dir
    assert env["EVAL_STATE_DIR"] == prepared.layout.state_dir
    assert env["EVAL_OUTPUT_DIR"] == prepared.layout.output_dir
    assert env["CUSTOM_FLAG"] == "1"

    # Command
    assert call["inner_command"] == "sh"
    assert call["inner_args"] == ["-lc", "npm install"]


@pytest.mark.asyncio
async def test_prepare_commands_follow_detected_executor_backend(
    isolated_run_dir: Path,
) -> None:
    prepared = _make_prepared_run(isolated_run_dir)
    runtime_config = RuntimeConfig(project_root=isolated_run_dir, executor_backend="docker")
    executor = FakeExecutor(runtime_config)

    await _run_prepare_commands(["python -V"], prepared, "openclaw", executor)

    assert len(executor.calls) == 1
    assert executor.calls[0]["harness"] == "openclaw"
