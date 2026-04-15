from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.adapters.interface import HarnessAdapter, PreparedRun
from agent_harness_eval.config.providers import ModelSpec
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.executor import ExecutionPolicy, Executor, WrappedCommand
from agent_harness_eval.graders.specs import GraderResult
from agent_harness_eval.runner import (
    RunPlanItem,
    RunRequest,
    _build_task_plan,
    _check_adapter_support,
    _execute_single_run,
    _finalize_prepared_run,
    _inject_cache_bust_nonce,
    _persist_run_result,
    _recover_existing_results,
)
from agent_harness_eval.task import NativeMemoryConfig, Task
from agent_harness_eval.types import CanonicalTraceEvent, EvalConfig, RunMetrics, RunResult
from agent_harness_eval.utils.subprocess import SubprocessResult
from agent_harness_eval.utils.workspace import RunLayout


@pytest.fixture
def isolated_run_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-runner-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class FakeExecutor(Executor):
    name = "fake"

    async def execute(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
        timeout_ms: int,
    ) -> SubprocessResult:
        return SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=False)

    def wrap_command(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
    ) -> WrappedCommand:
        return WrappedCommand(
            command=inner_command,
            args=list(inner_args),
            env=dict(inner_env),
            cwd=policy.cwd or policy.workspace_dir,
        )


class FakeAdapter(HarnessAdapter):
    name = "fake"

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        executor: Executor,
        *,
        prepared: PreparedRun | None = None,
        run_result: RunResult | None = None,
        prepare_error: Exception | None = None,
        run_error: Exception | None = None,
        cleanup_removes_workspace: bool = False,
    ) -> None:
        super().__init__(runtime_config, executor)
        self.prepared = prepared
        self.run_result = run_result
        self.prepare_error = prepare_error
        self.run_error = run_error
        self.cleanup_removes_workspace = cleanup_removes_workspace
        self.cleanup_calls = 0

    def prepare(self, task: Task, run_id: str) -> PreparedRun:
        if self.prepare_error is not None:
            raise self.prepare_error
        if self.prepared is None:
            raise AssertionError("prepared must be provided")
        return self.prepared

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        if self.run_error is not None:
            raise self.run_error
        if self.run_result is None:
            raise AssertionError("run_result must be provided")
        return self.run_result

    def cleanup(self, prepared: PreparedRun) -> None:
        self.cleanup_calls += 1
        if self.cleanup_removes_workspace:
            shutil.rmtree(prepared.layout.root_dir, ignore_errors=True)


def _make_layout(root: Path, name: str = "run") -> RunLayout:
    layout = RunLayout(
        root_dir=str(root / name),
        input_dir=str(root / name / "input"),
        workspace_seed_dir=str(root / name / "input" / "workspace"),
        workspace_dir=str(root / name / "runtime" / "workspace"),
        state_dir=str(root / name / "runtime" / "state"),
        output_dir=str(root / name / "output"),
    )
    for path in (
        layout.root_dir,
        layout.input_dir,
        layout.workspace_seed_dir,
        layout.workspace_dir,
        layout.state_dir,
        layout.output_dir,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
    return layout


def _make_prepared_run(root: Path, task: Task | None = None, name: str = "run") -> PreparedRun:
    layout = _make_layout(root, name)
    task = task or Task(
        id="coding.01",
        category="coding",
        description="runner test",
        user_query="Do the thing",
        timeout_sec=30,
    )
    return PreparedRun(
        task=task,
        layout=layout,
        env={"_TEST_RUNTIME_DIR": layout.state_dir},
        execution_policy=ExecutionPolicy(workspace_dir=layout.workspace_dir, timeout_sec=task.timeout_sec),
    )


def _make_result(task_id: str = "coding.01", *, status: str = "completed") -> RunResult:
    return RunResult(
        task_id=task_id,
        harness="fake",
        run_id="run-1",
        run_index=0,
        model="anthropic:claude-sonnet-4-6",
        status=status,  # type: ignore[arg-type]
        final_text="done" if status == "completed" else "",
        trace=[CanonicalTraceEvent(type="task_completed", ts="2026-01-01T00:00:00+00:00")],
        metrics=RunMetrics(latency_sec=1.2, total_tokens=10),
        grader_results=[GraderResult(grader_type="regex", name="regex:x", passed=True, score=1.0)],
    )


def test_build_task_plan_flattens_to_run_task_model_harness_tuples() -> None:
    """Plan is a flat list with run_index outermost and completed items filtered out.

    Ordering matters: run_index outermost guarantees all run=0 work is enqueued before
    any run=1 work, so two runs of the same (task, model, harness) are separated in
    time by an entire round of the other axes (defeats provider response caches).
    """
    task_one = Task(id="coding.01", category="coding", description="one", user_query="q1")
    task_two = Task(id="coding.02", category="coding", description="two", user_query="q2")
    model = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
    config = EvalConfig(
        model_spec=model,
        harnesses=["openclaw", "codex"],
        runs_per_task=2,
    )
    models = [model]
    all_results = [
        RunResult(
            task_id="coding.01",
            harness="openclaw",
            run_id="a",
            run_index=0,
            model="anthropic:claude-sonnet-4-6",
            status="completed",
            final_text="",
        ),
        RunResult(
            task_id="coding.01",
            harness="codex",
            run_id="b",
            run_index=0,
            model="anthropic:claude-sonnet-4-6",
            status="completed",
            final_text="",
        ),
        RunResult(
            task_id="coding.01",
            harness="openclaw",
            run_id="c",
            run_index=1,
            model="anthropic:claude-sonnet-4-6",
            status="completed",
            final_text="",
        ),
    ]

    plan = _build_task_plan(config, [task_one, task_two], models, all_results)

    assert plan == [
        # run=0 — task_one is fully recovered, only task_two remains
        RunPlanItem(task=task_two, model_spec=model, run_index=0, harness="openclaw"),
        RunPlanItem(task=task_two, model_spec=model, run_index=0, harness="codex"),
        # run=1 — task_one/openclaw already done; everything else missing
        RunPlanItem(task=task_one, model_spec=model, run_index=1, harness="codex"),
        RunPlanItem(task=task_two, model_spec=model, run_index=1, harness="openclaw"),
        RunPlanItem(task=task_two, model_spec=model, run_index=1, harness="codex"),
    ]


def test_build_task_plan_orders_all_run_zero_before_any_run_one() -> None:
    """Nothing recovered — entire plan should have run=0 grouped ahead of run=1."""
    task_one = Task(id="t1", category="coding", description="", user_query="q")
    task_two = Task(id="t2", category="coding", description="", user_query="q")
    model = ModelSpec(provider="anthropic", model="m")
    config = EvalConfig(
        model_spec=model,
        harnesses=["h1", "h2"],
        runs_per_task=2,
    )

    plan = _build_task_plan(config, [task_one, task_two], [model], all_results=[])

    run_indexes = [item.run_index for item in plan]
    assert run_indexes == [0, 0, 0, 0, 1, 1, 1, 1], run_indexes


def test_inject_cache_bust_nonce_appends_marker_preserving_other_fields() -> None:
    original = Task(
        id="x",
        category="coding",
        description="d",
        user_query="please do X",
        timeout_sec=42,
    )
    nonced = _inject_cache_bust_nonce(original, run_id="abcdef1234567890-uuid")

    assert nonced.id == "x"
    assert nonced.timeout_sec == 42
    assert nonced.user_query.startswith("please do X")
    assert "<!-- eval-nonce: abcdef123456 -->" in nonced.user_query
    # Original unchanged (we return a new Task, not a mutation)
    assert original.user_query == "please do X"


def test_check_adapter_support_flags_native_memory_and_conversation_history(
    isolated_run_dir: Path,
) -> None:
    runtime_config = RuntimeConfig(project_root=isolated_run_dir)
    adapter = FakeAdapter(
        runtime_config,
        FakeExecutor(runtime_config),
        prepared=_make_prepared_run(isolated_run_dir),
        run_result=_make_result(),
    )
    task = Task(
        id="memory.01",
        category="memory",
        description="memory",
        user_query="remember",
        native_memory=NativeMemoryConfig(files=[{"path": "MEMORY.md", "content": "deadline"}]),
        conversation_history=[{"role": "user", "content": "hello"}],
    )

    reasons = _check_adapter_support(adapter, task)

    assert reasons == [
        "native_memory required but not supported",
        "conversation_history required but not supported",
    ]


@pytest.mark.asyncio
async def test_persist_and_recover_run_result_round_trip(
    isolated_run_dir: Path,
) -> None:
    trace_dir = isolated_run_dir / "traces" / "run-1"
    trace_dir.mkdir(parents=True, exist_ok=True)
    results_file = isolated_run_dir / "data" / "runs.jsonl"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    trace_index_file = isolated_run_dir / "traces" / "index.jsonl"
    all_results: list[RunResult] = []
    result = RunResult(
        task_id="coding.01",
        harness="fake",
        run_id="run-1",
        run_index=0,
        model="anthropic:claude-sonnet-4-6",
        status="completed",
        final_text="done",
        trace=[
            CanonicalTraceEvent(
                type="tool_call_completed",
                ts="2026-01-01T00:00:00+00:00",
                tool_name="read_file",
                success=True,
                output="contents",
            )
        ],
        metrics=RunMetrics(latency_sec=2.5, input_tokens=10, output_tokens=5, total_tokens=15),
        grader_results=[GraderResult(grader_type="regex", name="regex:x", passed=True, score=1.0)],
    )
    task = Task(
        id="coding.01",
        category="coding",
        description="runner test",
        user_query="Do the thing",
        timeout_sec=30,
    )
    runtime_config = RuntimeConfig(project_root=isolated_run_dir)
    run_request = RunRequest(
        task=task,
        harness="fake",
        adapter=FakeAdapter(
            runtime_config,
            FakeExecutor(runtime_config),
            prepared=_make_prepared_run(isolated_run_dir),
            run_result=result,
        ),
        model_spec=ModelSpec(provider="anthropic", model="claude-sonnet-4-6"),
        model_label="anthropic:claude-sonnet-4-6",
        run_index=0,
        run_id="run-1",
        trace_dir=str(trace_dir),
    )

    await _persist_run_result(
        request=run_request,
        result=result,
        trace_dir=str(trace_dir),
        results_file=str(results_file),
        trace_index_file=str(trace_index_file),
        all_results=all_results,
        results_lock=asyncio.Lock(),
    )

    recovered = _recover_existing_results(str(results_file))

    assert len(all_results) == 1
    assert len(recovered) == 1
    assert recovered[0].task_id == "coding.01"
    assert recovered[0].metrics.total_tokens == 15
    assert recovered[0].grader_results[0].passed is True
    assert json.loads((trace_dir / "request.json").read_text(encoding="utf-8"))["task_id"] == "coding.01"
    assert json.loads((trace_dir / "trace.json").read_text(encoding="utf-8"))[0]["tool_name"] == "read_file"
    assert json.loads(trace_index_file.read_text(encoding="utf-8").strip())["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_execute_single_run_restores_workspace_before_grading(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(project_root=isolated_run_dir)
    task = Task(
        id="coding.restore.01",
        category="coding",
        description="restore before grading",
        user_query="fix it",
        timeout_sec=30,
    )
    prepared = _make_prepared_run(isolated_run_dir, task=task, name="restore")
    adapter = FakeAdapter(
        runtime_config,
        FakeExecutor(runtime_config),
        prepared=prepared,
        run_result=_make_result(task.id),
    )
    restore_calls: list[str] = []

    def fake_restore(workspace_dir: str) -> None:
        restore_calls.append(workspace_dir)

    adapter.executor.restore_workspace = fake_restore  # type: ignore[method-assign]

    async def fake_run_graders(*args, **kwargs):
        assert restore_calls == [prepared.workspace_dir]
        return [GraderResult(grader_type="regex", name="regex:x", passed=True, score=1.0)]

    monkeypatch.setattr("agent_harness_eval.runner.run_graders", fake_run_graders)

    result = await _execute_single_run(
        request=RunRequest(
            task=task,
            harness="fake",
            adapter=adapter,
            model_spec=ModelSpec(provider="anthropic", model="claude-sonnet-4-6"),
            model_label="anthropic:claude-sonnet-4-6",
            run_index=0,
            run_id="run-restore",
            trace_dir=str(isolated_run_dir / "traces" / "run-restore"),
        ),
        judge_llm=None,
        runtime_config=runtime_config,
        pricing_override=None,
    )

    assert result.status == "completed"
    assert restore_calls == [prepared.workspace_dir]


@pytest.mark.asyncio
async def test_finalize_prepared_run_keeps_workspace_on_failure(
    isolated_run_dir: Path,
) -> None:
    task = Task(id="coding.01", category="coding", description="keep", user_query="q")
    prepared = _make_prepared_run(isolated_run_dir, task, name="keep-run")
    runtime_config = RuntimeConfig(project_root=isolated_run_dir)
    adapter = FakeAdapter(
        runtime_config,
        FakeExecutor(runtime_config),
        prepared=prepared,
        run_result=_make_result(status="failed"),
    )
    trace_dir = isolated_run_dir / "trace-keep"
    trace_dir.mkdir()

    await _finalize_prepared_run(
        adapter=adapter,
        prepared=prepared,
        result=_make_result(status="failed"),
        trace_dir=str(trace_dir),
        keep_workspace="on_failure",
    )

    assert adapter.cleanup_calls == 0
    assert (trace_dir / "kept_workspace.txt").read_text(encoding="utf-8").strip() == prepared.workspace_dir
    assert Path(prepared.layout.root_dir).exists()


@pytest.mark.asyncio
async def test_finalize_prepared_run_cleans_workspace_on_success(
    isolated_run_dir: Path,
) -> None:
    task = Task(id="coding.01", category="coding", description="cleanup", user_query="q")
    prepared = _make_prepared_run(isolated_run_dir, task, name="cleanup-run")
    runtime_config = RuntimeConfig(project_root=isolated_run_dir)
    adapter = FakeAdapter(
        runtime_config,
        FakeExecutor(runtime_config),
        prepared=prepared,
        run_result=_make_result(),
        cleanup_removes_workspace=True,
    )
    trace_dir = isolated_run_dir / "trace-clean"
    trace_dir.mkdir()

    await _finalize_prepared_run(
        adapter=adapter,
        prepared=prepared,
        result=_make_result(),
        trace_dir=str(trace_dir),
        keep_workspace="on_failure",
    )

    assert adapter.cleanup_calls == 1
    assert not Path(prepared.layout.root_dir).exists()
    assert not (trace_dir / "kept_workspace.txt").exists()


@pytest.mark.asyncio
async def test_execute_single_run_wraps_prepare_errors_as_failed_results(
    isolated_run_dir: Path,
) -> None:
    runtime_config = RuntimeConfig(project_root=isolated_run_dir)
    adapter = FakeAdapter(
        runtime_config,
        FakeExecutor(runtime_config),
        prepare_error=RuntimeError("prepare exploded"),
    )
    task = Task(id="coding.01", category="coding", description="prepare error", user_query="q")
    trace_dir = isolated_run_dir / "trace-exec"
    trace_dir.mkdir()

    result = await _execute_single_run(
        request=__import__("agent_harness_eval.runner", fromlist=["RunRequest"]).RunRequest(
            task=task,
            harness="fake",
            adapter=adapter,
            model_spec=ModelSpec(provider="anthropic", model="claude-sonnet-4-6"),
            model_label="anthropic:claude-sonnet-4-6",
            run_index=0,
            run_id="run-prepare-error",
            trace_dir=str(trace_dir),
        ),
        judge_llm=None,
        runtime_config=runtime_config,
        pricing_override=None,
    )

    assert result.status == "failed"
    assert result.task_id == "coding.01"
    assert result.infra_error_details == "prepare exploded"
    assert result.trace[0].type == "task_failed"
