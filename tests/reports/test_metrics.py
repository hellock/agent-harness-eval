from __future__ import annotations

from agent_harness_eval.graders.specs import GraderResult
from agent_harness_eval.metrics import compute_category_metrics, compute_harness_metrics
from agent_harness_eval.types import RunMetrics, RunResult


def _make_run(
    *,
    task_id: str,
    harness: str = "codex",
    model: str = "openai:gpt-5.4",
    run_index: int,
    status: str = "completed",
    passed: bool = True,
    failure_origin: str | None = None,
    infra_error_code: str | None = None,
    latency_sec: float = 1.0,
    total_tokens: int = 10,
    usage_available: bool = True,
    cost_usd: float = 0.1,
    tool_calls: int = 1,
    quality_score: float | None = None,
) -> RunResult:
    grader_results: list[GraderResult] = []
    if status == "completed":
        grader_results.append(
            GraderResult(
                grader_type="test_pass",
                name="test_pass",
                passed=passed,
                score=1.0 if passed else 0.0,
            )
        )
    if quality_score is not None:
        grader_results.append(
            GraderResult(
                grader_type="rubric_judge",
                name="rubric_judge",
                passed=quality_score > 0,
                score=quality_score,
            )
        )

    return RunResult(
        task_id=task_id,
        harness=harness,
        run_id=f"{task_id}-{run_index}",
        run_index=run_index,
        model=model,
        status=status,
        final_text="done" if status == "completed" else "",
        metrics=RunMetrics(
            latency_sec=latency_sec,
            total_tokens=total_tokens,
            usage_available=usage_available,
            cost_usd=cost_usd,
            tool_calls=tool_calls,
        ),
        grader_results=grader_results,
        failure_origin=failure_origin,
        infra_error_code=infra_error_code,
    )


def test_compute_harness_metrics_limits_pass_at_3_to_first_three_runs() -> None:
    results = [
        _make_run(task_id="task.1", run_index=1, passed=False),
        _make_run(task_id="task.1", run_index=2, passed=False),
        _make_run(task_id="task.1", run_index=3, passed=False),
        _make_run(task_id="task.1", run_index=4, passed=True),
    ]

    metrics = compute_harness_metrics(results, ["codex"])[0]

    assert metrics.pass_at_1 == 0.0
    assert metrics.pass_at_3 == 0.0


def test_compute_harness_metrics_excludes_capability_unsupported_and_provider_failures() -> None:
    results = [
        _make_run(task_id="task.pass", run_index=1, passed=True),
        _make_run(
            task_id="task.provider",
            run_index=1,
            status="failed",
            passed=False,
            failure_origin="provider",
        ),
        _make_run(
            task_id="task.unsupported",
            run_index=1,
            status="not_applicable",
            passed=False,
            infra_error_code="capability_unsupported",
        ),
    ]

    metrics = compute_harness_metrics(results, ["codex"])[0]

    assert metrics.pass_at_1 == 1.0
    assert metrics.pass_at_3 == 1.0
    assert metrics.infra_failure_count == 1


def test_compute_category_metrics_keeps_harnesses_and_categories_separate() -> None:
    results = [
        _make_run(task_id="task.a", harness="codex", run_index=1, passed=True, latency_sec=3.0, total_tokens=30),
        _make_run(task_id="task.b", harness="codex", run_index=1, passed=False, latency_sec=9.0, total_tokens=90),
        _make_run(task_id="task.a", harness="zeroclaw", run_index=1, passed=True, latency_sec=5.0, total_tokens=50),
    ]
    tasks = [
        {"id": "task.a", "category": "coding"},
        {"id": "task.b", "category": "security"},
    ]

    metrics = compute_category_metrics(results, tasks, ["codex", "zeroclaw"])

    by_key = {(m.harness, m.category): m for m in metrics}

    assert by_key[("codex", "coding")].pass_rate == 1.0
    assert by_key[("codex", "coding")].median_latency_sec == 3.0
    assert by_key[("codex", "security")].pass_rate == 0.0
    assert by_key[("zeroclaw", "coding")].median_total_tokens == 50.0


def test_compute_metrics_marks_unavailable_token_and_cost_series() -> None:
    results = [
        _make_run(
            task_id="task.a",
            harness="zeroclaw",
            run_index=1,
            passed=True,
            total_tokens=50,
            usage_available=False,
            cost_usd=0.2,
        ),
    ]

    harness_metrics = compute_harness_metrics(results, ["zeroclaw"])[0]
    category_metrics = compute_category_metrics(results, [{"id": "task.a", "category": "coding"}], ["zeroclaw"])[0]

    assert harness_metrics.usage_metrics_available is False
    assert category_metrics.usage_metrics_available is False
