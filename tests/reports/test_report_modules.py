from __future__ import annotations

from agent_harness_eval.graders.specs import GraderResult, RubricDimensionResult
from agent_harness_eval.reports.case_review import generate_case_review_report
from agent_harness_eval.reports.failure_taxonomy import generate_failure_report
from agent_harness_eval.reports.judge_analysis import generate_judge_analysis_report
from agent_harness_eval.task import Task
from agent_harness_eval.types import RunMetrics, RunResult


def _make_result(
    *,
    task_id: str,
    harness: str,
    run_index: int = 1,
    status: str = "completed",
    grader_results: list[GraderResult] | None = None,
    infra_error_code: str | None = None,
    infra_error_details: str | None = None,
    final_text: str = "final answer",
) -> RunResult:
    return RunResult(
        task_id=task_id,
        harness=harness,
        run_id=f"{task_id}-{harness}-{run_index}",
        run_index=run_index,
        model="openai:gpt-5.4",
        status=status,
        final_text=final_text,
        metrics=RunMetrics(latency_sec=2.5, total_tokens=1530, cost_usd=0.1234, tool_calls=4),
        grader_results=grader_results or [],
        infra_error_code=infra_error_code,
        infra_error_details=infra_error_details,
    )


def test_generate_failure_report_groups_failures_and_skips_not_applicable() -> None:
    provider_failure = _make_result(
        task_id="task.provider",
        harness="codex",
        status="failed",
        infra_error_code="provider_api_error",
        infra_error_details="upstream provider api error",
    )
    grader_failure = _make_result(
        task_id="task.grader",
        harness="zeroclaw",
        grader_results=[
            GraderResult(grader_type="file_exists", name="file_exists:answer.txt", passed=False, score=0.0)
        ],
    )
    not_applicable = _make_result(
        task_id="task.na",
        harness="codex",
        status="not_applicable",
        infra_error_code="capability_unsupported",
        infra_error_details="shell disabled",
    )

    report = generate_failure_report([provider_failure, grader_failure, not_applicable])

    assert "# Failure Taxonomy Report" in report
    assert "**Total failures:** 2 / 2 runs" in report
    assert "### Grader Failures (1)" in report
    assert "### Provider Failures (1)" in report
    assert "**Zeroclaw** (1 failures):" in report
    assert "Failed graders: file_exists:answer.txt" in report
    assert "task.grader" in report
    assert "task.provider" in report
    assert "task.na" not in report


def test_generate_case_review_report_includes_dimensions_failure_details_and_snippet() -> None:
    result = _make_result(
        task_id="task.review",
        harness="codex",
        grader_results=[
            GraderResult(
                grader_type="rubric_judge",
                name="rubric_judge",
                passed=False,
                score=0.4,
                details="overall rubric failure",
                dimensions=[
                    RubricDimensionResult(
                        name="accuracy",
                        passed=False,
                        score=0.2,
                        reason="Missed the required file update",
                    )
                ],
            ),
            GraderResult(grader_type="file_exists", name="file_exists:answer.txt", passed=False, score=0.0),
        ],
        final_text="Line one\nLine two",
    )

    report = generate_case_review_report([result])

    assert "# Case Review Report" in report
    assert "## task.review" in report
    assert "**Run 1 - FAIL** (status: completed)" in report
    assert "- **Failure origin:** Agent" in report
    assert "- **Grader results:**" in report
    assert "    - [FAIL] accuracy (0.20)" in report
    assert "      > Missed the required file update" in report
    assert "  > overall rubric failure" in report
    assert "- **Response snippet:** Line one Line two" in report


def test_generate_judge_analysis_report_covers_distribution_and_secondary_agreement() -> None:
    tasks = [
        Task(id="task.1", category="coding", description="desc", user_query="query", timeout_sec=30),
        Task(id="task.2", category="coding", description="desc", user_query="query", timeout_sec=30),
    ]
    results = [
        _make_result(
            task_id="task.1",
            harness="codex",
            grader_results=[
                GraderResult(grader_type="file_exists", name="file_exists:a", passed=True, score=1.0),
                GraderResult(grader_type="rubric_judge", name="primary_judge", passed=True, score=0.9),
                GraderResult(grader_type="rubric_judge", name="secondary_judge", passed=True, score=0.8),
            ],
        ),
        _make_result(
            task_id="task.1",
            harness="zeroclaw",
            grader_results=[
                GraderResult(grader_type="file_exists", name="file_exists:a", passed=False, score=0.0),
                GraderResult(grader_type="rubric_judge", name="primary_judge", passed=False, score=0.3),
                GraderResult(grader_type="rubric_judge", name="secondary_judge", passed=False, score=0.1),
            ],
        ),
        _make_result(
            task_id="task.2",
            harness="codex",
            grader_results=[
                GraderResult(grader_type="file_exists", name="file_exists:b", passed=True, score=1.0),
                GraderResult(grader_type="rubric_judge", name="primary_judge", passed=True, score=0.7),
                GraderResult(grader_type="rubric_judge", name="secondary_judge", passed=True, score=0.75),
            ],
        ),
    ]

    report = generate_judge_analysis_report(
        results,
        tasks,
        judge_llm="openai:gpt-5.4",
        secondary_judge_llm="anthropic:claude-sonnet-4-6",
    )

    assert "# Judge Analysis Report" in report
    assert "**Primary judge:** openai:gpt-5.4" in report
    assert "**Secondary judge:** anthropic:claude-sonnet-4-6" in report
    assert "## Score Distribution" in report
    assert "## Per-Harness Judge Scores" in report
    assert "## Judge-Grader Agreement" in report
    assert "- **Passing runs:** mean judge score =" in report
    assert "- **Failing runs:** mean judge score =" in report
    assert "- **Score gap:**" in report
    assert "## Primary vs Secondary Judge" in report
    assert "- **Paired evaluations:** 3" in report
    assert "- **Agreement rate (within 0.2):** 100.0%" in report
