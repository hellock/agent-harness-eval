from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.config.providers import ModelSpec
from agent_harness_eval.graders.judge_json import extract_json
from agent_harness_eval.graders.specs import GraderResult
from agent_harness_eval.metrics import HarnessMetrics
from agent_harness_eval.reports.category_breakdown import generate_category_report
from agent_harness_eval.reports.generate import _generate_multi_model_summary, generate_reports
from agent_harness_eval.reports.summary import generate_summary_report
from agent_harness_eval.task import Task
from agent_harness_eval.types import EvalConfig, RunMetrics, RunResult
from agent_harness_eval.utils.failure_origin import (
    detect_failure_origin_from_error,
    format_failure_origin,
    infer_failure_origin,
)


@pytest.fixture
def isolated_temp_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _make_result(
    *,
    task_id: str = "task.1",
    harness: str = "codex",
    model: str = "openai:gpt-5.4",
    status: str = "completed",
    run_index: int = 1,
    passed: bool = True,
    failure_origin: str | None = None,
    infra_error_details: str | None = None,
    infra_error_code: str | None = None,
) -> RunResult:
    grader_results = []
    if status == "completed":
        grader_results.append(
            GraderResult(
                grader_type="test_pass",
                name="test_pass",
                passed=passed,
                score=1.0 if passed else 0.0,
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
        metrics=RunMetrics(latency_sec=2.0, total_tokens=1200, cost_usd=0.12, tool_calls=3),
        grader_results=grader_results,
        failure_origin=failure_origin,
        infra_error_code=infra_error_code,
        infra_error_details=infra_error_details,
    )


def _make_config(output_dir: str = "") -> EvalConfig:
    model_spec = ModelSpec(provider="openai", model="gpt-5.4")
    return EvalConfig(
        model_spec=model_spec,
        models=[model_spec],
        harnesses=["codex", "zeroclaw"],
        runs_per_task=1,
        output_dir=output_dir,
        timeout_sec=30,
    )


def test_extract_json_handles_fenced_and_inline_text() -> None:
    fenced = """```json\n{"pass": true, "score": 1.0}\n```"""
    inline = 'Judge said: {"pass": false, "score": 0.2}'

    assert extract_json(fenced) == {"pass": True, "score": 1.0}
    assert extract_json(inline) == {"pass": False, "score": 0.2}


def test_failure_origin_prefers_provider_and_infers_agent_for_boundary_failures() -> None:
    classified = detect_failure_origin_from_error("HTTP 429 parse failed upstream provider api error")
    assert classified == {"failure_origin": "provider", "infra_error_code": "provider_api_error"}

    boundary_result = RunResult(
        task_id="task.boundary",
        harness="codex",
        run_id="run-1",
        run_index=1,
        model="openai:gpt-5.4",
        status="completed",
        final_text="done",
        grader_results=[
            GraderResult(
                grader_type="trajectory",
                name="trajectory:boundary_respected(shell)",
                passed=False,
                score=0.0,
            )
        ],
    )
    grader_result = RunResult(
        task_id="task.grader",
        harness="codex",
        run_id="run-2",
        run_index=1,
        model="openai:gpt-5.4",
        status="completed",
        final_text="done",
        grader_results=[
            GraderResult(
                grader_type="file_exists",
                name="file_exists:answer.txt",
                passed=False,
                score=0.0,
            )
        ],
    )

    assert infer_failure_origin(boundary_result) == "agent"
    assert infer_failure_origin(grader_result) == "grader"
    assert format_failure_origin("sandbox") == "Sandbox"


def test_generate_summary_and_category_reports_include_key_sections() -> None:
    metrics = [
        HarnessMetrics(
            harness="codex",
            pass_at_1=0.5,
            pass_at_3=1.0,
            quality_score=0.8,
            mean_cost_usd=0.1,
            mean_latency_sec=2.0,
            mean_total_tokens=1200,
            mean_tool_calls=3.0,
            timeout_rate=0.0,
        ),
        HarnessMetrics(
            harness="zeroclaw",
            pass_at_1=0.5,
            pass_at_3=0.5,
            quality_score=0.7,
            mean_cost_usd=0.2,
            mean_latency_sec=3.0,
            mean_total_tokens=1500,
            mean_tool_calls=4.0,
            timeout_rate=0.1,
        ),
    ]
    config = _make_config()
    tasks = [
        Task(id="task.1", category="coding", description="task", user_query="answer", timeout_sec=30),
        Task(id="task.2", category="security", description="task", user_query="answer", timeout_sec=30),
    ]
    results = [
        _make_result(task_id="task.1", harness="codex", passed=True),
        _make_result(task_id="task.1", harness="zeroclaw", passed=False),
        _make_result(task_id="task.2", harness="codex", passed=False),
        _make_result(task_id="task.2", harness="zeroclaw", passed=True),
    ]

    summary = generate_summary_report(metrics, config, results, tasks)
    category = generate_category_report(results, tasks, config)

    assert "# Evaluation Summary" in summary
    assert "Codex" in summary
    assert "## 2. Headline Results" in summary
    assert "## 3. Failures" in summary
    assert "## 4. Category Breakdown" in summary
    assert "# Category Breakdown Report" in category
    assert "## Coding" in category
    assert "## Cross-Category Summary" in category


def test_generate_multi_model_summary_and_reports_write_expected_files(isolated_temp_dir: Path) -> None:
    output_dir = str(isolated_temp_dir / "reports")
    model_a = ModelSpec(provider="openai", model="gpt-5.4")
    model_b = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
    config = EvalConfig(
        model_spec=model_a,
        models=[model_a, model_b],
        harnesses=["codex", "zeroclaw"],
        runs_per_task=1,
        output_dir=output_dir,
        timeout_sec=30,
    )
    tasks = [
        Task(id="task.1", category="coding", description="task", user_query="answer", timeout_sec=30),
    ]
    results = [
        _make_result(task_id="task.1", harness="codex", model=model_a.label, passed=True),
        _make_result(task_id="task.1", harness="zeroclaw", model=model_a.label, passed=False),
        _make_result(task_id="task.1", harness="codex", model=model_b.label, passed=False),
        _make_result(task_id="task.1", harness="zeroclaw", model=model_b.label, passed=True),
    ]

    summary = _generate_multi_model_summary(
        {
            model_a.label: [
                HarnessMetrics(harness="codex", pass_at_1=1.0, quality_score=0.9),
                HarnessMetrics(harness="zeroclaw", pass_at_1=0.0, quality_score=0.2),
            ],
            model_b.label: [
                HarnessMetrics(harness="codex", pass_at_1=0.0, quality_score=0.3),
                HarnessMetrics(harness="zeroclaw", pass_at_1=1.0, quality_score=0.8),
            ],
        },
        config,
        results,
        tasks,
    )
    assert "# Multi-Model Evaluation Matrix" in summary
    assert "openai:gpt-5.4" in summary
    assert "anthropic:claude-sonnet-4-6" in summary

    generate_reports(results, tasks, config)

    root = Path(output_dir)
    assert (root / "manifest.json").is_file()
    assert (root / "reports" / "summary.md").is_file()
    assert (root / "reports" / "case-review.md").is_file()
    metrics_json = json.loads((root / "data" / "metrics.json").read_text())
    assert set(metrics_json.keys()) == {model_a.label, model_b.label}
    summary = (root / "reports" / "summary.md").read_text()
    assert "# Multi-Model Evaluation Matrix" in summary
    assert "## Failure Taxonomy" in summary


def test_generate_reports_single_harness_multi_model_uses_dedicated_layout(
    isolated_temp_dir: Path,
) -> None:
    """1 harness x N models should not degenerate into the 1-row 'matrix' layout.

    The dedicated layout must surface: Setup (with executor + per-model list),
    a per-model headline table with full metric columns, a task x model matrix
    with winner column, a category x model breakdown, and per-model judge stats.
    """
    output_dir = str(isolated_temp_dir / "single-harness-multi-model")
    model_a = ModelSpec(provider="openai", model="gpt-5.4")
    model_b = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
    judge = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
    config = EvalConfig(
        model_spec=model_a,
        models=[model_a, model_b],
        harnesses=["codex"],
        runs_per_task=1,
        judge_model_spec=judge,
        output_dir=output_dir,
        timeout_sec=30,
    )
    tasks = [
        Task(id="coding.01", category="coding", description="code", user_query="q", timeout_sec=30),
        Task(id="reasoning.02", category="reasoning", description="reason", user_query="q", timeout_sec=30),
    ]

    def _judged_result(*, task_id: str, model: str, passed: bool, judge_score: float) -> RunResult:
        r = _make_result(task_id=task_id, harness="codex", model=model, passed=passed)
        r.grader_results.append(
            GraderResult(
                grader_type="rubric_judge",
                name="quality",
                passed=judge_score >= 0.5,
                score=judge_score,
            )
        )
        return r

    results = [
        _judged_result(task_id="coding.01", model=model_a.label, passed=True, judge_score=0.9),
        _judged_result(task_id="coding.01", model=model_b.label, passed=True, judge_score=0.8),
        _judged_result(task_id="reasoning.02", model=model_a.label, passed=False, judge_score=0.3),
        _judged_result(task_id="reasoning.02", model=model_b.label, passed=True, judge_score=0.7),
    ]

    from agent_harness_eval.config.runtime import RuntimeConfig

    runtime_config = RuntimeConfig(
        project_root=isolated_temp_dir,
        executor_backend="docker",
        harness_config={"codex": {"version": "0.118.0"}},
    )

    generate_reports(results, tasks, config, runtime_config=runtime_config)

    root = Path(output_dir)
    summary = (root / "reports" / "summary.md").read_text()

    # Title / routing
    assert "Evaluation Summary — Codex across 2 models" in summary
    assert "# Multi-Model Evaluation Matrix" not in summary  # must NOT fall through to generic layout

    # 1. Setup — version, executor, per-model listing, judge, tasks count
    assert "## 1. Setup" in summary
    assert "| 0.118.0" in summary
    assert "**Executor:** docker" in summary
    assert "- **Models (2):**" in summary
    assert "- **Tasks:** 2 (coding, reasoning)" in summary
    assert f"- **Judge model:** `{judge.label}`" in summary

    # 2. Model Comparison — full metric columns
    assert "## 2. Model Comparison" in summary
    for col in ("Pass Rate", "Avg Quality", "Mean Time", "Mean Tokens", "Mean Cost", "Mean Tools", "Timeout%"):
        assert col in summary
    # Narrative winner line: model_b wins (2/2 pass, higher mean judge quality)
    assert f"`{model_b.label}` led on pass rate" in summary

    # 3. Per-Task Breakdown — rows per task, winner column
    assert "## 3. Per-Task Breakdown" in summary
    # coding.01: both pass, judge 0.9 vs 0.8 → model_a wins on quality tiebreak
    # reasoning.02: model_b wins outright
    assert "coding.01" in summary
    assert "reasoning.02" in summary
    assert f"`{model_a.label}`" in summary and f"`{model_b.label}`" in summary

    # 4. Category x Model
    assert "## 4. Category x Model" in summary
    assert "Coding" in summary and "Reasoning" in summary

    # 5. Judge Summary per model
    assert "## 5. Judge Summary" in summary
    # per-model rows with judge stats
    assert "Median" in summary
    # 4 judge evaluations split 2-per-model
    assert " | 2 " in summary  # Evaluations count


def test_generate_reports_single_harness_multi_model_does_not_crown_tied_models(
    isolated_temp_dir: Path,
) -> None:
    output_dir = str(isolated_temp_dir / "single-harness-multi-model-tie")
    model_a = ModelSpec(provider="openai", model="gpt-5.4")
    model_b = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
    config = EvalConfig(
        model_spec=model_a,
        models=[model_a, model_b],
        harnesses=["codex"],
        runs_per_task=1,
        output_dir=output_dir,
        timeout_sec=30,
    )
    tasks = [
        Task(id="reasoning.02", category="reasoning", description="reason", user_query="q", timeout_sec=30),
    ]
    results = [
        _make_result(task_id="reasoning.02", harness="codex", model=model_a.label, passed=False),
        _make_result(task_id="reasoning.02", harness="codex", model=model_b.label, passed=False),
    ]

    generate_reports(results, tasks, config)

    root = Path(output_dir)
    summary = (root / "reports" / "summary.md").read_text()

    assert f"`{model_a.label}` and `{model_b.label}` tied on pass rate and overall quality." in summary
    assert f"`{model_a.label}` led on pass rate" not in summary
    assert f"`{model_b.label}` led on pass rate" not in summary


def test_generate_summary_report_does_not_crown_tied_harnesses() -> None:
    metrics = [
        HarnessMetrics(
            harness="codex",
            pass_at_1=0.0,
            pass_at_3=0.0,
            quality_score=0.0,
            mean_cost_usd=0.0,
            mean_latency_sec=0.0,
            mean_total_tokens=0,
            mean_tool_calls=0.0,
            timeout_rate=0.0,
            usage_metrics_available=False,
        ),
        HarnessMetrics(
            harness="zeroclaw",
            pass_at_1=0.0,
            pass_at_3=0.0,
            quality_score=0.0,
            mean_cost_usd=0.0,
            mean_latency_sec=0.0,
            mean_total_tokens=0,
            mean_tool_calls=0.0,
            timeout_rate=0.0,
            usage_metrics_available=False,
        ),
    ]
    config = _make_config()
    tasks = [Task(id="task.1", category="coding", description="task", user_query="answer", timeout_sec=30)]
    results = [
        _make_result(task_id="task.1", harness="codex", passed=False),
        _make_result(task_id="task.1", harness="zeroclaw", passed=False),
    ]

    summary = generate_summary_report(metrics, config, results, tasks)

    assert "Codex and Zeroclaw tied on pass rate and overall quality." in summary
    assert "Codex led on pass rate and overall quality." not in summary
    assert "Zeroclaw was the closest runner-up." not in summary


def test_generate_summary_report_category_breakdown_marks_tied_best_harness() -> None:
    metrics = [
        HarnessMetrics(
            harness="codex",
            pass_at_1=0.0,
            pass_at_3=0.0,
            quality_score=0.0,
            mean_cost_usd=0.0,
            mean_latency_sec=0.0,
            mean_total_tokens=0,
            mean_tool_calls=0.0,
            timeout_rate=0.0,
            usage_metrics_available=False,
        ),
        HarnessMetrics(
            harness="zeroclaw",
            pass_at_1=0.0,
            pass_at_3=0.0,
            quality_score=0.0,
            mean_cost_usd=0.0,
            mean_latency_sec=0.0,
            mean_total_tokens=0,
            mean_tool_calls=0.0,
            timeout_rate=0.0,
            usage_metrics_available=False,
        ),
    ]
    config = _make_config()
    tasks = [Task(id="skills.02", category="skills", description="task", user_query="answer", timeout_sec=30)]
    results = [
        _make_result(task_id="skills.02", harness="codex", passed=False),
        _make_result(task_id="skills.02", harness="zeroclaw", passed=False),
    ]

    summary = generate_summary_report(metrics, config, results, tasks)

    assert "| Skills   | tied" in summary
    assert "| Skills   | Codex" not in summary
    assert "| Skills   | Zeroclaw" not in summary
