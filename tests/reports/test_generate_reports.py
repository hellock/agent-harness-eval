from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.config.providers import ModelSpec, ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.graders.specs import GraderResult
from agent_harness_eval.reports.generate import generate_reports
from agent_harness_eval.task import Task
from agent_harness_eval.types import EvalConfig, RunMetrics, RunResult


@pytest.fixture
def isolated_temp_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-generate-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _make_result(
    *,
    task_id: str = "coding.01",
    harness: str = "codex",
    model: str = "openai:gpt-5.4",
    passed: bool = True,
    usage_available: bool = True,
) -> RunResult:
    return RunResult(
        task_id=task_id,
        harness=harness,
        run_id=f"{task_id}-{harness}",
        run_index=0,
        model=model,
        status="completed",
        final_text="done",
        metrics=RunMetrics(
            latency_sec=2.0,
            total_tokens=1200,
            usage_available=usage_available,
            cost_usd=0.12,
            tool_calls=3,
        ),
        grader_results=[
            GraderResult(
                grader_type="test_pass",
                name="test_pass",
                passed=passed,
                score=1.0 if passed else 0.0,
            )
        ],
    )


def test_generate_reports_single_model_writes_expected_files_and_summary_metadata(
    isolated_temp_dir: Path,
) -> None:
    output_dir = isolated_temp_dir / "reports"
    model = ModelSpec(provider="openai", model="gpt-5.4")
    config = EvalConfig(
        model_spec=model,
        models=[model],
        harnesses=["codex", "zeroclaw"],
        runs_per_task=1,
        judge_model_spec=ModelSpec(provider="anthropic", model="claude-sonnet-4-6"),
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com",
                api_key="judge-key",
                api_format="anthropic",
            )
        },
        output_dir=str(output_dir),
        timeout_sec=30,
    )
    tasks = [Task(id="coding.01", category="coding", description="task", user_query="answer", timeout_sec=30)]
    results = [
        _make_result(harness="codex", passed=True),
        _make_result(harness="zeroclaw", passed=False),
    ]
    runtime_config = RuntimeConfig(
        project_root=isolated_temp_dir,
        executor_backend="docker",
        harness_config={
            "codex": {"version": "0.118.0"},
            "zeroclaw": {"version": "0.6.9"},
        },
    )

    generate_reports(results, tasks, config, judge_llm=object(), runtime_config=runtime_config)

    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "reports" / "summary.md").is_file()
    assert (output_dir / "reports" / "case-review.md").is_file()
    metrics_json = json.loads((output_dir / "data" / "metrics.json").read_text(encoding="utf-8"))
    assert isinstance(metrics_json, list)
    assert {row["harness"] for row in metrics_json} == {"codex", "zeroclaw"}

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["summary"] == "reports/summary.md"
    assert manifest["files"]["runs"] == "data/runs.jsonl"

    summary = (output_dir / "reports" / "summary.md").read_text(encoding="utf-8")
    assert "**Executor:** docker" in summary
    assert "| Codex" in summary and "| 0.118.0" in summary
    assert "| Zeroclaw" in summary and "| 0.6.9" in summary
    assert "**Judge model:** `anthropic:claude-sonnet-4-6`" in summary
    assert "**Preflight:**" not in summary
    assert "Mean Time" in summary
    assert "| Codex    | 100.0%    | 0.00        | 2.0s      | 1.2k" in summary
    assert "$0.1200" in summary
    assert "| 3.0        | 0.0%     |" in summary
    assert "## 3. Failures" in summary
    assert "## 4. Category Breakdown" in summary
    assert "## 5. Judge Summary" not in summary
    assert "## 6. Detailed Reports" in summary


def test_generate_reports_skips_judge_analysis_without_judge_model(
    isolated_temp_dir: Path,
) -> None:
    output_dir = isolated_temp_dir / "reports-no-judge"
    model = ModelSpec(provider="openai", model="gpt-5.4")
    config = EvalConfig(
        model_spec=model,
        models=[model],
        harnesses=["codex"],
        runs_per_task=1,
        judge_model_spec=None,
        output_dir=str(output_dir),
        timeout_sec=30,
    )
    tasks = [Task(id="coding.01", category="coding", description="task", user_query="answer", timeout_sec=30)]
    results = [_make_result(harness="codex", passed=True)]

    generate_reports(results, tasks, config, judge_llm=None, runtime_config=None)

    assert (output_dir / "reports" / "summary.md").is_file()
    summary = (output_dir / "reports" / "summary.md").read_text(encoding="utf-8")
    assert "## 5. Judge Summary" not in summary


def test_generate_reports_render_na_for_unavailable_token_and_cost_metrics(
    isolated_temp_dir: Path,
) -> None:
    output_dir = isolated_temp_dir / "reports-na"
    model = ModelSpec(provider="openai", model="gpt-5.4")
    config = EvalConfig(
        model_spec=model,
        models=[model],
        harnesses=["zeroclaw"],
        runs_per_task=1,
        output_dir=str(output_dir),
        timeout_sec=30,
    )
    tasks = [Task(id="coding.01", category="coding", description="task", user_query="answer", timeout_sec=30)]
    results = [_make_result(harness="zeroclaw", passed=True, usage_available=False)]

    generate_reports(results, tasks, config, judge_llm=None, runtime_config=None)

    summary = (output_dir / "reports" / "summary.md").read_text(encoding="utf-8")
    case_review = (output_dir / "reports" / "case-review.md").read_text(encoding="utf-8")

    assert "N/A" in summary
    assert "Tokens: N/A | Cost: N/A" in case_review
