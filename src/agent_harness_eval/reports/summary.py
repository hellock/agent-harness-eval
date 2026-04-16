"""Main summary report."""

from __future__ import annotations

from collections import defaultdict

from ..metrics import HarnessMetrics, compute_category_metrics, is_not_applicable_run
from ..task import Task
from ..types import EvalConfig, RunResult
from .formatting import (
    format_cost_cell,
    format_harness_name,
    format_latency_cell,
    format_pass_cell,
    format_token_cell,
    is_pass,
    markdown_table,
)


def generate_summary_report(
    metrics: list[HarnessMetrics],
    config: EvalConfig,
    results: list[RunResult] | None = None,
    tasks: list[Task] | None = None,
    *,
    harness_versions: dict[str, str] | None = None,
    executor_backend: str = "host",
    preflight_summary: str | None = None,
) -> str:
    """Generate the main summary report."""
    lines: list[str] = []
    lines.extend(_section_setup(config, tasks, harness_versions, executor_backend, preflight_summary))
    lines.extend(_section_results(metrics, results, config))

    if results and metrics:
        lines.extend(_section_failures(metrics, results))
        if tasks:
            lines.extend(_section_category_breakdown(metrics, results, tasks))
        if config.judge_model_spec:
            lines.extend(_section_judge_summary(results, config.judge_model_spec.label))

    lines.extend(
        [
            "## 6. Detailed Reports",
            "",
            "- Case review: `reports/case-review.md`",
            "- Machine-readable results: `data/runs.jsonl`",
            "- Metrics: `data/metrics.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _section_setup(
    config: EvalConfig,
    tasks: list[Task] | None,
    harness_versions: dict[str, str] | None,
    executor_backend: str,
    preflight_summary: str | None,
) -> list[str]:
    lines: list[str] = ["# Evaluation Summary", "", "## 1. Setup", ""]

    versions = harness_versions or {}
    rows = []
    for harness in config.harnesses:
        version = versions.get(harness, versions.get(harness.replace("-", "_"), "—"))
        rows.append([format_harness_name(harness), version])
    lines.append(markdown_table(["Harness", "Version"], rows))
    lines.append("")
    lines.append(f"- **Executor:** {executor_backend}")
    if tasks:
        categories = sorted({task.category for task in tasks if task.category})
        lines.append(f"- **Tasks:** {len(tasks)} ({', '.join(categories)})")
    lines.append(f"- **Runs per task:** {config.runs_per_task}")
    lines.append(f"- **Model:** `{config.model_spec.label}`")
    if config.judge_model_spec:
        lines.append(f"- **Judge model:** `{config.judge_model_spec.label}`")
    if preflight_summary:
        lines.append(f"- **Preflight:** {preflight_summary}")
    lines.append("")
    return lines


def _section_results(
    metrics: list[HarnessMetrics],
    results: list[RunResult] | None,
    config: EvalConfig,
) -> list[str]:
    lines = ["## 2. Headline Results", ""]
    multi_run = config.runs_per_task > 1

    headers = ["Harness", "Pass Rate"]
    if multi_run:
        headers.append("Avg Pass Rate")
    headers.extend(
        [
            "Avg Quality",
            "Mean Time",
            "Mean Tokens",
            "Mean Cost",
            "Mean Cost Without Cache",
            "Mean Tools",
            "Timeout%",
        ]
    )

    rows: list[list[str]] = []
    for metric in metrics:
        row = [format_harness_name(metric.harness), format_pass_cell(metric.pass_at_1)]
        if multi_run:
            row.append(format_pass_cell(metric.pass_at_3) if metric.pass_at_3 > 0 else "—")
        row.extend(
            [
                f"{metric.quality_score:.2f}",
                format_latency_cell(metric.mean_latency_sec),
                format_token_cell(metric.mean_total_tokens, available=metric.usage_metrics_available),
                format_cost_cell(metric.mean_cost_usd, available=metric.usage_metrics_available),
                format_cost_cell(
                    metric.mean_cost_usd_no_cache,
                    available=metric.usage_metrics_available,
                ),
                f"{metric.mean_tool_calls:.1f}",
                format_pass_cell(metric.timeout_rate),
            ]
        )
        rows.append(row)

    lines.append(markdown_table(headers, rows))
    lines.append("")

    if results:
        applicable_runs = [result for result in results if not is_not_applicable_run(result)]
        passed_runs = [result for result in applicable_runs if is_pass(result)]
        if len(metrics) == 1:
            harness_name = format_harness_name(metrics[0].harness)
            lines.append(f"{harness_name} passed {len(passed_runs)} of {len(applicable_runs)} applicable runs.")
        else:
            ranked = sorted(metrics, key=lambda item: (item.pass_at_1, item.quality_score), reverse=True)
            winner = ranked[0]
            if len(ranked) > 1:
                runner_up = ranked[1]
                lines.append(
                    f"{format_harness_name(winner.harness)} led on pass rate and overall quality. "
                    f"{format_harness_name(runner_up.harness)} was the closest runner-up."
                )
            else:
                lines.append(f"{format_harness_name(winner.harness)} led on pass rate.")
        lines.append("")

    return lines


def _section_failures(metrics: list[HarnessMetrics], results: list[RunResult]) -> list[str]:
    lines = ["## 3. Failures", ""]
    failed_runs = [result for result in results if not is_not_applicable_run(result) and not is_pass(result)]
    if not failed_runs:
        lines.append("No failures observed.")
        lines.append("")
        return lines

    grader_failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in results:
        for grader in result.grader_results:
            if not grader.passed:
                grader_failures[result.harness][grader.grader_type] += 1

    summary_rows: list[list[str]] = []
    for metric in metrics:
        failures = grader_failures.get(metric.harness)
        if not failures:
            summary_rows.append([format_harness_name(metric.harness), "—", "0", "no failures"])
            continue

        dominant = max(failures, key=failures.get)
        harness_results = [result for result in results if result.harness == metric.harness]
        notes: list[str] = []
        timeout_count = sum(1 for result in harness_results if result.status == "timed_out")
        failed_count = sum(1 for result in harness_results if result.status == "failed")
        if timeout_count:
            notes.append(f"{timeout_count} timeout(s)")
        if failed_count:
            notes.append(f"{failed_count} crash(es)")
        summary_rows.append(
            [
                format_harness_name(metric.harness),
                dominant,
                str(failures[dominant]),
                "; ".join(notes) if notes else f"{sum(failures.values())} total failure(s)",
            ]
        )

    lines.append(markdown_table(["Harness", "Dominant Failure Mode", "Count", "Notes"], summary_rows))
    lines.append("")

    task_failures: dict[str, list[str]] = defaultdict(list)
    for result in failed_runs:
        task_failures[result.task_id].append(result.harness)

    task_rows = []
    for task_id, harnesses in sorted(task_failures.items(), key=lambda item: (-len(item[1]), item[0])):
        task_rows.append(
            [
                task_id,
                str(len(harnesses)),
                ", ".join(format_harness_name(harness) for harness in sorted(set(harnesses))),
            ]
        )
    lines.append(markdown_table(["Task", "Failures", "Affected Harnesses"], task_rows[:15]))
    lines.append("")
    return lines


def _section_category_breakdown(
    metrics: list[HarnessMetrics],
    results: list[RunResult],
    tasks: list[Task],
) -> list[str]:
    lines = ["## 4. Category Breakdown", ""]
    category_metrics = compute_category_metrics(
        results,
        [{"id": task.id, "category": task.category} for task in tasks],
        [metric.harness for metric in metrics],
    )

    if len(metrics) == 1:
        harness = metrics[0].harness
        rows: list[list[str]] = []
        best: tuple[str, float] | None = None
        worst: tuple[str, float] | None = None
        for item in sorted(category_metrics, key=lambda item: item.category):
            if item.harness != harness:
                continue
            rows.append(
                [
                    item.category.title(),
                    format_pass_cell(item.pass_rate),
                    f"{item.avg_quality_score:.2f}",
                    format_latency_cell(item.median_latency_sec),
                    format_token_cell(item.median_total_tokens, available=item.usage_metrics_available),
                ]
            )
            if best is None or item.pass_rate > best[1]:
                best = (item.category, item.pass_rate)
            if worst is None or item.pass_rate < worst[1]:
                worst = (item.category, item.pass_rate)

        lines.append(markdown_table(["Category", "Pass Rate", "Avg Quality", "Median Latency", "Median Tokens"], rows))
        lines.append("")
        if best and worst:
            lines.append(f"Best category: {best[0].title()}. Weakest category: {worst[0].title()}.")
            lines.append("")
        return lines

    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for item in category_metrics:
        grouped[item.category].append((item.harness, item.pass_rate))

    rows = []
    largest_gap: tuple[str, float] | None = None
    for category in sorted(grouped):
        entries = grouped[category]
        best_harness, best_rate = max(entries, key=lambda item: item[1])
        worst_rate = min(rate for _, rate in entries)
        gap = best_rate - worst_rate
        gap_label = "high" if gap >= 0.5 else "moderate" if gap >= 0.2 else "low"
        rows.append([category.title(), format_harness_name(best_harness), format_pass_cell(best_rate), gap_label])
        if largest_gap is None or gap > largest_gap[1]:
            largest_gap = (category, gap)

    lines.append(markdown_table(["Category", "Best Harness", "Best Pass Rate", "Largest Gap"], rows))
    lines.append("")
    if largest_gap and largest_gap[1] > 0:
        lines.append(f"Largest divergence appeared in {largest_gap[0].title()}.")
        lines.append("")
    return lines


def _section_judge_summary(results: list[RunResult], judge_label: str) -> list[str]:
    lines = ["## 5. Judge Summary", ""]
    applicable = [result for result in results if not is_not_applicable_run(result)]

    all_scores: list[float] = []
    scores_by_harness: dict[str, list[float]] = defaultdict(list)
    scores_by_task: dict[str, list[float]] = defaultdict(list)
    pass_scores: list[float] = []
    fail_scores: list[float] = []

    for result in applicable:
        judge_scores = [
            grader.score
            for grader in result.grader_results
            if grader.grader_type == "rubric_judge" and grader.score is not None
        ]
        if not judge_scores:
            continue
        for score in judge_scores:
            all_scores.append(score)
            scores_by_harness[result.harness].append(score)
            scores_by_task[result.task_id].append(score)

        judge_avg = sum(judge_scores) / len(judge_scores)
        non_judge = [grader for grader in result.grader_results if grader.grader_type != "rubric_judge"]
        if non_judge and all(grader.passed for grader in non_judge) and result.status == "completed":
            pass_scores.append(judge_avg)
        else:
            fail_scores.append(judge_avg)

    if not all_scores:
        return []

    sorted_scores = sorted(all_scores)
    mid = len(sorted_scores) // 2
    median_score = sorted_scores[mid] if len(sorted_scores) % 2 else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2

    lines.append(f"- **Primary judge:** `{judge_label}`")
    lines.append(f"- **Judge evaluations:** {len(all_scores)}")
    lines.append(f"- **Mean score:** {sum(all_scores) / len(all_scores):.3f}")
    lines.append(f"- **Median score:** {median_score:.3f}")
    lines.append(f"- **Min / max:** {min(all_scores):.3f} / {max(all_scores):.3f}")
    if pass_scores:
        lines.append(f"- **Passing runs mean judge score:** {sum(pass_scores) / len(pass_scores):.3f}")
    if fail_scores:
        lines.append(f"- **Failing runs mean judge score:** {sum(fail_scores) / len(fail_scores):.3f}")
    lines.append("")

    if len(scores_by_harness) > 1:
        rows = []
        for harness in sorted(scores_by_harness):
            harness_scores = scores_by_harness[harness]
            rows.append(
                [
                    format_harness_name(harness),
                    f"{sum(harness_scores) / len(harness_scores):.3f}",
                    f"{min(harness_scores):.3f}",
                    f"{max(harness_scores):.3f}",
                ]
            )
        lines.append(markdown_table(["Harness", "Mean Judge Score", "Min", "Max"], rows))
        lines.append("")

    max_spread: tuple[str, float] | None = None
    for task_id, task_scores in scores_by_task.items():
        if len(task_scores) < 2:
            continue
        spread = max(task_scores) - min(task_scores)
        if max_spread is None or spread > max_spread[1]:
            max_spread = (task_id, spread)

    if max_spread and max_spread[1] >= 0.2:
        lines.append(f"The largest judge score spread appeared on `{max_spread[0]}` ({max_spread[1]:.3f}).")
    else:
        lines.append("No judge inconsistency signal was observed in this run.")
    lines.append("")
    return lines
