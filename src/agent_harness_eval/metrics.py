"""Aggregation metrics computation."""

from __future__ import annotations

from dataclasses import dataclass

from .task import TaskCategory
from .types import RunResult


@dataclass
class HarnessMetrics:
    harness: str
    pass_at_1: float = 0.0
    pass_at_3: float = 0.0
    median_latency_sec: float = 0.0
    median_total_tokens: float = 0.0
    median_cost_usd: float = 0.0
    median_cost_usd_no_cache: float = 0.0
    median_tool_calls: float = 0.0
    mean_latency_sec: float = 0.0
    mean_total_tokens: float = 0.0
    mean_cost_usd: float = 0.0
    mean_cost_usd_no_cache: float = 0.0
    mean_tool_calls: float = 0.0
    infra_failure_count: int = 0
    safety_failure_rate: float = 0.0
    timeout_rate: float = 0.0
    quality_score: float = 0.0
    usage_metrics_available: bool = True


@dataclass
class CategoryMetrics:
    category: TaskCategory
    harness: str
    pass_rate: float = 0.0
    avg_quality_score: float = 0.0
    median_latency_sec: float = 0.0
    median_total_tokens: float = 0.0
    usage_metrics_available: bool = True


def is_not_applicable_run(result: RunResult) -> bool:
    return result.status == "not_applicable" or result.infra_error_code == "capability_unsupported"


def is_reportable_failure(result: RunResult) -> bool:
    if is_not_applicable_run(result):
        return False
    return result.status != "completed" or any(not g.passed for g in result.grader_results)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _is_pass(result: RunResult) -> bool:
    if result.status != "completed":
        return False
    if not result.grader_results:
        return True
    return all(g.passed for g in result.grader_results)


def compute_harness_metrics(
    results: list[RunResult],
    harnesses: list[str],
) -> list[HarnessMetrics]:
    return [
        _compute_metrics_for_results(harness, [r for r in results if r.harness == harness]) for harness in harnesses
    ]


def _compute_metrics_for_results(
    harness: str,
    results: list[RunResult],
) -> HarnessMetrics:
    applicable = [r for r in results if not is_not_applicable_run(r)]
    eligible = [r for r in applicable if not (r.status != "completed" and r.failure_origin == "provider")]
    infra_failure_count = len(applicable) - len(eligible)

    if not eligible:
        return HarnessMetrics(
            harness=harness,
            infra_failure_count=infra_failure_count,
        )

    # Group by task_id + model
    by_task: dict[str, list[RunResult]] = {}
    for r in eligible:
        key = f"{r.task_id}@@{r.model}"
        by_task.setdefault(key, []).append(r)

    pass_at_1_count = 0
    pass_at_3_count = 0
    total_tasks = len(by_task)

    for task_results in by_task.values():
        sorted_results = sorted(task_results, key=lambda r: r.run_index)
        first_run = sorted_results[0]
        if _is_pass(first_run):
            pass_at_1_count += 1
        if any(_is_pass(r) for r in sorted_results[:3]):
            pass_at_3_count += 1

    # Efficiency metrics
    completed = [r for r in applicable if r.status == "completed"]
    latencies = [r.metrics.latency_sec for r in completed]
    usage_metrics_available = bool(completed) and all(r.metrics.usage_available for r in completed)
    tokens = [float(r.metrics.total_tokens) for r in completed if r.metrics.usage_available]
    costs = [r.metrics.cost_usd for r in completed if r.metrics.usage_available]
    costs_no_cache = [
        r.metrics.cost_usd_no_cache if r.metrics.cost_usd_no_cache is not None else r.metrics.cost_usd
        for r in completed
        if r.metrics.usage_available
    ]
    tool_calls = [float(r.metrics.tool_calls) for r in completed]

    # Safety failure rate
    safety_failures = len(
        [
            r
            for r in applicable
            if any(not g.passed and g.grader_type == "trajectory" and "dangerous" in g.name for g in r.grader_results)
        ]
    )

    # Timeout rate
    timeouts = len([r for r in applicable if r.status == "timed_out"])

    # Quality score
    judge_scores = [
        g.score for r in applicable for g in r.grader_results if g.grader_type == "rubric_judge" and g.score is not None
    ]
    avg_quality = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0

    n_applicable = len(applicable)
    return HarnessMetrics(
        harness=harness,
        pass_at_1=pass_at_1_count / total_tasks if total_tasks > 0 else 0.0,
        pass_at_3=pass_at_3_count / total_tasks if total_tasks > 0 else 0.0,
        median_latency_sec=_median(latencies),
        median_total_tokens=_median(tokens),
        median_cost_usd=_median(costs),
        median_cost_usd_no_cache=_median(costs_no_cache),
        median_tool_calls=_median(tool_calls),
        mean_latency_sec=_mean(latencies),
        mean_total_tokens=_mean(tokens),
        mean_cost_usd=_mean(costs),
        mean_cost_usd_no_cache=_mean(costs_no_cache),
        mean_tool_calls=_mean(tool_calls),
        infra_failure_count=infra_failure_count,
        safety_failure_rate=safety_failures / n_applicable if n_applicable > 0 else 0.0,
        timeout_rate=timeouts / n_applicable if n_applicable > 0 else 0.0,
        quality_score=avg_quality,
        usage_metrics_available=usage_metrics_available,
    )


def compute_category_metrics(
    results: list[RunResult],
    tasks: list[dict[str, str]],
    harnesses: list[str],
) -> list[CategoryMetrics]:
    task_categories = {t["id"]: t["category"] for t in tasks}
    metrics: list[CategoryMetrics] = []

    for harness in harnesses:
        h_results = [r for r in results if r.harness == harness]
        applicable = [r for r in h_results if not is_not_applicable_run(r)]

        by_category: dict[str, list[RunResult]] = {}
        for r in applicable:
            cat = task_categories.get(r.task_id)
            if not cat:
                continue
            by_category.setdefault(cat, []).append(r)

        for category, cat_results in by_category.items():
            completed = [r for r in cat_results if r.status == "completed"]
            passed = [r for r in cat_results if _is_pass(r)]
            judge_scores = [
                g.score
                for r in cat_results
                for g in r.grader_results
                if g.grader_type == "rubric_judge" and g.score is not None
            ]

            completed_usage_metrics_available = bool(completed) and all(r.metrics.usage_available for r in completed)
            metrics.append(
                CategoryMetrics(
                    category=category,
                    harness=harness,
                    pass_rate=len(passed) / len(cat_results) if cat_results else 0.0,
                    avg_quality_score=(sum(judge_scores) / len(judge_scores) if judge_scores else 0.0),
                    median_latency_sec=_median([r.metrics.latency_sec for r in completed]),
                    median_total_tokens=_median(
                        [float(r.metrics.total_tokens) for r in completed if r.metrics.usage_available]
                    ),
                    usage_metrics_available=completed_usage_metrics_available,
                )
            )

    return metrics
