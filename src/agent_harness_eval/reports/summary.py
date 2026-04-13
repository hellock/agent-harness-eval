"""Main summary report."""

from __future__ import annotations

from collections import defaultdict

from ..metrics import HarnessMetrics
from ..task import Task
from ..types import (
    EvalConfig,
    RunResult,
)
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
) -> str:
    """Generate the summary report.

    Structure:
      1. Evaluation Setup
      2. Results
      3. Analysis
      4. Appendix (links to detail reports)
    """
    lines: list[str] = []

    # ── 1. Evaluation Setup ──
    lines.extend(_section_setup(config, tasks, metrics, harness_versions, executor_backend))

    # ── 2. Results ──
    lines.extend(_section_results(metrics, results, config))

    # ── 3. Analysis ──
    if results and metrics:
        lines.extend(_section_analysis(metrics, results))

    return "\n".join(lines)


# ─── Section builders ───


def _section_setup(
    config: EvalConfig,
    tasks: list[Task] | None,
    metrics: list[HarnessMetrics],
    harness_versions: dict[str, str] | None,
    executor_backend: str,
) -> list[str]:
    lines: list[str] = ["# Evaluation Summary", ""]

    # Harness table with versions
    lines.append("## 1. Evaluation Setup")
    lines.append("")

    versions = harness_versions or {}
    h_headers = ["Harness", "Version"]
    h_rows = []
    for h in config.harnesses:
        v = versions.get(h, versions.get(h.replace("-", "_"), "—"))
        h_rows.append([format_harness_name(h), v])
    lines.append(markdown_table(h_headers, h_rows))
    lines.append("")

    lines.append(f"- **Executor:** {executor_backend}")
    if tasks:
        categories = sorted(set(t.category for t in tasks if t.category))
        lines.append(f"- **Tasks:** {len(tasks)} ({', '.join(categories)})")
    lines.append(f"- **Runs per task:** {config.runs_per_task}")
    lines.append(f"- **Model:** `{config.model_spec.label}`")
    if config.judge_model_spec:
        lines.append(f"- **Judge model:** `{config.judge_model_spec.label}`")
    lines.append("")
    return lines


def _section_results(
    metrics: list[HarnessMetrics],
    results: list[RunResult] | None,
    config: EvalConfig,
) -> list[str]:
    lines: list[str] = []
    lines.append("## 2. Results")
    lines.append("")

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
    for m in metrics:
        estimated = False
        if results:
            harness_results = [r for r in results if r.harness == m.harness]
            estimated = any(r.metrics.metrics_estimated for r in harness_results if r.metrics.metrics_estimated)

        row = [
            format_harness_name(m.harness),
            format_pass_cell(m.pass_at_1),
        ]
        if multi_run:
            row.append(format_pass_cell(m.pass_at_3) if m.pass_at_3 > 0 else "—")
            row.extend(
                [
                    f"{m.quality_score:.2f}",
                    format_latency_cell(m.mean_latency_sec),
                    format_token_cell(m.mean_total_tokens),
                    format_cost_cell(m.mean_cost_usd, estimated=estimated),
                    format_cost_cell(m.mean_cost_usd_no_cache, estimated=estimated),
                    f"{m.mean_tool_calls:.1f}",
                    format_pass_cell(m.timeout_rate),
                ]
            )
        rows.append(row)

    lines.append(markdown_table(headers, rows))
    lines.append("")

    # Cache token availability footnote
    if results:
        harness_has_cache: dict[str, bool] = {}
        for m in metrics:
            h_results = [r for r in results if r.harness == m.harness]
            all_zero = all(r.metrics.cache_read_tokens == 0 and r.metrics.cache_write_tokens == 0 for r in h_results)
            harness_has_cache[m.harness] = not all_zero

        has_cache_harnesses = [h for h, v in harness_has_cache.items() if v]
        no_cache_harnesses = [h for h, v in harness_has_cache.items() if not v]

        if has_cache_harnesses and no_cache_harnesses:
            names = ", ".join(format_harness_name(h) for h in no_cache_harnesses)
            lines.append(f"*† Cache token data unavailable for {names} — Cost (cached) equals Cost (no cache).*")
            lines.append("")

    # One-line takeaway
    if metrics:
        sorted_by_pass = sorted(metrics, key=lambda m: m.pass_at_1, reverse=True)
        winner = sorted_by_pass[0]
        parts = [f"**{format_harness_name(winner.harness)}** leads at {format_pass_cell(winner.pass_at_1)} pass rate"]
        if len(sorted_by_pass) > 1:
            runner = sorted_by_pass[1]
            if runner.pass_at_1 < winner.pass_at_1:
                parts.append(
                    f"runner-up **{format_harness_name(runner.harness)}** at {format_pass_cell(runner.pass_at_1)}"
                )
        lines.append(". ".join(parts) + ".")
        lines.append("")

    return lines


def _section_analysis(
    metrics: list[HarnessMetrics],
    results: list[RunResult],
) -> list[str]:
    lines: list[str] = []
    lines.append("## 3. Analysis")
    lines.append("")

    lines.extend(_failure_attribution(metrics, results))
    lines.extend(_category_divergence(results))
    lines.extend(_cost_quality(metrics))
    lines.extend(_tool_gaps(metrics))

    return lines


# ─── Analysis sub-sections ───


def _failure_attribution(
    metrics: list[HarnessMetrics],
    results: list[RunResult],
) -> list[str]:
    lines: list[str] = []
    lines.append("### Failure Attribution")
    lines.append("")

    grader_failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        for g in r.grader_results:
            if not g.passed:
                grader_failures[r.harness][g.grader_type] += 1

    if not grader_failures:
        lines.append("No failures to analyze.")
        lines.append("")
        return lines

    headers = ["Harness", "Dominant Failure Mode", "Count", "Notes"]
    rows: list[list[str]] = []

    for m in sorted(metrics, key=lambda x: x.pass_at_1, reverse=True):
        h = m.harness
        failures = grader_failures.get(h)
        if not failures:
            continue
        dominant = max(failures, key=failures.get)
        total_fails = sum(failures.values())

        harness_results = [r for r in results if r.harness == h]
        timed_out = sum(1 for r in harness_results if r.status == "timed_out")
        failed = sum(1 for r in harness_results if r.status == "failed")
        notes: list[str] = []
        if timed_out:
            notes.append(f"{timed_out} timeout(s)")
        if failed:
            notes.append(f"{failed} crash(es)")
        if failures.get("trajectory", 0) > 0:
            notes.append(f"{failures['trajectory']} trajectory violation(s)")
        rows.append(
            [
                format_harness_name(h),
                dominant,
                str(failures[dominant]),
                "; ".join(notes) if notes else f"{total_fails} total grader failures",
            ]
        )

    lines.append(markdown_table(headers, rows))
    lines.append("")

    # Hotspot tasks
    task_fail: dict[str, list[str]] = defaultdict(list)
    for r in results:
        if not is_pass(r):
            task_fail[r.task_id].append(r.harness)

    hotspots = [
        (tid, hs) for tid, hs in sorted(task_fail.items(), key=lambda x: len(x[1]), reverse=True) if len(hs) >= 3
    ]
    if hotspots:
        lines.append("**Task difficulty hotspots** (failed by 3+ harnesses):")
        lines.append("")
        for task_id, harnesses in hotspots:
            names = ", ".join(format_harness_name(h) for h in harnesses)
            lines.append(f"- `{task_id}`: {names}")
        lines.append("")

    # Suspicious grading: all harnesses failed the same grader on a task
    suspicious: list[tuple[str, int, str]] = []
    task_harness_results: dict[str, dict[str, list[RunResult]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        task_harness_results[r.task_id][r.harness].append(r)

    for task_id, harness_map in task_harness_results.items():
        harnesses_that_ran = list(harness_map.keys())
        if len(harnesses_that_ran) < 3:
            continue
        # Check if every harness failed this task
        all_failed = True
        failed_grader_types_per_harness: list[set[str]] = []
        for _, h_results in harness_map.items():
            h_all_fail = all(not is_pass(r) for r in h_results)
            if not h_all_fail:
                all_failed = False
                break
            grader_types: set[str] = set()
            for r in h_results:
                for g in r.grader_results:
                    if not g.passed:
                        grader_types.add(g.grader_type)
            failed_grader_types_per_harness.append(grader_types)
        if not all_failed:
            continue
        # Find common failed grader types across all harnesses
        common_graders = failed_grader_types_per_harness[0]
        for gs in failed_grader_types_per_harness[1:]:
            common_graders = common_graders & gs
        if common_graders:
            for grader_type in sorted(common_graders):
                suspicious.append((task_id, len(harnesses_that_ran), grader_type))

    if suspicious:
        lines.append("**Suspicious grading** (all harnesses failed the same grader):")
        lines.append("")
        for task_id, n_harnesses, grader_type in suspicious:
            lines.append(
                f"- `{task_id}`: all {n_harnesses} harnesses failed "
                f"`{grader_type}` — grader may be too strict or judge model unreliable"
            )
        lines.append("")

    return lines


def _category_divergence(results: list[RunResult]) -> list[str]:
    lines: list[str] = []
    lines.append("### Category Performance")
    lines.append("")

    cat_pass: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        cat = r.task_id.split(".")[0] if "." in r.task_id else "other"
        cat_pass[cat][r.harness].append(is_pass(r))

    entries: list[tuple[str, float, str, float, str, float]] = []
    for cat, hp in cat_pass.items():
        rates = {h: (sum(ps) / len(ps)) for h, ps in hp.items() if ps}
        if len(rates) < 2:
            continue
        spread = max(rates.values()) - min(rates.values())
        best_h = max(rates, key=rates.get)
        worst_h = min(rates, key=rates.get)
        entries.append((cat, spread, best_h, rates[best_h], worst_h, rates[worst_h]))

    entries.sort(key=lambda x: x[1], reverse=True)

    if entries:
        for cat, spread, best_h, best_r, worst_h, worst_r in entries:
            if spread < 0.01:
                continue
            lines.append(
                f"- **{cat}**: {format_pass_cell(spread)} spread — "
                f"best {format_harness_name(best_h)} ({format_pass_cell(best_r)}), "
                f"worst {format_harness_name(worst_h)} ({format_pass_cell(worst_r)})"
            )
        lines.append("")

    return lines


def _cost_quality(metrics: list[HarnessMetrics]) -> list[str]:
    lines: list[str] = []
    costed = [m for m in metrics if m.mean_cost_usd > 0]
    if len(costed) < 2:
        return lines

    lines.append("### Cost-Quality Tradeoff")
    lines.append("")

    best = max(costed, key=lambda m: m.pass_at_1 / m.mean_cost_usd if m.mean_cost_usd > 0 else 0)
    worst = max(costed, key=lambda m: m.mean_cost_usd)

    lines.append(
        f"- **Most cost-efficient:** {format_harness_name(best.harness)} "
        f"({format_pass_cell(best.pass_at_1)} pass rate at "
        f"{format_cost_cell(best.mean_cost_usd)}/task)"
    )
    lines.append(
        f"- **Most expensive:** {format_harness_name(worst.harness)} "
        f"({format_cost_cell(worst.mean_cost_usd)}/task, "
        f"{format_token_cell(worst.mean_total_tokens)} mean tokens)"
    )

    if worst.mean_cost_usd > 0 and best.mean_cost_usd > 0:
        ratio = worst.mean_cost_usd / best.mean_cost_usd
        diff = worst.pass_at_1 - best.pass_at_1
        if ratio > 1.5:
            lines.append(
                f"- {format_harness_name(worst.harness)} costs "
                f"{ratio:.1f}x more than {format_harness_name(best.harness)} "
                f"for {'+' if diff >= 0 else ''}{format_pass_cell(diff)} pass rate difference"
            )

    lines.append("")
    return lines


def _tool_gaps(metrics: list[HarnessMetrics]) -> list[str]:
    lines: list[str] = []
    no_tools = [m for m in metrics if m.mean_tool_calls == 0 and m.pass_at_1 < 0.5]
    if not no_tools:
        return lines

    lines.append("### Tool Usage Gaps")
    lines.append("")
    for m in no_tools:
        lines.append(
            f"- **{format_harness_name(m.harness)}** reports 0 tool calls — "
            f"tasks requiring file operations or shell commands will fail. "
            f"pass rate = {format_pass_cell(m.pass_at_1)}"
        )
    lines.append("")
    return lines
