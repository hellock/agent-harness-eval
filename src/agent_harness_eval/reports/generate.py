"""Top-level report generation orchestration."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Any

from ..metrics import HarnessMetrics, compute_harness_metrics, is_not_applicable_run
from ..task import Task
from ..types import EvalConfig, RunResult
from .case_review import generate_case_review_report
from .failure_taxonomy import generate_failure_report
from .formatting import (
    format_cost_cell,
    format_harness_name,
    format_latency_cell,
    format_pass_cell,
    format_task_count_label,
    format_token_cell,
    is_pass,
    markdown_table,
)
from .summary import generate_summary_report


def generate_reports(
    results: list[RunResult],
    tasks: list[Task],
    config: EvalConfig,
    judge_llm: Any | None = None,
    secondary_judge_llm: Any | None = None,
    *,
    runtime_config: Any | None = None,
) -> None:
    """Generate all report files for an evaluation run."""
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    reports_dir = os.path.join(output_dir, "reports")
    traces_dir = os.path.join(output_dir, "traces")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    models = config.models or [config.model_spec]
    is_multi_model = len(models) > 1
    harness_versions, executor_backend = _extract_runtime_info(runtime_config, config.harnesses)

    if is_multi_model:
        all_metrics: dict[str, list[HarnessMetrics]] = {}
        for model_spec in models:
            model_results = [r for r in results if r.model == model_spec.label]
            all_metrics[model_spec.label] = compute_harness_metrics(model_results, config.harnesses)

        if len(config.harnesses) == 1:
            # Single-harness x N-models: compare models under the same harness.
            # The multi-harness "matrix" collapses to a 1-row table here, so we
            # render a dedicated layout with model-as-row-dimension and the full
            # metric columns (latency, cost, tokens) that model comparison needs.
            summary_md = _generate_single_harness_multi_model_summary(
                all_metrics,
                config,
                results,
                tasks,
                harness_versions=harness_versions,
                executor_backend=executor_backend,
                preflight_summary=_describe_preflight(output_dir),
            )
        else:
            summary_md = _generate_multi_model_summary(all_metrics, config, results, tasks)
            summary_md = _append_embedded_sections(
                summary_md,
                preflight_summary=_generate_preflight_summary(output_dir),
                failure_report=generate_failure_report(results),
            )
        _write_report_files(
            reports_dir,
            {
                "summary.md": summary_md,
                "case-review.md": generate_case_review_report(results),
            },
        )
        _write_json_file(
            os.path.join(data_dir, "metrics.json"),
            {k: [asdict(m) for m in v] for k, v in all_metrics.items()},
        )
    else:
        metrics = compute_harness_metrics(results, config.harnesses)

        summary_md = generate_summary_report(
            metrics,
            config,
            results,
            tasks,
            harness_versions=harness_versions,
            executor_backend=executor_backend,
            preflight_summary=_describe_preflight(output_dir),
        )
        _write_report_files(
            reports_dir,
            {
                "summary.md": summary_md,
                "case-review.md": generate_case_review_report(results),
            },
        )
        _write_json_file(os.path.join(data_dir, "metrics.json"), [asdict(m) for m in metrics])

    _write_manifest(
        output_dir=output_dir,
        config=config,
        tasks=tasks,
        results=results,
        runtime_config=runtime_config,
    )


def _generate_multi_model_summary(
    all_metrics: dict[str, list[HarnessMetrics]],
    config: EvalConfig,
    results: list[RunResult],
    tasks: list[Task],
) -> str:
    lines: list[str] = []
    models = config.models or [config.model_spec]
    harnesses = config.harnesses

    lines.append("# Multi-Model Evaluation Matrix")
    lines.append("")
    lines.append(f"- **Models**: {', '.join('`' + m.label + '`' for m in models)}")
    lines.append(f"- **Harnesses**: {', '.join(format_harness_name(h) for h in harnesses)}")
    lines.append(f"- **Tasks**: {format_task_count_label(results)}")
    lines.append("")

    # pass@1 matrix
    lines.append("## 1. pass@1")
    lines.append("")
    headers = ["Harness", *(m.label for m in models)]
    rows = []
    for h in harnesses:
        row = [format_harness_name(h)]
        for m in models:
            hm = next((x for x in (all_metrics.get(m.label) or []) if x.harness == h), None)
            row.append(f"{hm.pass_at_1 * 100:.1f}%" if hm else "—")
        rows.append(row)
    lines.append(markdown_table(headers, rows))

    # Quality matrix
    lines.append("")
    lines.append("## 2. Average Quality")
    lines.append("")
    rows2 = []
    for h in harnesses:
        row = [format_harness_name(h)]
        for m in models:
            hm = next((x for x in (all_metrics.get(m.label) or []) if x.harness == h), None)
            row.append(f"{hm.quality_score:.2f}" if hm else "—")
        rows2.append(row)
    lines.append(markdown_table(headers, rows2))

    lines.append("")
    lines.append("---")
    lines.append("Detailed run-by-run diagnostics are available in `reports/case-review.md` and `traces/`.")

    return "\n".join(lines) + "\n"


def _generate_single_harness_multi_model_summary(
    all_metrics: dict[str, list[HarnessMetrics]],
    config: EvalConfig,
    results: list[RunResult],
    tasks: list[Task],
    *,
    harness_versions: dict[str, str] | None = None,
    executor_backend: str = "host",
    preflight_summary: str | None = None,
) -> str:
    """Summary layout for 1 harness x N models.

    Uses model as the row dimension (the multi-harness "matrix" degenerates here),
    keeps the full single-model metric columns, and adds task x model and
    category x model breakdowns that only make sense in this shape.
    """
    models = config.models or [config.model_spec]
    harness = config.harnesses[0]
    versions = harness_versions or {}
    version = versions.get(harness, versions.get(harness.replace("-", "_"), "—"))

    lines: list[str] = []
    lines.append(f"# Evaluation Summary — {format_harness_name(harness)} across {len(models)} models")
    lines.append("")

    # ─── 1. Setup ───
    lines.append("## 1. Setup")
    lines.append("")
    lines.append(markdown_table(["Harness", "Version"], [[format_harness_name(harness), version]]))
    lines.append("")
    lines.append(f"- **Executor:** {executor_backend}")
    if tasks:
        categories = sorted({task.category for task in tasks if task.category})
        lines.append(f"- **Tasks:** {len(tasks)} ({', '.join(categories)})")
    lines.append(f"- **Runs per task:** {config.runs_per_task}")
    lines.append(f"- **Models ({len(models)}):**")
    for m in models:
        lines.append(f"  - `{m.label}`")
    if config.judge_model_spec:
        lines.append(f"- **Judge model:** `{config.judge_model_spec.label}`")
    if preflight_summary:
        lines.append(f"- **Preflight:** {preflight_summary}")
    lines.append("")

    # ─── 2. Per-model headline ───
    lines.extend(_section_per_model_headline(harness, models, all_metrics, results, config))

    # ─── 3. Per-task x model ───
    lines.extend(_section_per_task_model_matrix(harness, models, results, tasks, config))

    # ─── 4. Category x model ───
    if tasks:
        lines.extend(_section_category_model_matrix(harness, models, results, tasks))

    # ─── 5. Judge summary per model ───
    if config.judge_model_spec:
        lines.extend(_section_judge_summary_per_model(harness, models, results, config.judge_model_spec.label))

    lines.append("## 6. Detailed Reports")
    lines.append("")
    lines.append("- Case review: `reports/case-review.md`")
    lines.append("- Machine-readable results: `data/runs.jsonl`")
    lines.append("- Metrics: `data/metrics.json`")
    lines.append("")
    return "\n".join(lines)


def _section_per_model_headline(
    harness: str,
    models: list[Any],
    all_metrics: dict[str, list[HarnessMetrics]],
    results: list[RunResult],
    config: EvalConfig,
) -> list[str]:
    lines = ["## 2. Model Comparison", ""]
    multi_run = config.runs_per_task > 1

    headers = ["Model", "Pass Rate"]
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
    metric_by_model: dict[str, HarnessMetrics] = {}
    for m in models:
        hm = next((x for x in (all_metrics.get(m.label) or []) if x.harness == harness), None)
        if hm is None:
            continue
        metric_by_model[m.label] = hm
        model_results = [r for r in results if r.model == m.label and r.harness == harness]
        estimated = any(r.metrics.metrics_estimated for r in model_results if r.metrics.metrics_estimated)
        row = [f"`{m.label}`", format_pass_cell(hm.pass_at_1)]
        if multi_run:
            row.append(format_pass_cell(hm.pass_at_3) if hm.pass_at_3 > 0 else "—")
        row.extend(
            [
                f"{hm.quality_score:.2f}",
                format_latency_cell(hm.mean_latency_sec),
                format_token_cell(hm.mean_total_tokens),
                format_cost_cell(hm.mean_cost_usd, estimated=estimated),
                format_cost_cell(hm.mean_cost_usd_no_cache, estimated=estimated),
                f"{hm.mean_tool_calls:.1f}",
                format_pass_cell(hm.timeout_rate),
            ]
        )
        rows.append(row)

    lines.append(markdown_table(headers, rows))
    lines.append("")

    # Narrative winner line (with cheaper-model callout if meaningful)
    if metric_by_model:
        ranked = sorted(
            metric_by_model.items(),
            key=lambda kv: (kv[1].pass_at_1, kv[1].quality_score),
            reverse=True,
        )
        winner_label, winner = ranked[0]
        if len(ranked) == 1:
            lines.append(f"`{winner_label}` passed {winner.pass_at_1 * 100:.1f}% of applicable runs.")
        else:
            leaders = [
                label
                for label, metric in ranked
                if metric.pass_at_1 == winner.pass_at_1 and metric.quality_score == winner.quality_score
            ]
            if len(leaders) > 1:
                leader_text = ", ".join(f"`{label}`" for label in leaders[:-1]) + f" and `{leaders[-1]}`"
                lines.append(f"{leader_text} tied on pass rate and overall quality.")
            else:
                runner_label, _runner = ranked[1]
                cost_note = ""
                eligible = [(label, m.mean_cost_usd) for label, m in metric_by_model.items() if m.mean_cost_usd > 0]
                if eligible:
                    cheapest_label, cheapest_cost = min(eligible, key=lambda kv: kv[1])
                    if cheapest_label != winner_label and winner.mean_cost_usd > 0:
                        ratio = winner.mean_cost_usd / cheapest_cost
                        if ratio >= 1.5:
                            cost_note = f" `{cheapest_label}` cost ~{ratio:.1f}x less per run."
                lines.append(
                    f"`{winner_label}` led on pass rate and overall quality. "
                    f"`{runner_label}` was the closest runner-up.{cost_note}"
                )
        lines.append("")

    return lines


def _section_per_task_model_matrix(
    harness: str,
    models: list[Any],
    results: list[RunResult],
    tasks: list[Task],
    config: EvalConfig,
) -> list[str]:
    lines = ["## 3. Per-Task Breakdown", ""]
    if not tasks:
        lines.append("_No tasks recorded._")
        lines.append("")
        return lines

    headers = ["Task", *[f"`{m.label}`" for m in models], "Winner"]
    rows: list[list[str]] = []

    for task in sorted(tasks, key=lambda t: t.id):
        row: list[str] = [task.id]
        cells: list[tuple[str, float, float, int]] = []  # (label, pass_rate, quality, applicable_count)

        for m in models:
            relevant = [
                r
                for r in results
                if r.task_id == task.id and r.harness == harness and r.model == m.label and not is_not_applicable_run(r)
            ]
            if not relevant:
                row.append("—")
                cells.append((m.label, -1.0, -1.0, 0))
                continue
            passed = [r for r in relevant if is_pass(r)]
            pass_rate = len(passed) / len(relevant)
            judge_scores = [
                g.score
                for r in relevant
                for g in r.grader_results
                if g.grader_type == "rubric_judge" and g.score is not None
            ]
            quality = sum(judge_scores) / len(judge_scores) if judge_scores else -1.0

            if len(relevant) == 1:
                pass_part = "✓" if pass_rate >= 1.0 else "✗"
            else:
                pass_part = f"{len(passed)}/{len(relevant)}"
            cell_text = f"{pass_part} ({quality:.2f})" if quality >= 0 else pass_part
            row.append(cell_text)
            cells.append((m.label, pass_rate, quality, len(relevant)))

        # Winner: highest pass rate; tiebreak by quality. Ignore models with no runs.
        # If nobody passed this task, there is no winner — we don't want to crown
        # the less-bad model and mislead readers scanning the column.
        contenders = [c for c in cells if c[3] > 0]
        if not contenders or max(c[1] for c in contenders) == 0.0:
            row.append("—")
        else:
            best_pr = max(c[1] for c in contenders)
            top = [c for c in contenders if c[1] == best_pr]
            if len(top) == 1:
                row.append(f"`{top[0][0]}`")
            else:
                qualities = [c[2] for c in top if c[2] >= 0]
                if qualities and max(qualities) > min(qualities):
                    best_q = max(qualities)
                    q_top = [c for c in top if c[2] == best_q]
                    row.append(f"`{q_top[0][0]}`" if len(q_top) == 1 else "tied")
                else:
                    row.append("tied")

        rows.append(row)

    lines.append(markdown_table(headers, rows))
    lines.append("")
    return lines


def _section_category_model_matrix(
    harness: str,
    models: list[Any],
    results: list[RunResult],
    tasks: list[Task],
) -> list[str]:
    lines = ["## 4. Category x Model", ""]

    tasks_by_category: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        if task.category:
            tasks_by_category[task.category].add(task.id)
    if not tasks_by_category:
        return []

    headers = ["Category", *[f"`{m.label}`" for m in models], "Gap"]
    rows: list[list[str]] = []
    for category in sorted(tasks_by_category):
        task_ids = tasks_by_category[category]
        row: list[str] = [category.title()]
        pass_rates: list[float] = []
        for m in models:
            relevant = [
                r
                for r in results
                if r.task_id in task_ids
                and r.harness == harness
                and r.model == m.label
                and not is_not_applicable_run(r)
            ]
            if not relevant:
                row.append("—")
                continue
            passed = [r for r in relevant if is_pass(r)]
            pass_rate = len(passed) / len(relevant)
            pass_rates.append(pass_rate)
            row.append(format_pass_cell(pass_rate))

        if len(pass_rates) >= 2:
            gap_pp = (max(pass_rates) - min(pass_rates)) * 100
            row.append(f"{gap_pp:.0f}pp" if gap_pp >= 0.5 else "0pp")
        else:
            row.append("—")
        rows.append(row)

    lines.append(markdown_table(headers, rows))
    lines.append("")
    return lines


def _section_judge_summary_per_model(
    harness: str,
    models: list[Any],
    results: list[RunResult],
    judge_label: str,
) -> list[str]:
    lines = ["## 5. Judge Summary", ""]
    lines.append(f"- **Primary judge:** `{judge_label}`")
    lines.append("")

    headers = ["Model", "Evaluations", "Mean", "Median", "Min", "Max"]
    rows: list[list[str]] = []
    any_scores = False
    for m in models:
        relevant = [r for r in results if r.model == m.label and r.harness == harness and not is_not_applicable_run(r)]
        scores = [
            g.score
            for r in relevant
            for g in r.grader_results
            if g.grader_type == "rubric_judge" and g.score is not None
        ]
        if not scores:
            rows.append([f"`{m.label}`", "0", "—", "—", "—", "—"])
            continue
        any_scores = True
        sorted_scores = sorted(scores)
        mid = len(sorted_scores) // 2
        median = sorted_scores[mid] if len(sorted_scores) % 2 else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
        rows.append(
            [
                f"`{m.label}`",
                str(len(scores)),
                f"{sum(scores) / len(scores):.3f}",
                f"{median:.3f}",
                f"{min(scores):.3f}",
                f"{max(scores):.3f}",
            ]
        )

    if not any_scores:
        # Nothing to say — drop the section entirely rather than print an empty table.
        return []

    lines.append(markdown_table(headers, rows))
    lines.append("")
    return lines


def _extract_runtime_info(
    runtime_config: Any | None,
    harnesses: list[str],
) -> tuple[dict[str, str], str]:
    """Pull harness versions and executor backend out of RuntimeConfig."""
    harness_versions: dict[str, str] = {}
    executor_backend = "host"
    if runtime_config is None:
        return harness_versions, executor_backend

    executor_backend = getattr(runtime_config, "executor_backend", "host")
    hc = getattr(runtime_config, "harness_config", {})
    for h in harnesses:
        key = h.replace("-", "_")
        cfg = hc.get(key) or hc.get(h) or {}
        if isinstance(cfg, dict) and cfg.get("version"):
            harness_versions[h] = cfg["version"]
    return harness_versions, executor_backend


def _write_report_files(output_dir: str, reports: dict[str, str]) -> None:
    """Write a dict of {filename: content} to output_dir and print each."""
    os.makedirs(output_dir, exist_ok=True)
    for filename, content in reports.items():
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(content)
        print(f"  Written: {filename}")


def _write_json_file(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Written: {os.path.relpath(path, os.path.dirname(path))}")


def _strip_title(report_text: str) -> str:
    lines = report_text.splitlines()
    while lines and (not lines[0].strip() or lines[0].startswith("# ")):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def _append_embedded_sections(
    summary_md: str,
    *,
    preflight_summary: str | None = None,
    category_report: str | None = None,
    failure_report: str | None = None,
    judge_report: str | None = None,
) -> str:
    sections = [summary_md.rstrip()]
    if preflight_summary:
        sections.append(preflight_summary.strip())
    if category_report:
        sections.append("## Category Breakdown\n\n" + _strip_title(category_report))
    if failure_report:
        sections.append("## Failure Taxonomy\n\n" + _strip_title(failure_report))
    if judge_report:
        sections.append("## Judge Analysis\n\n" + _strip_title(judge_report))
    return "\n\n".join(section for section in sections if section) + "\n"


def _generate_preflight_summary(output_dir: str) -> str | None:
    path = os.path.join(output_dir, "data", "preflight.json")
    if not os.path.exists(path):
        return None

    with open(path) as f:
        rows = json.load(f)

    if not rows:
        return None

    total = len(rows)
    failed = [row for row in rows if row.get("status") != "passed"]
    stage_counts: dict[str, int] = {}
    for row in failed:
        stage = str(row.get("stage") or "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    lines = ["## Preflight", ""]
    lines.append(f"- **Entries:** {total}")
    lines.append(f"- **Failures:** {len(failed)}")
    if stage_counts:
        stage_summary = ", ".join(f"{stage}={count}" for stage, count in sorted(stage_counts.items()))
        lines.append(f"- **Failure stages:** {stage_summary}")
    return "\n".join(lines)


def _describe_preflight(output_dir: str) -> str | None:
    path = os.path.join(output_dir, "data", "preflight.json")
    if not os.path.exists(path):
        return None

    with open(path) as f:
        rows = json.load(f)

    if not rows:
        return None

    failed = [row for row in rows if row.get("status") != "passed"]
    if not failed:
        return "passed"

    stage_counts: dict[str, int] = {}
    for row in failed:
        stage = str(row.get("stage") or "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    stage_summary = ", ".join(f"{stage}={count}" for stage, count in sorted(stage_counts.items()))
    return f"{len(failed)} failure(s) ({stage_summary})"


def _write_manifest(
    *,
    output_dir: str,
    config: EvalConfig,
    tasks: list[Task],
    results: list[RunResult],
    runtime_config: Any | None,
) -> None:
    models = config.models or [config.model_spec]
    executor_backend = getattr(runtime_config, "executor_backend", "host") if runtime_config is not None else "host"
    payload = {
        "schema_version": 1,
        "output_layout_version": 1,
        "models": [model.label for model in models],
        "judge_model": config.judge_model_spec.label if config.judge_model_spec else None,
        "secondary_judge_model": (
            config.secondary_judge_model_spec.label if config.secondary_judge_model_spec else None
        ),
        "harnesses": list(config.harnesses),
        "runs_per_task": config.runs_per_task,
        "executor": executor_backend,
        "task_count": len(tasks),
        "result_count": len(results),
        "files": {
            "preflight": "data/preflight.json",
            "runs": "data/runs.jsonl",
            "metrics": "data/metrics.json",
            "summary": "reports/summary.md",
            "case_review": "reports/case-review.md",
            "trace_index": "traces/index.jsonl",
        },
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("  Written: manifest.json")
