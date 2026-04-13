"""Top-level report generation orchestration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from ..metrics import HarnessMetrics, compute_harness_metrics
from ..task import Task
from ..types import EvalConfig, RunResult
from .case_review import generate_case_review_report
from .category_breakdown import generate_category_report
from .failure_taxonomy import generate_failure_report
from .formatting import format_harness_name, format_task_count_label, markdown_table
from .judge_analysis import generate_judge_analysis_report
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

    if is_multi_model:
        all_metrics: dict[str, list[HarnessMetrics]] = {}
        for model_spec in models:
            model_results = [r for r in results if r.model == model_spec.label]
            all_metrics[model_spec.label] = compute_harness_metrics(model_results, config.harnesses)

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

        # Extract harness versions and executor from runtime_config if available
        harness_versions: dict[str, str] = {}
        executor_backend = "host"
        if runtime_config is not None:
            executor_backend = getattr(runtime_config, "executor_backend", "host")
            hc = getattr(runtime_config, "harness_config", {})
            for h in config.harnesses:
                key = h.replace("-", "_")
                cfg = hc.get(key) or hc.get(h) or {}
                if isinstance(cfg, dict) and cfg.get("version"):
                    harness_versions[h] = cfg["version"]

        summary_md = generate_summary_report(
            metrics,
            config,
            results,
            tasks,
            harness_versions=harness_versions,
            executor_backend=executor_backend,
        )
        judge_label = config.judge_model_spec.label if config.judge_model_spec else None
        judge_report = None
        if judge_label:
            judge_report = generate_judge_analysis_report(
                results,
                tasks,
                judge_label,
            )
        summary_md = _append_embedded_sections(
            summary_md,
            preflight_summary=_generate_preflight_summary(output_dir),
            category_report=generate_category_report(results, tasks, config),
            failure_report=generate_failure_report(results),
            judge_report=judge_report,
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
