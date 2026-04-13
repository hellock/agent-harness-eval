"""Failure analysis report."""

from __future__ import annotations

from collections import defaultdict

from ..metrics import is_not_applicable_run, is_reportable_failure
from ..types import RunResult
from ..utils.failure_origin import format_failure_origin, infer_failure_origin
from .formatting import (
    format_harness_name,
    markdown_table,
)


def generate_failure_report(results: list[RunResult]) -> str:
    """Generate failure taxonomy report."""
    lines: list[str] = []

    lines.append("# Failure Taxonomy Report")
    lines.append("")

    # Filter to reportable failures only
    failures = [r for r in results if is_reportable_failure(r)]

    if not failures:
        lines.append("No failures to report. All runs passed or were not applicable.")
        return "\n".join(lines)

    total_runs = len([r for r in results if not is_not_applicable_run(r)])
    lines.append(f"**Total failures:** {len(failures)} / {total_runs} runs")
    lines.append("")

    # ─── Group by failure origin ───
    lines.append("## Failures by Origin")
    lines.append("")

    by_origin: dict[str, list[RunResult]] = defaultdict(list)
    for r in failures:
        origin = infer_failure_origin(r)
        by_origin[origin].append(r)

    # Summary table
    headers = ["Origin", "Count", "% of Failures"]
    rows: list[list[str]] = []

    for origin in sorted(by_origin.keys()):
        count = len(by_origin[origin])
        pct = count / len(failures) * 100 if failures else 0
        rows.append(
            [
                format_failure_origin(origin),
                str(count),
                f"{pct:.1f}%",
            ]
        )

    lines.append(markdown_table(headers, rows))
    lines.append("")

    # ─── Details per origin ───
    for origin in sorted(by_origin.keys()):
        origin_results = by_origin[origin]
        lines.append(f"### {format_failure_origin(origin)} Failures ({len(origin_results)})")
        lines.append("")

        # Group by harness
        by_harness: dict[str, list[RunResult]] = defaultdict(list)
        for r in origin_results:
            by_harness[r.harness].append(r)

        for harness in sorted(by_harness.keys()):
            harness_results = by_harness[harness]
            lines.append(f"**{format_harness_name(harness)}** ({len(harness_results)} failures):")
            lines.append("")

            for r in harness_results:
                status_label = r.status
                if r.status == "timed_out":
                    status_label = "timed out"

                # Get failure details
                detail = ""
                if r.infra_error_details:
                    detail = r.infra_error_details[:200]
                elif r.status == "completed":
                    # Failed graders
                    failed_graders = [g for g in r.grader_results if not g.passed]
                    if failed_graders:
                        grader_names = ", ".join(g.name for g in failed_graders)
                        detail = f"Failed graders: {grader_names}"

                lines.append(f"- **{r.task_id}** (run {r.run_index}): {status_label}")
                if detail:
                    lines.append(f"  {detail}")

            lines.append("")

    # ─── Failures per Task ───
    lines.append("## Failures by Task")
    lines.append("")

    by_task: dict[str, list[RunResult]] = defaultdict(list)
    for r in failures:
        by_task[r.task_id].append(r)

    headers = ["Task", "Failures", "Harnesses Affected", "Origins"]
    rows = []

    for task_id in sorted(by_task.keys()):
        task_failures = by_task[task_id]
        harnesses_affected = sorted(set(r.harness for r in task_failures))
        origins = sorted(set(format_failure_origin(infer_failure_origin(r)) for r in task_failures))

        rows.append(
            [
                task_id,
                str(len(task_failures)),
                ", ".join(format_harness_name(h) for h in harnesses_affected),
                ", ".join(origins),
            ]
        )

    lines.append(markdown_table(headers, rows))
    lines.append("")

    return "\n".join(lines)
