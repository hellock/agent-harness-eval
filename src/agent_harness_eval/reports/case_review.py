"""Detailed case review report."""

from __future__ import annotations

from collections import defaultdict

from ..metrics import is_not_applicable_run
from ..types import RunResult
from ..utils.failure_origin import format_failure_origin, infer_failure_origin
from .formatting import (
    format_cost_cell,
    format_harness_name,
    format_latency_cell,
    format_token_cell,
    is_pass,
)


def generate_case_review_report(results: list[RunResult]) -> str:
    """Generate detailed case-by-case review report."""
    lines: list[str] = []

    lines.append("# Case Review Report")
    lines.append("")

    applicable = [r for r in results if not is_not_applicable_run(r)]

    if not applicable:
        lines.append("No applicable results to review.")
        return "\n".join(lines)

    lines.append(f"**Total runs reviewed:** {len(applicable)}")
    lines.append("")

    # Group by task_id
    by_task: dict[str, list[RunResult]] = defaultdict(list)
    for r in applicable:
        by_task[r.task_id].append(r)

    for task_id in sorted(by_task.keys()):
        task_results = by_task[task_id]
        lines.append(f"## {task_id}")
        lines.append("")

        # Summary row for this task
        pass_count = sum(1 for r in task_results if is_pass(r))
        total = len(task_results)
        lines.append(f"**Overall:** {pass_count}/{total} runs passed")
        lines.append("")

        # Per-harness details
        by_harness: dict[str, list[RunResult]] = defaultdict(list)
        for r in task_results:
            by_harness[r.harness].append(r)

        for harness in sorted(by_harness.keys()):
            harness_runs = sorted(by_harness[harness], key=lambda r: r.run_index)
            lines.append(f"### {format_harness_name(harness)}")
            lines.append("")

            for r in harness_runs:
                passed = is_pass(r)
                status_emoji = "PASS" if passed else "FAIL"
                lines.append(f"**Run {r.run_index} - {status_emoji}** (status: {r.status})")
                lines.append("")

                # Metrics
                lines.append(
                    f"- Latency: {format_latency_cell(r.metrics.latency_sec)} | "
                    f"Tokens: {format_token_cell(float(r.metrics.total_tokens))} | "
                    f"Cost: {format_cost_cell(r.metrics.cost_usd)} | "
                    f"Tools: {r.metrics.tool_calls}"
                )

                # Grader outcomes
                if r.grader_results:
                    lines.append("- **Grader results:**")
                    for g in r.grader_results:
                        g_status = "PASS" if g.passed else "FAIL"
                        score_str = f" (score: {g.score:.2f})" if g.score is not None else ""
                        lines.append(f"  - [{g_status}] {g.name}{score_str}")

                        # Show rubric dimensions if present
                        if g.dimensions:
                            for dim in g.dimensions:
                                dim_status = "PASS" if dim.passed else "FAIL"
                                dim_score = f" ({dim.score:.2f})" if dim.score is not None else ""
                                lines.append(f"    - [{dim_status}] {dim.name}{dim_score}")
                                if dim.reason and not dim.passed:
                                    reason_short = dim.reason[:150]
                                    lines.append(f"      > {reason_short}")

                        # Show grader details for failures
                        if not g.passed and g.details:
                            detail_short = g.details[:200]
                            lines.append(f"  > {detail_short}")

                # Failure info
                if not passed:
                    origin = infer_failure_origin(r)
                    lines.append(f"- **Failure origin:** {format_failure_origin(origin)}")

                    if r.infra_error_code:
                        lines.append(f"- **Error code:** {r.infra_error_code}")
                    if r.infra_error_details:
                        detail_short = r.infra_error_details[:300]
                        lines.append(f"- **Error details:** {detail_short}")

                # Final text snippet
                if r.final_text:
                    snippet = r.final_text[:200].replace("\n", " ")
                    lines.append(f"- **Response snippet:** {snippet}")

                lines.append("")

    return "\n".join(lines)
