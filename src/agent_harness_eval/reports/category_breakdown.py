"""Per-category metrics report."""

from __future__ import annotations

from ..metrics import CategoryMetrics, compute_category_metrics
from ..task import Task
from ..types import (
    EvalConfig,
    RunResult,
)
from .formatting import (
    format_category_name,
    format_harness_name,
    format_latency_cell,
    format_pass_cell,
    format_token_cell,
    markdown_table,
)


def generate_category_report(
    results: list[RunResult],
    tasks: list[Task],
    config: EvalConfig,
) -> str:
    """Generate per-category breakdown report."""
    lines: list[str] = []

    lines.append("# Category Breakdown Report")
    lines.append("")

    # Build task dicts for compute_category_metrics
    task_dicts = [{"id": t.id, "category": t.category} for t in tasks]
    cat_metrics = compute_category_metrics(results, task_dicts, config.harnesses)

    if not cat_metrics:
        lines.append("No category metrics available.")
        return "\n".join(lines)

    # Group by category
    categories: dict[str, list[CategoryMetrics]] = {}
    for cm in cat_metrics:
        categories.setdefault(cm.category, []).append(cm)

    # Sort categories alphabetically
    for category in sorted(categories.keys()):
        cat_data = categories[category]
        lines.append(f"## {format_category_name(category)}")
        lines.append("")

        headers = ["Harness", "Pass Rate", "Avg Quality", "Median Latency", "Median Tokens"]
        rows: list[list[str]] = []

        for cm in sorted(cat_data, key=lambda c: c.pass_rate, reverse=True):
            rows.append(
                [
                    format_harness_name(cm.harness),
                    format_pass_cell(cm.pass_rate),
                    f"{cm.avg_quality_score:.2f}",
                    format_latency_cell(cm.median_latency_sec),
                    format_token_cell(cm.median_total_tokens),
                ]
            )

        lines.append(markdown_table(headers, rows))
        lines.append("")

    # ─── Cross-category comparison ───
    lines.append("## Cross-Category Summary")
    lines.append("")

    # For each harness, show which categories are strongest/weakest
    for harness in config.harnesses:
        harness_cats = [cm for cm in cat_metrics if cm.harness == harness]
        if not harness_cats:
            continue

        sorted_cats = sorted(harness_cats, key=lambda c: c.pass_rate, reverse=True)
        best = sorted_cats[0]
        worst = sorted_cats[-1]

        lines.append(f"**{format_harness_name(harness)}:**")
        lines.append(f"  Best: {format_category_name(best.category)} ({format_pass_cell(best.pass_rate)})")
        if len(sorted_cats) > 1:
            lines.append(f"  Weakest: {format_category_name(worst.category)} ({format_pass_cell(worst.pass_rate)})")
        lines.append("")

    return "\n".join(lines)
