"""Report formatting helpers."""

from __future__ import annotations

from ..types import RunResult


def format_harness_name(name: str) -> str:
    """Display name for harness (e.g., 'claude-code' -> 'Claude Code')."""
    return " ".join(word.capitalize() for word in name.replace("-", " ").replace("_", " ").split())


def format_cost_cell(cost: float, estimated: bool = False, decimals: int = 4) -> str:
    """Format cost."""
    prefix = "~" if estimated else ""
    return f"{prefix}${cost:.{decimals}f}"


def format_latency_cell(seconds: float) -> str:
    """Format latency — always in seconds for consistent comparison."""
    return f"{seconds:.1f}s"


def format_pass_cell(pass_rate: float) -> str:
    """Format pass rate as percentage."""
    return f"{pass_rate * 100:.1f}%"


def format_token_cell(tokens: float) -> str:
    """Format token count (e.g., '12.3k')."""
    if tokens < 1000:
        return str(int(tokens))
    if tokens < 1_000_000:
        return f"{tokens / 1000:.1f}k"
    return f"{tokens / 1_000_000:.1f}M"


def format_category_name(category: str) -> str:
    """Capitalize category name."""
    return category.capitalize()


def format_task_count_label(results: list[RunResult]) -> str:
    """Count unique task IDs."""
    unique_tasks = len(set(r.task_id for r in results))
    return f"{unique_tasks} task{'s' if unique_tasks != 1 else ''}"


def is_pass(result: RunResult) -> bool:
    """Check if a RunResult passes all graders."""
    if result.status != "completed":
        return False
    if not result.grader_results:
        return True
    return all(g.passed for g in result.grader_results)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build markdown table string."""
    if not headers:
        return ""

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def pad_row(cells: list[str]) -> str:
        padded = []
        for i, cell in enumerate(cells):
            w = widths[i] if i < len(widths) else len(cell)
            padded.append(cell.ljust(w))
        return "| " + " | ".join(padded) + " |"

    lines = [
        pad_row(headers),
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    for row in rows:
        # Ensure row has enough columns
        padded_row = row + [""] * (len(headers) - len(row))
        lines.append(pad_row(padded_row))

    return "\n".join(lines)
