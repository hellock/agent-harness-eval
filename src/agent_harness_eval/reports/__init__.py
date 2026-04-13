"""Report generation modules."""

from .case_review import generate_case_review_report
from .category_breakdown import generate_category_report
from .failure_taxonomy import generate_failure_report
from .formatting import (
    format_category_name,
    format_cost_cell,
    format_harness_name,
    format_latency_cell,
    format_pass_cell,
    format_task_count_label,
    format_token_cell,
    is_pass,
    markdown_table,
)
from .generate import generate_reports
from .judge_analysis import generate_judge_analysis_report
from .summary import generate_summary_report

__all__ = [
    "format_category_name",
    "format_cost_cell",
    "format_harness_name",
    "format_latency_cell",
    "format_pass_cell",
    "format_task_count_label",
    "format_token_cell",
    "generate_case_review_report",
    "generate_category_report",
    "generate_failure_report",
    "generate_judge_analysis_report",
    "generate_reports",
    "generate_summary_report",
    "is_pass",
    "markdown_table",
]
