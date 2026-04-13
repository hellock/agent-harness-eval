"""Judge confidence and agreement analysis report."""

from __future__ import annotations

from collections import defaultdict

from ..metrics import is_not_applicable_run
from ..task import Task
from ..types import (
    RunResult,
)
from .formatting import (
    format_harness_name,
    format_pass_cell,
    markdown_table,
)


def generate_judge_analysis_report(
    results: list[RunResult],
    tasks: list[Task],
    judge_llm: str,
    secondary_judge_llm: str | None = None,
) -> str:
    """Generate judge confidence and agreement analysis report."""
    lines: list[str] = []

    lines.append("# Judge Analysis Report")
    lines.append("")
    lines.append(f"**Primary judge:** {judge_llm}")
    if secondary_judge_llm:
        lines.append(f"**Secondary judge:** {secondary_judge_llm}")
    lines.append("")

    applicable = [r for r in results if not is_not_applicable_run(r)]

    # Extract all rubric judge scores
    all_scores: list[float] = []
    scores_by_harness: dict[str, list[float]] = defaultdict(list)
    scores_by_task: dict[str, list[float]] = defaultdict(list)

    for r in applicable:
        for g in r.grader_results:
            if g.grader_type == "rubric_judge" and g.score is not None:
                all_scores.append(g.score)
                scores_by_harness[r.harness].append(g.score)
                scores_by_task[r.task_id].append(g.score)

    if not all_scores:
        lines.append("No rubric judge scores found in results.")
        return "\n".join(lines)

    # ─── Score Distribution ───
    lines.append("## Score Distribution")
    lines.append("")

    avg_score = sum(all_scores) / len(all_scores)
    min_score = min(all_scores)
    max_score = max(all_scores)
    sorted_scores = sorted(all_scores)
    mid = len(sorted_scores) // 2
    median_score = sorted_scores[mid] if len(sorted_scores) % 2 else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2

    lines.append(f"- **Total judge evaluations:** {len(all_scores)}")
    lines.append(f"- **Mean score:** {avg_score:.3f}")
    lines.append(f"- **Median score:** {median_score:.3f}")
    lines.append(f"- **Min score:** {min_score:.3f}")
    lines.append(f"- **Max score:** {max_score:.3f}")
    lines.append("")

    # Score buckets
    buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for s in all_scores:
        if s < 0.2:
            buckets["0.0-0.2"] += 1
        elif s < 0.4:
            buckets["0.2-0.4"] += 1
        elif s < 0.6:
            buckets["0.4-0.6"] += 1
        elif s < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1

    headers = ["Score Range", "Count", "Percentage"]
    rows: list[list[str]] = []
    for bucket, count in buckets.items():
        pct = count / len(all_scores) * 100 if all_scores else 0
        rows.append([bucket, str(count), f"{pct:.1f}%"])

    lines.append(markdown_table(headers, rows))
    lines.append("")

    # ─── Per-Harness Judge Scores ───
    lines.append("## Per-Harness Judge Scores")
    lines.append("")

    headers = ["Harness", "Evaluations", "Mean Score", "Min", "Max"]
    rows = []

    for harness in sorted(scores_by_harness.keys()):
        h_scores = scores_by_harness[harness]
        h_mean = sum(h_scores) / len(h_scores)
        h_min = min(h_scores)
        h_max = max(h_scores)
        rows.append(
            [
                format_harness_name(harness),
                str(len(h_scores)),
                f"{h_mean:.3f}",
                f"{h_min:.3f}",
                f"{h_max:.3f}",
            ]
        )

    lines.append(markdown_table(headers, rows))
    lines.append("")

    # ─── Score Variance Analysis ───
    lines.append("## Score Variance by Task")
    lines.append("")
    lines.append("Tasks with highest score variance across harnesses (potential judge inconsistency):")
    lines.append("")

    task_variance: list[tuple[str, float, float, float]] = []
    for task_id, t_scores in scores_by_task.items():
        if len(t_scores) < 2:
            continue
        t_mean = sum(t_scores) / len(t_scores)
        variance = sum((s - t_mean) ** 2 for s in t_scores) / len(t_scores)
        score_range = max(t_scores) - min(t_scores)
        task_variance.append((task_id, variance, t_mean, score_range))

    task_variance.sort(key=lambda x: x[1], reverse=True)

    if task_variance:
        headers = ["Task", "Variance", "Mean Score", "Score Range"]
        rows = []
        for task_id, var, mean, rng in task_variance[:15]:
            rows.append([task_id, f"{var:.4f}", f"{mean:.3f}", f"{rng:.3f}"])

        lines.append(markdown_table(headers, rows))
    else:
        lines.append("Not enough data for variance analysis.")

    lines.append("")

    # ─── Judge Agreement (Grader pass vs judge score) ───
    lines.append("## Judge-Grader Agreement")
    lines.append("")
    lines.append("Comparison of binary grader pass/fail with judge quality scores:")
    lines.append("")

    pass_scores: list[float] = []
    fail_scores: list[float] = []

    for r in applicable:
        # Determine binary pass from non-judge graders
        non_judge_graders = [g for g in r.grader_results if g.grader_type != "rubric_judge"]
        judge_graders = [g for g in r.grader_results if g.grader_type == "rubric_judge" and g.score is not None]

        if not non_judge_graders or not judge_graders:
            continue

        binary_pass = all(g.passed for g in non_judge_graders) and r.status == "completed"
        judge_avg = sum(g.score for g in judge_graders) / len(judge_graders)

        if binary_pass:
            pass_scores.append(judge_avg)
        else:
            fail_scores.append(judge_avg)

    if pass_scores or fail_scores:
        if pass_scores:
            lines.append(
                f"- **Passing runs:** mean judge score = "
                f"{sum(pass_scores) / len(pass_scores):.3f} (n={len(pass_scores)})"
            )
        if fail_scores:
            lines.append(
                f"- **Failing runs:** mean judge score = "
                f"{sum(fail_scores) / len(fail_scores):.3f} (n={len(fail_scores)})"
            )

        if pass_scores and fail_scores:
            gap = (sum(pass_scores) / len(pass_scores)) - (sum(fail_scores) / len(fail_scores))
            lines.append(f"- **Score gap:** {gap:.3f}")
            if gap < 0.1:
                lines.append(
                    "- **Warning:** Low score gap between passing and failing runs "
                    "suggests judge may not differentiate well."
                )
    else:
        lines.append("Not enough data for agreement analysis.")

    lines.append("")

    # ─── Secondary Judge Comparison ───
    if secondary_judge_llm:
        lines.append("## Primary vs Secondary Judge")
        lines.append("")
        lines.append(
            "Note: Secondary judge comparison requires paired scores from both judges. "
            "If secondary judge data is embedded in grader results, analysis follows."
        )
        lines.append("")

        # Look for secondary judge graders (convention: name contains "secondary")
        primary_by_key: dict[str, float] = {}
        secondary_by_key: dict[str, float] = {}

        for r in applicable:
            for g in r.grader_results:
                if g.grader_type != "rubric_judge" or g.score is None:
                    continue
                key = f"{r.task_id}@@{r.harness}@@{r.run_index}"
                if "secondary" in g.name.lower():
                    secondary_by_key[key] = g.score
                else:
                    primary_by_key[key] = g.score

        shared_keys = set(primary_by_key.keys()) & set(secondary_by_key.keys())

        if shared_keys:
            diffs = [abs(primary_by_key[k] - secondary_by_key[k]) for k in shared_keys]
            agreements = sum(1 for k in shared_keys if abs(primary_by_key[k] - secondary_by_key[k]) < 0.2)

            lines.append(f"- **Paired evaluations:** {len(shared_keys)}")
            lines.append(f"- **Mean absolute difference:** {sum(diffs) / len(diffs):.3f}")
            lines.append(f"- **Agreement rate (within 0.2):** {format_pass_cell(agreements / len(shared_keys))}")

            # Identify large disagreements
            large_disagreements = [
                (k, primary_by_key[k], secondary_by_key[k])
                for k in shared_keys
                if abs(primary_by_key[k] - secondary_by_key[k]) >= 0.4
            ]
            if large_disagreements:
                lines.append(f"- **Large disagreements (diff >= 0.4):** {len(large_disagreements)}")
                for key, p_score, s_score in sorted(large_disagreements, key=lambda x: abs(x[1] - x[2]), reverse=True)[
                    :10
                ]:
                    parts = key.split("@@")
                    task_id = parts[0] if parts else key
                    lines.append(f"  - {task_id}: primary={p_score:.3f}, secondary={s_score:.3f}")
        else:
            lines.append(
                "No paired primary/secondary judge scores found. Secondary judge may use separate result storage."
            )

        lines.append("")

    return "\n".join(lines)
