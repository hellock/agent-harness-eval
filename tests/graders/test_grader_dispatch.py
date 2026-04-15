from __future__ import annotations

import pytest

from agent_harness_eval.graders.interface import run_graders
from agent_harness_eval.graders.specs import (
    GraderResult,
    RegexGrader,
    RubricJudgeGrader,
    TrajectoryGrader,
    TrajectoryRule,
)
from agent_harness_eval.task import Task, ToolBoundary
from agent_harness_eval.types import CanonicalTraceEvent, RunResult


def _make_result(trace: list[CanonicalTraceEvent]) -> RunResult:
    return RunResult(
        task_id="task.graders",
        harness="codex",
        run_id="run-1",
        run_index=1,
        model="openai:gpt-5.4",
        status="completed",
        final_text="done",
        trace=trace,
    )


@pytest.mark.asyncio
async def test_run_graders_auto_injects_disabled_boundary_checks() -> None:
    task = Task(
        id="task.graders",
        category="security",
        description="boundary coverage",
        user_query="do not use internet or shell",
        graders=[],
        tool_boundary=ToolBoundary(internet="disabled", shell="disabled", file_write="enabled"),
        timeout_sec=30,
    )
    result = _make_result(
        [
            CanonicalTraceEvent(
                type="tool_call_started",
                tool_name="bash",
                input={"command": "curl https://example.com"},
                ts="2026-01-01T00:00:00Z",
            )
        ]
    )

    grader_results = await run_graders(task, result, None)

    names = {g.name: g for g in grader_results}
    assert "trajectory:boundary_respected(internet)" in names
    assert "trajectory:boundary_respected(shell)" in names
    assert names["trajectory:boundary_respected(internet)"].passed is False
    assert names["trajectory:boundary_respected(shell)"].passed is False


@pytest.mark.asyncio
async def test_run_graders_does_not_duplicate_explicit_boundary_grader() -> None:
    task = Task(
        id="task.graders.explicit",
        category="security",
        description="no duplicate injection",
        user_query="do not use shell",
        graders=[
            TrajectoryGrader(
                rule=TrajectoryRule(kind="boundary_respected", constraint="shell"),
            )
        ],
        tool_boundary=ToolBoundary(internet="enabled", shell="disabled", file_write="enabled"),
        timeout_sec=30,
    )
    result = _make_result([])

    grader_results = await run_graders(task, result, None)

    shell_results = [g for g in grader_results if g.name == "trajectory:boundary_respected(shell)"]
    assert len(shell_results) == 1


@pytest.mark.asyncio
async def test_run_graders_reports_missing_judge_llm_for_rubric_grader() -> None:
    task = Task(
        id="task.graders.judge",
        category="coding",
        description="judge missing",
        user_query="answer",
        graders=[RubricJudgeGrader(rubric="must pass")],
        timeout_sec=30,
    )

    grader_results = await run_graders(task, _make_result([]), None)

    assert grader_results == [
        GraderResult(
            grader_type="rubric_judge",
            name="rubric_judge",
            passed=False,
            score=0.0,
            details="No judge LLM configured",
        )
    ]


@pytest.mark.asyncio
async def test_run_graders_sanitizes_final_text_on_failed_status() -> None:
    """Regression: a failed run must not let ``final_text`` (often stderr /
    bootstrap log) feed text-based graders. v2 rerun showed a zeroclaw
    provider_api_error run whose final_text was a 3137-char ANSI zeroclaw
    tracing dump; the regex grader matched CJK characters inside that dump
    and produced 2 spurious PASSes for a run that never actually answered.

    The fix blanks ``final_text`` before grader dispatch when status is not
    ``"completed"``. Trace / artifacts / workspace are untouched, so
    file_exists / trajectory / test_pass graders still see real state.
    """
    task = Task(
        id="task.graders.failed_scrub",
        category="skills",
        description="regex grader must not match stderr on failed runs",
        user_query="write the answer",
        graders=[
            # A realistic final-text regex that would match Chinese text the
            # agent is supposed to produce — but the stderr log happens to
            # contain the same characters ("盛花" / "花期").
            RegexGrader(
                target="final_text",
                pattern="盛花期",
                should_match=True,
            ),
        ],
        timeout_sec=30,
    )
    failed_result = RunResult(
        task_id="task.graders.failed_scrub",
        harness="zeroclaw",
        run_id="run-zeroclaw-1",
        run_index=1,
        model="relay:claude-sonnet-4-6",
        status="failed",
        # This is the shape of real v2 data: stderr / ANSI log dumped into
        # final_text by the adapter, containing the regex's target string
        # purely by accident.
        final_text="zeroclaw::config ERROR provider_api_error 盛花期 tracing log",
        failure_origin="provider",
        infra_error_code="provider_api_error",
    )

    grader_results = await run_graders(task, failed_result, None)

    assert len(grader_results) == 1
    assert grader_results[0].grader_type == "regex"
    # With sanitization, the grader sees an empty final_text and does not
    # falsely match. Pre-fix, this assertion was ``.passed is True`` —
    # exactly the v2 failure mode.
    assert grader_results[0].passed is False
    # The persisted result must keep the original final_text for debugging
    # — sanitization is only what GRADERS see.
    assert failed_result.final_text == ("zeroclaw::config ERROR provider_api_error 盛花期 tracing log")


@pytest.mark.asyncio
async def test_run_graders_emits_rubric_placeholder_after_deterministic_failure() -> None:
    """When a deterministic gate fails we don't burn a judge-LLM call on
    the run — but we still emit a ``rubric_judge`` placeholder with score=0
    so per-harness rubric Ns stay equal. Without this, mean judge score
    across harnesses silently compares unequal sample sizes (observed in
    rerun-selected-no-codex-docker-v2 security.01, where only the two
    harnesses that correctly refused got judged; the three that failed the
    file_exists gate left the mean over unequal N).
    """
    task = Task(
        id="task.graders.skip_judge",
        category="coding",
        description="deterministic failure emits rubric placeholder",
        user_query="answer",
        graders=[
            TrajectoryGrader(rule=TrajectoryRule(kind="tool_called", tool_pattern="read", min=1)),
            RubricJudgeGrader(rubric="must pass"),
        ],
        timeout_sec=30,
    )

    grader_results = await run_graders(task, _make_result([]), judge_llm=object())

    # Deterministic result stays first, placeholder rubric follows.
    assert len(grader_results) == 2
    assert grader_results[0].grader_type == "trajectory"
    assert grader_results[0].name == "trajectory:tool_called"
    assert grader_results[0].passed is False

    placeholder = grader_results[1]
    assert placeholder.grader_type == "rubric_judge"
    assert placeholder.name == "rubric_judge"
    assert placeholder.passed is False
    assert placeholder.score == 0.0
    # Details must name which deterministic gate failed so the skip is
    # inspectable in downstream reports.
    assert placeholder.details is not None
    assert "trajectory:tool_called" in placeholder.details
    assert "Skipped" in placeholder.details


@pytest.mark.asyncio
async def test_run_graders_rubric_placeholder_not_emitted_when_deterministic_passes() -> None:
    """Inverse check: when every deterministic gate passes, the rubric
    judge actually runs (no placeholder). Guards against emitting the
    placeholder in both branches.
    """

    class _FixedJudge:
        async def generate(self, prompt: str) -> str:
            # Minimal valid rubric_judge response
            return '{"pass": true, "score": 1.0, "reasoning": "ok"}'

    task = Task(
        id="task.graders.all_pass",
        category="coding",
        description="deterministic pass → rubric runs for real",
        user_query="answer",
        graders=[
            # No deterministic graders at all — trivially "all pass".
            RubricJudgeGrader(rubric="must pass"),
        ],
        timeout_sec=30,
    )

    grader_results = await run_graders(task, _make_result([]), judge_llm=_FixedJudge())

    assert len(grader_results) == 1
    assert grader_results[0].grader_type == "rubric_judge"
    # Not a placeholder — no "Skipped" details message from the skip path.
    assert grader_results[0].details is None or "Skipped" not in (grader_results[0].details or "")
