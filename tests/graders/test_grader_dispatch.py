from __future__ import annotations

import pytest

from agent_harness_eval.graders.interface import run_graders
from agent_harness_eval.graders.specs import GraderResult, RubricJudgeGrader, TrajectoryGrader, TrajectoryRule
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
async def test_run_graders_skips_rubric_judge_after_deterministic_failure() -> None:
    task = Task(
        id="task.graders.skip_judge",
        category="coding",
        description="deterministic failure should short-circuit judge",
        user_query="answer",
        graders=[
            TrajectoryGrader(rule=TrajectoryRule(kind="tool_called", tool_pattern="read", min=1)),
            RubricJudgeGrader(rubric="must pass"),
        ],
        timeout_sec=30,
    )

    grader_results = await run_graders(task, _make_result([]), judge_llm=object())

    assert len(grader_results) == 1
    assert grader_results[0].grader_type == "trajectory"
    assert grader_results[0].name == "trajectory:tool_called"
    assert grader_results[0].passed is False
