from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.graders.rubric_judge import run_rubric_judge
from agent_harness_eval.graders.specs import RubricJudgeGrader
from agent_harness_eval.task import Task
from agent_harness_eval.types import CanonicalTraceEvent, RunResult


class FakeJudgeLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


@pytest.fixture
def isolated_temp_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _make_result() -> RunResult:
    return RunResult(
        task_id="task.rubric",
        harness="codex",
        run_id="run-1",
        run_index=1,
        model="openai:gpt-5.4",
        status="completed",
        final_text="final answer",
        trace=[
            CanonicalTraceEvent(
                type="tool_call_started",
                tool_name="read_file",
                input={"path": "README.md"},
                ts="2026-01-01T00:00:00Z",
            )
        ],
    )


def _make_result_with_final_text(final_text: str) -> RunResult:
    result = _make_result()
    result.final_text = final_text
    return result


def _make_task() -> Task:
    return Task(
        id="task.rubric",
        category="skills",
        description="judge a time-sensitive answer",
        user_query="Plan a trip for 2026-04-18 with current-date awareness.",
        timeout_sec=30,
    )


@pytest.mark.asyncio
async def test_run_rubric_judge_parses_flat_json_response() -> None:
    judge = FakeJudgeLLM('{"pass": true, "score": 0.75, "reason": "solid"}')
    spec = RubricJudgeGrader(rubric="must pass")

    result = await run_rubric_judge(spec, _make_task(), _make_result(), judge, None)

    assert result.passed is True
    assert result.score == 0.75
    assert result.details == "solid"


@pytest.mark.asyncio
async def test_run_rubric_judge_parses_dimension_response_and_weights_bonus() -> None:
    judge = FakeJudgeLLM(
        """
        {
          "dimensions": {
            "correctness": {"pass": true, "score": 1.0, "reason": "correct"},
            "style": {"pass": false, "score": 0.5, "reason": "bonus miss"}
          }
        }
        """
    )
    spec = RubricJudgeGrader(rubric="grade it", dimensions=["correctness", "style (bonus)"])

    result = await run_rubric_judge(spec, _make_task(), _make_result(), judge, None)

    assert result.passed is True
    assert result.score == pytest.approx((1.0 * 1.0 + 0.5 * 0.5) / 1.5)
    assert result.dimensions is not None
    assert result.dimensions[1].required is False
    assert result.dimensions[1].weight == 0.5


@pytest.mark.asyncio
async def test_run_rubric_judge_includes_snapshots_in_prompt(isolated_temp_dir: Path) -> None:
    workspace = isolated_temp_dir / "workspace-eval"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("artifact body\n")

    judge = FakeJudgeLLM('{"pass": true, "score": 1.0, "reason": "ok"}')
    spec = RubricJudgeGrader(
        rubric="check snapshot",
        snapshot_paths=["answer.txt", "missing.txt"],
    )

    result = await run_rubric_judge(spec, _make_task(), _make_result(), judge, str(isolated_temp_dir))

    assert result.passed is True
    prompt = judge.prompts[0]
    assert "=== answer.txt ===\nartifact body" in prompt
    assert "=== missing.txt === [not found]" in prompt


@pytest.mark.asyncio
async def test_run_rubric_judge_prompt_has_no_prior_results_section() -> None:
    """Judge prompt should not contain prior grader results to avoid bias."""
    judge = FakeJudgeLLM('{"pass": true, "score": 1.0, "reason": "ok"}')
    spec = RubricJudgeGrader(rubric="evaluate independently")

    await run_rubric_judge(spec, _make_task(), _make_result(), judge, None)

    prompt = judge.prompts[0]
    assert "Prior Grader Results" not in prompt
    assert "Canonical evaluation time (UTC):" in prompt
    assert "Local evaluation time (informational only):" in prompt
    assert "Local evaluation timezone:" in prompt
    assert "Original User Task" in prompt
    assert "Plan a trip for 2026-04-18 with current-date awareness." in prompt
    assert "primary current-date/current-time" in prompt


@pytest.mark.asyncio
async def test_run_rubric_judge_prompt_preserves_full_final_text() -> None:
    long_tail = "TAIL_MARKER_" + ("x" * 6000)
    final_text = "prefix\n" + long_tail
    judge = FakeJudgeLLM('{"pass": true, "score": 1.0, "reason": "ok"}')
    spec = RubricJudgeGrader(rubric="evaluate the full answer")

    await run_rubric_judge(spec, _make_task(), _make_result_with_final_text(final_text), judge, None)

    prompt = judge.prompts[0]
    assert long_tail in prompt
    assert "## Agent's Final Response" in prompt


@pytest.mark.asyncio
async def test_run_rubric_judge_fails_on_invalid_json_response() -> None:
    judge = FakeJudgeLLM("not json")
    spec = RubricJudgeGrader(rubric="must be valid json")

    result = await run_rubric_judge(spec, _make_task(), _make_result(), judge, None)

    assert result.passed is False
    assert "Could not parse judge response as JSON" in (result.details or "")
