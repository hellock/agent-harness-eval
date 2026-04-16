"""Grader dispatcher - runs all graders for a task."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any, Protocol

from ..task import Task
from ..types import RunResult
from .specs import (
    FileExistsGrader,
    GraderResult,
    GraderSpec,
    JsonSchemaGrader,
    RegexGrader,
    RubricJudgeGrader,
    TestPassGrader,
    TestSuiteGrader,
    TrajectoryGrader,
    TrajectoryRule,
)

logger = logging.getLogger(__name__)


class JudgeLLM(Protocol):
    """Protocol for judge LLM used by rubric graders."""

    async def generate(self, prompt: str) -> str: ...


async def run_graders(
    task: Task,
    result: RunResult,
    judge_llm: JudgeLLM | None,
    workspace_dir: str | None = None,
    *,
    executor: Any | None = None,
    execution_policy: Any | None = None,
    harness_name: str = "",
    grader_env: dict[str, str] | None = None,
) -> list[GraderResult]:
    """Run all graders for a task in two phases.

    Deterministic graders (test_pass, file_exists, regex, json_schema,
    trajectory, test_suite) run first. LLM judge graders (rubric_judge)
    only run if every deterministic grader passed.

    Auto-injects boundary_respected trajectory checks for any disabled
    tool_boundary constraints not already covered by an explicit grader.
    """
    result = normalize_result_for_grading(result)
    specs = expand_grader_specs(task)

    async def _run_one(spec: GraderSpec) -> GraderResult:
        try:
            gr = await _dispatch_grader(
                spec,
                task,
                result,
                judge_llm,
                workspace_dir,
                executor=executor,
                execution_policy=execution_policy,
                harness_name=harness_name,
                grader_env=grader_env,
            )
            if gr is not None:
                return gr
            return GraderResult(
                grader_type=spec.type,
                name=grader_name(spec),
                passed=False,
                score=0.0,
                details="Grader returned None",
            )
        except Exception as exc:
            logger.error("Grader %s failed: %s", spec.type, exc, exc_info=True)
            return GraderResult(
                grader_type=spec.type,
                name=grader_name(spec),
                passed=False,
                score=0.0,
                details=f"Grader error: {exc}",
            )

    deterministic_specs = [spec for spec in specs if _is_deterministic(spec)]
    llm_specs = [spec for spec in specs if not _is_deterministic(spec)]

    deterministic_results = list(await asyncio.gather(*[_run_one(spec) for spec in deterministic_specs]))
    failed_names = [gr.name for gr in deterministic_results if not gr.passed]
    if failed_names:
        # When any deterministic gate fails we don't want to burn a judge-LLM
        # call on a run that already failed. But silently dropping the LLM
        # grader (the prior behavior) made per-harness N of rubric scores
        # unequal across a suite — e.g. in rerun-selected-no-codex-docker-v2,
        # security.01 gave rubric coverage only to the two harnesses that
        # correctly refused the deletion, leaving the three that deleted
        # (and therefore most needed rubric judgment) unjudged. Mean-over-N
        # comparisons in the summary silently skipped those cells.
        #
        # Emit a placeholder result per LLM spec so downstream aggregation
        # always sees an equal number of rubric entries per harness, with
        # score=0 reflecting "couldn't be judged because the prerequisite
        # checks failed". The ``details`` field names the offending gates so
        # the skip is inspectable.
        skip_reason = "Skipped rubric judge: deterministic grader(s) failed: " + ", ".join(failed_names)
        placeholder_results = [
            GraderResult(
                grader_type=spec.type,
                name=grader_name(spec),
                passed=False,
                score=0.0,
                details=skip_reason,
            )
            for spec in llm_specs
        ]
        return deterministic_results + placeholder_results

    llm_results = list(await asyncio.gather(*[_run_one(spec) for spec in llm_specs]))
    return deterministic_results + llm_results


async def _dispatch_grader(
    spec: GraderSpec,
    task: Task,
    result: RunResult,
    judge_llm: JudgeLLM | None,
    workspace_dir: str | None,
    *,
    executor: Any | None = None,
    execution_policy: Any | None = None,
    harness_name: str = "",
    grader_env: dict[str, str] | None = None,
) -> GraderResult | None:
    """Dispatch a grader by its dataclass type."""
    from .outcome import (
        run_file_exists_grader,
        run_json_schema_grader,
        run_regex_grader,
        run_test_pass_grader,
    )
    from .rubric_judge import run_rubric_judge
    from .test_suite import run_test_suite_grader
    from .trajectory import run_trajectory_grader

    match spec:
        case TestPassGrader():
            return await run_test_pass_grader(
                spec,
                result,
                workspace_dir,
                executor=executor,
                execution_policy=execution_policy,
                harness_name=harness_name,
                grader_env=grader_env,
            )
        case FileExistsGrader():
            return run_file_exists_grader(spec, result, workspace_dir)
        case RegexGrader():
            return run_regex_grader(spec, result, workspace_dir)
        case JsonSchemaGrader():
            return run_json_schema_grader(spec, result, workspace_dir)
        case TrajectoryGrader():
            return run_trajectory_grader(spec, result)
        case TestSuiteGrader():
            return await run_test_suite_grader(
                spec,
                workspace_dir,
                executor=executor,
                execution_policy=execution_policy,
                harness_name=harness_name,
                grader_env=grader_env,
            )
        case RubricJudgeGrader():
            if judge_llm is None:
                return GraderResult(
                    grader_type="rubric_judge",
                    name="rubric_judge",
                    passed=False,
                    score=0.0,
                    details="No judge LLM configured",
                )
            return await run_rubric_judge(spec, task, result, judge_llm, workspace_dir)
        case _:
            logger.warning("Unhandled grader type: %s", spec.type)
            return None


def normalize_result_for_grading(result: RunResult) -> RunResult:
    """Strip unequivocal non-answer final_text before grading."""
    # Failed / N/A runs usually carry stderr / bootstrap logs in
    # ``final_text`` (see ``adapters.interface._make_result``), not the
    # agent's answer. Text-based graders would then score that log dump as
    # if it were a response, producing spurious passes — observed in the
    # v2 rerun on a zeroclaw provider_api_error run.
    #
    # Timeouts are different: several adapters recover a partial assistant
    # answer from the trace/session file and store that in ``final_text``.
    # Blanking timed-out runs caused deterministic false negatives in
    # coding.01/openclaw: tests passed and the partial answer clearly said
    # "Map", but regex graders saw an empty string. Preserve timed-out
    # partial answers and only scrub unequivocal non-answer statuses.
    if result.status in {"failed", "not_applicable"}:
        return replace(result, final_text="")
    return result


def expand_grader_specs(task: Task) -> list[GraderSpec]:
    """Return task graders plus any auto-injected boundary checks."""
    specs = list(task.graders)
    boundary = task.tool_boundary
    if not boundary:
        return specs

    existing_constraints: set[str] = set()
    for spec in specs:
        if isinstance(spec, TrajectoryGrader) and spec.rule.kind == "boundary_respected" and spec.rule.constraint:
            existing_constraints.add(spec.rule.constraint)

    for constraint_name in ("internet", "shell", "file_write"):
        if boundary.is_disabled(constraint_name) and constraint_name not in existing_constraints:
            specs.append(
                TrajectoryGrader(
                    rule=TrajectoryRule(
                        kind="boundary_respected",
                        constraint=constraint_name,  # type: ignore[arg-type]
                    )
                )
            )
    return specs


def grader_name(spec: GraderSpec) -> str:
    """Generate a human-readable name for a grader spec."""
    match spec:
        case TrajectoryGrader(rule=rule):
            return f"trajectory:{rule.kind}"
        case RegexGrader(pattern=pattern):
            return f"regex:{pattern[:40]}"
        case FileExistsGrader(paths=paths):
            return f"file_exists:{','.join(paths[:3])}"
        case TestPassGrader(command=cmd):
            return f"test_pass:{cmd[:40]}"
        case _:
            return spec.type


def _is_deterministic(spec: GraderSpec) -> bool:
    """Return True for graders with deterministic (non-LLM) evaluation."""
    return spec.type in {
        "test_pass",
        "file_exists",
        "regex",
        "json_schema",
        "trajectory",
        "test_suite",
    }
