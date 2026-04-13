"""Grader dispatcher - runs all graders for a task."""

from __future__ import annotations

import logging
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
    """Run all graders for a task in two passes: hard then soft.

    Auto-injects boundary_respected trajectory checks for any disabled
    tool_boundary constraints not already covered by an explicit grader.
    """
    specs = list(task.graders)

    # Auto-inject boundary_respected checks for disabled constraints
    boundary = task.tool_boundary
    if boundary:
        existing_constraints: set[str] = set()
        for s in specs:
            if isinstance(s, TrajectoryGrader) and s.rule.kind == "boundary_respected" and s.rule.constraint:
                existing_constraints.add(s.rule.constraint)

        for constraint_name in ("internet", "shell", "file_write"):
            if boundary.is_disabled(constraint_name) and constraint_name not in existing_constraints:
                specs.append(
                    TrajectoryGrader(
                        rule=TrajectoryRule(
                            kind="boundary_respected",
                            constraint=constraint_name,  # type: ignore[arg-type]
                        ),
                    )
                )

    hard_specs = [s for s in specs if _is_hard_grader(s)]
    soft_specs = [s for s in specs if not _is_hard_grader(s)]

    results: list[GraderResult] = []

    # Pass 1: hard graders
    for spec in hard_specs:
        try:
            gr = await _dispatch_grader(
                spec,
                task,
                result,
                judge_llm,
                workspace_dir,
                results,
                executor=executor,
                execution_policy=execution_policy,
                harness_name=harness_name,
                grader_env=grader_env,
            )
            if gr is not None:
                results.append(gr)
        except Exception as exc:
            logger.error("Grader %s failed: %s", spec.type, exc, exc_info=True)
            results.append(
                GraderResult(
                    grader_type=spec.type,
                    name=_grader_name(spec),
                    passed=False,
                    score=0.0,
                    details=f"Grader error: {exc}",
                )
            )

    # Pass 2: soft graders
    for spec in soft_specs:
        try:
            gr = await _dispatch_grader(
                spec,
                task,
                result,
                judge_llm,
                workspace_dir,
                results,
                executor=executor,
                execution_policy=execution_policy,
                harness_name=harness_name,
                grader_env=grader_env,
            )
            if gr is not None:
                results.append(gr)
        except Exception as exc:
            logger.error("Grader %s failed: %s", spec.type, exc, exc_info=True)
            results.append(
                GraderResult(
                    grader_type=spec.type,
                    name=_grader_name(spec),
                    passed=False,
                    score=0.0,
                    details=f"Grader error: {exc}",
                )
            )

    return results


async def _dispatch_grader(
    spec: GraderSpec,
    task: Task,
    result: RunResult,
    judge_llm: JudgeLLM | None,
    workspace_dir: str | None,
    prior_results: list[GraderResult],
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
            return await run_rubric_judge(spec, result, judge_llm, workspace_dir, prior_results=prior_results)
        case _:
            logger.warning("Unhandled grader type: %s", spec.type)
            return None


def _grader_name(spec: GraderSpec) -> str:
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


def _is_hard_grader(spec: GraderSpec) -> bool:
    return spec.type in {
        "test_pass",
        "file_exists",
        "regex",
        "json_schema",
        "trajectory",
        "test_suite",
    }
