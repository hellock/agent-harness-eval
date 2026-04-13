"""LLM rubric judge grader."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..types import (
    CanonicalTraceEvent,
    RunResult,
)
from .judge_json import extract_json
from .specs import (
    GraderResult,
    GraderSpec,
    RubricDimensionDef,
    RubricDimensionResult,
)

if TYPE_CHECKING:
    from .interface import JudgeLLM

logger = logging.getLogger(__name__)

# Subdirectory patterns for reading workspace snapshots (same as outcome.py)
_SNAPSHOT_SUBDIRS = [
    "",
    "users/eval-user/workspace-eval",
    "workspace-eval",
    "workspace",
]


def _read_snapshot(path: str, workspace_dir: str | None) -> str | None:
    """Read a workspace snapshot file, trying subdirectory patterns."""
    if os.path.isabs(path) and os.path.isfile(path):
        try:
            with open(path, errors="replace") as f:
                return f.read()
        except OSError:
            pass

    if workspace_dir is None:
        return None

    for subdir in _SNAPSHOT_SUBDIRS:
        candidate = os.path.join(workspace_dir, subdir, path) if subdir else os.path.join(workspace_dir, path)
        if os.path.isfile(candidate):
            try:
                with open(candidate, errors="replace") as f:
                    return f.read()
            except OSError:
                continue

    return None


def _normalize_dimensions(raw: list[str | RubricDimensionDef] | None) -> list[RubricDimensionDef]:
    """Normalize dimension specs.

    Supports plain strings (optionally with "(bonus)" suffix to mark as
    non-required with weight 0.5) and full RubricDimensionDef objects.
    """
    if not raw:
        return []

    dims: list[RubricDimensionDef] = []
    for item in raw:
        if isinstance(item, RubricDimensionDef):
            dims.append(item)
        elif isinstance(item, str):
            name = item.strip()
            required = True
            weight = 1.0
            if name.lower().endswith("(bonus)"):
                name = name[: -len("(bonus)")].strip()
                required = False
                weight = 0.5
            dims.append(RubricDimensionDef(name=name, required=required, weight=weight))
        elif isinstance(item, dict):
            dims.append(
                RubricDimensionDef(
                    name=item.get("name", "unknown"),
                    required=item.get("required", True),
                    weight=item.get("weight", 1.0),
                )
            )
    return dims


def _build_tool_summary(trace: Sequence[CanonicalTraceEvent]) -> str:
    """Build a summary of tool calls for the judge prompt."""
    tool_calls = [e for e in trace if e.type == "tool_call_started"]
    if not tool_calls:
        return "No tool calls recorded."

    lines: list[str] = []
    for i, tc in enumerate(tool_calls[:50], 1):  # Cap at 50
        name = tc.tool_name or "unknown"
        inp = tc.input
        if isinstance(inp, dict):
            # Truncate large inputs
            inp_str = json.dumps(inp, ensure_ascii=False)
            if len(inp_str) > 500:
                inp_str = inp_str[:500] + "..."
        elif isinstance(inp, str):
            inp_str = inp[:500] + ("..." if len(inp) > 500 else "")
        else:
            inp_str = str(inp)[:500] if inp is not None else ""
        lines.append(f"{i}. {name}({inp_str})")

    if len(tool_calls) > 50:
        lines.append(f"... and {len(tool_calls) - 50} more tool calls")

    return "\n".join(lines)


def _build_tool_sequence(trace: Sequence[CanonicalTraceEvent]) -> str:
    """Build a compact tool call sequence string."""
    tool_calls = [e for e in trace if e.type == "tool_call_started"]
    names = [tc.tool_name or "?" for tc in tool_calls[:100]]
    return " -> ".join(names) if names else "none"


async def run_rubric_judge(
    spec: GraderSpec,
    result: RunResult,
    judge_llm: JudgeLLM,
    workspace_dir: str | None,
) -> GraderResult:
    """Run the LLM rubric judge grader.

    Builds a structured prompt with:
    - The rubric text
    - Tool call summary and sequence
    - Workspace snapshots (ground truth)
    - Dimension definitions

    Parses per-dimension or flat (single pass/fail) responses.
    """
    rubric = spec.rubric or ""
    dimensions = _normalize_dimensions(spec.dimensions)

    # Build prompt components
    tool_summary = _build_tool_summary(result.trace)
    tool_sequence = _build_tool_sequence(result.trace)

    # Read workspace snapshots for ground truth
    snapshots_str = ""
    if spec.snapshot_paths and workspace_dir:
        snapshot_parts: list[str] = []
        for sp in spec.snapshot_paths:
            content = _read_snapshot(sp, workspace_dir)
            if content is not None:
                # Truncate very large snapshots
                if len(content) > 10_000:
                    content = content[:10_000] + "\n... [truncated]"
                snapshot_parts.append(f"=== {sp} ===\n{content}")
            else:
                snapshot_parts.append(f"=== {sp} === [not found]")
        snapshots_str = "\n\n".join(snapshot_parts)

    # Build the judge prompt
    if dimensions:
        dim_desc = "\n".join(f"- {d.name} (required={d.required}, weight={d.weight})" for d in dimensions)
        response_format = (
            "Respond with a JSON object with this structure:\n"
            "{\n"
            '  "dimensions": {\n'
            '    "<dimension_name>": { "pass": true/false, "score": 0.0-1.0, "reason": "..." },\n'
            "    ...\n"
            "  }\n"
            "}"
        )
    else:
        dim_desc = "No specific dimensions. Evaluate holistically."
        response_format = 'Respond with a JSON object:\n{ "pass": true/false, "score": 0.0-1.0, "reason": "..." }'

    prompt = f"""You are an expert judge evaluating an AI agent's task completion.

## Rubric
{rubric}

## Dimensions
{dim_desc}

## Agent's Final Response
{result.final_text[:5000] if result.final_text else "(empty)"}

## Tool Call Sequence
{tool_sequence}

## Tool Call Details
{tool_summary}
"""

    if snapshots_str:
        prompt += f"""
## Ground Truth / Workspace Snapshots
{snapshots_str}
"""

    prompt += f"""
## Instructions
Evaluate whether the agent successfully completed the task according to the rubric.
Be strict but fair. Consider both the final output and the approach taken.

{response_format}
"""

    # Call judge LLM
    try:
        response_text = await judge_llm.generate(prompt)
    except Exception as exc:
        logger.error("Judge LLM call failed: %s", exc)
        return GraderResult(
            grader_type="rubric_judge",
            name="rubric_judge",
            passed=False,
            score=0.0,
            details=f"Judge LLM error: {exc}",
        )

    # Parse response
    parsed = extract_json(response_text)
    if parsed is None:
        return GraderResult(
            grader_type="rubric_judge",
            name="rubric_judge",
            passed=False,
            score=0.0,
            details=f"Could not parse judge response as JSON. Raw: {response_text[:500]}",
        )

    # Per-dimension response
    if dimensions and "dimensions" in parsed:
        dim_results: list[RubricDimensionResult] = []
        raw_dims = parsed["dimensions"]

        for dim_def in dimensions:
            dim_data = raw_dims.get(dim_def.name, {})
            if not isinstance(dim_data, dict):
                dim_data = {}

            dim_pass = bool(dim_data.get("pass", False))
            dim_score = float(dim_data.get("score", 1.0 if dim_pass else 0.0))
            dim_reason = dim_data.get("reason", "")

            dim_results.append(
                RubricDimensionResult(
                    name=dim_def.name,
                    passed=dim_pass,
                    score=dim_score,
                    reason=str(dim_reason) if dim_reason else None,
                    required=dim_def.required,
                    weight=dim_def.weight,
                )
            )

        # Overall pass: all required dimensions must pass
        required_dims = [d for d in dim_results if d.required]
        all_required_pass = all(d.passed for d in required_dims) if required_dims else True

        # Weighted score
        total_weight = sum(d.weight for d in dim_results) or 1.0
        weighted_score = sum(d.score * d.weight for d in dim_results) / total_weight

        return GraderResult(
            grader_type="rubric_judge",
            name="rubric_judge",
            passed=all_required_pass,
            score=weighted_score,
            details=response_text[:2000],
            dimensions=dim_results,
        )

    # Flat response (no dimensions)
    flat_pass = bool(parsed.get("pass", False))
    flat_score = float(parsed.get("score", 1.0 if flat_pass else 0.0))
    flat_reason = parsed.get("reason", "")

    return GraderResult(
        grader_type="rubric_judge",
        name="rubric_judge",
        passed=flat_pass,
        score=flat_score,
        details=str(flat_reason)[:2000] if flat_reason else response_text[:2000],
    )
