"""Hard objective graders: test_pass, file_exists, regex, json_schema."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from ..constants import TEST_PASS_GRADER_TIMEOUT_MS, TOOL_OUTPUT_MAX_CHARS
from ..executor import policy_for_grader
from ..types import RunResult
from ..utils.subprocess import run_subprocess
from .specs import GraderResult, GraderSpec

logger = logging.getLogger(__name__)

# Subdirectory patterns to try when looking for artifacts
_ARTIFACT_SUBDIRS = [
    "",
    "users/eval-user/workspace-eval",
    "workspace-eval",
    "workspace",
]


def _read_artifact(artifact_path: str, workspace_dir: str | None) -> str | None:
    """Read an artifact file, trying direct path first then subdirectory patterns.

    Tries the path as-is first. If not found and workspace_dir is provided,
    tries joining workspace_dir with various subdirectory prefixes.
    """
    # Try direct/absolute path
    if os.path.isabs(artifact_path) and os.path.isfile(artifact_path):
        try:
            with open(artifact_path, errors="replace") as f:
                return f.read()
        except OSError:
            pass

    if workspace_dir is None:
        return None

    # Try workspace_dir + subdirectory patterns
    for subdir in _ARTIFACT_SUBDIRS:
        candidate = (
            os.path.join(workspace_dir, subdir, artifact_path) if subdir else os.path.join(workspace_dir, artifact_path)
        )
        if os.path.isfile(candidate):
            try:
                with open(candidate, errors="replace") as f:
                    return f.read()
            except OSError:
                continue

    return None


# ─── test_pass ───


async def run_test_pass_grader(
    spec: GraderSpec,
    result: RunResult,
    workspace_dir: str | None,
    *,
    executor: Any | None = None,
    execution_policy: Any | None = None,
    harness_name: str = "",
    grader_env: dict[str, str] | None = None,
) -> GraderResult:
    """Run a shell command and check for exit code 0."""
    command = spec.command
    if not command:
        return GraderResult(
            grader_type="test_pass",
            name="test_pass",
            passed=False,
            score=0.0,
            details="No command specified",
        )

    cwd = spec.cwd or workspace_dir
    if cwd and not os.path.isdir(cwd):
        return GraderResult(
            grader_type="test_pass",
            name=f"test_pass:{command[:40]}",
            passed=False,
            score=0.0,
            details=f"Working directory does not exist: {cwd}",
        )

    if executor is not None and execution_policy is not None:
        policy = policy_for_grader(execution_policy, cwd=cwd)
        env = grader_env if grader_env is not None else dict(os.environ)
        proc_result = await executor.execute(
            harness_name or "_grader",
            policy,
            "bash",
            ["-c", command],
            env,
            timeout_ms=TEST_PASS_GRADER_TIMEOUT_MS,
        )
    else:
        proc_result = await run_subprocess(
            "bash",
            ["-c", command],
            cwd=cwd,
            timeout_ms=TEST_PASS_GRADER_TIMEOUT_MS,
        )

    passed = proc_result.exit_code == 0 and not proc_result.timed_out
    details_parts: list[str] = []
    if proc_result.stdout.strip():
        details_parts.append(f"stdout: {proc_result.stdout.strip()[:TOOL_OUTPUT_MAX_CHARS]}")
    if proc_result.stderr.strip():
        details_parts.append(f"stderr: {proc_result.stderr.strip()[:TOOL_OUTPUT_MAX_CHARS]}")
    if proc_result.timed_out:
        details_parts.append("Command timed out")
    details_parts.append(f"exit_code: {proc_result.exit_code}")

    return GraderResult(
        grader_type="test_pass",
        name=f"test_pass:{command[:40]}",
        passed=passed,
        score=1.0 if passed else 0.0,
        details="\n".join(details_parts),
    )


# ─── file_exists ───


def run_file_exists_grader(
    spec: GraderSpec,
    result: RunResult,
    workspace_dir: str | None,
) -> GraderResult:
    """Check that specified file paths exist."""
    paths = spec.paths or []
    if not paths:
        return GraderResult(
            grader_type="file_exists",
            name="file_exists",
            passed=False,
            score=0.0,
            details="No paths specified",
        )

    missing: list[str] = []
    found: list[str] = []

    for file_path in paths:
        resolved = False

        # Try absolute path
        if os.path.isabs(file_path) and os.path.exists(file_path):
            resolved = True
        elif workspace_dir:
            # Try workspace + subdirectory patterns
            for subdir in _ARTIFACT_SUBDIRS:
                candidate = (
                    os.path.join(workspace_dir, subdir, file_path) if subdir else os.path.join(workspace_dir, file_path)
                )
                if os.path.exists(candidate):
                    resolved = True
                    break

        if resolved:
            found.append(file_path)
        else:
            missing.append(file_path)

    passed = len(missing) == 0
    score = len(found) / len(paths) if paths else 0.0

    details_parts: list[str] = []
    if found:
        details_parts.append(f"Found: {', '.join(found)}")
    if missing:
        details_parts.append(f"Missing: {', '.join(missing)}")

    return GraderResult(
        grader_type="file_exists",
        name=f"file_exists:{','.join(paths[:3])}",
        passed=passed,
        score=score,
        details="\n".join(details_parts),
    )


# ─── regex ───


def run_regex_grader(
    spec: GraderSpec,
    result: RunResult,
    workspace_dir: str | None,
) -> GraderResult:
    """Regex matching on final_text or an artifact file."""
    pattern = spec.pattern
    if not pattern:
        return GraderResult(
            grader_type="regex",
            name="regex",
            passed=False,
            score=0.0,
            details="No pattern specified",
        )

    target = spec.target or "final_text"
    should_match = spec.should_match if spec.should_match is not None else True

    # Determine case sensitivity default: True for final_text, False for artifact
    if spec.case_insensitive is not None:
        case_insensitive = spec.case_insensitive
    else:
        case_insensitive = target == "final_text"

    # Get the text to search
    if target == "final_text":
        text = result.final_text or ""
    elif target == "artifact":
        artifact_path = spec.artifact_path
        if not artifact_path:
            return GraderResult(
                grader_type="regex",
                name=f"regex:{pattern[:40]}",
                passed=False,
                score=0.0,
                details="No artifact_path specified for artifact target",
            )
        text = _read_artifact(artifact_path, workspace_dir)
        if text is None:
            return GraderResult(
                grader_type="regex",
                name=f"regex:{pattern[:40]}",
                passed=False,
                score=0.0,
                details=f"Artifact not found: {artifact_path}",
            )
    else:
        return GraderResult(
            grader_type="regex",
            name=f"regex:{pattern[:40]}",
            passed=False,
            score=0.0,
            details=f"Unknown target: {target}",
        )

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        match = re.search(pattern, text, flags)
    except re.error as e:
        return GraderResult(
            grader_type="regex",
            name=f"regex:{pattern[:40]}",
            passed=False,
            score=0.0,
            details=f"Invalid regex: {e}",
        )

    found = match is not None
    passed = found == should_match

    details = (
        f"pattern=/{pattern}/ target={target} "
        f"should_match={should_match} found={found} "
        f"case_insensitive={case_insensitive}"
    )
    if match:
        details += f" matched='{match.group(0)[:200]}'"

    return GraderResult(
        grader_type="regex",
        name=f"regex:{pattern[:40]}",
        passed=passed,
        score=1.0 if passed else 0.0,
        details=details,
    )


# ─── json_schema ───


def run_json_schema_grader(
    spec: GraderSpec,
    result: RunResult,
    workspace_dir: str | None,
) -> GraderResult:
    """Validate a JSON artifact or final_text against a JSON Schema."""
    import jsonschema

    schema = spec.schema
    if not schema:
        return GraderResult(
            grader_type="json_schema",
            name="json_schema",
            passed=False,
            score=0.0,
            details="No schema specified",
        )

    # Get text to parse
    if spec.target == "artifact" and spec.artifact_path:
        text = _read_artifact(spec.artifact_path, workspace_dir)
        if text is None:
            return GraderResult(
                grader_type="json_schema",
                name="json_schema",
                passed=False,
                score=0.0,
                details=f"Artifact not found: {spec.artifact_path}",
            )
    else:
        text = result.final_text or ""

    # Parse JSON
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        from .judge_json import extract_json

        data = extract_json(text)
        if data is None:
            return GraderResult(
                grader_type="json_schema",
                name="json_schema",
                passed=False,
                score=0.0,
                details="Could not parse JSON from output",
            )

    # Validate against schema
    try:
        jsonschema.validate(data, schema)
        return GraderResult(
            grader_type="json_schema",
            name="json_schema",
            passed=True,
            score=1.0,
            details="Schema validated",
        )
    except jsonschema.ValidationError as e:
        return GraderResult(
            grader_type="json_schema",
            name="json_schema",
            passed=False,
            score=0.0,
            details=f"Validation failed: {e.message}",
        )
    except jsonschema.SchemaError as e:
        return GraderResult(
            grader_type="json_schema",
            name="json_schema",
            passed=False,
            score=0.0,
            details=f"Invalid schema: {e.message}",
        )
