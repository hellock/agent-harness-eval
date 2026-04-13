"""Error classification and failure origin detection."""

from __future__ import annotations

import re

from ..types import FailureOrigin, RunResult


def detect_failure_origin_from_error(
    error_text: str,
) -> dict[str, str | None]:
    """Classify an error string into a failure origin.

    Returns dict with keys: failure_origin, infra_error_code
    """
    text = error_text.lower()

    # Provider-side errors
    if any(
        pattern in text
        for pattern in [
            "validationexception",
            "invalid beta flag",
            "api error",
            "unauthorized",
            "authentication",
            "invalid config",
            "provider error",
            "provider api",
            "upstream provider",
            "all providers/models failed",
            "system memory overloaded",
            "new_api_error",
            "bedrock",
        ]
    ) or re.search(r"\b(401|403|429|5\d\d)\b", text):
        return {"failure_origin": "provider", "infra_error_code": "provider_api_error"}

    if any(
        pattern in text
        for pattern in [
            "eacces",
            "permission denied",
            "sandbox",
            "docker",
            "read-only file system",
        ]
    ):
        return {"failure_origin": "sandbox", "infra_error_code": "sandbox_permission_error"}

    if any(
        pattern in text
        for pattern in [
            "no result event",
            "failed to parse",
            "parse",
            "session file not found",
            "adapter",
        ]
    ):
        return {"failure_origin": "adapter", "infra_error_code": "adapter_output_error"}

    return {"failure_origin": "unknown", "infra_error_code": None}


def infer_failure_origin(result: RunResult) -> FailureOrigin:
    if result.failure_origin:
        return result.failure_origin

    if result.status == "completed" and any(not g.passed for g in result.grader_results):
        failed_graders = [g for g in result.grader_results if not g.passed]
        if any("boundary_respected" in g.name for g in failed_graders):
            return "agent"
        if any(g.grader_type in ("trajectory", "rubric_judge") for g in failed_graders):
            return "agent"
        return "grader"

    error_text = "\n".join(
        filter(
            None,
            [
                result.infra_error_details,
                *(e.error for e in result.trace if e.type == "task_failed" and e.error),
                result.final_text,
            ],
        )
    )
    return detect_failure_origin_from_error(error_text)["failure_origin"]


def format_failure_origin(origin: FailureOrigin) -> str:
    labels: dict[str, str] = {
        "agent": "Agent",
        "adapter": "Adapter",
        "provider": "Provider",
        "sandbox": "Sandbox",
        "grader": "Grader",
        "unknown": "Unknown",
    }
    return labels.get(origin, "Unknown")
