"""Shared runtime and result types for the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .config.providers import ModelSpec, ProviderConfig
from .graders.specs import GraderResult, RubricDimensionResult, grader_result_from_dict

TraceEventType = Literal[
    "message",
    "tool_call_started",
    "tool_call_completed",
    "artifact_written",
    "task_completed",
    "task_failed",
]


@dataclass
class CanonicalTraceEvent:
    type: TraceEventType
    # ts contract: RFC 3339 / ISO 8601 string, UTC (``+00:00`` offset),
    # millisecond precision (3 fractional-second digits), produced by
    # ``utils.timestamps.to_canonical_ts``. Adapters MUST funnel any harness-
    # native timestamp (epoch numbers, Z-suffixed strings, naive datetimes,
    # ns-precision strings) through that helper rather than emitting raw
    # values, so the persisted trace is comparable across harnesses and
    # round-trippable through ``datetime.fromisoformat``.
    ts: str
    role: Literal["user", "assistant"] | None = None
    text: str | None = None
    tool_name: str | None = None
    input: Any | None = None
    success: bool | None = None
    output: str | None = None
    path: str | None = None
    error: str | None = None


@dataclass
class RunMetrics:
    latency_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    usage_available: bool = True
    cost_usd: float = 0.0
    cost_usd_no_cache: float | None = None
    tool_calls: int = 0
    turns: int = 0


FailureOrigin = Literal["agent", "adapter", "provider", "sandbox", "grader", "unknown"]


@dataclass
class RunResult:
    task_id: str
    harness: str
    run_id: str
    run_index: int
    model: str
    status: Literal["completed", "failed", "timed_out", "not_applicable"]
    final_text: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    trace: list[CanonicalTraceEvent] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)
    grader_results: list[GraderResult] = field(default_factory=list)
    failure_origin: FailureOrigin | None = None
    infra_error_code: str | None = None
    infra_error_details: str | None = None


@dataclass
class EvalConfig:
    model_spec: ModelSpec
    models: list[ModelSpec] = field(default_factory=list)
    harnesses: list[str] = field(default_factory=list)
    runs_per_task: int = 1
    max_concurrency: int = 1
    judge_model_spec: ModelSpec | None = None
    secondary_judge_model_spec: ModelSpec | None = None
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    task_filter: dict[str, Any] | None = None
    output_dir: str = ""
    timeout_sec: int = 1800
    # When ``runs_per_task > 1``, append a unique nonce to ``user_query`` per run so
    # providers with response-level or prompt-level caching can't return an earlier
    # run's reply for the same prompt. Default True — set False only if you want to
    # intentionally test cache behavior.
    runs_bust_cache: bool = True


def run_result_from_dict(data: dict[str, Any]) -> RunResult:
    trace = [
        CanonicalTraceEvent(
            **{key: value for key, value in event.items() if key in CanonicalTraceEvent.__dataclass_fields__}
        )
        for event in data.get("trace", [])
    ]

    grader_results = [grader_result_from_dict(grader) for grader in data.get("grader_results", [])]

    return RunResult(
        task_id=data["task_id"],
        harness=data["harness"],
        run_id=data.get("run_id", ""),
        run_index=data.get("run_index", 0),
        model=data.get("model", ""),
        status=data["status"],
        final_text=data.get("final_text", ""),
        artifacts=list(data.get("artifacts", [])),
        trace=trace,
        metrics=RunMetrics(
            **{key: value for key, value in data.get("metrics", {}).items() if key in RunMetrics.__dataclass_fields__}
        ),
        grader_results=grader_results,
        failure_origin=data.get("failure_origin"),
        infra_error_code=data.get("infra_error_code"),
        infra_error_details=data.get("infra_error_details"),
    )


def trace_event_to_dict(event: CanonicalTraceEvent) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "type": event.type,
            "ts": event.ts,
            "role": event.role,
            "text": event.text,
            "tool_name": event.tool_name,
            "input": event.input,
            "success": event.success,
            "output": event.output,
            "path": event.path,
            "error": event.error,
        }.items()
        if value is not None
    }


def grader_result_to_dict(result: GraderResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "grader_type": result.grader_type,
        "name": result.name,
        "pass": result.passed,
    }
    if result.score is not None:
        data["score"] = result.score
    if result.details is not None:
        data["details"] = result.details
    if result.dimensions is not None:
        data["dimensions"] = [_rubric_dimension_to_dict(dim) for dim in result.dimensions]
    return data


def _rubric_dimension_to_dict(dim: RubricDimensionResult) -> dict[str, Any]:
    return {
        "name": dim.name,
        "pass": dim.passed,
        "score": dim.score,
        "reason": dim.reason,
        "required": dim.required,
        "weight": dim.weight,
    }


def run_result_to_dict(result: RunResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "harness": result.harness,
        "run_id": result.run_id,
        "run_index": result.run_index,
        "model": result.model,
        "status": result.status,
        "final_text": result.final_text,
        "artifacts": list(result.artifacts),
        "trace": [trace_event_to_dict(event) for event in result.trace],
        "metrics": {
            "latency_sec": result.metrics.latency_sec,
            "input_tokens": result.metrics.input_tokens,
            "output_tokens": result.metrics.output_tokens,
            "cache_read_tokens": result.metrics.cache_read_tokens,
            "cache_write_tokens": result.metrics.cache_write_tokens,
            "total_tokens": result.metrics.total_tokens,
            "usage_available": result.metrics.usage_available,
            "cost_usd": result.metrics.cost_usd,
            "cost_usd_no_cache": result.metrics.cost_usd_no_cache,
            "tool_calls": result.metrics.tool_calls,
            "turns": result.metrics.turns,
        },
        "grader_results": [grader_result_to_dict(grader) for grader in result.grader_results],
        "failure_origin": result.failure_origin,
        "infra_error_code": result.infra_error_code,
        "infra_error_details": result.infra_error_details,
    }
