"""Tests for the adapter base class (HarnessAdapter._make_result).

Covers framework-level invariants that every adapter inherits — especially
the post-mortem fields (``failure_origin``, ``infra_error_code``,
``infra_error_details``) that downstream metrics aggregation depends on.
"""

from __future__ import annotations

from agent_harness_eval.adapters.interface import HarnessAdapter
from agent_harness_eval.task import Task
from agent_harness_eval.types import CanonicalTraceEvent, RunMetrics


class _StubAdapter(HarnessAdapter):
    """Minimal HarnessAdapter that lets us call ``_make_result`` directly.

    We only need ``name`` and the ``_make_result`` method itself; the other
    abstract methods are stubbed out so the class is instantiable.
    """

    name = "stub"

    def __init__(self) -> None:  # bypass HarnessAdapter.__init__ deps
        pass

    def prepare(self, task, run_id):  # type: ignore[override]
        raise NotImplementedError

    async def run(self, prepared, model):  # type: ignore[override]
        raise NotImplementedError

    def cleanup(self, prepared):  # type: ignore[override]
        raise NotImplementedError


def _task(timeout_sec: int = 30) -> Task:
    return Task(
        id="stub.test",
        category="smoke",
        description="stub task",
        user_query="hi",
        workspace_files=[],
        timeout_sec=timeout_sec,
    )


def test_make_result_timed_out_stamps_default_failure_origin_and_ec() -> None:
    """Regression: pre-fix, every ``timed_out`` result returned with
    ``failure_origin=None`` and ``infra_error_code=None``, so downstream
    ``infra_failure_count`` reported 0 even when 20% of runs hit the timeout
    (observed in the rerun-selected-no-codex-docker-v2 audit). The base class
    must default-stamp these fields so adapters don't each have to remember.
    """
    adapter = _StubAdapter()
    result = adapter._make_result(
        _task(timeout_sec=42),
        "relay:gpt-5.4",
        "timed_out",
        "",
        [],
        RunMetrics(latency_sec=42),
    )
    assert result.status == "timed_out"
    assert result.failure_origin == "adapter"
    assert result.infra_error_code == "harness_timeout"
    assert result.infra_error_details is not None
    assert "42s" in result.infra_error_details


def test_make_result_timed_out_respects_explicit_overrides() -> None:
    """Adapters that have a more specific cause (e.g. nanobot's session
    recovery failure) must be able to override the default fields without
    being silently clobbered.
    """
    adapter = _StubAdapter()
    result = adapter._make_result(
        _task(),
        "relay:gpt-5.4",
        "timed_out",
        "",
        [],
        RunMetrics(latency_sec=30),
        failure_origin="adapter",
        infra_error_code="session_recovery_failed",
        infra_error_details="session.jsonl recovery failed (ValueError): bad json",
    )
    assert result.failure_origin == "adapter"
    assert result.infra_error_code == "session_recovery_failed"
    assert "session.jsonl recovery failed" in (result.infra_error_details or "")


def test_make_result_failed_keeps_existing_behavior() -> None:
    """The pre-existing failed path (final_text → infra_error_details) must
    keep working — the timeout fix is additive, not a behavior change for
    failed runs.
    """
    adapter = _StubAdapter()
    result = adapter._make_result(
        _task(),
        "relay:gpt-5.4",
        "failed",
        "stderr line one\nstderr line two\n",
        [
            CanonicalTraceEvent(
                type="task_failed",
                error="boom",
                ts="2026-04-15T00:00:00.000+00:00",
            )
        ],
        RunMetrics(latency_sec=1.2),
        failure_origin="adapter",
        infra_error_code="adapter_empty_output",
    )
    assert result.failure_origin == "adapter"
    assert result.infra_error_code == "adapter_empty_output"
    assert result.infra_error_details is not None
    assert "stderr line one" in result.infra_error_details


def test_make_result_completed_strips_infra_fields() -> None:
    """A completed run must not carry post-mortem fields even if a caller
    accidentally passes them — guarded so report aggregations don't double-count.
    """
    adapter = _StubAdapter()
    result = adapter._make_result(
        _task(),
        "relay:gpt-5.4",
        "completed",
        "the answer",
        [],
        RunMetrics(latency_sec=2.5),
        failure_origin="adapter",
        infra_error_code="adapter_empty_output",
        infra_error_details="leaked",
    )
    assert result.failure_origin is None
    assert result.infra_error_code is None
    assert result.infra_error_details is None
