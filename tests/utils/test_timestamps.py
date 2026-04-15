"""Tests for ``utils.timestamps`` — canonical ts helpers used by every adapter."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_harness_eval.types import CanonicalTraceEvent
from agent_harness_eval.utils.timestamps import task_completion_ts, to_canonical_ts


def test_task_completion_ts_empty_trace_returns_now() -> None:
    """Empty trace has no ts to be monotonic w.r.t. — just return now()."""
    result = task_completion_ts([])
    parsed = datetime.fromisoformat(result)
    # Result is a valid canonical ts (ms precision, tz-aware).
    assert parsed.tzinfo is not None
    assert parsed.microsecond % 1000 == 0
    # And it's close to now — within 60s is plenty generous.
    now = datetime.now(UTC)
    assert abs((now - parsed).total_seconds()) < 60


def test_task_completion_ts_past_trace_returns_now() -> None:
    """When the trace's last event is in the past, now() wins — no bump
    needed, because now() is already strictly greater.
    """
    trace = [CanonicalTraceEvent(type="message", ts="2020-01-01T00:00:00.000+00:00")]
    result = task_completion_ts(trace)
    # Strictly greater than the trace event's ts.
    assert result > trace[0].ts


def test_task_completion_ts_future_trace_bumps_by_one_ms() -> None:
    """When the trace's last event is somehow in the future (upstream clock
    drift, or claude-code's monotonic +1ms bump pushing past now()), the
    helper must still produce a ts strictly greater than it — specifically
    ``last_ts + 1ms`` at canonical granularity.
    """
    future_ts = "2099-01-01T00:00:00.500+00:00"
    trace = [CanonicalTraceEvent(type="message", ts=future_ts)]
    result = task_completion_ts(trace)
    assert result == "2099-01-01T00:00:00.501+00:00"


def test_task_completion_ts_uses_max_not_last_index() -> None:
    """Defensive: if the trace has out-of-order ts (not strictly monotonic by
    index), the helper must bump past the MAX ts, not the last-indexed one.
    Otherwise task_completed could still land between two prior events.
    """
    trace = [
        CanonicalTraceEvent(type="message", ts="2099-01-01T00:00:00.100+00:00"),
        CanonicalTraceEvent(type="message", ts="2099-01-01T00:00:00.050+00:00"),
    ]
    result = task_completion_ts(trace)
    # +1ms past the TRUE max (.100), not the last-indexed (.050).
    assert result == "2099-01-01T00:00:00.101+00:00"


def test_task_completion_ts_accepts_dict_shaped_events() -> None:
    """Regression-guard: callers sometimes pass a dict-list instead of a
    dataclass-list (e.g. in tests or deserialization paths). The helper
    reads ts from both shapes.
    """
    trace = [{"type": "message", "ts": "2099-01-01T00:00:00.500+00:00"}]
    result = task_completion_ts(trace)
    assert result == "2099-01-01T00:00:00.501+00:00"


def test_task_completion_ts_strictly_after_ts_ms_precision() -> None:
    """Monotonicity property: for any trace, the returned ts must be
    strictly greater than every ts already in the trace.
    """
    trace = [
        CanonicalTraceEvent(type="message", ts=to_canonical_ts()),
        CanonicalTraceEvent(
            type="tool_call_started",
            ts=to_canonical_ts(datetime(2099, 12, 31, 23, 59, 59, 999000, tzinfo=UTC)),
        ),
    ]
    result = task_completion_ts(trace)
    for event in trace:
        assert result > event.ts, f"{result} must be > {event.ts}"
