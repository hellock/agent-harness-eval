"""Canonical timestamp utilities for CanonicalTraceEvent.ts.

The canonical contract: every ``ts`` written into a CanonicalTraceEvent (and
therefore into ``runs.jsonl``) is an RFC 3339 / ISO 8601 string in UTC with
``+00:00`` offset and **millisecond precision** (3 fractional-second digits)::

    2026-04-15T09:49:54.401+00:00

Why milliseconds and not microseconds:
  * Every harness source we ingest (claude-code stream, hermes session,
    openclaw payloads, nanobot session.json, codex JSONL) emits ms-or-coarser
    timestamps natively. Only zeroclaw goes finer (nanoseconds), and the
    extra precision is information we never use — events are reasoned about
    at human / network scale (10s of ms apart at minimum).
  * The single sub-ms dependency was claude_code's monotonic-bump fallback
    for events that share a parent message timestamp; bumping by 1ms instead
    of 1µs is fine — a tight burst of N synthetic events drifts forward by
    N ms, which is still far below the gap to the next real event.
  * 3 fewer characters per ts x thousands of events x hundreds of runs is
    real bytes saved in the persisted reports, and ms reads naturally to
    humans skimming a trace.

Use ``to_canonical_ts(raw)`` to produce a canonical ts from anything an
adapter has lying around (None, an epoch number, a parsed ``datetime``, an
ISO string with any tz suffix or none, garbage). Use it everywhere you would
have written ``datetime.now(UTC).isoformat()``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

CANONICAL_TIMESPEC = "milliseconds"


def to_canonical_ts(raw: Any = None) -> str:
    """Return a canonical millisecond-precision UTC ISO timestamp.

    Accepts:
      * ``None`` (or omitted) — returns now().
      * ``datetime`` — converted to UTC; naive inputs assumed UTC.
      * ``int`` / ``float`` — epoch in **seconds** (≤ 1e12) or
        **milliseconds** (> 1e12), heuristic on magnitude.
      * ``str`` — any ``datetime.fromisoformat``-parseable form (Python 3.11+
        accepts the ``Z`` suffix natively); naive strings are assumed UTC;
        sub-millisecond fractions are silently truncated.
      * Anything else / unparseable — falls back to now().
    """
    if raw is None:
        return _format(datetime.now(UTC))
    if isinstance(raw, datetime):
        dt = raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
        return _format(dt.astimezone(UTC))
    if isinstance(raw, bool):
        # bool is a subclass of int; explicitly treat as garbage.
        return _format(datetime.now(UTC))
    if isinstance(raw, (int, float)):
        seconds = raw / 1000 if raw > 1e12 else raw
        try:
            return _format(datetime.fromtimestamp(seconds, tz=UTC))
        except (OverflowError, OSError, ValueError):
            return _format(datetime.now(UTC))
    if isinstance(raw, str) and raw:
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return _format(datetime.now(UTC))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return _format(parsed.astimezone(UTC))
    return _format(datetime.now(UTC))


def task_completion_ts(trace: Iterable[Any]) -> str:
    """Return a canonical ts for a trailing ``task_completed`` / ``task_failed``
    event that is strictly after every existing ts in ``trace``.

    ``task_completed`` is stamped with local parse time (``now()``), but
    trace event ts often come from upstream harness clocks (Claude CLI
    stream, zeroclaw runtime_trace). Without this guard, ``task_completed``
    occasionally appears BEFORE the final trace event — observed in v2,
    where 4/4 claude-code completed runs had ``task_completed`` 6-20 ms
    earlier than the last message.

    Call this when appending ``task_completed`` (or ``task_failed`` for
    mid-trace failures where a real last_ts exists) so the final event's ts
    is strictly greater than every preceding ts.
    """
    now_ts = to_canonical_ts()
    last_ts = None
    for event in trace:
        ev_ts = getattr(event, "ts", None) or (event.get("ts") if isinstance(event, dict) else None)
        if ev_ts and (last_ts is None or ev_ts > last_ts):
            last_ts = ev_ts
    if last_ts is None or now_ts > last_ts:
        return now_ts
    # now() falls on or before the trace's high-water mark — bump past it
    # by one ms (canonical granularity, matching claude_code's monotonic
    # bump used for parallel in-stream events).
    try:
        parsed = datetime.fromisoformat(last_ts)
    except ValueError:
        return now_ts
    return to_canonical_ts(parsed + timedelta(milliseconds=1))


def _format(dt: datetime) -> str:
    return dt.isoformat(timespec=CANONICAL_TIMESPEC)
