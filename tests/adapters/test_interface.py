"""Tests for the adapter base class (HarnessAdapter._make_result).

Covers framework-level invariants that every adapter inherits — especially
the post-mortem fields (``failure_origin``, ``infra_error_code``,
``infra_error_details``) that downstream metrics aggregation depends on.
"""

from __future__ import annotations

from agent_harness_eval.adapters.interface import (
    HarnessAdapter,
    SubprocessFailure,
    detect_empty_output_silent_failure,
    detect_subprocess_failure,
)
from agent_harness_eval.task import Task
from agent_harness_eval.types import CanonicalTraceEvent, RunMetrics
from agent_harness_eval.utils.subprocess import SubprocessResult


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


# --- detect_subprocess_failure gating ---------------------------------------


def _sp_result(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = 0,
    timed_out: bool = False,
) -> SubprocessResult:
    return SubprocessResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
    )


def test_detect_subprocess_failure_returns_none_on_timeout() -> None:
    """Timeout handling belongs to the caller; this helper only classifies
    non-zero-exit failures. Without this gate, every timed-out run would
    also get wrapped as a SubprocessFailure and misclassified."""
    result = _sp_result(timed_out=True, exit_code=137)
    assert detect_subprocess_failure(result, command_label="Codex") is None


def test_detect_subprocess_failure_returns_none_on_clean_exit() -> None:
    """Exit 0 and None both mean 'no subprocess-level failure' — the
    adapter's parse-path will decide whether the output is usable."""
    assert detect_subprocess_failure(_sp_result(exit_code=0), command_label="X") is None
    assert detect_subprocess_failure(_sp_result(exit_code=None), command_label="X") is None


def test_detect_subprocess_failure_wraps_nonzero_exit_with_classification() -> None:
    """Non-zero exit + stderr with a known classifier keyword yields a
    SubprocessFailure with the wrapper error message AND the
    failure_origin/infra_error_code from detect_failure_origin_from_error.
    """
    result = _sp_result(
        exit_code=1,
        stderr="HTTP 429 Too Many Requests from provider",
    )
    failure = detect_subprocess_failure(result, command_label="ClaudeCode")
    assert isinstance(failure, SubprocessFailure)
    assert "ClaudeCode exited with code 1" in failure.error
    assert "stderr:" in failure.error
    assert "429" in failure.error
    assert failure.failure_origin == "provider"
    assert failure.infra_error_code == "provider_api_error"


def test_detect_subprocess_failure_uses_stdout_when_stderr_empty() -> None:
    """Some harnesses emit diagnostics to stdout. Keep both tails in the
    wrapped error so the classifier has context."""
    result = _sp_result(
        exit_code=2,
        stdout="Failed to parse JSONL",
    )
    failure = detect_subprocess_failure(result, command_label="Codex")
    assert failure is not None
    assert "stdout:" in failure.error
    # Hits the adapter branch because of "parse"
    assert failure.failure_origin == "adapter"
    assert failure.infra_error_code == "adapter_output_error"


def test_detect_subprocess_failure_unknown_when_no_keywords() -> None:
    """Non-zero exit with unclassifiable stderr still returns a
    SubprocessFailure — but with unknown origin, not None. Callers rely on
    getting a non-None result whenever exit code is non-zero so they know
    the subprocess path was the failure source.
    """
    result = _sp_result(exit_code=1, stderr="core dumped at 0x1337")
    failure = detect_subprocess_failure(result, command_label="Zeroclaw")
    assert failure is not None
    assert failure.failure_origin == "unknown"
    assert failure.infra_error_code is None


def test_detect_subprocess_failure_truncates_long_error() -> None:
    """Error is capped at 800 chars so a pathological stderr dump doesn't
    overflow downstream fields (infra_error_details, trace event .error).
    """
    result = _sp_result(exit_code=1, stderr="x" * 10_000)
    failure = detect_subprocess_failure(result, command_label="Label")
    assert failure is not None
    assert len(failure.error) <= 800


# --- detect_empty_output_silent_failure ---------------------------------------


def test_empty_output_guard_classifies_bare_run_as_failure() -> None:
    """Run with no trace events and no final_text = silent failure, not
    completion. v1 openclaw regression that now applies universally."""
    failure = detect_empty_output_silent_failure(
        trace=[],
        final_text="",
        command_label="Codex",
    )
    assert failure is not None
    assert failure.failure_origin == "adapter"
    assert failure.infra_error_code == "adapter_empty_output"
    assert "Codex produced no output" in failure.error


def test_empty_output_guard_lets_run_with_final_text_pass() -> None:
    """A non-empty final_text counts as content — don't flip to failed."""
    failure = detect_empty_output_silent_failure(
        trace=[],
        final_text="hello",
        command_label="Hermes",
    )
    assert failure is None


def test_empty_output_guard_lets_run_with_message_event_pass() -> None:
    """A trace containing a real message event counts as content."""
    failure = detect_empty_output_silent_failure(
        trace=[
            CanonicalTraceEvent(
                type="message",
                role="assistant",
                text="ok",
                ts="2026-01-01T00:00:00.000+00:00",
            )
        ],
        final_text="",
        command_label="ZeroClaw",
    )
    assert failure is None


def test_empty_output_guard_lets_run_with_tool_call_pass() -> None:
    """A tool call started counts as content even without assistant message —
    agent did something, even if it crashed before finishing the reply."""
    failure = detect_empty_output_silent_failure(
        trace=[
            CanonicalTraceEvent(
                type="tool_call_started",
                tool_name="read_file",
                ts="2026-01-01T00:00:00.000+00:00",
            )
        ],
        final_text="",
        command_label="OpenClaw",
    )
    assert failure is None


def test_empty_output_guard_ignores_only_task_completed() -> None:
    """Synthesized task_completed alone does NOT count as content — it's
    just the terminator the adapter appends. This is the realistic failure
    mode: subprocess exits 0, parser finds no real events, adapter adds a
    bare task_completed, and we must still classify as failed."""
    failure = detect_empty_output_silent_failure(
        trace=[
            CanonicalTraceEvent(
                type="task_completed",
                ts="2026-01-01T00:00:00.000+00:00",
            )
        ],
        final_text="",
        command_label="OpenClaw",
    )
    assert failure is not None
    assert failure.infra_error_code == "adapter_empty_output"


def test_empty_output_guard_keeps_explicit_task_failure() -> None:
    """A parsed task_failed event is already meaningful output and must not be
    reclassified as an adapter-empty-output failure."""
    failure = detect_empty_output_silent_failure(
        trace=[
            CanonicalTraceEvent(
                type="task_failed",
                error="provider returned 401",
                ts="2026-01-01T00:00:00.000+00:00",
            )
        ],
        final_text="",
        command_label="Codex",
    )
    assert failure is None


def test_empty_output_guard_lets_run_with_tool_completed_pass() -> None:
    """P2 review finding: openclaw's session parser can emit standalone
    ``tool_call_completed`` events without a matching ``tool_call_started``
    (e.g. when only the toolResult line was captured). A ``tool_call_completed``
    is still evidence the agent did real work — must not be classified as
    adapter_empty_output."""
    failure = detect_empty_output_silent_failure(
        trace=[
            CanonicalTraceEvent(
                type="tool_call_completed",
                tool_name="read_file",
                success=True,
                output="file contents",
                ts="2026-01-01T00:00:00.000+00:00",
            )
        ],
        final_text="",
        command_label="OpenClaw",
    )
    assert failure is None


def test_empty_output_guard_includes_stderr_tail_when_provided() -> None:
    """Stderr context is useful for post-mortem — include it in the error."""
    failure = detect_empty_output_silent_failure(
        trace=[],
        final_text="",
        command_label="ZeroClaw",
        stderr="connection refused after 3 retries",
    )
    assert failure is not None
    assert "connection refused" in failure.error
