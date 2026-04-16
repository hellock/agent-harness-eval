"""Claude Code adapter.

Uses `claude --bare --print --output-format stream-json` to run tasks.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

from ..config.providers import parse_model_spec
from ..constants import STDERR_PREVIEW_MAX_CHARS, TOOL_OUTPUT_MAX_CHARS
from ..executor import attach_run_layout_mounts, filter_env, policy_from_task
from ..task import Task
from ..types import CanonicalTraceEvent, RunMetrics, RunResult
from ..utils.conversation import format_task_message
from ..utils.cost import calculate_cost
from ..utils.failure_origin import detect_failure_origin_from_error
from ..utils.subprocess import (
    SubprocessResult,
    _install_subprocess_cleanup_handlers,
    _kill_process_tree,
    _preexec_new_pgroup,
    _register_active_subprocess,
    _terminate_process_tree,
    _unregister_active_subprocess,
)
from ..utils.timestamps import task_completion_ts, to_canonical_ts
from ..utils.workspace import create_run_layout, remove_workspace
from . import register_adapter
from .interface import (
    HarnessAdapter,
    PreparedRun,
    _write_subprocess_debug_artifacts,
    detect_subprocess_failure,
)


@register_adapter
class ClaudeCodeAdapter(HarnessAdapter):
    name = "claude-code"
    cli_binary = "claude"
    managed_docker_image = True
    emits_paired_trace_events = True

    required_env_vars: ClassVar[list[list[str]]] = [["ANTHROPIC_API_KEY"]]
    supported_api_formats: ClassVar[list[str]] = ["anthropic"]

    def prepare(self, task: Task, run_id: str) -> PreparedRun:
        layout = create_run_layout(run_id, workspace_files=task.workspace_files)
        workspace_dir = layout.workspace_dir
        execution_policy = policy_from_task(task, workspace_dir, task.timeout_sec)
        attach_run_layout_mounts(execution_policy, layout)

        # Create isolated runtime directories (matches TS adapter)
        runtime_dir = layout.state_dir
        config_dir = os.path.join(runtime_dir, "config")
        home_dir = os.path.join(runtime_dir, "home")
        for d in [config_dir, home_dir]:
            os.makedirs(d, exist_ok=True)

        # Write minimal config so Claude Code doesn't require login
        import json as _json

        with open(os.path.join(config_dir, ".claude.json"), "w") as f:
            _json.dump({"firstStartTime": datetime.now(UTC).isoformat()}, f)

        session_id = str(uuid.uuid4())
        return PreparedRun(
            task=task,
            layout=layout,
            env={
                "_EVAL_SESSION_ID": session_id,
                "CLAUDE_CONFIG_DIR": config_dir,
                "_RUNTIME_DIR": runtime_dir,
                "_HOME_DIR": home_dir,
            },
            execution_policy=execution_policy,
        )

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        task = prepared.task
        execution_policy = prepared.execution_policy
        model_spec = parse_model_spec(model)
        start_time = asyncio.get_running_loop().time()
        timeout_ms = task.timeout_sec * 1000

        message_text = format_task_message(task)

        # Resolve provider via harness override or model spec
        provider = self.resolve_provider(model_spec)
        provider_env: dict[str, str] = {}
        if provider.api_key:
            provider_env["ANTHROPIC_API_KEY"] = provider.api_key
        if provider.base_url:
            provider_env["ANTHROPIC_BASE_URL"] = provider.base_url

        inner_env = filter_env(
            self.runtime_config.subprocess_env,
            {
                **prepared.env,
                **provider_env,
                "HOME": prepared.env.get("_HOME_DIR", ""),
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            },
            [
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
                "CLAUDE_CONFIG_DIR",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
                "HOME",
            ],
        )
        claude_bin = self.resolve_binary()
        session_id = prepared.env.get("_EVAL_SESSION_ID", str(uuid.uuid4()))
        inner_args = [
            "--bare",
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            model_spec.model,
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--session-id",
            session_id,
            "--max-turns",
            "100",
            message_text,
        ]

        wrapped = self.executor.wrap_command(
            self.name,
            execution_policy,
            claude_bin,
            inner_args,
            inner_env,
        )
        helper_result = await _run_claude_streaming(
            wrapped.command,
            wrapped.args,
            cwd=wrapped.cwd,
            env=wrapped.env,
            timeout_ms=timeout_ms,
        )
        result = SubprocessResult(
            stdout=helper_result["stdout"],
            stderr=helper_result["stderr"],
            exit_code=helper_result["exit_code"],
            timed_out=helper_result["timed_out"],
        )

        latency_sec = asyncio.get_running_loop().time() - start_time

        if result.timed_out:
            events = _parse_stream_json(result.stdout or "")
            trace = _events_to_trace(events)
            final_text = _extract_final_text(events)
            usage = _aggregate_usage(events)
            tool_calls = len([e for e in trace if e.type == "tool_call_started"])
            turns = usage["turns"] or (1 if events else 0)
            artifacts = _write_subprocess_debug_artifacts(
                prepared,
                stem="claude-code-timeout",
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
            return self._make_result(
                task,
                model,
                "timed_out",
                final_text,
                trace,
                RunMetrics(
                    latency_sec=latency_sec,
                    input_tokens=usage["input"],
                    output_tokens=usage["output"],
                    cache_read_tokens=usage["cache_read"],
                    cache_write_tokens=usage["cache_write"],
                    total_tokens=usage["total"],
                    cost_usd=calculate_cost(
                        model,
                        usage["input"],
                        usage["output"],
                        usage["cache_read"],
                        usage["cache_write"],
                        pricing=self.pricing_override(),
                    ),
                    tool_calls=tool_calls,
                    turns=turns,
                ),
                artifacts=artifacts,
            )

        subprocess_failure = detect_subprocess_failure(result, command_label="Claude Code")
        if subprocess_failure:
            artifacts = _write_subprocess_debug_artifacts(
                prepared,
                stem="claude-code-failed",
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
            return self._make_result(
                task,
                model,
                "failed",
                result.stdout or "",
                [
                    CanonicalTraceEvent(
                        type="task_failed",
                        error=subprocess_failure.error,
                        ts=to_canonical_ts(),
                    )
                ],
                RunMetrics(latency_sec=latency_sec),
                failure_origin=subprocess_failure.failure_origin,
                infra_error_code=subprocess_failure.infra_error_code,
                artifacts=artifacts,
            )

        try:
            events = _parse_stream_json(result.stdout)
            trace = _events_to_trace(events)
            final_text = _extract_final_text(events)
            usage = _aggregate_usage(events)

            input_tokens = usage["input"]
            output_tokens = usage["output"]
            cache_read_tokens = usage["cache_read"]
            cache_write_tokens = usage["cache_write"]
            total_tokens = usage["total"]
            tool_calls = len([e for e in trace if e.type == "tool_call_started"])
            turns = usage["turns"] or max(1, tool_calls)

            trace.append(
                CanonicalTraceEvent(
                    type="task_completed",
                    ts=task_completion_ts(trace),
                )
            )

            return self._make_result(
                task,
                model,
                "completed",
                final_text,
                trace,
                RunMetrics(
                    latency_sec=latency_sec,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    total_tokens=total_tokens,
                    cost_usd=calculate_cost(
                        model,
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_write_tokens,
                        pricing=self.pricing_override(),
                    ),
                    tool_calls=tool_calls,
                    turns=turns,
                ),
            )
        except Exception as parse_error:
            exception_msg = f"{type(parse_error).__name__}: {parse_error}"
            stderr_tail = (result.stderr or "")[:STDERR_PREVIEW_MAX_CHARS]
            error = f"Claude Code output parse error: {exception_msg}"
            if stderr_tail:
                error += f"\nstderr: {stderr_tail}"

            failure = detect_failure_origin_from_error(error)
            return self._make_result(
                task,
                model,
                "failed",
                result.stdout or "",
                [
                    CanonicalTraceEvent(
                        type="task_failed",
                        error=error[:800],
                        ts=to_canonical_ts(),
                    )
                ],
                RunMetrics(latency_sec=latency_sec),
                failure_origin=failure.get("failure_origin"),
                infra_error_code=failure.get("infra_error_code"),
            )

    def cleanup(self, prepared: PreparedRun) -> None:
        self.executor.restore_workspace(prepared.workspace_dir)
        remove_workspace(prepared.workspace_dir)


async def _run_claude_streaming(
    command: str,
    args: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_ms: int,
) -> dict[str, Any]:
    """Run Claude Code subprocess and collect streaming JSON output."""
    use_pgroup = sys.platform != "win32"
    _install_subprocess_cleanup_handlers()

    proc = await asyncio.create_subprocess_exec(
        command,
        *args,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=_preexec_new_pgroup if use_pgroup else None,
    )
    _register_active_subprocess(proc, use_pgroup)

    stdout = ""
    stderr = ""
    timed_out = False

    async def read_stream(stream: asyncio.StreamReader, is_stderr: bool = False) -> None:
        nonlocal stdout, stderr
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode(errors="replace")
            if is_stderr:
                stderr += text
            else:
                stdout += text

    try:
        await asyncio.wait_for(
            asyncio.gather(
                read_stream(proc.stdout, False),
                read_stream(proc.stderr, True),
            ),
            timeout=timeout_ms / 1000,
        )
    except asyncio.CancelledError:
        _terminate_process_tree(proc, use_pgroup)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            _kill_process_tree(proc, use_pgroup)
        raise
    except TimeoutError:
        timed_out = True
        _terminate_process_tree(proc, use_pgroup)

    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        _kill_process_tree(proc, use_pgroup)
    finally:
        _unregister_active_subprocess(proc)

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
    }


def _parse_stream_json(output: str) -> list[dict[str, Any]]:
    """Parse newline-delimited stream-json output from Claude Code."""
    events: list[dict[str, Any]] = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _events_to_trace(events: list[dict[str, Any]]) -> list[CanonicalTraceEvent]:
    """Convert Claude Code stream-json events into canonical trace events.

    Claude Code stream-json format (--output-format stream-json --verbose):
      {type: "system",    subtype: "init", tools: [...]}
      {type: "assistant", message: {content: [{type: "thinking"}, {type: "tool_use", name, input}, ...]}}
      {type: "user",      tool_use_result: {content: "...", is_error: false}, parent_tool_use_id: "..."}
      {type: "assistant", message: {content: [{type: "text", text: "..."}]}}
      {type: "result",    result: "...", subtype: "success"}

    Tool calls are nested inside assistant message content blocks, NOT top-level events.
    """
    trace: list[CanonicalTraceEvent] = []
    # Track pending tool_use blocks to pair with their results
    pending_tools: list[dict[str, Any]] = []
    last_ts: datetime | None = None

    for event in events:
        event_type = event.get("type", "")
        ts, last_ts = _normalize_canonical_timestamp(event.get("timestamp"), last_ts)

        if event_type == "assistant":
            # assistant events have message.content[] with blocks
            msg = event.get("message", {})
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                btype = block.get("type", "")

                if btype == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input")
                    tool_id = block.get("id", "")
                    trace.append(
                        CanonicalTraceEvent(
                            type="tool_call_started",
                            tool_name=tool_name,
                            input=tool_input,
                            ts=ts,
                        )
                    )
                    pending_tools.append({"tool_name": tool_name, "id": tool_id, "ts": ts})

                elif btype == "text":
                    text = block.get("text", "")
                    if text:
                        trace.append(
                            CanonicalTraceEvent(
                                type="message",
                                role="assistant",
                                text=text,
                                ts=ts,
                            )
                        )
                # Skip "thinking" blocks

        elif event_type == "user":
            # user events carry tool results in message.content[] blocks.
            # Each block has type="tool_result" with content (the output text)
            # and tool_use_id (to match against pending tool calls).
            msg = event.get("message", {})
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype != "tool_result":
                    continue

                # Extract result text
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = "\n".join(b.get("text", "") for b in result_content if isinstance(b, dict))
                is_error = block.get("is_error", False)

                # Pair with pending tool
                pending = pending_tools.pop(0) if pending_tools else None
                tool_name = pending["tool_name"] if pending else "unknown"

                trace.append(
                    CanonicalTraceEvent(
                        type="tool_call_completed",
                        tool_name=tool_name,
                        success=not is_error,
                        output=str(result_content)[:TOOL_OUTPUT_MAX_CHARS] if result_content else None,
                        ts=ts,
                    )
                )

        # NOTE: claude-code's stream emits the final assistant text TWICE —
        # once as the trailing ``assistant`` event's text block (handled
        # above), and once more as the ``result`` event's ``result`` field.
        # Adding a message for the ``result`` event produces an adjacent
        # duplicate in the canonical trace (observed in v2, 4/4 claude-code
        # completed runs had a 1ms-apart dup). We deliberately do NOT emit a
        # message for the ``result`` event — its role is final-answer
        # extraction (_extract_final_text reads it separately).

    # Orphaned pending tools (agent crashed before result)
    for pending in pending_tools:
        trace.append(
            CanonicalTraceEvent(
                type="tool_call_completed",
                tool_name=pending["tool_name"],
                success=False,
                output="<no result recorded>",
                ts=pending["ts"],
            )
        )

    return trace


def _normalize_canonical_timestamp(raw_ts: Any, last_ts: datetime | None) -> tuple[str, datetime]:
    """Return a monotonic canonical timestamp for a streamed Claude event.

    Claude's stream timestamps are useful provenance, but they are not stable
    enough to use directly as canonical event ordering: ``assistant`` events
    lack a top-level ``timestamp`` while ``user`` events carrying tool_result
    blocks do, and their real ts is usually earlier than the synthetic fill
    we'd pick for the preceding assistant event. Canonical traces should
    reflect the order we received and processed events, so we treat the
    source timestamp as a lower-bound hint and enforce a strictly
    non-decreasing sequence.

    When an event has no usable timestamp, inherit from ``last_ts`` rather
    than ``datetime.now(UTC)``: now() reflects PARSE time (often 30+s after
    the stream arrived), which would poison every subsequent real ts by
    pushing the monotonic floor far into the future. Falling back to
    last_ts keeps synthetic events close to their neighbors so that real
    timestamps coming in later can still be preserved as-is.

    Bumps colliding events by 1ms (canonical contract is millisecond
    precision — see ``utils/timestamps.py``). A burst of N synthetic events
    drifts forward by N ms, which is still below the gap to the next real
    harness event.
    """
    parsed: datetime | None = None
    if isinstance(raw_ts, (int, float)):
        parsed = datetime.fromtimestamp(raw_ts / 1000, tz=UTC)
    elif isinstance(raw_ts, str) and raw_ts:
        try:
            parsed = datetime.fromisoformat(raw_ts)
        except ValueError:
            parsed = None

    if parsed is None:
        parsed = last_ts if last_ts is not None else datetime.now(UTC)

    # Truncate to ms (canonical granularity) BEFORE the monotonicity check so
    # the emitted ts strings stay strictly increasing — comparing at μs would
    # let two events at e.g. 12.345678 / 12.345679 both emit "12.345".
    parsed = parsed.replace(microsecond=(parsed.microsecond // 1000) * 1000)

    if last_ts is not None and parsed <= last_ts:
        parsed = last_ts + timedelta(milliseconds=1)

    return to_canonical_ts(parsed), parsed


def _extract_final_text(events: list[dict[str, Any]]) -> str:
    """Extract the final assistant text from stream events.

    Priority: result event > last assistant text block.
    """
    # Check for result event first (most reliable)
    for event in reversed(events):
        if event.get("type") == "result" and event.get("result"):
            return event["result"]

    # Fallback: last assistant message with text content
    for event in reversed(events):
        if event.get("type") != "assistant":
            continue
        msg = event.get("message", {})
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        joined = "\n".join(t for t in texts if t)
        if joined:
            return joined

    return ""


def _aggregate_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate token usage across all stream events.

    Claude Code stream-json puts usage inside assistant message objects
    (message.usage) and also in the top-level result event.
    """
    # Canonical parser-output shape: ``turns`` (not ``calls``), ``total``
    # derived from in+out+cr+cw. claude-code's ``_aggregate_usage`` is
    # called from run() which composes it with trace + final_text; the
    # probe's replay reuses the same composition.
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0, "turns": 0}
    for event in events:
        # Check top-level usage (result events)
        u = event.get("usage")
        # Check message-level usage (assistant events)
        if not u:
            u = event.get("message", {}).get("usage")
        if not u:
            continue
        usage["input"] += int(u.get("input_tokens", 0) or u.get("input", 0))
        usage["output"] += int(u.get("output_tokens", 0) or u.get("output", 0))
        usage["cache_read"] += int(u.get("cache_read_input_tokens", 0) or u.get("cache_read", 0))
        usage["cache_write"] += int(u.get("cache_creation_input_tokens", 0) or u.get("cache_write", 0))
        usage["turns"] += 1
    usage["total"] = usage["input"] + usage["output"] + usage["cache_read"] + usage["cache_write"]
    return usage
