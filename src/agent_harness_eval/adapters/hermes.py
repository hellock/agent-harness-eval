"""Hermes adapter.

Uses the Hermes agent CLI.  Reads back session state from the SQLite
database that Hermes writes to ``$HERMES_HOME/state.db`` to extract
tool-call trace events and token usage.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from typing import Any, ClassVar

from ..config.providers import parse_model_spec
from ..constants import STDERR_PREVIEW_MAX_CHARS, TOOL_OUTPUT_MAX_CHARS
from ..executor import attach_run_layout_mounts, filter_env, policy_from_task
from ..task import Task
from ..types import CanonicalTraceEvent, RunMetrics, RunResult
from ..utils.conversation import format_task_message
from ..utils.cost import calculate_cost, calculate_cost_no_cache
from ..utils.failure_origin import detect_failure_origin_from_error
from ..utils.timestamps import task_completion_ts, to_canonical_ts
from ..utils.workspace import create_run_layout, remove_workspace
from . import register_adapter
from .interface import (
    HarnessAdapter,
    NativeMemoryFile,
    PreparedRun,
    detect_empty_output_silent_failure,
)


@register_adapter
class HermesAdapter(HarnessAdapter):
    name = "hermes"
    managed_docker_image = True
    supports_native_memory = True
    emits_paired_trace_events = True

    required_env_vars: ClassVar[list[list[str]]] = [["ANTHROPIC_API_KEY"], ["OPENAI_API_KEY"]]
    supported_api_formats: ClassVar[list[str]] = [
        "anthropic",
        "openai-chat-completions",
        "openai-responses",
    ]

    def install_memory(self, prepared: PreparedRun, files: list[NativeMemoryFile]) -> None:
        if not files:
            return
        memory_dir = os.path.join(prepared.workspace_dir, ".hermes", "memory")
        os.makedirs(memory_dir, exist_ok=True)
        for f in files:
            rel_path = f.path.lstrip("/")
            target = os.path.join(memory_dir, rel_path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w") as fh:
                fh.write(f.content)

    def prepare(self, task: Task, run_id: str) -> PreparedRun:
        layout = create_run_layout(run_id, workspace_files=task.workspace_files)
        workspace_dir = layout.workspace_dir
        execution_policy = policy_from_task(task, workspace_dir, task.timeout_sec)
        attach_run_layout_mounts(execution_policy, layout)

        runtime_dir = layout.state_dir
        os.makedirs(runtime_dir, exist_ok=True)

        return PreparedRun(
            task=task,
            layout=layout,
            env={
                "_EVAL_RUNTIME_DIR": runtime_dir,
            },
            execution_policy=execution_policy,
        )

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        task = prepared.task
        model_spec = parse_model_spec(model)
        runtime_dir = prepared.env["_EVAL_RUNTIME_DIR"]
        start_time = asyncio.get_running_loop().time()

        # --- Provider ---
        provider = self.resolve_provider(model_spec)

        is_anthropic = provider.api_format == "anthropic"
        if is_anthropic:
            api_key_env = "ANTHROPIC_API_KEY"
            hermes_provider = "anthropic"
        else:
            api_key_env = "OPENAI_API_KEY"
            hermes_provider = "custom"

        provider_env: dict[str, str] = {api_key_env: provider.api_key}
        base_url = provider.base_url if provider.base_url else ""

        # --- Env ---
        inner_env = filter_env(
            self.runtime_config.subprocess_env,
            {**prepared.env, **provider_env, "HOME": runtime_dir, "HERMES_HOME": runtime_dir},
            ["HOME", "HERMES_HOME", api_key_env],
        )

        # --- Command ---
        import shlex

        hq = shlex.quote(self.resolve_binary())
        config_cmds = [
            f"{hq} config set model.provider {shlex.quote(hermes_provider)} >/dev/null",
            f"{hq} config set model.default {shlex.quote(model_spec.model)} >/dev/null",
            f"{hq} config set model.api_key_env {shlex.quote(api_key_env)} >/dev/null",
        ]
        if base_url:
            config_cmds.append(f"{hq} config set model.base_url {shlex.quote(base_url)} >/dev/null")

        chat_cmd = (
            f"{hq} chat --quiet --query {shlex.quote(format_task_message(task))} --source tool --max-turns 60 --yolo"
        )
        script = " && ".join(config_cmds) + f" && {chat_cmd}"

        # --- Execute ---
        result = await self._run_via_executor(prepared, model, "sh", ["-c", script], inner_env)
        if isinstance(result, RunResult):
            if result.status == "failed":
                return self._recover_failed_run_from_session(runtime_dir, model, result)
            return result

        # --- Parse output ---
        latency_sec = asyncio.get_running_loop().time() - start_time
        final_text = _extract_response_text(result.stdout)
        session_id = _extract_session_id(result.stdout)

        try:
            session_data = _read_hermes_session(runtime_dir, session_id)
            trace = session_data["trace"]
            usage = session_data["usage"]

            if final_text and not any(
                event.type == "message" and event.role == "assistant" and event.text == final_text for event in trace
            ):
                trace.append(
                    CanonicalTraceEvent(
                        type="message",
                        role="assistant",
                        text=final_text,
                        ts=to_canonical_ts(),
                    )
                )

            # Silent-failure guard: exit 0 + empty session + empty final_text
            # = failure, not completion. Happens when state.db is empty
            # (hermes init failed silently) or the session wasn't flushed.
            empty_failure = detect_empty_output_silent_failure(
                trace, final_text, command_label="Hermes", stderr=result.stderr or ""
            )
            if empty_failure is not None:
                return self._make_result(
                    task,
                    model,
                    "failed",
                    result.stdout or "",
                    [
                        CanonicalTraceEvent(
                            type="task_failed",
                            error=empty_failure.error,
                            ts=to_canonical_ts(),
                        )
                    ],
                    RunMetrics(latency_sec=latency_sec),
                    failure_origin=empty_failure.failure_origin,
                    infra_error_code=empty_failure.infra_error_code,
                )

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
                    tool_calls=session_data["tool_calls"],
                    turns=usage["turns"],
                ),
            )
        except Exception as parse_error:
            exception_msg = f"{type(parse_error).__name__}: {parse_error}"
            stderr_tail = (result.stderr or "")[:STDERR_PREVIEW_MAX_CHARS]
            error = f"Hermes session parse error: {exception_msg}"
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

    def _recover_failed_run_from_session(self, runtime_dir: str, model: str, failed: RunResult) -> RunResult:
        """Merge Hermes session state back into failed subprocess results.

        Hermes may already have persisted tool/message rows in ``state.db`` before
        the CLI exits non-zero on a later provider error. Returning the raw
        subprocess failure loses those trace events and incorrectly reports
        ``tool_calls=0``. Recover what we can from the session, then preserve the
        original failure event as the terminal trace entry.
        """
        session_id = _extract_session_id(failed.final_text)
        try:
            session_data = _read_hermes_session(runtime_dir, session_id)
        except Exception:
            return failed

        recovered_trace = list(session_data["trace"])
        if not recovered_trace:
            return failed

        usage = session_data["usage"]
        failed.trace = recovered_trace + failed.trace
        failed.metrics.input_tokens = usage["input"]
        failed.metrics.output_tokens = usage["output"]
        failed.metrics.cache_read_tokens = usage["cache_read"]
        failed.metrics.cache_write_tokens = usage["cache_write"]
        failed.metrics.total_tokens = usage["total"]
        failed.metrics.tool_calls = session_data["tool_calls"]
        failed.metrics.turns = usage["turns"]
        failed.metrics.cost_usd = calculate_cost(
            model,
            usage["input"],
            usage["output"],
            usage["cache_read"],
            usage["cache_write"],
            pricing=self.pricing_override(),
        )
        failed.metrics.cost_usd_no_cache = calculate_cost_no_cache(
            model,
            usage["input"],
            usage["output"],
            usage["cache_read"],
            usage["cache_write"],
            pricing=self.pricing_override(),
        )
        return failed

    def cleanup(self, prepared: PreparedRun) -> None:
        import shutil

        runtime_dir = prepared.env.get("_EVAL_RUNTIME_DIR")
        if runtime_dir:
            shutil.rmtree(runtime_dir, ignore_errors=True)
        self.executor.restore_workspace(prepared.workspace_dir)
        remove_workspace(prepared.workspace_dir)


_SESSION_ID_RE = re.compile(r"session_id:\s*(\S+)")

# Hermes tool progress lines use the box-drawing character ┊ or start with tool emojis.
_TOOL_PROGRESS_RE = re.compile(r"┊|^\s*[⚡📖🔎📋✅❌⏳🔧]")


def _extract_session_id(stdout: str) -> str | None:
    """Extract the session_id from hermes quiet-mode output.

    Hermes prints ``session_id: <id>`` at the end of quiet-mode output.
    """
    if not stdout:
        return None
    match = _SESSION_ID_RE.search(stdout)
    return match.group(1) if match else None


def _extract_response_text(stdout: str) -> str:
    """Extract the agent's response from hermes output.

    Hermes --quiet outputs a box like:
        ╭─ ⚕ Hermes ─────╮
        <response text>
        session_id: ...

    We strip the box decorations and session metadata.
    """
    if not stdout:
        return ""

    lines = stdout.strip().split("\n")
    result_lines: list[str] = []
    in_response = False

    for line in lines:
        stripped = line.strip()
        # Skip box borders and decorations
        if stripped.startswith("╭") or stripped.startswith("╰") or stripped.startswith("╮") or stripped.startswith("╯"):
            in_response = True
            continue
        # Skip session metadata
        if stripped.startswith("session_id:"):
            continue
        # Skip empty box-border-like lines
        if set(stripped) <= {"─", "│", "╭", "╮", "╰", "╯", " ", ""}:
            continue
        # Skip hermes tool progress lines (┊ indicator or tool emojis)
        if _TOOL_PROGRESS_RE.search(stripped):
            continue
        if in_response or not any(c in line for c in "╭╮╰╯"):
            result_lines.append(line)

    text = "\n".join(result_lines).strip()
    if text:
        return text

    # Fallback: return everything
    return stdout.strip()


def _read_hermes_session(
    runtime_dir: str,
    session_id: str | None,
) -> dict[str, Any]:
    """Read session data from the Hermes SQLite database.

    Hermes stores all session state in ``$HERMES_HOME/state.db``.  The
    adapter sets ``HERMES_HOME`` to runtime_dir, so the database is at
    ``runtime_dir/state.db``.

    Returns a dict with ``trace`` (list of CanonicalTraceEvent) and
    ``usage`` (dict of token counts and tool_calls).
    """
    # Canonical parser-output shape (see codex/zeroclaw): ``total`` (not
    # ``total_tokens``); ``tool_calls`` returned top-level alongside usage.
    trace: list[CanonicalTraceEvent] = []
    usage = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total": 0,
        "turns": 0,
    }
    # Outer ``tool_calls_count`` avoids a name collision with the
    # inner-loop local ``tool_calls`` (per-message JSON list). Before this
    # rename the session-level count was silently clobbered by the last
    # message row's per-row list, making the final return's ``tool_calls``
    # field sometimes a ``list`` and sometimes ``None``.
    tool_calls_count = 0

    db_path = os.path.join(runtime_dir, "state.db")
    if not os.path.exists(db_path):
        return {"trace": trace, "usage": usage, "tool_calls": tool_calls_count}

    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        # Fall back to most recent session if no session_id was parsed from stdout
        if not session_id:
            row = conn.execute("SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()
            if not row:
                return {"trace": trace, "usage": usage, "tool_calls": tool_calls_count}
            session_id = row["id"]

        # --- Session-level token counts ---
        session_row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if session_row:
            session = dict(session_row)
            usage["input"] = int(session.get("input_tokens") or 0)
            usage["output"] = int(session.get("output_tokens") or 0)
            usage["cache_read"] = int(session.get("cache_read_tokens") or 0)
            usage["cache_write"] = int(session.get("cache_write_tokens") or 0)
            usage["total"] = usage["input"] + usage["output"]
            tool_calls_count = int(session.get("tool_call_count") or 0)

        # --- Messages → trace events ---
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id",
            (session_id,),
        ).fetchall()

        pending_tools: dict[str, str] = {}

        for row in rows:
            msg = dict(row)
            role = msg.get("role", "")
            content = msg.get("content") or ""
            ts_epoch = msg.get("timestamp")
            ts = to_canonical_ts(ts_epoch if isinstance(ts_epoch, (int, float)) and ts_epoch > 0 else None)

            # Parse tool_calls JSON if present
            raw_tool_calls = msg.get("tool_calls")
            tool_calls: list[dict[str, Any]] | None = None
            if isinstance(raw_tool_calls, str) and raw_tool_calls:
                try:
                    tool_calls = json.loads(raw_tool_calls)
                except json.JSONDecodeError:
                    pass
            elif isinstance(raw_tool_calls, list):
                tool_calls = raw_tool_calls

            if role == "user":
                trace.append(
                    CanonicalTraceEvent(
                        type="message",
                        role="user",
                        text=content,
                        ts=ts,
                    )
                )
                continue

            if role == "assistant":
                usage["turns"] += 1
                if isinstance(tool_calls, list) and tool_calls:
                    text = content.strip()
                    if text:
                        trace.append(
                            CanonicalTraceEvent(
                                type="message",
                                role="assistant",
                                text=text,
                                ts=ts,
                            )
                        )
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict):
                            continue
                        function = tool_call.get("function", {})
                        if not isinstance(function, dict):
                            continue
                        tool_id = tool_call.get("id")
                        tool_name = function.get("name", "unknown")
                        raw_arguments = function.get("arguments")
                        tool_input: Any = raw_arguments
                        if isinstance(raw_arguments, str):
                            try:
                                tool_input = json.loads(raw_arguments)
                            except json.JSONDecodeError:
                                tool_input = raw_arguments
                        trace.append(
                            CanonicalTraceEvent(
                                type="tool_call_started",
                                tool_name=tool_name,
                                input=tool_input,
                                ts=ts,
                            )
                        )
                        if isinstance(tool_id, str):
                            pending_tools[tool_id] = tool_name
                    continue

                trace.append(
                    CanonicalTraceEvent(
                        type="message",
                        role="assistant",
                        text=content,
                        ts=ts,
                    )
                )
                continue

            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                tool_name = pending_tools.pop(tool_call_id, msg.get("tool_name", "unknown"))
                trace.append(
                    CanonicalTraceEvent(
                        type="tool_call_completed",
                        tool_name=tool_name,
                        success=True,
                        output=content[:TOOL_OUTPUT_MAX_CHARS] if content else None,
                        ts=ts,
                    )
                )

        # Close out any tool calls that never got a result
        for tool_name in pending_tools.values():
            trace.append(
                CanonicalTraceEvent(
                    type="tool_call_completed",
                    tool_name=tool_name,
                    success=False,
                    output="<no result recorded>",
                    ts=to_canonical_ts(),
                )
            )
    finally:
        conn.close()

    return {"trace": trace, "usage": usage, "tool_calls": tool_calls_count}
