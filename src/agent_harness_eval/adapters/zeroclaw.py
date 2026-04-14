"""ZeroClaw adapter.

Uses `zeroclaw --session-file` to run a Rust-based agent.
Parses runtime trace JSONL for token usage and tool call events.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from ..config.providers import parse_model_spec
from ..constants import STDERR_PREVIEW_MAX_CHARS, TOOL_OUTPUT_MAX_CHARS
from ..executor import VolumeMount, attach_run_layout_mounts, filter_env, policy_from_task
from ..task import Task
from ..types import CanonicalTraceEvent, RunMetrics, RunResult
from ..utils.conversation import format_task_message
from ..utils.cost import calculate_cost
from ..utils.failure_origin import detect_failure_origin_from_error
from ..utils.workspace import create_run_layout, remove_workspace
from . import register_adapter
from .interface import (
    HarnessAdapter,
    PreparedRun,
    detect_subprocess_failure,
)


@register_adapter
class ZeroClawAdapter(HarnessAdapter):
    name = "zeroclaw"
    managed_docker_image = True

    required_env_vars: ClassVar[list[list[str]]] = [["API_KEY"], ["ZEROCLAW_API_KEY"]]

    def prepare(self, task: Task, run_id: str) -> PreparedRun:
        layout = create_run_layout(run_id, workspace_files=task.workspace_files)
        workspace_dir = layout.workspace_dir
        execution_policy = policy_from_task(task, workspace_dir, task.timeout_sec)
        attach_run_layout_mounts(execution_policy, layout)

        runtime_dir = layout.state_dir
        config_dir = str(Path(workspace_dir).parent / ".zeroclaw")
        os.makedirs(config_dir, exist_ok=True)
        execution_policy.extra_mounts.append(VolumeMount(source=config_dir, target=config_dir, mode="rw"))
        return PreparedRun(
            task=task,
            layout=layout,
            env={"_EVAL_CONFIG_DIR": config_dir, "_EVAL_RUNTIME_DIR": runtime_dir},
            execution_policy=execution_policy,
        )

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        task = prepared.task
        workspace_dir = prepared.workspace_dir
        execution_policy = prepared.execution_policy
        model_spec = parse_model_spec(model)
        config_dir = prepared.env["_EVAL_CONFIG_DIR"]
        runtime_dir = prepared.env["_EVAL_RUNTIME_DIR"]
        start_time = asyncio.get_running_loop().time()
        timeout_ms = task.timeout_sec * 1000

        message_text = format_task_message(task)

        # Resolve provider via harness override or model spec.
        provider = self.resolve_provider(model_spec)

        # ZeroClaw uses `anthropic-custom:<url>` for Anthropic-compatible relays
        # and `custom:<url>` for OpenAI-compatible relays.
        base = provider.base_url.rstrip("/")
        if provider.api_format == "anthropic":
            zc_provider_flag = f"anthropic-custom:{base}"
        else:
            zc_provider_flag = f"custom:{base}"

        extra_env: dict[str, str] = {
            **prepared.env,
            "HOME": runtime_dir,
            "ZEROCLAW_WORKSPACE": workspace_dir,
            "API_KEY": provider.api_key,
        }
        passthrough = ["HOME", "ZEROCLAW_WORKSPACE", "API_KEY"]

        inner_env = filter_env(self.runtime_config.subprocess_env, extra_env, passthrough)
        zeroclaw_bin = self.resolve_binary()

        # Step 1: Onboard
        onboard_args = [
            "onboard",
            "--quick",
            "--force",
            "--provider",
            zc_provider_flag,
            "--model",
            model_spec.model,
        ]

        onboard_result = await self.executor.execute(
            self.name,
            execution_policy,
            zeroclaw_bin,
            onboard_args,
            inner_env,
            timeout_ms=timeout_ms,
        )

        if onboard_result.timed_out:
            return self._make_result(
                task,
                model,
                "timed_out",
                "",
                [],
                RunMetrics(latency_sec=task.timeout_sec),
            )

        onboard_failure = detect_subprocess_failure(onboard_result, command_label="ZeroClaw onboard")
        if onboard_failure:
            latency_sec = asyncio.get_running_loop().time() - start_time
            return self._make_result(
                task,
                model,
                "failed",
                onboard_result.stdout or "",
                [
                    CanonicalTraceEvent(
                        type="task_failed",
                        error=onboard_failure.error,
                        ts=datetime.now(UTC).isoformat(),
                    )
                ],
                RunMetrics(latency_sec=latency_sec),
                failure_origin=onboard_failure.failure_origin,
                infra_error_code=onboard_failure.infra_error_code,
            )

        # Relax autonomy for eval (allow workspace access, no approval needed).
        # In Docker, harness-eval's container boundary is the primary safety
        # control, so when shell access is enabled we also relax ZeroClaw's
        # command allowlist to avoid fighting the outer executor policy.
        _patch_zeroclaw_autonomy_config(
            config_dir,
            workspace_dir,
            relax_shell_commands=execution_policy.shell and self.runtime_config.executor_backend == "docker",
        )

        # Step 2: Agent
        agent_args = [
            "agent",
            "--provider",
            zc_provider_flag,
            "--model",
            model_spec.model,
            "--message",
            message_text,
        ]

        result = await self.executor.execute(
            self.name,
            execution_policy,
            zeroclaw_bin,
            agent_args,
            inner_env,
            timeout_ms=timeout_ms,
        )

        latency_sec = asyncio.get_running_loop().time() - start_time

        if result.timed_out:
            return self._make_result(
                task,
                model,
                "timed_out",
                "",
                [],
                RunMetrics(latency_sec=task.timeout_sec),
            )

        subprocess_failure = detect_subprocess_failure(result, command_label="ZeroClaw")
        if subprocess_failure:
            return self._make_result(
                task,
                model,
                "failed",
                result.stdout or "",
                [
                    CanonicalTraceEvent(
                        type="task_failed",
                        error=subprocess_failure.error,
                        ts=datetime.now(UTC).isoformat(),
                    )
                ],
                RunMetrics(latency_sec=latency_sec),
                failure_origin=subprocess_failure.failure_origin,
                infra_error_code=subprocess_failure.infra_error_code,
            )

        try:
            # Sanitize stdout: strip ANSI codes and trim
            final_text = _strip_zeroclaw_logs(result.stdout)

            # Parse runtime trace for token usage and tool call events.
            trace_path = os.path.join(config_dir, "runtime_trace.jsonl")
            session_data = _read_zeroclaw_runtime_trace(trace_path)
            trace = session_data["trace"]
            usage = session_data["usage"]

            trace.append(
                CanonicalTraceEvent(
                    type="task_completed",
                    ts=datetime.now(UTC).isoformat(),
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
                    total_tokens=usage["total_tokens"],
                    cost_usd=calculate_cost(
                        model,
                        usage["input"],
                        usage["output"],
                        usage["cache_read"],
                        usage["cache_write"],
                        pricing=self.pricing_override(),
                    ),
                    tool_calls=usage["tool_calls"],
                    turns=usage["turns"],
                ),
            )
        except Exception as parse_error:
            exception_msg = f"{type(parse_error).__name__}: {parse_error}"
            stderr_tail = (result.stderr or "")[:STDERR_PREVIEW_MAX_CHARS]
            error = f"ZeroClaw output parse error: {exception_msg}"
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
                        ts=datetime.now(UTC).isoformat(),
                    )
                ],
                RunMetrics(latency_sec=latency_sec),
                failure_origin=failure.get("failure_origin"),
                infra_error_code=failure.get("infra_error_code"),
            )

    def cleanup(self, prepared: PreparedRun) -> None:
        import shutil

        runtime_dir = prepared.env.get("_EVAL_RUNTIME_DIR")
        if runtime_dir:
            shutil.rmtree(runtime_dir, ignore_errors=True)
        self.executor.restore_workspace(prepared.workspace_dir)
        remove_workspace(prepared.workspace_dir)


def _patch_zeroclaw_autonomy_config(
    config_dir: str,
    workspace_dir: str,
    *,
    relax_shell_commands: bool = False,
) -> None:
    """Patch config.toml to allow tool use in the eval workspace.

    ZeroClaw's default security config blocks /tmp (where eval workspaces live)
    and uses 'supervised' autonomy which requires interactive approval.
    """
    cfg_path = os.path.join(config_dir, "config.toml")
    if not os.path.exists(cfg_path):
        return

    with open(cfg_path) as f:
        content = f.read()

    # Set autonomy to full (non-interactive, no approval needed)
    content = re.sub(
        r'^level\s*=\s*"supervised"',
        'level = "full"',
        content,
        flags=re.MULTILINE,
    )

    # Remove /tmp from forbidden_paths (workspace lives there)
    content = re.sub(
        r'"/tmp",?\s*\n?',
        "",
        content,
    )

    # Set allowed_roots to include workspace
    content = re.sub(
        r"^allowed_roots\s*=\s*\[\]",
        f'allowed_roots = ["{workspace_dir}"]',
        content,
        flags=re.MULTILINE,
    )

    # Disable require_approval_for_medium_risk
    content = re.sub(
        r"^require_approval_for_medium_risk\s*=\s*true",
        "require_approval_for_medium_risk = false",
        content,
        flags=re.MULTILINE,
    )

    if relax_shell_commands:
        content = re.sub(
            r"^allowed_commands\s*=\s*\[[^\]]*\]",
            'allowed_commands = ["*"]',
            content,
            flags=re.MULTILINE,
        )
        content = re.sub(
            r"^block_high_risk_commands\s*=\s*true",
            "block_high_risk_commands = false",
            content,
            flags=re.MULTILINE,
        )

    # Increase max_actions_per_hour for eval
    content = re.sub(
        r"^max_actions_per_hour\s*=\s*\d+",
        "max_actions_per_hour = 200",
        content,
        flags=re.MULTILINE,
    )

    # Point zeroclaw's workspace to the actual eval workspace
    # (onboard creates a separate workspace inside config_dir)
    content = re.sub(
        r"^workspace_only\s*=\s*true",
        "workspace_only = false",
        content,
        flags=re.MULTILINE,
    )

    # Enable runtime trace for observability (token usage + tool call events).
    trace_path = os.path.join(config_dir, "runtime_trace.jsonl")
    if "[observability]" not in content:
        content += f'\n[observability]\nruntime_trace_mode = "full"\nruntime_trace_path = "{trace_path}"\n'
    else:
        # Section exists (from onboard) — override mode and path.
        content = re.sub(
            r'^runtime_trace_mode\s*=\s*"[^"]*"',
            'runtime_trace_mode = "full"',
            content,
            flags=re.MULTILINE,
        )
        content = re.sub(
            r'^runtime_trace_path\s*=\s*"[^"]*"',
            f'runtime_trace_path = "{trace_path}"',
            content,
            flags=re.MULTILINE,
        )

    with open(cfg_path, "w") as f:
        f.write(content)


def _read_zeroclaw_runtime_trace(trace_path: str) -> dict[str, Any]:
    """Parse a ZeroClaw runtime_trace.jsonl file for token usage and tool events.

    ZeroClaw emits JSONL events with ``event_type`` fields such as:
    - ``llm_request`` / ``llm_response`` — contain token counts
    - ``tool_call_start`` / ``tool_result`` — contain tool call info
    - ``agent_start`` / ``agent_end`` — session boundaries
    """
    trace: list[CanonicalTraceEvent] = []
    usage: dict[str, int] = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 0,
        "tool_calls": 0,
        "turns": 0,
    }

    if not os.path.exists(trace_path):
        return {"trace": trace, "usage": usage}

    pending_tools: dict[str, str] = {}

    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("event_type") or event.get("type") or ""
            ts = event.get("timestamp") or event.get("ts") or datetime.now(UTC).isoformat()
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts, tz=UTC).isoformat()

            # ZeroClaw nests data under "payload"; fall back to top-level.
            payload = event.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {}

            if event_type == "llm_response":
                usage["turns"] += 1
                input_t = int(payload.get("input_tokens", 0))
                output_t = int(payload.get("output_tokens", 0))
                cache_read = int(payload.get("cached_input_tokens", 0))
                cache_write = int(payload.get("cache_creation_input_tokens", 0))
                usage["input"] += input_t
                usage["output"] += output_t
                usage["cache_read"] += cache_read
                usage["cache_write"] += cache_write
                usage["total_tokens"] += input_t + output_t + cache_read + cache_write

                content = payload.get("raw_response") or payload.get("text")
                if isinstance(content, str) and content.strip():
                    trace.append(
                        CanonicalTraceEvent(
                            type="message",
                            role="assistant",
                            text=content.strip(),
                            ts=ts,
                        )
                    )

            elif event_type == "tool_call_start":
                tool_name = payload.get("tool") or "unknown"
                tool_input = payload.get("arguments")
                if isinstance(tool_input, str):
                    try:
                        tool_input = json.loads(tool_input)
                    except json.JSONDecodeError:
                        pass
                trace.append(
                    CanonicalTraceEvent(
                        type="tool_call_started",
                        tool_name=tool_name,
                        input=tool_input,
                        ts=ts,
                    )
                )
                usage["tool_calls"] += 1

            elif event_type == "tool_call_result":
                tool_name = payload.get("tool") or "unknown"
                output = payload.get("output") or ""
                if not isinstance(output, str):
                    output = json.dumps(output) if output else ""
                trace.append(
                    CanonicalTraceEvent(
                        type="tool_call_completed",
                        tool_name=tool_name,
                        success=True,
                        output=output[:TOOL_OUTPUT_MAX_CHARS] if output else None,
                        ts=ts,
                    )
                )

            elif event_type == "turn_final_response":
                content = payload.get("text")
                if isinstance(content, str) and content.strip():
                    trace.append(
                        CanonicalTraceEvent(
                            type="message",
                            role="assistant",
                            text=content.strip(),
                            ts=ts,
                        )
                    )

    # Close any pending tool calls that never got a result.
    for tool_name in pending_tools.values():
        trace.append(
            CanonicalTraceEvent(
                type="tool_call_completed",
                tool_name=tool_name,
                success=False,
                output="<no result recorded>",
                ts=datetime.now(UTC).isoformat(),
            )
        )

    return {"trace": trace, "usage": usage}


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
# ZeroClaw log lines: timestamp + level + module + message
_ZEROCLAW_LOG_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+(INFO|WARN|ERROR|DEBUG|TRACE)\s+",
)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return _ANSI_RE.sub("", text)


def _strip_zeroclaw_logs(text: str) -> str:
    """Remove zeroclaw log lines from stdout, keeping only the actual response."""
    cleaned = _strip_ansi(text)
    lines = cleaned.split("\n")
    output_lines = [line for line in lines if not _ZEROCLAW_LOG_RE.match(line)]
    return "\n".join(output_lines).strip()
