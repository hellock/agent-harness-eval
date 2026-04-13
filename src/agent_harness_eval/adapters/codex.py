"""Codex adapter.

Uses `codex run` to execute tasks.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from typing import Any, ClassVar

from ..config.providers import parse_model_spec
from ..constants import STDERR_PREVIEW_MAX_CHARS, TOOL_OUTPUT_MAX_CHARS
from ..executor import attach_run_layout_mounts, filter_env, policy_from_task
from ..task import Task
from ..types import CanonicalTraceEvent, RunMetrics, RunResult
from ..utils.conversation import format_task_message
from ..utils.failure_origin import detect_failure_origin_from_error
from ..utils.workspace import create_run_layout, remove_workspace
from . import register_adapter
from .interface import (
    HarnessAdapter,
    PreparedRun,
    detect_subprocess_failure,
)


@register_adapter
class CodexAdapter(HarnessAdapter):
    name = "codex"
    managed_docker_image = True

    required_env_vars: ClassVar[list[list[str]]] = [["OPENAI_API_KEY"]]
    supported_api_formats: ClassVar[list[str]] = ["openai-responses"]

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
            env={"_EVAL_RUNTIME_DIR": runtime_dir},
            execution_policy=execution_policy,
        )

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        task = prepared.task
        workspace_dir = prepared.workspace_dir
        execution_policy = prepared.execution_policy
        model_spec = parse_model_spec(model)
        runtime_dir = prepared.env["_EVAL_RUNTIME_DIR"]
        start_time = asyncio.get_running_loop().time()
        timeout_ms = task.timeout_sec * 1000

        message_text = format_task_message(task)

        # Resolve provider via harness override or model spec.
        # Codex requires the OpenAI Responses API (/v1/responses).
        provider = self.resolve_provider(model_spec)
        config_toml_path = os.path.join(runtime_dir, "config.toml")
        provider_id = "relay"
        base_url = provider.base_url if provider and provider.base_url else ""
        with open(config_toml_path, "w") as f:
            f.write(f'model = "{model_spec.model}"\n')
            f.write(f'model_provider = "{provider_id}"\n\n')
            f.write(f"[model_providers.{provider_id}]\n")
            f.write('name = "Eval Relay"\n')
            if base_url:
                f.write(f'base_url = "{base_url}"\n')
            f.write('env_key = "OPENAI_API_KEY"\n')

        extra_env: dict[str, str] = {
            **prepared.env,
            "CODEX_HOME": runtime_dir,
        }
        if provider and provider.api_key:
            extra_env["OPENAI_API_KEY"] = provider.api_key

        inner_env = filter_env(
            self.runtime_config.subprocess_env,
            extra_env,
            [
                "OPENAI_API_KEY",
                "CODEX_HOME",
                "HOME",
            ],
        )
        codex_bin = self.resolve_binary()
        inner_args = [
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--cd",
            workspace_dir,
            "--sandbox",
            "workspace-write" if execution_policy.file_write else "read-only",
            "--model",
            model_spec.model,
            message_text,
        ]

        result = await self.executor.execute(
            self.name,
            execution_policy,
            codex_bin,
            inner_args,
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

        subprocess_failure = detect_subprocess_failure(result, command_label="Codex")
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
            # Parse JSONL stdout for codex events
            parsed = _parse_codex_jsonl(result.stdout)
            final_text = parsed["final_text"]
            trace = parsed["trace"]

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
                RunMetrics(latency_sec=latency_sec),
            )
        except Exception as parse_error:
            exception_msg = f"{type(parse_error).__name__}: {parse_error}"
            stderr_tail = (result.stderr or "")[:STDERR_PREVIEW_MAX_CHARS]
            error = f"Codex output parse error: {exception_msg}"
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
        self.executor.restore_workspace(prepared.workspace_dir)
        remove_workspace(prepared.workspace_dir)


def _parse_codex_jsonl(stdout: str) -> dict[str, Any]:
    """Parse Codex JSONL output.

    Looks for 'item.completed' events where item.type='agent_message' to
    extract final_text (from item.text). Also builds trace from tool_use
    and tool_result events.
    """
    trace: list[CanonicalTraceEvent] = []
    final_text = ""

    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")
        ts = event.get("ts") or event.get("timestamp") or datetime.now(UTC).isoformat()
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts / 1000, tz=UTC).isoformat()

        # Extract final text from item.completed events with agent_message items
        if event_type == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
                if text:
                    final_text = text

        # Build trace from tool_use / tool_result events
        elif event_type in ("function_call", "tool_call", "tool_use"):
            tool_name = event.get("name") or event.get("tool_name", "unknown")
            tool_input = event.get("input") or event.get("arguments")
            trace.append(CanonicalTraceEvent(type="tool_call_started", tool_name=tool_name, input=tool_input, ts=ts))

        elif event_type in ("function_call_output", "tool_result"):
            tool_name = event.get("name") or event.get("tool_name", "unknown")
            output = event.get("output") or event.get("content", "")
            is_error = event.get("is_error", False) or event.get("status") == "error"
            trace.append(
                CanonicalTraceEvent(
                    type="tool_call_completed",
                    tool_name=tool_name,
                    success=not is_error,
                    output=str(output)[:TOOL_OUTPUT_MAX_CHARS] if output else None,
                    ts=ts,
                )
            )

    return {"final_text": final_text, "trace": trace}
