"""Codex adapter.

Uses `codex run` to execute tasks.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, ClassVar

from ..config.providers import parse_model_spec
from ..constants import STDERR_PREVIEW_MAX_CHARS, TOOL_OUTPUT_MAX_CHARS
from ..executor import attach_run_layout_mounts, filter_env, policy_from_task, resolve_executor_backend
from ..task import Task
from ..types import CanonicalTraceEvent, RunMetrics, RunResult
from ..utils.conversation import format_task_message
from ..utils.cost import calculate_cost
from ..utils.failure_origin import detect_failure_origin_from_error
from ..utils.timestamps import task_completion_ts, to_canonical_ts
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
            "--model",
            model_spec.model,
        ]
        # Codex's internal shell sandbox relies on bwrap user namespaces. In our
        # Docker eval environment that nested sandbox is the wrong boundary and
        # fails even for benign command_execution calls. Let the outer container
        # enforce filesystem/network isolation instead, but only when shell tools
        # are enabled for the task. Host runs still use Codex's native sandbox.
        if resolve_executor_backend(self.runtime_config) == "docker" and execution_policy.shell:
            inner_args.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            inner_args.extend(
                [
                    "--sandbox",
                    "workspace-write" if execution_policy.file_write else "read-only",
                ]
            )
        inner_args.append(message_text)

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
            # Even on timeout, codex may have emitted partial JSONL events to
            # stdout before being killed — recover whatever turn.completed /
            # item.completed events we can so token usage isn't silently zero.
            # The parser already survives a truncated final line via per-line
            # JSONDecodeError handling.
            partial = _parse_codex_jsonl(result.stdout or "")
            return self._make_result(
                task,
                model,
                "timed_out",
                partial["final_text"],
                partial["trace"],
                RunMetrics(
                    latency_sec=task.timeout_sec,
                    input_tokens=partial["usage"]["input"],
                    output_tokens=partial["usage"]["output"],
                    cache_read_tokens=partial["usage"]["cache_read"],
                    cache_write_tokens=partial["usage"]["cache_write"],
                    total_tokens=partial["usage"]["total"],
                    cost_usd=calculate_cost(
                        model,
                        partial["usage"]["input"],
                        partial["usage"]["output"],
                        partial["usage"]["cache_read"],
                        partial["usage"]["cache_write"],
                        pricing=self.pricing_override(),
                    ),
                    tool_calls=partial["tool_calls"],
                    turns=partial["usage"]["turns"],
                ),
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
                        ts=to_canonical_ts(),
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
            usage = parsed["usage"]

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
                    tool_calls=parsed["tool_calls"],
                    turns=usage["turns"],
                ),
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


# Item types codex emits that are NOT tool calls or user-visible messages.
# - agent_message: the assistant's text reply — handled specially.
# - reasoning: codex's internal chain-of-thought. Not a tool, not something
#   the end user sees. Drop it from the canonical trace.
_CODEX_NON_TOOL_ITEM_TYPES = frozenset({"agent_message", "reasoning"})


def _parse_codex_jsonl(stdout: str) -> dict[str, Any]:
    """Parse ``codex exec --json`` output.

    Schema (verified empirically against codex 0.x binary, April 2026):

    Top-level event types
        thread.started    — session open; ignored
        turn.started      — new turn; ignored (counted when turn.completed fires)
        turn.completed    — carries ``usage.{input_tokens,cached_input_tokens,output_tokens}``
        turn.failed       — provider-side error; recorded as a task_failed event
        item.started      — a tool invocation began (item.type in {command_execution,
                            file_change, mcp_tool_call, web_search, ...}; NOT
                            emitted for agent_message — those arrive only as
                            item.completed)
        item.completed    — a tool invocation finished, OR an assistant message
        item.failed       — a tool invocation crashed before producing a completion

    Item types
        agent_message     — {id, type, text} — assistant's user-visible reply
        reasoning         — internal chain-of-thought; dropped
        command_execution — {id, type, command, aggregated_output, exit_code, status}
        file_change, mcp_tool_call, web_search, ... — other tool kinds; treated
            generically (item.type becomes tool_name; first of
            {command, arguments, query, input} becomes trace input; first of
            {aggregated_output, output, content, result} becomes trace output)

    Note: codex uses the OpenAI prompt-caching model — ``cached_input_tokens``
    reports cache-read hits but there is NO cache-creation / cache-write
    counterpart in the stream. We therefore leave ``cache_write`` at 0 for
    codex, the same architectural constraint zeroclaw has.
    """
    trace: list[CanonicalTraceEvent] = []
    final_text = ""
    usage = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,  # never populated from codex — see docstring
        "total": 0,
        "turns": 0,
    }
    # item.id -> tool_name for pairing item.started with item.completed.
    pending: dict[str, str] = {}

    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")
        ts = to_canonical_ts(event.get("ts") or event.get("timestamp"))

        if event_type == "item.started":
            item = event.get("item") or {}
            it_type = item.get("type", "")
            if not it_type or it_type in _CODEX_NON_TOOL_ITEM_TYPES:
                # agent_message only arrives as item.completed; reasoning is
                # silent. Either way, nothing to emit for item.started.
                continue
            item_id = str(item.get("id", ""))
            tool_input = item.get("command") or item.get("arguments") or item.get("query") or item.get("input")
            trace.append(
                CanonicalTraceEvent(
                    type="tool_call_started",
                    tool_name=it_type,
                    input=tool_input,
                    ts=ts,
                )
            )
            if item_id:
                pending[item_id] = it_type

        elif event_type == "item.completed":
            item = event.get("item") or {}
            it_type = item.get("type", "")
            if it_type == "agent_message":
                text = item.get("text", "")
                if text:
                    final_text = text
                    # Emit the assistant message into the canonical trace too
                    # — every other adapter does this, and downstream graders
                    # / trajectory checks expect message events in-line.
                    trace.append(
                        CanonicalTraceEvent(
                            type="message",
                            role="assistant",
                            text=text,
                            ts=ts,
                        )
                    )
            elif it_type == "reasoning":
                # Internal chain-of-thought — drop.
                continue
            elif it_type:
                # Tool completion.
                item_id = str(item.get("id", ""))
                tool_name = pending.pop(item_id, it_type)
                output = (
                    item.get("aggregated_output")
                    or item.get("output")
                    or item.get("content")
                    or item.get("result")
                    or ""
                )
                if not isinstance(output, str):
                    output = json.dumps(output) if output else ""
                status = item.get("status", "")
                exit_code = item.get("exit_code")
                # codex marks status="failed" directly on non-zero exit_code
                # (observed empirically with `cat /nonexistent` → status="failed"
                # exit_code=1). Treat completed+clean exit as success; anything
                # else as failure.
                success = (status == "completed") and (exit_code in (None, 0))
                trace.append(
                    CanonicalTraceEvent(
                        type="tool_call_completed",
                        tool_name=tool_name,
                        success=success,
                        output=output[:TOOL_OUTPUT_MAX_CHARS] if output else None,
                        ts=ts,
                    )
                )

        elif event_type == "item.failed":
            # Tool crashed before writing an item.completed (e.g. sandbox
            # killed it). Pair against pending, synthesize a failed completion.
            item = event.get("item") or {}
            it_type = item.get("type", "")
            if not it_type or it_type in _CODEX_NON_TOOL_ITEM_TYPES:
                continue
            item_id = str(item.get("id", ""))
            tool_name = pending.pop(item_id, it_type)
            error_msg = event.get("error") or event.get("message") or item.get("error") or ""
            trace.append(
                CanonicalTraceEvent(
                    type="tool_call_completed",
                    tool_name=tool_name,
                    success=False,
                    output=(str(error_msg)[:TOOL_OUTPUT_MAX_CHARS] if error_msg else "<item.failed>"),
                    ts=ts,
                )
            )

        elif event_type == "turn.completed":
            usage["turns"] += 1
            u = event.get("usage") or {}
            input_t = int(u.get("input_tokens", 0))
            output_t = int(u.get("output_tokens", 0))
            cache_read_t = int(u.get("cached_input_tokens", 0))
            usage["input"] += input_t
            usage["output"] += output_t
            usage["cache_read"] += cache_read_t
            usage["total"] += input_t + output_t + cache_read_t

        elif event_type == "turn.failed":
            err = event.get("error") or event.get("message") or ""
            trace.append(
                CanonicalTraceEvent(
                    type="task_failed",
                    error=str(err)[:800] if err else "turn.failed",
                    ts=ts,
                )
            )

    # Orphan handling: tools whose item.started arrived but no item.completed /
    # item.failed ever did (stream truncated mid-tool). Emit synthetic failed
    # completions in parity with claude_code / zeroclaw orphan behaviour.
    for _item_id, tool_name in pending.items():
        trace.append(
            CanonicalTraceEvent(
                type="tool_call_completed",
                tool_name=tool_name,
                success=False,
                output="<no result recorded>",
                ts=to_canonical_ts(),
            )
        )

    return {
        "final_text": final_text,
        "trace": trace,
        "usage": usage,
        "tool_calls": sum(1 for e in trace if e.type == "tool_call_started"),
    }
