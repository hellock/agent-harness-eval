"""Nanobot adapter.

Uses `nanobot agent -m <query>` to run a Python-based agent.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import textwrap
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from ..config.providers import parse_model_spec
from ..constants import STDERR_PREVIEW_MAX_CHARS, TOOL_OUTPUT_MAX_CHARS
from ..executor import attach_run_layout_mounts, filter_env, policy_from_task, resolve_executor_backend
from ..task import Task
from ..types import CanonicalTraceEvent, RunMetrics, RunResult
from ..utils.conversation import format_task_message
from ..utils.cost import calculate_cost
from ..utils.failure_origin import detect_failure_origin_from_error
from ..utils.workspace import create_run_layout, remove_workspace
from . import register_adapter
from .interface import (
    HarnessAdapter,
    NativeMemoryFile,
    PreparedRun,
    detect_subprocess_failure,
)

_NANOBOT_WORKSPACE_TEMPLATES = {
    "SOUL.md": """# Soul

I am nanobot 🐈, a personal AI assistant.

## Personality

- Helpful and friendly
- Concise and to the point
- Curious and eager to learn

## Values

- Accuracy over speed
- User privacy and safety
- Transparency in actions

## Communication Style

- Be clear and direct
- Explain reasoning when helpful
- Ask clarifying questions when needed
""",
    "AGENTS.md": """# Agent Instructions

You are a helpful AI assistant. Be concise, accurate, and friendly.

## Scheduled Reminders

Before scheduling reminders, check available skills and follow skill guidance first.
Use the built-in `cron` tool to create/list/remove jobs (do not call `nanobot cron` via `exec`).
Get USER_ID and CHANNEL from the current session (e.g., `8281248569` and `telegram` from `telegram:8281248569`).

**Do NOT just write reminders to MEMORY.md** — that won't trigger actual notifications.

## Heartbeat Tasks

`HEARTBEAT.md` is checked on the configured heartbeat interval. Use file tools to manage periodic tasks:

- **Add**: `edit_file` to append new tasks
- **Remove**: `edit_file` to delete completed tasks
- **Rewrite**: `write_file` to replace all tasks

When the user asks for a recurring/periodic task, update `HEARTBEAT.md` instead of creating a one-time cron reminder.
""",
    "TOOLS.md": """# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace

## glob — File Discovery

- Use `glob` to find files by pattern before falling back to shell commands
- Simple patterns like `*.py` match recursively by filename
- Use `entry_type="dirs"` when you need matching directories instead of files
- Use `head_limit` and `offset` to page through large result sets
- Prefer this over `exec` when you only need file paths

## grep — Content Search

- Use `grep` to search file contents inside the workspace
- Default behavior returns only matching file paths (`output_mode="files_with_matches"`)
- Supports optional `glob` filtering plus `context_before` / `context_after`
- Supports `type="py"`, `type="ts"`, `type="md"` and similar shorthand filters
- Use `fixed_strings=true` for literal keywords containing regex characters
- Use `output_mode="files_with_matches"` to get only matching file paths
- Use `output_mode="count"` to size a search before reading full matches
- Use `head_limit` and `offset` to page across results
- Prefer this over `exec` for code and history searches
- Binary or oversized files may be skipped to keep results readable

## cron — Scheduled Reminders

- Please refer to cron skill for usage.
""",
    "USER.md": """# User Profile

Information about the user to help personalize interactions.

## Basic Information

- **Name**: (your name)
- **Timezone**: (your timezone, e.g., UTC+8)
- **Language**: (preferred language)

## Preferences

### Communication Style

- [ ] Casual
- [ ] Professional
- [ ] Technical

### Response Length

- [ ] Brief and concise
- [ ] Detailed explanations
- [ ] Adaptive based on question

### Technical Level

- [ ] Beginner
- [ ] Intermediate
- [ ] Expert

## Work Context

- **Primary Role**: (your role, e.g., developer, researcher)
- **Main Projects**: (what you're working on)
- **Tools You Use**: (IDEs, languages, frameworks)

## Topics of Interest

-
-
-

## Special Instructions

(Any specific instructions for how the assistant should behave)

---

*Edit this file to customize nanobot's behavior for your needs.*
""",
    "HEARTBEAT.md": """# Heartbeat Tasks

This file is checked every 30 minutes by your nanobot agent.
Add tasks below that you want the agent to work on periodically.

If this file has no tasks (only headers and comments), the agent will skip the heartbeat.

## Active Tasks

<!-- Add your periodic tasks below this line -->


## Completed

<!-- Move completed tasks here or delete them -->
""",
    "memory/MEMORY.md": """# Long-term Memory

This file stores important information that should persist across sessions.

## User Information

(Important facts about the user)

## Preferences

(User preferences learned over time)

## Project Context

(Information about ongoing projects)

## Important Notes

(Things to remember)

---

*This file is automatically updated by nanobot when important information should be remembered.*
""",
}

_NANOBOT_NATIVE_PROVIDERS = {
    "anthropic",
    "openai",
    "openrouter",
    "deepseek",
    "groq",
    "zhipu",
    "dashscope",
    "gemini",
    "moonshot",
    "minimax",
    "mistral",
    "stepfun",
    "qianfan",
}


@register_adapter
class NanobotAdapter(HarnessAdapter):
    name = "nanobot"
    managed_docker_image = True
    supports_native_memory = True
    emits_paired_trace_events = True

    required_env_vars: ClassVar[list[list[str]]] = [["ANTHROPIC_API_KEY"], ["OPENAI_API_KEY"]]

    def install_memory(self, prepared: PreparedRun, files: list[NativeMemoryFile]) -> None:
        if not files:
            return
        memory_dir = os.path.join(prepared.workspace_dir, ".nanobot", "memory")
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
        _prime_nanobot_workspace(Path(workspace_dir))
        execution_policy = policy_from_task(task, workspace_dir, task.timeout_sec)
        attach_run_layout_mounts(execution_policy, layout)

        session_id = f"eval-{uuid.uuid4().hex[:16]}"
        runtime_dir = layout.state_dir
        os.makedirs(runtime_dir, exist_ok=True)
        return PreparedRun(
            task=task,
            layout=layout,
            env={"_EVAL_SESSION_ID": session_id, "_EVAL_RUNTIME_DIR": runtime_dir},
            execution_policy=execution_policy,
        )

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        task = prepared.task
        workspace_dir = prepared.workspace_dir
        execution_policy = prepared.execution_policy
        model_spec = parse_model_spec(model)
        start_time = asyncio.get_running_loop().time()
        timeout_ms = task.timeout_sec * 1000

        message_text = format_task_message(task)
        session_id = prepared.env["_EVAL_SESSION_ID"]
        runtime_dir = prepared.env["_EVAL_RUNTIME_DIR"]

        provider = self.resolve_provider(model_spec)
        config_path = os.path.join(runtime_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                _build_nanobot_runtime_config(
                    model_spec,
                    workspace_dir,
                    provider,
                ),
                f,
                indent=2,
            )

        inner_env = filter_env(
            self.runtime_config.subprocess_env,
            prepared.env,
            [
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
            ],
        )
        nanobot_python = (
            "python"
            if resolve_executor_backend(self.runtime_config) == "docker"
            else _resolve_nanobot_runtime_python(str(self.runtime_config.harness_state_dir("nanobot")))
        )
        wrapper_path = _write_nanobot_eval_wrapper(runtime_dir)
        inner_args = [
            wrapper_path,
            "-m",
            message_text,
            "-s",
            session_id,
            "-w",
            workspace_dir,
            "-c",
            config_path,
            "--state-dir",
            runtime_dir,
            "--no-markdown",
        ]

        result = await self.executor.execute(
            self.name,
            execution_policy,
            nanobot_python,
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

        subprocess_failure = detect_subprocess_failure(result, command_label="Nanobot")
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
            final_text = _strip_ansi(result.stdout).strip()
            session_data = _read_nanobot_session(runtime_dir, session_id)
            trace = session_data["trace"]
            if final_text and not any(
                event.type == "message" and event.role == "assistant" and event.text == final_text for event in trace
            ):
                trace.append(
                    CanonicalTraceEvent(
                        type="message",
                        role="assistant",
                        text=final_text,
                        ts=datetime.now(UTC).isoformat(),
                    )
                )
            trace.append(
                CanonicalTraceEvent(
                    type="task_completed",
                    ts=datetime.now(UTC).isoformat(),
                )
            )
            usage = session_data["usage"]

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
            error = f"Nanobot output parse error: {exception_msg}"
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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return _ANSI_RE.sub("", text)


def _resolve_nanobot_runtime_python(nanobot_home: str) -> str:
    return os.path.join(nanobot_home, "nanobot-ai", "bin", "python")


def _build_nanobot_runtime_config(
    model_spec: Any,
    workspace_dir: str,
    provider_config: Any,
) -> dict[str, Any]:
    from ..config.providers import ModelSpec, ProviderConfig

    assert isinstance(model_spec, ModelSpec)
    assert isinstance(provider_config, ProviderConfig)

    provider_name = _resolve_nanobot_provider_name(model_spec.provider, provider_config.api_format)
    provider_payload: dict[str, Any] = {
        "apiKey": provider_config.api_key,
    }
    if provider_config.base_url:
        provider_payload["apiBase"] = provider_config.base_url
    if provider_config.extra_headers:
        provider_payload["extraHeaders"] = dict(provider_config.extra_headers)

    return {
        "providers": {
            provider_name: provider_payload,
        },
        "agents": {
            "defaults": {
                "model": model_spec.model,
                "provider": provider_name,
                "workspace": workspace_dir,
            }
        },
        "tools": {
            "exec": {
                "sandbox": "",
            }
        },
    }


def _resolve_nanobot_provider_name(provider_name: str, api_format: str) -> str:
    if provider_name in _NANOBOT_NATIVE_PROVIDERS:
        return provider_name
    if api_format == "anthropic":
        return "anthropic"
    return "custom"


def _write_nanobot_eval_wrapper(runtime_dir: str) -> str:
    wrapper_path = Path(runtime_dir) / "nanobot_eval_agent.py"
    wrapper_path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import argparse
            import asyncio
            from pathlib import Path


            def _parse_args() -> argparse.Namespace:
                parser = argparse.ArgumentParser()
                parser.add_argument("-m", "--message", required=True)
                parser.add_argument("-s", "--session", required=True)
                parser.add_argument("-w", "--workspace", required=True)
                parser.add_argument("-c", "--config", required=True)
                parser.add_argument("--state-dir", required=True)
                parser.add_argument("--no-markdown", action="store_true")
                return parser.parse_args()


            async def _run() -> int:
                args = _parse_args()

                from nanobot.agent.loop import AgentLoop
                from nanobot.bus.queue import MessageBus
                from nanobot.cli.commands import _make_provider
                from nanobot.config.loader import (
                    load_config,
                    resolve_config_env_vars,
                    set_config_path,
                )
                from nanobot.cron.service import CronService
                from nanobot.session.manager import SessionManager
                from nanobot.utils.helpers import sync_workspace_templates

                config_path = Path(args.config).expanduser().resolve()
                set_config_path(config_path)
                runtime_config = resolve_config_env_vars(load_config(config_path))
                runtime_config.agents.defaults.workspace = args.workspace
                sync_workspace_templates(runtime_config.workspace_path)

                bus = MessageBus()
                provider = _make_provider(runtime_config)
                state_dir = Path(args.state_dir).expanduser().resolve()
                session_manager = SessionManager(state_dir)
                cron = CronService(state_dir / "cron" / "jobs.json")

                agent_loop = AgentLoop(
                    bus=bus,
                    provider=provider,
                    workspace=runtime_config.workspace_path,
                    model=runtime_config.agents.defaults.model,
                    max_iterations=runtime_config.agents.defaults.max_tool_iterations,
                    context_window_tokens=runtime_config.agents.defaults.context_window_tokens,
                    web_config=runtime_config.tools.web,
                    context_block_limit=runtime_config.agents.defaults.context_block_limit,
                    max_tool_result_chars=runtime_config.agents.defaults.max_tool_result_chars,
                    provider_retry_mode=runtime_config.agents.defaults.provider_retry_mode,
                    exec_config=runtime_config.tools.exec,
                    cron_service=cron,
                    restrict_to_workspace=runtime_config.tools.restrict_to_workspace,
                    mcp_servers=runtime_config.tools.mcp_servers,
                    channels_config=runtime_config.channels,
                    timezone=runtime_config.agents.defaults.timezone,
                    session_manager=session_manager,
                )

                try:
                    response = await agent_loop.process_direct(args.message, args.session)
                    if response and response.content:
                        print(response.content)
                    return 0
                finally:
                    await agent_loop.close_mcp()


            def main() -> None:
                raise SystemExit(asyncio.run(_run()))


            if __name__ == "__main__":
                main()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return str(wrapper_path)


def _read_nanobot_session(runtime_dir: str, session_id: str) -> dict[str, Any]:
    session_path = Path(runtime_dir) / "sessions" / f"{session_id}.jsonl"
    trace: list[CanonicalTraceEvent] = []
    usage = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 0,
        "tool_calls": 0,
        "turns": 0,
    }
    if not session_path.exists():
        return {"trace": trace, "usage": usage}

    pending_tools: dict[str, str] = {}
    with open(session_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_type = event.get("_type")
            if event_type == "metadata":
                last_usage = event.get("metadata", {}).get("last_usage", {})
                usage["input"] = int(last_usage.get("prompt_tokens", 0))
                usage["output"] = int(last_usage.get("completion_tokens", 0))
                usage["cache_read"] = int(last_usage.get("cache_read_input_tokens", last_usage.get("cached_tokens", 0)))
                usage["cache_write"] = int(last_usage.get("cache_creation_input_tokens", 0))
                usage["total_tokens"] = int(last_usage.get("total_tokens", 0))
                continue

            role = event.get("role")
            content = event.get("content", "")
            if not isinstance(content, str):
                content = ""
            ts = event.get("timestamp")
            if not isinstance(ts, str) or not ts:
                ts = datetime.now(UTC).isoformat()

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
                tool_calls = event.get("tool_calls")
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
                        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                        tool_id = tool_call.get("id") if isinstance(tool_call, dict) else None
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
                        usage["tool_calls"] += 1
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
                tool_call_id = event.get("tool_call_id")
                tool_name = pending_tools.pop(tool_call_id, event.get("name", "unknown"))
                trace.append(
                    CanonicalTraceEvent(
                        type="tool_call_completed",
                        tool_name=tool_name,
                        success=True,
                        output=content[:TOOL_OUTPUT_MAX_CHARS] if content else None,
                        ts=ts,
                    )
                )

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


def _prime_nanobot_workspace(workspace_dir: Path) -> None:
    for name in ("SOUL.md", "AGENTS.md", "TOOLS.md", "USER.md", "HEARTBEAT.md"):
        target = workspace_dir / name
        if not target.exists():
            target.write_text(_NANOBOT_WORKSPACE_TEMPLATES[name], encoding="utf-8")

    memory_dir = workspace_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_file = memory_dir / "MEMORY.md"
    if not memory_file.exists():
        memory_file.write_text(_NANOBOT_WORKSPACE_TEMPLATES["memory/MEMORY.md"], encoding="utf-8")
    history_file = memory_dir / "history.jsonl"
    if not history_file.exists():
        history_file.write_text("", encoding="utf-8")

    (workspace_dir / "skills").mkdir(exist_ok=True)

    if not (workspace_dir / ".git").exists():
        subprocess.run(
            ["git", "init", "--quiet", str(workspace_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
