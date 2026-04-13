# Adding a New Harness Adapter

This guide walks through integrating a new agent harness into the evaluation framework. By the end you will have a single Python file that plugs into the CLI, preflight, execution backend, and grading pipeline automatically.

## Overview

The framework evaluates how different agent harnesses orchestrate the same LLM model on identical tasks. Each harness adapter translates between the framework's unified interface and a specific agent CLI.

**What you need to implement:** one Python file with a class that inherits `HarnessAdapter` and implements three methods — `prepare()`, `run()`, `cleanup()`.

**What the framework handles for you:** task loading, workspace isolation, execution backend (docker/host), environment filtering, grading, reporting, and concurrent execution.

## Quick Start (Minimal Adapter)

Create `src/agent_harness_eval/adapters/my_agent.py`:

```python
"""MyAgent adapter."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from ..executor import (
    attach_run_layout_mounts,
    filter_env,
    policy_from_task,
)
from ..task import Task
from ..types import CanonicalTraceEvent, RunMetrics, RunResult
from ..utils.conversation import format_task_message
from ..config.providers import parse_model_spec
from ..utils.workspace import create_run_layout, remove_workspace
from . import register_adapter
from .interface import HarnessAdapter, PreparedRun, detect_subprocess_failure


@register_adapter
class MyAgentAdapter(HarnessAdapter):
    name = "my-agent"                    # CLI name: --harness my-agent
    cli_binary = "my-agent"              # Binary name (if different from name)
    managed_docker_image = True          # True if you provide a Dockerfile
    required_env_vars = [["OPENAI_API_KEY"]]  # At least one group must be satisfied
    supported_api_formats = ["openai-chat-completions"]     # Reject providers with wrong wire protocol

    def prepare(self, task: Task, run_id: str) -> PreparedRun:
        layout = create_run_layout(run_id, workspace_files=task.workspace_files)
        execution_policy = policy_from_task(task, layout.workspace_dir, task.timeout_sec)
        attach_run_layout_mounts(execution_policy, layout)
        os.makedirs(layout.state_dir, exist_ok=True)
        return PreparedRun(
            task=task,
            layout=layout,
            env={"_EVAL_RUNTIME_DIR": layout.state_dir},
            execution_policy=execution_policy,
        )

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        task = prepared.task
        model_spec = parse_model_spec(model)
        start = asyncio.get_running_loop().time()

        # 1. Resolve provider (uses harness-level override if configured)
        provider = self.resolve_provider(model_spec)
        provider_env = {"OPENAI_API_KEY": provider.api_key}
        passthrough = ["HOME", "OPENAI_API_KEY"]

        # 2. Build filtered environment
        inner_env = filter_env(
            self.runtime_config.subprocess_env,
            {**prepared.env, **provider_env},
            passthrough,
        )

        # 3. Build command
        binary = self.resolve_binary()
        inner_args = [
            "run",
            "--model", model_spec.model,
            "--query", format_task_message(task),
        ]

        # 4. Execute via the executor
        result = await self.executor.execute(
            self.name,
            prepared.execution_policy,
            binary,
            inner_args,
            inner_env,
            timeout_ms=task.timeout_sec * 1000,
        )

        latency = asyncio.get_running_loop().time() - start

        # 5. Handle timeout
        if result.timed_out:
            return self._make_result(task, model, "timed_out", "", [], RunMetrics(latency_sec=task.timeout_sec))

        # 6. Handle subprocess failure
        failure = detect_subprocess_failure(result, command_label="MyAgent")
        if failure:
            return self._make_result(
                task, model, "failed", result.stdout or "",
                [CanonicalTraceEvent(type="task_failed", error=failure.error, ts=datetime.now(timezone.utc).isoformat())],
                RunMetrics(latency_sec=latency),
                failure_origin=failure.failure_origin,
                infra_error_code=failure.infra_error_code,
            )

        # 7. Return success
        return self._make_result(
            task, model, "completed", result.stdout.strip(),
            [CanonicalTraceEvent(type="task_completed", ts=datetime.now(timezone.utc).isoformat())],
            RunMetrics(latency_sec=latency),
        )

    def cleanup(self, prepared: PreparedRun) -> None:
        import shutil
        runtime_dir = prepared.env.get("_EVAL_RUNTIME_DIR")
        if runtime_dir:
            shutil.rmtree(runtime_dir, ignore_errors=True)
        self.executor.restore_workspace(prepared.workspace_dir)
        remove_workspace(prepared.workspace_dir)
```

Then register the import in `adapters/__init__.py`:

```python
from . import my_agent as _my_agent  # noqa: E402, F401
```

That's it. Run `uv run agent-harness-eval run --harness my-agent --task-id reasoning.01`.

---

## Step-by-Step Walkthrough

### 1. Class Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | CLI-facing harness name (`--harness <name>`) |
| `cli_binary` | `str \| None` | No | Binary name if different from `name` (e.g. `"claude"` for `"claude-code"`). Defaults to `name`. |
| `managed_docker_image` | `bool` | No | `True` if you provide `docker/<name>/Dockerfile`. Default `False`. |
| `required_env_vars` | `list[list[str]] \| None` | No | Disjunctive groups — at least one group must have all vars set. E.g. `[["ANTHROPIC_API_KEY"], ["OPENAI_API_KEY"]]` means either key suffices. |
| `supports_native_memory` | `bool` | No | `True` if the harness supports durable memory injection. |
| `emits_paired_trace_events` | `bool` | No | `True` if `run()` returns `tool_call_started` + `tool_call_completed` trace events. Affects preflight validation strictness. |

### 2. `prepare(task, run_id) -> PreparedRun`

Called on the host before any execution backend. Must only do file I/O — no CLI calls.

```python
def prepare(self, task: Task, run_id: str) -> PreparedRun:
    # 1. Create isolated filesystem layout
    layout = create_run_layout(run_id, workspace_files=task.workspace_files)

    # 2. Derive execution policy from task boundaries
    execution_policy = policy_from_task(task, layout.workspace_dir, task.timeout_sec)
    attach_run_layout_mounts(execution_policy, layout)

    # 3. Any additional file setup (configs, profiles, etc.)
    runtime_dir = layout.state_dir
    os.makedirs(runtime_dir, exist_ok=True)
    # ... write config files to runtime_dir ...

    # 4. Return PreparedRun with private state in env dict
    return PreparedRun(
        task=task,
        layout=layout,
        env={"_MY_RUNTIME_DIR": runtime_dir},
        execution_policy=execution_policy,
    )
```

**Key points:**
- `layout.workspace_dir` — mutable workspace with task input files
- `layout.state_dir` — harness-private mutable state (configs, sessions)
- `layout.input_dir` — immutable copy of task inputs (read-only in docker)
- `layout.output_dir` — for durable output artifacts
- Store adapter-private state in `env` dict (prefixed with `_` by convention)

### 3. `run(prepared, model) -> RunResult`

The actual agent execution. In docker mode, this runs inside the container.

**The typical flow is:**

```
resolve provider -> build env -> resolve binary -> construct command -> self.executor.execute() -> parse output -> _make_result()
```

#### Provider Resolution

```python
model_spec = parse_model_spec(model)  # "relay:claude-sonnet-4-6" -> ModelSpec(provider="relay", model="claude-sonnet-4-6")

# resolve_provider() checks the harness-level provider override first,
# then falls back to model_spec.provider. It also validates api_format
# against the adapter's supported_api_formats.
provider = self.resolve_provider(model_spec)
```

Declare `supported_api_formats` on the adapter class to restrict which providers are compatible:
- `["anthropic"]` — Anthropic Messages API (`/v1/messages`)
- `["openai-chat-completions"]` — OpenAI Chat Completions (`/v1/chat/completions`)
- `["openai-responses"]` — OpenAI Responses API (`/v1/responses`)
- `["openai-chat-completions", "openai-responses"]` — either OpenAI variant
- `None` — accept any api_format (adapter handles branching internally)

Users configure a per-harness provider override in `eval.yaml` when the default provider's wire protocol doesn't match:

```yaml
harnesses:
  my-agent:
    provider: relay_openai   # overrides the default provider for this harness only
```

Use `provider.is_openai_compat` to check for either OpenAI variant.

#### Environment Filtering

```python
inner_env = filter_env(
    self.runtime_config.subprocess_env,   # base env from runtime config
    {                                     # extra vars to inject
        **prepared.env,
        "MY_API_KEY": provider.api_key,
        "HOME": runtime_dir,
    },
    [                                     # passthrough list (bypasses sensitive-key filtering)
        "MY_API_KEY",
        "HOME",
    ],
)
```

`filter_env()` strips sensitive variables (API keys, tokens, passwords) from the base environment. Use the `passthrough` list to explicitly allow variables your harness needs.

#### Binary Resolution

```python
binary = self.resolve_binary()
# looks in: .harnesses/<name>/node_modules/.bin/<binary>
#           .harnesses/<name>/bin/<binary>
#           PATH (fallback)
```

#### Executor-based Execution

```python
result = await self.executor.execute(
    self.name,                       # harness name
    prepared.execution_policy,       # ExecutionPolicy (network, file_write, shell constraints)
    binary,                          # inner command
    ["run", "--query", message],     # inner args
    inner_env,                       # filtered environment
    timeout_ms=task.timeout_sec * 1000,
)
```

In docker mode, `self.executor.execute()` wraps your command in `docker run` with resource limits, volume mounts, and env vars. In host mode, it passes through with env filtering and optional filesystem permission restrictions.

#### Compound Commands (sh -c)

If your harness needs multi-step setup (e.g. configure then run):

```python
import shlex
hq = shlex.quote(binary)
script = f"{hq} config set model {shlex.quote(model_spec.model)} >/dev/null && {hq} run --query {shlex.quote(message)}"

result = await self.executor.execute(
    self.name, prepared.execution_policy, "sh", ["-c", script], inner_env,
    timeout_ms=task.timeout_sec * 1000,
)
```

#### Result Construction

Use `self._make_result()` (provided by the base class):

```python
# Success
return self._make_result(task, model, "completed", final_text, trace, RunMetrics(latency_sec=latency))

# Failure
return self._make_result(task, model, "failed", stdout, trace, metrics, failure_origin="provider", infra_error_code="api_auth_error")

# Timeout
return self._make_result(task, model, "timed_out", "", [], RunMetrics(latency_sec=task.timeout_sec))
```

### 4. `cleanup(prepared) -> None`

Remove temporary files. Standard pattern:

```python
async def cleanup(self, prepared: PreparedRun) -> None:
    import shutil
    runtime_dir = prepared.env.get("_MY_RUNTIME_DIR")
    if runtime_dir:
        shutil.rmtree(runtime_dir, ignore_errors=True)
    self.executor.restore_workspace(prepared.workspace_dir)
    remove_workspace(prepared.workspace_dir)
```

### 5. Optional: Trace Events

If your harness can report tool usage, emit `CanonicalTraceEvent` entries:

```python
trace = [
    CanonicalTraceEvent(type="tool_call_started", tool_name="read_file", input={"path": "data.csv"}, ts=ts),
    CanonicalTraceEvent(type="tool_call_completed", tool_name="read_file", success=True, output="col1,col2...", ts=ts),
    CanonicalTraceEvent(type="message", role="assistant", text="Here is my analysis...", ts=ts),
    CanonicalTraceEvent(type="task_completed", ts=ts),
]
```

Set `emits_paired_trace_events = True` on your adapter class if you provide paired `tool_call_started` / `tool_call_completed` events. This enables stricter preflight validation and richer trajectory grading.

### 6. Optional: Native Memory

If your harness supports durable memory, override `install_memory()`:

```python
supports_native_memory = True

async def install_memory(self, prepared: PreparedRun, files: list[NativeMemoryFile]) -> None:
    if not files:
        return
    memory_dir = os.path.join(prepared.workspace_dir, ".my-agent", "memory")
    os.makedirs(memory_dir, exist_ok=True)
    for f in files:
        target = os.path.join(memory_dir, f.path.lstrip("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as fh:
            fh.write(f.content)
```

---

## Docker Image Setup

If `managed_docker_image = True`, create two files:

### `docker/my-agent/Dockerfile`

```dockerfile
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir my-agent==1.0.0
```

### `docker/my-agent/build_docker_image.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERSION="${1:-1.0.0}"
IMAGE_TAG="${2:-agent-harness-eval-my-agent:${VERSION}}"

docker build --tag "${IMAGE_TAG}" "${PROJECT_ROOT}/docker/my-agent"
```

### `eval.yaml` entry

```yaml
harnesses:
  my_agent:              # underscore in YAML key
    version: "1.0.0"
```

The framework automatically builds the image on first run if it doesn't exist.

---

## Checklist

1. [ ] Create `src/agent_harness_eval/adapters/my_agent.py` with `@register_adapter`
2. [ ] Add import in `adapters/__init__.py`
3. [ ] Create `docker/my-agent/Dockerfile` + `build_docker_image.sh` (if managed)
4. [ ] Add entry in `eval.yaml` under `harnesses:`
5. [ ] Test: `uv run agent-harness-eval run --harness my-agent --task-id reasoning.01 --timeout 120`
6. [ ] Verify both modes: `EVAL_EXECUTOR=host` and `EVAL_EXECUTOR=docker`

---

## Existing Adapters as Reference

| Adapter | Complexity | Good reference for |
|---------|------------|-------------------|
| `hermes.py` | Simple | Minimal adapter, compound `sh -c` command |
| `codex.py` | Simple | Config file generation, JSONL output parsing |
| `zeroclaw.py` | Medium | Two-step CLI (onboard + agent), config patching, log stripping |
| `nanobot.py` | Medium | Python-based harness, streaming output |
| `claude_code.py` | Complex | Stream-JSON parsing, auth config, trace extraction |
| `openclaw.py` | Complex | Agent registration, session management, compound docker commands |
