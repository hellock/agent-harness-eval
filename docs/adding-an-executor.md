# Adding a Custom Executor

This guide walks through adding a new execution backend to the evaluation framework. An executor controls **where and how** agent commands run — on the host, in a Docker container, on a Kubernetes cluster, or any other environment.

## Overview

The framework uses an `Executor` abstraction to decouple agent adapters from the execution environment. Each adapter calls `executor.execute()` without knowing whether the command runs locally or remotely. The executor handles isolation, resource limits, environment injection, and result collection.

**Built-in executors:**

| Name | Module | Description |
|------|--------|-------------|
| `host` | `executor/host.py` | Runs commands directly on the host machine |
| `docker` | `executor/docker.py` | Runs commands inside Docker containers |

**What you need to implement:** one Python file with a class that inherits `Executor`, decorated with `@register_executor`, and implements one required method — `execute()`.

## Quick Start

Create `src/agent_harness_eval/executor/my_backend.py`:

```python
"""My custom executor."""

from __future__ import annotations

from ..utils.subprocess import SubprocessResult
from . import ExecutionPolicy, Executor, register_executor


@register_executor
class MyBackendExecutor(Executor):
    name = "my-backend"  # used in eval.yaml: executor: my-backend

    async def execute(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
        timeout_ms: int,
    ) -> SubprocessResult:
        # Your execution logic here.
        # Must return a SubprocessResult with stdout, stderr, exit_code, timed_out.
        ...
```

Register the import in `executor/__init__.py`:

```python
# At the bottom, alongside existing imports:
from . import my_backend as _my_backend  # noqa: E402, F401
```

Set it in `eval.yaml`:

```yaml
executor: my-backend
```

That's it. All adapters and the runner will use your executor automatically.

## The Executor Interface

```python
class Executor(ABC):
    name: str  # unique identifier, matches eval.yaml executor: value

    def __init__(self, runtime_config: RuntimeConfig):
        self.runtime_config = runtime_config

    @abstractmethod
    async def execute(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
        timeout_ms: int,
    ) -> SubprocessResult: ...

    def resolve_binary(self, harness: str, binary: str) -> str: ...
    def restore_workspace(self, workspace_dir: str) -> None: ...
```

### `execute()` (required)

The core method. Takes a command intent and returns a `SubprocessResult`.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `harness` | Harness name (e.g. `"nanobot"`) — useful for image selection or labeling |
| `policy` | `ExecutionPolicy` with resource limits, boundary constraints, workspace path, and volume mounts |
| `inner_command` | The command to run (e.g. `"sh"`, `"python"`, or a binary path) |
| `inner_args` | Command arguments (e.g. `["-c", "agent run --query ..."]`) |
| `inner_env` | Pre-filtered environment variables for the command |
| `timeout_ms` | Maximum execution time in milliseconds |

**Return value:** `SubprocessResult(stdout, stderr, exit_code, timed_out)`

For local executors (host, docker), this is the direct subprocess result. For remote executors (k8s, cloud), synthesize a `SubprocessResult` from the remote execution's logs and exit code:

```python
return SubprocessResult(
    stdout=pod_logs,
    stderr="",
    exit_code=pod_exit_code,
    timed_out=(pod_exit_code is None),
)
```

### `resolve_binary()` (optional override)

Resolves a harness binary path for this executor's context. The default implementation looks up the host-installed binary via `runtime_config.resolve_harness_bin()`.

Override this when your executor has binaries available in a different location:

```python
def resolve_binary(self, harness: str, binary: str) -> str:
    if self._image_has_binary(harness):
        return binary  # bare name, available inside the execution environment
    return super().resolve_binary(harness, binary)
```

The Docker executor overrides this to return bare binary names for managed images (where the CLI is pre-installed inside the container).

### `restore_workspace()` (optional override)

Post-run cleanup hook. Called after each run to restore workspace state. Default is a no-op.

The Host executor overrides this to restore file permissions (`chmod -R u+w`) after runs where `policy.file_write=False` restricted write access.

## ExecutionPolicy

The `ExecutionPolicy` dataclass carries all constraints for a run:

```python
@dataclass
class ExecutionPolicy:
    network: bool = True          # allow outbound network access
    file_write: bool = True       # allow workspace writes
    shell: bool = True            # allow shell execution
    workspace_dir: str = ""       # absolute path to the workspace
    extra_mounts: list[VolumeMount] = ...  # additional volume mounts
    timeout_sec: int = 600
    limits: dict[str, int] = ...  # memory_mb, cpu_cores, max_pids, max_disk_mb
```

How to translate these into your backend:

| Policy field | Docker | K8s example |
|-------------|--------|-------------|
| `limits["memory_mb"]` | `--memory 4096m` | `resources.limits.memory: 4096Mi` |
| `limits["cpu_cores"]` | `--cpus 2` | `resources.limits.cpu: "2"` |
| `limits["max_pids"]` | `--pids-limit 256` | SecurityContext or cgroup |
| `network=False` | `EVAL_BOUNDARY_INTERNET=disabled` env var | NetworkPolicy |
| `file_write=False` | Read-only workspace mount | ReadOnlyRootFilesystem |
| `extra_mounts` | `-v source:target:mode` | PVC or hostPath volumes |
| `workspace_dir` | `-v path:path:rw -w path` | Pod working directory + volume |

## Example: Kubernetes Executor

A sketch of what a k8s executor might look like:

```python
"""Kubernetes executor: submits pods to a k8s cluster."""

from __future__ import annotations

import os

from ..utils.subprocess import SubprocessResult
from . import ExecutionPolicy, Executor, filter_env, register_executor


@register_executor
class K8sExecutor(Executor):
    name = "k8s"

    def __init__(self, runtime_config):
        super().__init__(runtime_config)
        # Read executor-specific settings from env, or extend RuntimeConfig first
        self.namespace = os.environ.get("EVAL_K8S_NAMESPACE", "eval")

    async def execute(
        self, harness, policy, inner_command, inner_args, inner_env, timeout_ms,
    ) -> SubprocessResult:
        pod_spec = self._build_pod_spec(harness, policy, inner_command, inner_args, inner_env)
        pod_name = await self._create_pod(pod_spec)
        try:
            exit_code = await self._wait_for_completion(pod_name, timeout_ms)
            logs = await self._get_logs(pod_name)
            return SubprocessResult(
                stdout=logs,
                stderr="",
                exit_code=exit_code if exit_code is not None else -1,
                timed_out=(exit_code is None),
            )
        finally:
            await self._delete_pod(pod_name)

    def _build_pod_spec(self, harness, policy, cmd, args, env):
        return {
            "metadata": {"generateName": f"eval-{harness}-"},
            "spec": {
                "containers": [{
                    "name": "eval",
                    "image": self._resolve_image(harness),
                    "command": [cmd, *args],
                    "env": [{"name": k, "value": v} for k, v in env.items()],
                    "resources": {
                        "limits": {
                            "memory": f"{policy.limits['memory_mb']}Mi",
                            "cpu": str(policy.limits['cpu_cores']),
                        },
                    },
                }],
                "restartPolicy": "Never",
            },
        }
```

If your executor needs first-class config fields instead of reading from env,
add them explicitly to `RuntimeConfig` and `build_runtime_config()` before using
them in adapter or executor code.

## Shared Utilities

The `executor` package provides utilities you can use in your implementation:

| Function | Description |
|----------|-------------|
| `filter_env(env, extra_vars, passthrough)` | Filter sensitive env vars (API keys, tokens, passwords) |
| `policy_from_task(task, workspace_dir, timeout)` | Derive an `ExecutionPolicy` from a task definition |
| `attach_run_layout_mounts(policy, layout)` | Add input/state/output directory mounts to a policy |

These are used by adapters before calling `executor.execute()`, but your executor may also need `filter_env()` for additional env sanitization.

## Checklist

1. [ ] Create `src/agent_harness_eval/executor/my_backend.py` with `@register_executor`
2. [ ] Add import in `executor/__init__.py`
3. [ ] Set `executor: my-backend` in `eval.yaml`
4. [ ] Test: `uv run agent-harness-eval run --harness nanobot --task-id reasoning.01`
5. [ ] Verify `resolve_binary()` returns correct paths for your environment
6. [ ] Verify `restore_workspace()` if your backend modifies workspace permissions

## Existing Executors as Reference

| Executor | Lines | Good reference for |
|----------|-------|-------------------|
| `host.py` | ~85 | Minimal implementation, filesystem permission handling |
| `docker.py` | ~200 | Image resolution, volume mounts, resource limits, env injection |
