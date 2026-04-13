"""Execution backend abstraction for eval runs.

Each backend (host, docker, k8s) implements the Executor interface.
Shared types, env filtering, and policy construction live here.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from ..task import Task
from ..utils.workspace import RunLayout

if TYPE_CHECKING:
    from ..config.runtime import RuntimeConfig
    from ..utils.subprocess import SubprocessResult


# ─── Shared types ───


@dataclass
class ExecutionPolicy:
    network: bool = True
    strict_network: bool = False
    file_write: bool = True
    shell: bool = True
    workspace_dir: str = ""
    cwd: str = ""
    extra_mounts: list[VolumeMount] = field(default_factory=list)
    timeout_sec: int = 600
    limits: dict[str, int] = field(
        default_factory=lambda: {
            "memory_mb": 4096,
            "cpu_cores": 2,
            "max_pids": 256,
            "max_disk_mb": 2048,
        }
    )


@dataclass(frozen=True)
class VolumeMount:
    source: str
    target: str
    mode: Literal["ro", "rw"] = "rw"


# ─── Policy construction ───


def policy_from_task(
    task: Task,
    workspace_dir: str,
    timeout_sec: int,
) -> ExecutionPolicy:
    """Derive an execution policy from a task's tool_boundary."""
    tb = task.tool_boundary
    return ExecutionPolicy(
        network=not (tb and tb.internet == "disabled"),
        file_write=not (tb and tb.file_write == "disabled"),
        shell=not (tb and tb.shell == "disabled"),
        workspace_dir=workspace_dir,
        cwd=workspace_dir,
        timeout_sec=timeout_sec,
    )


def attach_run_layout_mounts(policy: ExecutionPolicy, layout: RunLayout) -> ExecutionPolicy:
    """Expose the non-workspace parts of a run layout."""
    mounts = list(policy.extra_mounts)
    mounts.extend(
        [
            VolumeMount(source=layout.input_dir, target=layout.input_dir, mode="ro"),
            VolumeMount(source=layout.state_dir, target=layout.state_dir, mode="rw"),
            VolumeMount(source=layout.output_dir, target=layout.output_dir, mode="rw"),
        ]
    )
    policy.extra_mounts = mounts
    return policy


def policy_for_grader(
    policy: ExecutionPolicy,
    *,
    cwd: str | None = None,
) -> ExecutionPolicy:
    """Derive verifier-side execution policy from an agent policy."""
    return replace(
        policy,
        network=True,
        strict_network=False,
        shell=True,
        file_write=True,
        cwd=cwd or policy.workspace_dir,
        extra_mounts=list(policy.extra_mounts),
        limits=dict(policy.limits),
    )


# ─── Env filtering ───

SENSITIVE_PATTERNS = [
    re.compile(r"^ANTHROPIC_API_KEY$", re.IGNORECASE),
    re.compile(r"^OPENAI_API_KEY$", re.IGNORECASE),
    re.compile(r"^OPENAI_BASE_URL$", re.IGNORECASE),
    re.compile(r"^AWS_SECRET", re.IGNORECASE),
    re.compile(r"^AWS_SESSION", re.IGNORECASE),
    re.compile(r"^GITHUB_TOKEN$", re.IGNORECASE),
    re.compile(r"^GH_TOKEN$", re.IGNORECASE),
    re.compile(r"^NPM_TOKEN$", re.IGNORECASE),
    re.compile(r"^DOCKER_PASSWORD$", re.IGNORECASE),
    re.compile(r"^SSH_AUTH_SOCK$", re.IGNORECASE),
    re.compile(r"^GPG_", re.IGNORECASE),
    re.compile(r"SECRET", re.IGNORECASE),
    re.compile(r"PASSWORD", re.IGNORECASE),
    re.compile(r"CREDENTIAL", re.IGNORECASE),
    re.compile(r"PRIVATE_KEY", re.IGNORECASE),
    re.compile(r"TOKEN$", re.IGNORECASE),
]

SAFE_ALLOWLIST = {
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "TZ",
    "NODE_ENV",
    "NODE_OPTIONS",
    "EDITOR",
    "XDG_RUNTIME_DIR",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
}


def filter_env(
    env: dict[str, str | None],
    extra_vars: dict[str, str] | None = None,
    passthrough: list[str] | None = None,
) -> dict[str, str]:
    """Filter environment variables, blocking sensitive patterns."""
    result: dict[str, str] = {}
    passthrough_set = set(passthrough or [])

    for key, value in env.items():
        if value is None:
            continue
        if key in passthrough_set:
            result[key] = value
            continue
        if key in SAFE_ALLOWLIST:
            result[key] = value
            continue
        if any(p.search(key) for p in SENSITIVE_PATTERNS):
            continue
        if re.match(r"^(AGIXOS_EVAL_|OPENCLAW_|EVAL_)", key):
            if re.match(r"^EVAL_PROVIDER_.*_API_KEY$", key):
                continue
            result[key] = value

    if extra_vars:
        result.update(extra_vars)

    return result


# ─── Executor ABC ───


class Executor(ABC):
    """Wraps and executes commands in an isolation backend."""

    name: str

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
    ) -> SubprocessResult:
        """Execute a command under this backend's isolation."""
        ...

    def resolve_binary(self, harness: str, binary: str) -> str:
        """Resolve a harness binary path for this executor's context.

        Default: look up the host-installed binary.
        Docker overrides to return bare name for managed images.
        """
        return self.runtime_config.resolve_harness_bin(harness, binary)

    def restore_workspace(self, workspace_dir: str) -> None:
        """Post-run cleanup. Default: no-op. Host overrides."""


# ─── Registry ───

_REGISTRY: dict[str, type[Executor]] = {}
_EXECUTOR_MODULES_LOADED = False


def register_executor(cls: type[Executor]) -> type[Executor]:
    """Class decorator that registers an executor by its ``name``."""
    _REGISTRY[cls.name] = cls
    return cls


def _ensure_executor_modules_loaded() -> None:
    global _EXECUTOR_MODULES_LOADED
    if _EXECUTOR_MODULES_LOADED:
        return

    from . import docker as _docker  # noqa: F401
    from . import host as _host  # noqa: F401

    _EXECUTOR_MODULES_LOADED = True


def resolve_executor_backend(runtime_config: RuntimeConfig) -> str:
    """Resolve the configured executor backend."""
    _ensure_executor_modules_loaded()
    override = runtime_config.executor_backend
    if override in _REGISTRY:
        return override
    raise ValueError(f"Unknown executor backend: {override!r}. Available: {sorted(_REGISTRY)}")


def create_executor(runtime_config: RuntimeConfig) -> Executor:
    """Create the appropriate executor for the configured backend."""
    _ensure_executor_modules_loaded()
    backend = resolve_executor_backend(runtime_config)
    cls = _REGISTRY.get(backend)
    if cls is None:
        raise ValueError(f"Unknown executor backend: {backend!r}. Available: {sorted(_REGISTRY)}")
    return cls(runtime_config)
