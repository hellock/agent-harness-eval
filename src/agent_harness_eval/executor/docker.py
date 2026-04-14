"""Docker executor: runs commands inside Docker containers."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils.subprocess import SubprocessResult, run_subprocess
from . import ExecutionPolicy, Executor, filter_env, register_executor

if TYPE_CHECKING:
    from ..config.runtime import RuntimeConfig


@dataclass
class _DockerCommand:
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str


@register_executor
class DockerExecutor(Executor):
    """Execute commands inside Docker containers with resource limits and isolation."""

    name = "docker"

    async def execute(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
        timeout_ms: int,
    ) -> SubprocessResult:
        wrapped = self._wrap(harness, policy, inner_command, inner_args, inner_env)
        return await run_subprocess(
            wrapped.command,
            wrapped.args,
            cwd=wrapped.cwd,
            env=wrapped.env,
            timeout_ms=timeout_ms,
        )

    def resolve_binary(self, harness: str, binary: str) -> str:
        """For managed images, return bare binary name (available inside container)."""
        if is_docker_managed_image(harness, self.runtime_config):
            return binary
        return super().resolve_binary(harness, binary)

    def _wrap(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
    ) -> _DockerCommand:
        docker_image = resolve_docker_image(harness, self.runtime_config)
        managed = is_docker_managed_image(harness, self.runtime_config)
        cwd = policy.cwd or policy.workspace_dir

        args: list[str] = [
            "run",
            "--rm",
            "--memory",
            f"{policy.limits['memory_mb']}m",
            "--cpus",
            str(policy.limits["cpu_cores"]),
            "--pids-limit",
            str(policy.limits["max_pids"]),
        ]

        # Agent runs may still need outbound access for model API calls even when
        # the declared boundary says "no internet". Verifier/grader runs opt into
        # hard isolation with strict_network=True. RuntimeConfig can also force it.
        if not policy.network and (policy.strict_network or self.runtime_config.docker_network_none):
            args.append("--network=none")

        # Boundary env vars (defense in depth alongside OS enforcement)
        if not policy.network:
            args.extend(["-e", "EVAL_BOUNDARY_INTERNET=disabled"])
        if not policy.shell:
            args.extend(["-e", "EVAL_BOUNDARY_SHELL=disabled"])
        if not policy.file_write:
            args.extend(["-e", "EVAL_BOUNDARY_FILE_WRITE=disabled"])

        # Non-root user
        uid = os.getuid() if hasattr(os, "getuid") else 1000
        gid = os.getgid() if hasattr(os, "getgid") else 1000
        args.extend(["--user", f"{uid}:{gid}"])

        # Workspace mount — same path inside container as on host
        mode = "rw" if policy.file_write else "ro"
        args.extend(["-v", f"{policy.workspace_dir}:{policy.workspace_dir}:{mode}"])
        args.extend(["-w", cwd])
        for mount in policy.extra_mounts:
            args.extend(["-v", f"{mount.source}:{mount.target}:{mount.mode}"])

        if not managed:
            harness_state_dir = str(self.runtime_config.harness_state_dir(harness))
            args.extend(["-v", f"{harness_state_dir}:{harness_state_dir}:rw"])

        # uv-managed Python virtualenvs
        uv_home = Path.home() / ".local" / "share" / "uv"
        if uv_home.exists():
            args.extend(["-v", f"{uv_home}:{uv_home}:ro"])

        # Environment variables
        for key, value in inner_env.items():
            args.extend(["--env", f"{key}={value}"])

        # Graceful shutdown timeout
        args.extend(["--stop-timeout", str(min(policy.timeout_sec, 30))])

        # Container name for traceability
        args.extend(["--name", f"eval-{harness}-{uuid.uuid4().hex[:8]}"])

        # Image + command
        if inner_command.endswith(".js") or inner_command.endswith(".mjs"):
            args.extend([docker_image, "node", inner_command, *inner_args])
        else:
            args.extend([docker_image, inner_command, *inner_args])

        return _DockerCommand(
            command="docker",
            args=args,
            env=filter_env(self.runtime_config.subprocess_env),
            cwd=policy.workspace_dir,
        )


# ─── Docker image helpers ───


def resolve_docker_image(harness: str, runtime_config: RuntimeConfig) -> str:
    """Resolve the Docker image for a harness."""
    yaml_key = harness.replace("-", "_")
    cfg = runtime_config.harness_config.get(yaml_key) or runtime_config.harness_config.get(harness)
    if cfg and cfg.get("docker_image"):
        return cfg["docker_image"]

    if runtime_config.docker_image:
        return runtime_config.docker_image

    managed = get_managed_harness_image(harness, runtime_config)
    if managed:
        return managed

    raise ValueError(f"No docker image configured for harness: {harness}")


def is_docker_managed_image(harness: str, runtime_config: RuntimeConfig) -> bool:
    """Whether the resolved docker image is managed by this repo."""
    image = resolve_docker_image(harness, runtime_config)
    return selected_image_is_managed(harness, image, runtime_config)


def get_managed_harness_image(
    harness: str | None,
    runtime_config: RuntimeConfig,
) -> str | None:
    """Return the managed image tag for a harness, if this repo owns it."""
    if harness is None:
        return None
    cls = _get_managed_harness_adapter_cls(harness)
    if cls is None:
        return None
    version = _get_harness_version(harness, runtime_config)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", version).strip(".-") or "latest"
    return f"agent-harness-eval-{harness}:{tag}"


def _get_managed_harness_adapter_cls(
    harness: str,
):
    from ..adapters import list_registered_adapters

    cls = list_registered_adapters().get(harness)
    if not getattr(cls, "managed_docker_image", False):
        return None
    return cls


def selected_image_is_managed(
    harness: str | None,
    selected_image: str,
    runtime_config: RuntimeConfig,
) -> bool:
    managed = get_managed_harness_image(harness, runtime_config)
    return managed is not None and managed == selected_image


_MANAGED_BASE_IMAGE = "agent-harness-eval-base:latest"


async def ensure_managed_harness_images(
    project_root: Path,
    harnesses: list[str],
    runtime_config: RuntimeConfig,
    selected_images: dict[str, str] | None = None,
) -> None:
    """Build missing managed images for the requested harnesses."""
    for harness in harnesses:
        managed_image = get_managed_harness_image(harness, runtime_config)
        if managed_image is None:
            continue
        selected_image = (selected_images or {}).get(harness)
        if selected_image != managed_image:
            continue

        inspect_result = await run_subprocess(
            "docker",
            ["image", "inspect", managed_image],
            cwd=str(project_root),
            timeout_ms=10_000,
            filtered_env=False,
        )
        if inspect_result.exit_code == 0:
            continue

        build_script_path = project_root / "docker" / harness / "build_docker_image.sh"
        if not build_script_path.is_file():
            raise FileNotFoundError(f"Managed docker build script not found for {harness}: {build_script_path}")
        await _ensure_managed_base_image(project_root)
        build_env = _managed_harness_build_env(harness, runtime_config)
        print(f"Building Docker image for {harness}: {managed_image}")
        build_result = await run_subprocess(
            "bash",
            [str(build_script_path), _get_harness_version(harness, runtime_config), managed_image],
            cwd=str(project_root),
            env=build_env,
            timeout_ms=10 * 60 * 1000,
            filtered_env=False,
        )
        if build_result.exit_code != 0:
            detail = (build_result.stderr or build_result.stdout or "").strip()
            raise RuntimeError(f"Failed to build managed docker image for {harness}: {managed_image}\n{detail[:1000]}")
        print(f"Built Docker image for {harness}: {managed_image}")


async def _ensure_managed_base_image(project_root: Path) -> None:
    inspect_result = await run_subprocess(
        "docker",
        ["image", "inspect", _MANAGED_BASE_IMAGE],
        cwd=str(project_root),
        timeout_ms=10_000,
        filtered_env=False,
    )
    if inspect_result.exit_code == 0:
        return

    build_script_path = project_root / "docker" / "base" / "build_docker_image.sh"
    if not build_script_path.is_file():
        raise FileNotFoundError(f"Managed docker base build script not found: {build_script_path}")

    print(f"Building shared Docker base image: {_MANAGED_BASE_IMAGE}")
    build_result = await run_subprocess(
        "bash",
        [str(build_script_path), _MANAGED_BASE_IMAGE],
        cwd=str(project_root),
        timeout_ms=10 * 60 * 1000,
        filtered_env=False,
    )
    if build_result.exit_code != 0:
        detail = (build_result.stderr or build_result.stdout or "").strip()
        raise RuntimeError(f"Failed to build managed docker base image: {_MANAGED_BASE_IMAGE}\n{detail[:1000]}")
    print(f"Built shared Docker base image: {_MANAGED_BASE_IMAGE}")


def _get_harness_version(harness: str, runtime_config: RuntimeConfig) -> str:
    yaml_key = harness.replace("-", "_")
    cfg = runtime_config.harness_config.get(yaml_key) or runtime_config.harness_config.get(harness) or {}
    version = str(cfg.get("version") or "latest").strip()
    return version or "latest"


def _managed_harness_build_env(
    harness: str,
    runtime_config: RuntimeConfig,
) -> dict[str, str] | None:
    cls = _get_managed_harness_adapter_cls(harness)
    if cls is None:
        return None
    return cls.managed_docker_build_env(runtime_config)
