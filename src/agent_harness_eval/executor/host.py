"""Host executor: runs commands directly on the host machine."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from ..utils.subprocess import SubprocessResult, run_subprocess
from . import ExecutionPolicy, Executor, register_executor


@dataclass
class _HostCommand:
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str


@register_executor
class HostExecutor(Executor):
    """Execute commands directly on the host with env filtering and filesystem restrictions."""

    name = "host"

    async def execute(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
        timeout_ms: int,
    ) -> SubprocessResult:
        wrapped = self._wrap(policy, inner_command, inner_args, inner_env)
        return await run_subprocess(
            wrapped.command,
            wrapped.args,
            cwd=wrapped.cwd,
            env=wrapped.env,
            timeout_ms=timeout_ms,
        )

    def restore_workspace(self, workspace_dir: str) -> None:
        """Restore write permissions after a host run."""
        try:
            subprocess.run(
                ["chmod", "-R", "u+w", workspace_dir],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

    def _wrap(
        self,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
    ) -> _HostCommand:
        cwd = policy.cwd or policy.workspace_dir
        if not policy.file_write:
            try:
                subprocess.run(
                    ["chmod", "-R", "a-w", policy.workspace_dir],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

        env_with_constraints = dict(inner_env)
        if not policy.network:
            env_with_constraints["EVAL_BOUNDARY_INTERNET"] = "disabled"
        if not policy.shell:
            env_with_constraints["EVAL_BOUNDARY_SHELL"] = "disabled"
        if not policy.file_write:
            env_with_constraints["EVAL_BOUNDARY_FILE_WRITE"] = "disabled"

        return _HostCommand(
            command=inner_command,
            args=inner_args,
            env=env_with_constraints,
            cwd=cwd,
        )
