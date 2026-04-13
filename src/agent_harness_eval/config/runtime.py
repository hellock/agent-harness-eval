"""Runtime configuration context for a single CLI invocation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .providers import ProviderConfig, resolve_providers


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable runtime configuration for a single evaluation run."""

    project_root: Path
    executor_backend: str = "host"
    eval_state_dir: Path | None = None
    judge_timeout_ms: int = 60_000
    judge_max_attempts: int = 4
    judge_retry_base_ms: int = 1000
    preflight_max_attempts: int = 3
    skip_capability_probes: bool = False
    keep_workspace: str = ""
    docker_image: str | None = None
    docker_network_none: bool = False
    custom_pricing: dict[str, float] | None = None
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    harness_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    _process_env: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def state_dir(self) -> Path:
        """Resolved eval state directory."""
        if self.eval_state_dir:
            return self.eval_state_dir
        return self.project_root / ".harnesses"

    def harness_state_dir(self, harness: str) -> Path:
        """Per-harness state directory."""
        d = self.state_dir / harness
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolve_harness_bin(self, harness: str, binary: str) -> str:
        """Resolve binary path for a harness CLI."""
        local_bin = self.state_dir / harness / "node_modules" / ".bin" / binary
        if local_bin.exists():
            return str(Path(os.path.abspath(local_bin)))
        custom_bin = self.state_dir / harness / "bin" / binary
        if custom_bin.exists():
            return str(Path(os.path.abspath(custom_bin)))
        return binary

    @property
    def subprocess_env(self) -> dict[str, str]:
        """Env dict for subprocesses, copied from the bootstrap snapshot."""
        return dict(self._process_env)


def build_runtime_config(
    project_root: Path,
    eval_yaml: dict[str, Any],
) -> RuntimeConfig:
    """Build RuntimeConfig from loaded YAML config + environment."""
    env_snapshot = dict(os.environ)
    providers_config = dict(eval_yaml.get("providers") or {})
    harnesses_config = dict(eval_yaml.get("harnesses") or {})
    pricing_config = eval_yaml.get("pricing")

    return RuntimeConfig(
        project_root=Path(os.path.abspath(project_root)),
        executor_backend=str(eval_yaml["executor"]).strip(),
        eval_state_dir=(Path(env_snapshot["EVAL_STATE_DIR"]) if env_snapshot.get("EVAL_STATE_DIR") else None),
        judge_timeout_ms=int(env_snapshot.get("EVAL_JUDGE_TIMEOUT_MS", "60000")),
        judge_max_attempts=int(env_snapshot.get("EVAL_JUDGE_MAX_ATTEMPTS", "4")),
        judge_retry_base_ms=int(env_snapshot.get("EVAL_JUDGE_RETRY_BASE_MS", "1000")),
        preflight_max_attempts=int(env_snapshot.get("EVAL_PREFLIGHT_MAX_ATTEMPTS", "3")),
        skip_capability_probes=env_snapshot.get("EVAL_SKIP_CAPABILITY_PROBES") == "1",
        keep_workspace=env_snapshot.get("EVAL_KEEP_WORKSPACE", ""),
        docker_image=env_snapshot.get("EVAL_DOCKER_IMAGE"),
        docker_network_none=env_snapshot.get("EVAL_DOCKER_NETWORK_NONE") == "1",
        custom_pricing=(pricing_config if pricing_config else None),
        providers=resolve_providers(env_snapshot, providers_config),
        harness_config=harnesses_config,
        _process_env=env_snapshot,
    )
