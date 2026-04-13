"""Harness adapter interface and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar

from ..config.providers import ApiFormat, ModelSpec, ProviderConfig
from ..config.runtime import RuntimeConfig
from ..constants import (
    ERROR_DETAIL_MAX_CHARS,
    STDERR_PREVIEW_MAX_CHARS,
    VERIFY_INSTALL_TIMEOUT_MS,
    VERSION_ERROR_MAX_CHARS,
)
from ..executor import ExecutionPolicy, Executor
from ..task import Task
from ..types import CanonicalTraceEvent, FailureOrigin, RunMetrics, RunResult
from ..utils.cost import ModelPricing
from ..utils.failure_origin import detect_failure_origin_from_error
from ..utils.subprocess import SubprocessResult
from ..utils.workspace import RunLayout


@dataclass(slots=True)
class PreparedRun:
    task: Task
    layout: RunLayout
    execution_policy: ExecutionPolicy
    env: dict[str, str] = field(default_factory=dict)
    debug_artifacts: list[dict[str, str]] = field(default_factory=list)

    @property
    def workspace_dir(self) -> str:
        return self.layout.workspace_dir


@dataclass(slots=True)
class NativeMemoryFile:
    path: str
    content: str


@dataclass(slots=True)
class VerifyInstallResult:
    ok: bool
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SubprocessFailure:
    error: str
    failure_origin: FailureOrigin | None
    infra_error_code: str | None


def detect_subprocess_failure(
    result: SubprocessResult,
    *,
    command_label: str,
) -> SubprocessFailure | None:
    """Convert a non-zero subprocess exit into a structured adapter failure."""
    if result.timed_out or result.exit_code in (None, 0):
        return None

    stderr_tail = (result.stderr or "").strip()[:STDERR_PREVIEW_MAX_CHARS]
    stdout_tail = (result.stdout or "").strip()[:STDERR_PREVIEW_MAX_CHARS]

    details = [f"{command_label} exited with code {result.exit_code}"]
    if stderr_tail:
        details.append(f"stderr: {stderr_tail}")
    if stdout_tail:
        details.append(f"stdout: {stdout_tail}")

    error = "\n".join(details)
    failure = detect_failure_origin_from_error(error)
    return SubprocessFailure(
        error=error[:800],
        failure_origin=failure.get("failure_origin"),
        infra_error_code=failure.get("infra_error_code"),
    )


def _harness_config_key(harness_name: str) -> str:
    """Map CLI harness name to YAML config key (e.g. 'claude-code' → 'claude_code')."""
    return harness_name.replace("-", "_")


class HarnessAdapter(ABC):
    name: ClassVar[str]
    cli_binary: ClassVar[str | None] = None
    managed_docker_image: ClassVar[bool] = False
    required_env_vars: ClassVar[list[list[str]] | None] = None
    supported_api_formats: ClassVar[list[ApiFormat] | None] = None
    supports_native_memory: ClassVar[bool] = False
    supports_conversation_history_replay: ClassVar[bool] = False
    emits_paired_trace_events: ClassVar[bool] = False

    def __init__(self, runtime_config: RuntimeConfig, executor: Executor):
        self.runtime_config = runtime_config
        self.executor = executor

    def resolve_provider(self, model_spec: ModelSpec) -> ProviderConfig:
        """Resolve the effective provider for this adapter and model.

        Resolution order:
          1. ``harnesses.<name>.provider`` override in eval.yaml
          2. ``model_spec.provider`` (from the top-level or matrix entry)

        Raises immediately if the provider is missing or its ``api_format``
        does not match this adapter's ``supported_api_formats``.
        """
        yaml_key = _harness_config_key(self.name)
        harness_cfg = (
            self.runtime_config.harness_config.get(yaml_key) or self.runtime_config.harness_config.get(self.name) or {}
        )
        override_name = harness_cfg.get("provider")
        provider_name = str(override_name).strip() if override_name else model_spec.provider

        provider = self.runtime_config.providers.get(provider_name)
        if not provider:
            source = "harness provider override" if override_name else "model spec"
            raise ValueError(
                f'{self.name}: no provider "{provider_name}" configured '
                f"(from {source}). Available: {list(self.runtime_config.providers.keys())}"
            )

        if self.supported_api_formats and provider.api_format not in self.supported_api_formats:
            raise ValueError(
                f'{self.name}: provider "{provider_name}" has api_format '
                f'"{provider.api_format}" but {self.name} requires one of '
                f"{self.supported_api_formats}"
            )

        return provider

    def pricing_override(self) -> ModelPricing | None:
        pricing = self.runtime_config.custom_pricing
        if pricing is None:
            return None
        return ModelPricing(
            input=pricing["input"],
            output=pricing["output"],
            cache_read=pricing["cache_read"],
            cache_write=pricing["cache_write"],
        )

    async def verify_install(self) -> VerifyInstallResult:
        """Smoke test to verify the harness CLI is installed.

        Runs ``<binary> --version`` through the executor so it works in
        both host and docker modes.
        """
        try:
            import tempfile

            bin_path = self.resolve_binary()
            policy = ExecutionPolicy(
                workspace_dir=tempfile.gettempdir(),
                timeout_sec=VERIFY_INSTALL_TIMEOUT_MS // 1000,
            )
            result = await self.executor.execute(
                self.name,
                policy,
                bin_path,
                ["--version"],
                {},
                timeout_ms=VERIFY_INSTALL_TIMEOUT_MS,
            )
            if result.exit_code == 0:
                runtime_check = await self._verify_custom_docker_runtime(policy)
                if runtime_check is not None:
                    return runtime_check
                return VerifyInstallResult(ok=True, version=(result.stdout.strip() or result.stderr.strip()))
            return VerifyInstallResult(ok=False, error=result.stderr[:VERSION_ERROR_MAX_CHARS] or "non-zero exit")
        except Exception as e:
            return VerifyInstallResult(ok=False, error=str(e))

    async def _verify_custom_docker_runtime(
        self,
        policy: ExecutionPolicy,
    ) -> VerifyInstallResult | None:
        """Validate verifier-side runtime requirements for custom Docker images."""
        if self.executor.name != "docker":
            return None

        from ..executor.docker import is_docker_managed_image

        if is_docker_managed_image(self.name, self.runtime_config):
            return None

        for binary in ("bash",):
            result = await self.executor.execute(
                self.name,
                policy,
                binary,
                ["--version"],
                {},
                timeout_ms=VERIFY_INSTALL_TIMEOUT_MS,
            )
            if result.timed_out:
                return VerifyInstallResult(
                    ok=False,
                    error=f'Custom docker image for {self.name} timed out while verifying required runtime "{binary}".',
                )
            if result.exit_code != 0:
                detail = (result.stderr or result.stdout or "").strip()[:VERSION_ERROR_MAX_CHARS]
                error = f'Custom docker image for {self.name} must include "{binary}" for verifier/grader execution.'
                if detail:
                    error += f" {detail}"
                return VerifyInstallResult(ok=False, error=error)

        return None

    def resolve_binary(self) -> str:
        """Resolve the harness binary via the executor."""
        return self.executor.resolve_binary(self.name, self.cli_binary or self.name)

    async def _run_via_executor(
        self,
        prepared: PreparedRun,
        model: str,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
    ) -> SubprocessResult | RunResult:
        """Execute via the executor with standard timeout/failure handling.

        Returns:
          SubprocessResult — on successful subprocess exit (caller parses output)
          RunResult        — on timeout or subprocess failure (caller returns as-is)
        """
        result = await self.executor.execute(
            self.name,
            prepared.execution_policy,
            inner_command,
            inner_args,
            inner_env,
            timeout_ms=prepared.task.timeout_sec * 1000,
        )

        if result.timed_out:
            return self._make_result(
                prepared.task,
                model,
                "timed_out",
                "",
                [],
                RunMetrics(latency_sec=prepared.task.timeout_sec),
            )

        failure = detect_subprocess_failure(result, command_label=self.name.title())
        if failure:
            return self._make_result(
                prepared.task,
                model,
                "failed",
                result.stdout or "",
                [
                    CanonicalTraceEvent(
                        type="task_failed",
                        error=failure.error,
                        ts=datetime.now(UTC).isoformat(),
                    )
                ],
                RunMetrics(latency_sec=0),
                failure_origin=failure.failure_origin,
                infra_error_code=failure.infra_error_code,
            )

        return result

    def _make_result(
        self,
        task: Task,
        model: str,
        status: str,
        final_text: str,
        trace: list[CanonicalTraceEvent],
        metrics: RunMetrics,
        failure_origin: FailureOrigin | None = None,
        infra_error_code: str | None = None,
    ) -> RunResult:
        return RunResult(
            task_id=task.id,
            harness=self.name,
            run_id="",
            run_index=0,
            model=model,
            status=status,
            final_text=final_text,
            artifacts=[],
            trace=trace,
            metrics=metrics,
            grader_results=[],
            failure_origin=failure_origin if status == "failed" else None,
            infra_error_code=infra_error_code if status == "failed" else None,
            infra_error_details=final_text[:ERROR_DETAIL_MAX_CHARS] if status == "failed" else None,
        )

    @abstractmethod
    def prepare(self, task: Task, run_id: str) -> PreparedRun:
        """Create an isolated workspace for this run."""
        ...

    def install_memory(self, prepared: PreparedRun, files: list[NativeMemoryFile]) -> None:
        """Install durable memory entries into the harness's native memory store."""

    @abstractmethod
    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        """Execute the task and return a unified result."""
        ...

    @abstractmethod
    def cleanup(self, prepared: PreparedRun) -> None:
        """Clean up the workspace after the run."""
        ...
