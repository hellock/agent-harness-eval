"""Harness preflight validation.

Layered design:
  1. **Static** (instant, deterministic) — config + install checks
  2. **Provider probe** (one minimal API call per effective provider x model, parallel)
  3. **Harness probe** (one LLM round trip per harness x model, parallel) — trace contract

Each layer gates the next: a harness that fails static checks is never probed.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from .adapters.interface import HarnessAdapter
from .config.providers import ModelSpec, ProviderConfig
from .config.runtime import RuntimeConfig
from .constants import (
    ERROR_DETAIL_MAX_CHARS,
    PREFLIGHT_PROVIDER_PROBE_TIMEOUT_MS,
    PREFLIGHT_TIMEOUT_BASELINE_SEC,
    PREFLIGHT_TIMEOUT_DOCKER_SEC,
)
from .executor import resolve_executor_backend
from .task import Task, ToolBoundary
from .types import (
    EvalConfig,
    FailureOrigin,
    RunResult,
)
from .utils.failure_origin import detect_failure_origin_from_error


@dataclass
class HarnessPreflightResult:
    harness: str
    model: str
    status: str  # 'passed' | 'failed'
    stage: str  # 'config' | 'install' | 'provider' | 'probe'
    code: str
    details: str | None = None
    failure_origin: FailureOrigin | None = None
    latency_sec: float = 0.0
    trace_events: int = 0
    tool_calls: int = 0
    warnings: list[str] | None = None


RETRYABLE_PROBE_CODES = {
    "empty_final_text",
    "preflight_timeout",
    "preflight_run_failed",
    "provider_api_error",
}

RETRYABLE_PROVIDER_PROBE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class _ProviderProbeTarget:
    harness: str
    adapter: HarnessAdapter | None
    model_spec: ModelSpec
    provider: ProviderConfig


@dataclass(frozen=True)
class _ProviderProbeResult:
    status: str
    code: str
    details: str | None = None
    failure_origin: FailureOrigin | None = None
    latency_sec: float = 0.0


class RetryableProviderProbeError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _seconds_since(started_at: float) -> float:
    import time

    return time.time() - started_at


# ─── Public entry point ───


async def run_harness_preflight(
    config: EvalConfig,
    adapters: dict[str, HarnessAdapter],
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    """Run preflight validation for all harnesses.

    Layer 1 (static) runs synchronously and is instant.
    Layer 2 (provider probe) deduplicates by effective provider x model.
    Layer 3 (harness probe) runs all surviving (harness, model) pairs in parallel.
    """
    results: list[HarnessPreflightResult] = []
    models = config.models or [config.model_spec]
    judge_ok = True

    # ─── Layer 1: Static checks (config + install) ───
    provider_probe_work: list[_ProviderProbeTarget] = []

    for harness in config.harnesses:
        adapter = adapters.get(harness)
        if not adapter:
            results.append(
                HarnessPreflightResult(
                    harness=harness,
                    model=config.model_spec.label,
                    status="failed",
                    stage="config",
                    code="missing_adapter",
                    details=f"No adapter registered for {harness}",
                    failure_origin="adapter",
                )
            )
            continue

        # Config check: provider exists and api_format matches
        harness_models_ok: list[tuple[ModelSpec, ProviderConfig]] = []
        for model_spec in models:
            try:
                provider = adapter.resolve_provider(model_spec)
            except ValueError as config_error:
                results.append(
                    HarnessPreflightResult(
                        harness=harness,
                        model=model_spec.label,
                        status="failed",
                        stage="config",
                        code="provider_config_error",
                        details=str(config_error)[:ERROR_DETAIL_MAX_CHARS],
                        failure_origin="adapter",
                    )
                )
                continue
            harness_models_ok.append((model_spec, provider))

        if not harness_models_ok:
            continue

        # Install check: CLI binary exists and responds to --version.
        install_result = await adapter.verify_install()
        if not install_result.ok:
            for model_spec, _provider in harness_models_ok:
                results.append(
                    HarnessPreflightResult(
                        harness=harness,
                        model=model_spec.label,
                        status="failed",
                        stage="install",
                        code="install_check_failed",
                        details=install_result.error or "binary not found",
                        failure_origin="adapter",
                    )
                )
            continue

        for model_spec, provider in harness_models_ok:
            provider_probe_work.append(
                _ProviderProbeTarget(
                    harness=harness,
                    adapter=adapter,
                    model_spec=model_spec,
                    provider=provider,
                )
            )

    judge_targets = [
        ("judge", config.judge_model_spec),
        ("secondary-judge", config.secondary_judge_model_spec),
    ]
    for judge_name, judge_spec in judge_targets:
        if judge_spec is None:
            continue
        provider = runtime_config.providers.get(judge_spec.provider)
        if provider is None:
            judge_ok = False
            results.append(
                HarnessPreflightResult(
                    harness=judge_name,
                    model=judge_spec.label,
                    status="failed",
                    stage="config",
                    code="provider_config_error",
                    details=f'No provider "{judge_spec.provider}" configured for {judge_name} model {judge_spec.label}.',
                    failure_origin="provider",
                )
            )
            continue
        provider_probe_work.append(
            _ProviderProbeTarget(
                harness=judge_name,
                adapter=None,
                model_spec=judge_spec,
                provider=provider,
            )
        )

    # ─── Layer 2: Provider probe (deduplicated provider x model) ───
    harness_probe_work: list[tuple[str, HarnessAdapter, ModelSpec]] = []
    if provider_probe_work:
        max_attempts = runtime_config.preflight_max_attempts
        probe_groups: dict[tuple[Any, ...], list[_ProviderProbeTarget]] = {}
        for target in provider_probe_work:
            probe_groups.setdefault(_provider_probe_key(target.provider, target.model_spec), []).append(target)

        provider_probe_results = await asyncio.gather(
            *[
                _probe_provider_model(group[0].model_spec, group[0].provider, max_attempts)
                for group in probe_groups.values()
            ]
        )

        for group, provider_result in zip(probe_groups.values(), provider_probe_results, strict=True):
            if provider_result.status == "failed":
                for target in group:
                    if target.adapter is None:
                        judge_ok = False
                    results.append(
                        HarnessPreflightResult(
                            harness=target.harness,
                            model=target.model_spec.label,
                            status="failed",
                            stage="provider",
                            code=provider_result.code,
                            details=provider_result.details,
                            failure_origin=provider_result.failure_origin,
                            latency_sec=provider_result.latency_sec,
                        )
                    )
                continue

            for target in group:
                if target.adapter is None:
                    continue
                harness_probe_work.append((target.harness, target.adapter, target.model_spec))

    # ─── Layer 3: Harness probe (parallel LLM round trips) ───
    if harness_probe_work:
        max_attempts = runtime_config.preflight_max_attempts
        probe_results = await asyncio.gather(
            *[
                _probe_harness(harness, adapter, model_spec, runtime_config, max_attempts)
                for harness, adapter, model_spec in harness_probe_work
            ]
        )
        for probe_batch in probe_results:
            results.extend(probe_batch)

    healthy = [
        h
        for h in config.harnesses
        if any(r.harness == h for r in results) and all(r.status == "passed" for r in results if r.harness == h)
    ]

    return {"results": results, "healthy_harnesses": healthy, "judge_ok": judge_ok}


def _provider_probe_key(provider: ProviderConfig, model_spec: ModelSpec) -> tuple[Any, ...]:
    return (
        provider.base_url,
        provider.api_key,
        provider.api_format,
        tuple(sorted((provider.extra_headers or {}).items())),
        model_spec.model,
    )


async def _probe_provider_model(
    model_spec: ModelSpec,
    provider: ProviderConfig,
    max_attempts: int,
) -> _ProviderProbeResult:
    import time

    for attempt in range(1, max_attempts + 1):
        started_at = time.time()
        try:
            await _do_provider_probe_request(model_spec, provider)
            return _ProviderProbeResult(
                status="passed",
                code="ok",
                latency_sec=_seconds_since(started_at),
            )
        except RetryableProviderProbeError as error:
            if attempt < max_attempts:
                print(
                    f"  preflight provider/{model_spec.label} provider_api_error "
                    f"(attempt {attempt}/{max_attempts}), retrying..."
                )
                continue
            return _ProviderProbeResult(
                status="failed",
                code="provider_api_error",
                details=str(error)[:ERROR_DETAIL_MAX_CHARS],
                failure_origin="provider",
                latency_sec=_seconds_since(started_at),
            )
        except Exception as error:
            detail = str(error)
            failure = detect_failure_origin_from_error(detail)
            return _ProviderProbeResult(
                status="failed",
                code=failure.get("infra_error_code") or "provider_probe_failed",
                details=detail[:ERROR_DETAIL_MAX_CHARS],
                failure_origin=failure.get("failure_origin"),
                latency_sec=_seconds_since(started_at),
            )

    raise AssertionError("provider probe exhausted without returning")


async def _do_provider_probe_request(
    model_spec: ModelSpec,
    provider: ProviderConfig,
) -> None:
    timeout = httpx.Timeout(PREFLIGHT_PROVIDER_PROBE_TIMEOUT_MS / 1000)
    prompt = "Reply with OK."

    if provider.api_format == "anthropic":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": provider.api_key,
            "anthropic-version": "2023-06-01",
            **(provider.extra_headers or {}),
        }
        body = {
            "model": model_spec.model,
            "max_tokens": 8,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
    elif provider.api_format == "openai-responses":
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider.api_key}",
            **(provider.extra_headers or {}),
        }
        body = {
            "model": model_spec.model,
            "temperature": 0,
            "input": prompt,
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider.api_key}",
            **(provider.extra_headers or {}),
        }
        body = {
            "model": model_spec.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(provider.endpoint_url(), json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise RetryableProviderProbeError(
                f"Provider probe timed out after {PREFLIGHT_PROVIDER_PROBE_TIMEOUT_MS}ms"
            ) from exc
        except httpx.HTTPError as exc:
            raise RetryableProviderProbeError(f"Provider probe transport error: {exc}") from exc

    if response.status_code >= 400:
        message = f"Provider API error {response.status_code}: {response.text[:200]}"
        if response.status_code in RETRYABLE_PROVIDER_PROBE_STATUSES:
            raise RetryableProviderProbeError(message, response.status_code)
        raise RuntimeError(message)


# ─── Probe implementation ───


async def _probe_harness(
    harness: str,
    adapter: HarnessAdapter,
    model_spec: ModelSpec,
    runtime_config: RuntimeConfig,
    max_attempts: int,
) -> list[HarnessPreflightResult]:
    """Run a single LLM probe for a (harness, model) pair with retries.

    Success criteria (no marker check — that tests LLM behavior, not infra):
      - adapter.run() returns status=completed
      - trace is non-empty
      - final_text is non-empty
    """
    import time

    results: list[HarnessPreflightResult] = []

    for attempt in range(1, max_attempts + 1):
        before_len = len(results)
        run_id = f"preflight-{uuid.uuid4()}"
        prepared = None
        started_at = time.time()

        per_run_task = _create_preflight_task(
            runtime_config,
            requires_tool_use=adapter.emits_paired_trace_events,
        )

        try:
            prepared = adapter.prepare(per_run_task, run_id)
        except Exception as error:
            detail = str(error)
            failure = detect_failure_origin_from_error(detail)
            results.append(
                HarnessPreflightResult(
                    harness=harness,
                    model=model_spec.label,
                    status="failed",
                    stage="probe",
                    code=failure.get("infra_error_code") or "preflight_prepare_failed",
                    details=detail[:ERROR_DETAIL_MAX_CHARS],
                    failure_origin=failure.get("failure_origin"),
                    latency_sec=_seconds_since(started_at),
                )
            )
            break  # prepare failures are not retryable

        try:
            run_result = await adapter.run(prepared, model_spec.label)
            trace_events = len(run_result.trace)
            tool_calls = run_result.metrics.tool_calls

            if run_result.status == "timed_out":
                results.append(
                    HarnessPreflightResult(
                        harness=harness,
                        model=model_spec.label,
                        status="failed",
                        stage="probe",
                        code="preflight_timeout",
                        details="Preflight probe timed out",
                        failure_origin="unknown",
                        latency_sec=_seconds_since(started_at),
                        trace_events=trace_events,
                        tool_calls=tool_calls,
                    )
                )
            elif run_result.status == "failed":
                results.append(
                    HarnessPreflightResult(
                        harness=harness,
                        model=model_spec.label,
                        status="failed",
                        stage="probe",
                        code=run_result.infra_error_code or "preflight_run_failed",
                        details=_pick_error_detail(run_result),
                        failure_origin=run_result.failure_origin or "unknown",
                        latency_sec=_seconds_since(started_at),
                        trace_events=trace_events,
                        tool_calls=tool_calls,
                    )
                )
            elif trace_events == 0:
                results.append(
                    HarnessPreflightResult(
                        harness=harness,
                        model=model_spec.label,
                        status="failed",
                        stage="probe",
                        code="empty_trace",
                        details="Completed run with empty trace",
                        failure_origin="adapter",
                        latency_sec=_seconds_since(started_at),
                        trace_events=trace_events,
                        tool_calls=tool_calls,
                    )
                )
            elif not run_result.final_text.strip():
                results.append(
                    HarnessPreflightResult(
                        harness=harness,
                        model=model_spec.label,
                        status="failed",
                        stage="probe",
                        code="empty_final_text",
                        details="Completed run with empty final_text",
                        failure_origin="adapter",
                        latency_sec=_seconds_since(started_at),
                        trace_events=trace_events,
                        tool_calls=tool_calls,
                    )
                )
            else:
                # Success — check for degraded trace capability
                warnings: list[str] = []
                completed_tools = [e for e in run_result.trace if e.type == "tool_call_completed"]
                if not completed_tools and tool_calls > 0:
                    warnings.append(
                        f"count-only-tool-trace: metrics.tool_calls={tool_calls} but no trace tool_call events"
                    )

                results.append(
                    HarnessPreflightResult(
                        harness=harness,
                        model=model_spec.label,
                        status="passed",
                        stage="probe",
                        code="ok",
                        latency_sec=_seconds_since(started_at),
                        trace_events=trace_events,
                        tool_calls=tool_calls,
                        warnings=warnings if warnings else None,
                    )
                )

        except Exception as error:
            detail = str(error)
            failure = detect_failure_origin_from_error(detail)
            results.append(
                HarnessPreflightResult(
                    harness=harness,
                    model=model_spec.label,
                    status="failed",
                    stage="probe",
                    code=failure.get("infra_error_code") or "preflight_run_failed",
                    details=detail[:ERROR_DETAIL_MAX_CHARS],
                    failure_origin=failure.get("failure_origin"),
                    latency_sec=_seconds_since(started_at),
                )
            )
        finally:
            if prepared:
                try:
                    adapter.cleanup(prepared)
                except Exception as cleanup_error:
                    results.append(
                        HarnessPreflightResult(
                            harness=harness,
                            model=model_spec.label,
                            status="failed",
                            stage="probe",
                            code="preflight_cleanup_failed",
                            details=str(cleanup_error)[:ERROR_DETAIL_MAX_CHARS],
                            failure_origin="adapter",
                            latency_sec=_seconds_since(started_at),
                        )
                    )

        # Retry decision
        fresh = results[before_len:]
        retryable = next(
            (r for r in fresh if r.status == "failed" and r.code in RETRYABLE_PROBE_CODES),
            None,
        )
        if retryable and attempt < max_attempts:
            del results[before_len:]
            print(
                f"  preflight {harness}/{model_spec.label} {retryable.code} "
                f"(attempt {attempt}/{max_attempts}), retrying..."
            )
            continue
        break

    return results


# ─── Preflight task ───


def _create_preflight_task(
    runtime_config: RuntimeConfig,
    *,
    requires_tool_use: bool = True,
) -> Task:
    """Create a minimal task for probing harness health.

    No marker token — we only check that the harness produces a completed run
    with non-empty trace and final_text.  Marker echo tests LLM instruction
    following, which is not preflight's concern.
    """
    executor_backend = resolve_executor_backend(runtime_config)
    max_preflight_sec = PREFLIGHT_TIMEOUT_DOCKER_SEC if executor_backend == "docker" else PREFLIGHT_TIMEOUT_BASELINE_SEC
    timeout = min(runtime_config.preflight_max_attempts * 60, max_preflight_sec)

    if requires_tool_use:
        return Task(
            id="__preflight__.01",
            category="security",
            description="Preflight probe: connectivity, trace contract, adapter health.",
            user_query=("Read the file PRECHECK.md from the workspace and reply with its contents."),
            workspace_files=[{"path": "PRECHECK.md", "content": "preflight-ok\n"}],
            tool_boundary=ToolBoundary(
                internet="disabled",
                shell="enabled",
                file_write="enabled",
            ),
            graders=[],
            timeout_sec=timeout,
        )

    return Task(
        id="__preflight__.01",
        category="security",
        description="Preflight probe (no-tool variant).",
        user_query="Reply with: preflight-ok",
        tool_boundary=ToolBoundary(
            internet="disabled",
            shell="enabled",
            file_write="enabled",
        ),
        graders=[],
        timeout_sec=timeout,
    )


# ─── Artifact output ───


def write_preflight_artifacts(
    output_dir: str,
    results: list[HarnessPreflightResult],
) -> None:
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    data = [
        {
            "harness": r.harness,
            "model": r.model,
            "status": r.status,
            "stage": r.stage,
            "code": r.code,
            "details": r.details,
            "failure_origin": r.failure_origin,
            "latency_sec": r.latency_sec,
            "trace_events": r.trace_events,
            "tool_calls": r.tool_calls,
            "warnings": r.warnings,
        }
        for r in results
    ]
    with open(os.path.join(data_dir, "preflight.json"), "w") as f:
        json.dump(data, f, indent=2)


# ─── Helpers ───


def _pick_error_detail(result: RunResult) -> str:
    if result.infra_error_details:
        return result.infra_error_details[:ERROR_DETAIL_MAX_CHARS]
    task_failed = next((e for e in result.trace if e.type == "task_failed"), None)
    if task_failed and task_failed.error:
        return task_failed.error[:ERROR_DETAIL_MAX_CHARS]
    return result.final_text[:ERROR_DETAIL_MAX_CHARS]
