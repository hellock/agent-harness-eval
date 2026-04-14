"""Task execution orchestration — runs all tasks across all harnesses."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, TypeVar

from .adapters.interface import HarnessAdapter, PreparedRun
from .config.runtime import RuntimeConfig
from .constants import (
    ERROR_DETAIL_MAX_CHARS,
    GRADER_PASS_WEIGHT,
    PREPARE_COMMAND_TIMEOUT_MS,
    STDERR_PREVIEW_MAX_CHARS,
    SUSPICIOUS_LOW_TOKEN_THRESHOLD,
)
from .executor import Executor
from .graders.interface import JudgeLLM, run_graders
from .graders.specs import GraderResult
from .task import Task
from .types import (
    CanonicalTraceEvent,
    EvalConfig,
    ModelSpec,
    RunMetrics,
    RunResult,
    run_result_from_dict,
)
from .utils.conversation import format_task_message
from .utils.cost import ModelPricing, calculate_cost_no_cache
from .utils.failure_origin import detect_failure_origin_from_error

T = TypeVar("T")
R = TypeVar("R")


async def _map_with_concurrency(
    items: list[T],
    concurrency: int,
    fn: Callable[[T, int], Awaitable[R]],
) -> list[R]:
    """Execute an async function over items with bounded concurrency."""
    results: list[R | None] = [None] * len(items)
    next_index = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal next_index
        while True:
            async with lock:
                if next_index >= len(items):
                    return
                i = next_index
                next_index += 1
            results[i] = await fn(items[i], i)

    workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(items)))]
    await asyncio.gather(*workers)
    return results  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RunPlanItem:
    """One unit of work in the flattened eval plan.

    The plan is flattened across ``(run_index, task, model, harness)`` so a single
    bounded-concurrency worker pool controls the entire eval's parallelism. See
    :func:`_build_task_plan` for the iteration order and its rationale (run_index is
    outermost so same-prompt runs are separated in time to defeat provider caches).
    """

    task: Task
    model_spec: ModelSpec
    run_index: int
    harness: str


@dataclass(frozen=True, slots=True)
class RunRequest:
    task: Task
    harness: str
    adapter: HarnessAdapter
    model_spec: ModelSpec
    model_label: str
    run_index: int
    run_id: str
    trace_dir: str


async def _run_prepare_commands(
    commands: list[str],
    prepared: PreparedRun,
    harness: str,
    executor: Executor,
) -> None:
    """Run task prepare_commands inside the execution environment.

    These are framework-level setup (e.g. `npm install typescript`) that must
    complete before the agent starts. They run with full network access
    regardless of the task's tool_boundary, since they're not agent behavior.
    """
    setup_policy = replace(
        prepared.execution_policy,
        network=True,
        file_write=True,
        shell=True,
        extra_mounts=list(prepared.execution_policy.extra_mounts),
        limits=dict(prepared.execution_policy.limits),
    )
    setup_env = _build_prepare_command_env(prepared)
    for cmd in commands:
        result = await executor.execute(
            harness,
            setup_policy,
            "sh",
            ["-lc", cmd],
            setup_env,
            timeout_ms=PREPARE_COMMAND_TIMEOUT_MS,
        )
        if result.timed_out:
            raise RuntimeError(f"prepare_commands timed out ({cmd})")
        if result.exit_code != 0:
            stderr_tail = (result.stderr or "").strip()
            stdout_tail = (result.stdout or "").strip()
            details = stderr_tail or stdout_tail or "<no output>"
            raise RuntimeError(f"prepare_commands failed ({cmd}): {details[:STDERR_PREVIEW_MAX_CHARS]}")


def _build_prepare_command_env(prepared: PreparedRun) -> dict[str, str]:
    env = dict(prepared.env)
    layout = prepared.layout
    state_dir = layout.state_dir
    xdg_config_home = os.path.join(state_dir, ".config")
    xdg_data_home = os.path.join(state_dir, ".local", "share")
    xdg_cache_home = os.path.join(state_dir, ".cache")
    for path in (xdg_config_home, xdg_data_home, xdg_cache_home):
        os.makedirs(path, exist_ok=True)

    env.setdefault("HOME", state_dir)
    env.setdefault("USERPROFILE", state_dir)
    env.setdefault("XDG_CONFIG_HOME", xdg_config_home)
    env.setdefault("XDG_DATA_HOME", xdg_data_home)
    env.setdefault("XDG_CACHE_HOME", xdg_cache_home)
    env.setdefault("EVAL_RUN_ROOT", layout.root_dir)
    env.setdefault("EVAL_INPUT_DIR", layout.input_dir)
    env.setdefault("EVAL_WORKSPACE_DIR", layout.workspace_dir)
    env.setdefault("EVAL_STATE_DIR", layout.state_dir)
    env.setdefault("EVAL_OUTPUT_DIR", layout.output_dir)
    return env


def _persist_debug_artifacts(
    prepared: PreparedRun | None,
    trace_dir: str,
) -> None:
    if not prepared:
        return

    raw_dir = os.path.join(trace_dir, "raw")
    output_dir = prepared.layout.output_dir
    copied_any = False

    if output_dir and os.path.isdir(output_dir) and os.listdir(output_dir):
        os.makedirs(raw_dir, exist_ok=True)
        shutil.copytree(output_dir, os.path.join(raw_dir, "run-output"), dirs_exist_ok=True)
        copied_any = True

    if not prepared.debug_artifacts:
        return

    if not copied_any:
        os.makedirs(raw_dir, exist_ok=True)
    for artifact in prepared.debug_artifacts:
        src = artifact.get("path", "")
        dest_name = artifact.get("dest_name") or os.path.basename(src)
        dest = os.path.join(raw_dir, dest_name)
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)
        except Exception:
            pass


def _score_result(result: RunResult) -> float:
    passed_graders = sum(1 for g in result.grader_results if g.passed)
    quality_score = sum(
        g.score for g in result.grader_results if g.grader_type == "rubric_judge" and g.score is not None
    )
    return passed_graders * GRADER_PASS_WEIGHT + quality_score


def _failed_run_result(
    *,
    task_id: str,
    harness: str,
    run_id: str,
    run_index: int,
    model_label: str,
    error_text: str,
    failure_origin: str | None,
    infra_error_code: str | None,
) -> RunResult:
    return RunResult(
        task_id=task_id,
        harness=harness,
        run_id=run_id,
        run_index=run_index,
        model=model_label,
        status="failed",
        final_text="",
        trace=[
            CanonicalTraceEvent(
                type="task_failed",
                error=error_text,
                ts=datetime.now(UTC).isoformat(),
            )
        ],
        metrics=RunMetrics(),
        grader_results=[],
        failure_origin=failure_origin,
        infra_error_code=infra_error_code,
        infra_error_details=error_text[:ERROR_DETAIL_MAX_CHARS],
    )


def _check_adapter_support(
    adapter: HarnessAdapter,
    task: Task,
) -> list[str]:
    skip_reasons: list[str] = []
    native_memory = task.native_memory
    if native_memory and native_memory.files and not adapter.supports_native_memory:
        skip_reasons.append("native_memory required but not supported")

    if task.conversation_history and not adapter.supports_conversation_history_replay:
        skip_reasons.append("conversation_history required but not supported")

    return skip_reasons


async def _finalize_prepared_run(
    *,
    adapter: HarnessAdapter,
    prepared: PreparedRun | None,
    result: RunResult,
    trace_dir: str,
    keep_workspace: str,
) -> None:
    """Persist debug artifacts and clean up the workspace.

    The sync filesystem work (``shutil.copytree`` for debug artifacts,
    ``shutil.rmtree`` inside ``adapter.cleanup``) is offloaded to the default
    thread pool so it doesn't block the event loop while other concurrent runs
    are active.
    """
    if not prepared:
        return

    await asyncio.to_thread(_persist_debug_artifacts, prepared, trace_dir)

    any_fail = result.status != "completed" or any(not grader.passed for grader in result.grader_results)
    should_keep = keep_workspace == "always" or (keep_workspace == "on_failure" and any_fail)

    if not should_keep:
        try:
            await asyncio.to_thread(adapter.cleanup, prepared)
        except Exception:
            pass
        return

    try:
        with open(os.path.join(trace_dir, "kept_workspace.txt"), "w") as f:
            f.write(prepared.workspace_dir + "\n")
        with open(os.path.join(trace_dir, "kept_run_root.txt"), "w") as f:
            f.write(prepared.layout.root_dir + "\n")
    except Exception:
        pass


async def _prepare_run(
    adapter: HarnessAdapter,
    task: Task,
    run_id: str,
    harness: str,
    runtime_config: RuntimeConfig,
) -> PreparedRun:
    """Prepare the workspace: adapter.prepare + prepare_commands.

    ``adapter.prepare`` is sync and can do heavy filesystem work (``shutil.copytree``,
    many ``mkdir``/file writes for ``workspace_files``). Offloading to a thread keeps
    the event loop responsive while other concurrent runs progress.
    """
    prepared = await asyncio.to_thread(adapter.prepare, task, run_id)

    if task.prepare_commands:
        await _run_prepare_commands(
            task.prepare_commands,
            prepared,
            harness,
            adapter.executor,
        )

    return prepared


def _install_run_memory(
    adapter: HarnessAdapter,
    prepared: PreparedRun,
    task: Task,
) -> None:
    """Install native memory files if the task defines them."""
    native_memory = task.native_memory
    if native_memory and native_memory.files:
        from .adapters.interface import NativeMemoryFile

        memory_files = [
            NativeMemoryFile(path=file_info["path"], content=file_info["content"]) for file_info in native_memory.files
        ]
        adapter.install_memory(prepared, memory_files)


async def _execute_and_grade(
    adapter: HarnessAdapter,
    prepared: PreparedRun,
    model_spec: ModelSpec,
    model_label: str,
    run_id: str,
    run_index: int,
    task: Task,
    judge_llm: JudgeLLM | None,
    pricing_override: ModelPricing | None,
    runtime_config: RuntimeConfig,
) -> RunResult:
    """Run the adapter, compute cost, and grade the result."""
    provider_name = adapter.resolve_provider_name(model_spec)
    async with runtime_config.provider_slot(provider_name):
        result = await adapter.run(prepared, model_label)
    result.model = model_label
    result.run_id = run_id
    result.run_index = run_index

    # Normalize total_tokens centrally — always in + out + cache_read + cache_write.
    expected_total = (
        result.metrics.input_tokens
        + result.metrics.output_tokens
        + result.metrics.cache_read_tokens
        + result.metrics.cache_write_tokens
    )
    if expected_total > 0:
        result.metrics.total_tokens = expected_total

    # Compute cost centrally — adapters only need to populate token counts.
    if result.metrics.cost_usd == 0.0 and result.metrics.total_tokens > 0:
        from .utils.cost import calculate_cost

        result.metrics.cost_usd = calculate_cost(
            model_label,
            result.metrics.input_tokens,
            result.metrics.output_tokens,
            result.metrics.cache_read_tokens,
            result.metrics.cache_write_tokens,
            pricing=pricing_override,
        )

    result.metrics.cost_usd_no_cache = calculate_cost_no_cache(
        model_label,
        result.metrics.input_tokens,
        result.metrics.output_tokens,
        result.metrics.cache_read_tokens,
        result.metrics.cache_write_tokens,
        pricing=pricing_override,
    )

    adapter.executor.restore_workspace(prepared.workspace_dir)
    grader_env = _build_prepare_command_env(prepared) if prepared else None
    result.grader_results = await run_graders(
        task,
        result,
        judge_llm,
        prepared.workspace_dir,
        executor=adapter.executor,
        execution_policy=prepared.execution_policy,
        harness_name=adapter.name,
        grader_env=grader_env,
    )

    if (
        not result.metrics.metrics_estimated
        and result.metrics.total_tokens > 0
        and result.metrics.total_tokens < SUSPICIOUS_LOW_TOKEN_THRESHOLD
        and result.metrics.latency_sec > 10
    ):
        result.infra_error_code = result.infra_error_code or "token_count_implausibly_low"

    return result


async def _finalize_run(
    adapter: HarnessAdapter,
    prepared: PreparedRun | None,
    result: RunResult,
    trace_dir: str,
    runtime_config: RuntimeConfig,
) -> None:
    """Persist debug artifacts and clean up the workspace."""
    await _finalize_prepared_run(
        adapter=adapter,
        prepared=prepared,
        result=result,
        trace_dir=trace_dir,
        keep_workspace=runtime_config.keep_workspace,
    )


async def _execute_single_run(
    *,
    request: RunRequest,
    judge_llm: JudgeLLM | None,
    runtime_config: RuntimeConfig,
    pricing_override: ModelPricing | None,
) -> RunResult:
    task = request.task
    harness = request.harness
    adapter = request.adapter
    model_spec = request.model_spec
    model_label = request.model_label
    run_index = request.run_index
    run_id = request.run_id
    trace_dir = request.trace_dir
    prepared: PreparedRun | None = None
    result: RunResult | None = None

    try:
        skip_reasons = _check_adapter_support(adapter, task)
        if skip_reasons:
            result = RunResult(
                task_id=task.id,
                harness=harness,
                run_id=run_id,
                run_index=run_index,
                model=model_label,
                status="not_applicable",
                final_text="",
                trace=[
                    CanonicalTraceEvent(
                        type="task_failed",
                        error=f"harness {harness} unsupported: {'; '.join(skip_reasons)}",
                        ts=datetime.now(UTC).isoformat(),
                    )
                ],
                metrics=RunMetrics(),
                grader_results=[],
                failure_origin="adapter",
                infra_error_code="capability_unsupported",
                infra_error_details=f"harness {harness} unsupported: {'; '.join(skip_reasons)}"[
                    :ERROR_DETAIL_MAX_CHARS
                ],
            )
            return result

        prepared = await _prepare_run(
            adapter,
            task,
            run_id,
            harness,
            runtime_config,
        )

        _install_run_memory(adapter, prepared, task)

        result = await _execute_and_grade(
            adapter,
            prepared,
            model_spec,
            model_label,
            run_id,
            run_index,
            task,
            judge_llm,
            pricing_override,
            runtime_config,
        )

        return result
    except Exception as err:
        error_text = str(err)
        failure = detect_failure_origin_from_error(error_text)
        result = _failed_run_result(
            task_id=task.id,
            harness=harness,
            run_id=run_id,
            run_index=run_index,
            model_label=model_label,
            error_text=error_text,
            failure_origin=failure.get("failure_origin"),
            infra_error_code=failure.get("infra_error_code"),
        )
        return result
    finally:
        if result is not None:
            await _finalize_run(
                adapter=adapter,
                prepared=prepared,
                result=result,
                trace_dir=trace_dir,
                runtime_config=runtime_config,
            )


async def execute_eval(
    config: EvalConfig,
    tasks: list[Task],
    adapters: dict[str, HarnessAdapter],
    judge_llm: JudgeLLM | None,
    runtime_config: RuntimeConfig,
) -> list[RunResult]:
    """Execute the full evaluation.

    The plan is flattened across ``(run_index, task, model, harness)`` so a single
    ``max_concurrency`` knob bounds total parallelism — no more unbounded inner gather
    over ``harnesses``. Ordering puts ``run_index`` outermost so that two runs of the
    same prompt are separated in time by an entire round of the other axes, which
    mitigates provider response caching between runs.
    """
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    traces_dir = os.path.join(output_dir, "traces")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    results_file = os.path.join(data_dir, "runs.jsonl")
    trace_index_file = os.path.join(traces_dir, "index.jsonl")
    all_results = _recover_existing_results(results_file)
    if all_results:
        print(f"Recovered {len(all_results)} existing results")

    models = config.models or [config.model_spec]
    task_plan = _build_task_plan(config, tasks, models, all_results)

    # Validate adapters
    for h in config.harnesses:
        if h not in adapters:
            raise RuntimeError(f"Missing adapter: {h}")

    print(
        f"Run plan: {len(task_plan)} runs ({len(config.harnesses)} harnesses x "
        f"{len(models)} model{'s' if len(models) > 1 else ''} x {len(tasks)} tasks x "
        f"{config.runs_per_task} run(s), concurrency={config.max_concurrency})"
    )
    pricing_override = _build_pricing_override(runtime_config)

    results_lock = asyncio.Lock()

    async def run_one_item(item: RunPlanItem, _idx: int) -> None:
        task = item.task.materialize()
        model_spec = item.model_spec
        model_label = model_spec.label
        run_index = item.run_index
        harness = item.harness
        adapter = adapters[harness]

        log_prefix = f"[{harness}/{model_label}]" if len(models) > 1 else f"[{harness}]"
        print(f"{log_prefix} {task.id} run {run_index + 1}/{config.runs_per_task} starting...")

        request_run_id = str(uuid.uuid4())
        trace_dir = _build_trace_dir(output_dir, request_run_id)
        os.makedirs(trace_dir, exist_ok=True)

        # Inject a per-run nonce into the prompt so same-prompt runs bust any
        # provider-side response / prompt caching. Only matters when runs_per_task > 1.
        if config.runs_per_task > 1 and config.runs_bust_cache:
            task = _inject_cache_bust_nonce(task, request_run_id)

        request = RunRequest(
            task=task,
            harness=harness,
            adapter=adapter,
            model_spec=model_spec,
            model_label=model_label,
            run_index=run_index,
            run_id=request_run_id,
            trace_dir=trace_dir,
        )
        result = await _execute_single_run(
            request=request,
            judge_llm=judge_llm,
            runtime_config=runtime_config,
            pricing_override=pricing_override,
        )

        await _persist_run_result(
            request=request,
            result=result,
            trace_dir=trace_dir,
            results_file=results_file,
            trace_index_file=trace_index_file,
            all_results=all_results,
            results_lock=results_lock,
        )

        pass_count = sum(1 for g in result.grader_results if g.passed)
        total_count = len(result.grader_results)
        print(f"{log_prefix} {task.id} run {run_index + 1}: {result.status} (graders: {pass_count}/{total_count})")

    await _map_with_concurrency(task_plan, config.max_concurrency, run_one_item)

    return all_results


def _inject_cache_bust_nonce(task: Task, run_id: str) -> Task:
    """Return a Task whose ``user_query`` ends with a unique per-run marker.

    The marker is an HTML-style comment so models generally ignore it but any
    cache keyed off the full prompt hash / prefix hash still sees a different
    input. We keep it 12 chars of the run UUID — short enough to stay out of
    the way, unique enough to defeat collision.
    """
    nonce = run_id[:12]
    marker = f"\n\n<!-- eval-nonce: {nonce} -->"
    return Task(
        id=task.id,
        category=task.category,
        description=task.description,
        user_query=task.user_query + marker,
        graders=task.graders,
        timeout_sec=task.timeout_sec,
        task_dir=task.task_dir,
        workspace_dir=task.workspace_dir,
        workspace_files=task.workspace_files,
        history_file=task.history_file,
        conversation_history=task.conversation_history,
        native_memory=task.native_memory,
        memory_state=task.memory_state,
        prepare_commands=task.prepare_commands,
        tool_boundary=task.tool_boundary,
    )


def _recover_existing_results(results_file: str) -> list[RunResult]:
    results: list[RunResult] = []
    try:
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(run_result_from_dict(json.loads(line)))
    except FileNotFoundError:
        pass
    return results


def _build_task_plan(
    config: EvalConfig,
    tasks: list[Task],
    models: list[ModelSpec],
    all_results: list[RunResult],
) -> list[RunPlanItem]:
    """Flatten the eval space into ``(task, model, run, harness)`` work items.

    Iteration order: ``run_index`` outermost, then task, model, harness. That means
    all of run=0 completes before any of run=1 starts (once concurrency admits them in
    order), maximizing the wall-clock gap between two same-prompt runs. Short of
    appending a nonce (see ``runs_bust_cache``), this is the single cheapest hedge
    against provider-side prompt / response caches collapsing ``runs > 1`` into one
    real sample plus N-1 replays.

    Items already present in ``all_results`` are filtered out (resume-from-crash).
    """
    completed_keys: set[tuple[str, str, int, str]] = {(r.task_id, r.model, r.run_index, r.harness) for r in all_results}

    task_plan: list[RunPlanItem] = []
    for run_index in range(config.runs_per_task):
        for task in tasks:
            for model_spec in models:
                for harness in config.harnesses:
                    if (task.id, model_spec.label, run_index, harness) in completed_keys:
                        continue
                    task_plan.append(
                        RunPlanItem(
                            task=task,
                            model_spec=model_spec,
                            run_index=run_index,
                            harness=harness,
                        )
                    )
    return task_plan


def _build_pricing_override(runtime_config: RuntimeConfig) -> ModelPricing | None:
    if runtime_config.custom_pricing is None:
        return None
    return ModelPricing(
        input=runtime_config.custom_pricing["input"],
        output=runtime_config.custom_pricing["output"],
        cache_read=runtime_config.custom_pricing["cache_read"],
        cache_write=runtime_config.custom_pricing["cache_write"],
    )


def _build_trace_dir(
    output_dir: str,
    run_id: str,
) -> str:
    return os.path.join(output_dir, "traces", run_id)


async def _persist_run_result(
    *,
    request: RunRequest,
    result: RunResult,
    trace_dir: str,
    results_file: str,
    trace_index_file: str,
    all_results: list[RunResult],
    results_lock: asyncio.Lock,
) -> None:
    with open(os.path.join(trace_dir, "request.json"), "w") as f:
        json.dump(_run_request_to_dict(request), f, indent=2)
    with open(os.path.join(trace_dir, "trace.json"), "w") as f:
        json.dump([_trace_event_to_dict(event) for event in result.trace], f, indent=2)

    async with results_lock:
        all_results.append(result)
        with open(results_file, "a") as f:
            f.write(json.dumps(_run_result_to_dict(result)) + "\n")
        with open(trace_index_file, "a") as f:
            f.write(json.dumps(_trace_index_entry(request, result, trace_dir)) + "\n")


def _run_request_to_dict(request: RunRequest) -> dict[str, Any]:
    task = request.task
    payload: dict[str, Any] = {
        "task_id": task.id,
        "run_id": request.run_id,
        "run_index": request.run_index,
        "harness": request.harness,
        "model": request.model_label,
        "prompt": format_task_message(task),
        "task": {
            "description": task.description,
            "user_query": task.user_query,
            "timeout_sec": task.timeout_sec,
            "tool_boundary": {
                "internet": task.tool_boundary.internet,
                "shell": task.tool_boundary.shell,
                "file_write": task.tool_boundary.file_write,
            },
        },
    }
    if task.category:
        payload["task"]["category"] = task.category
    return payload


def _trace_index_entry(
    request: RunRequest,
    result: RunResult,
    trace_dir: str,
) -> dict[str, Any]:
    return {
        "run_id": request.run_id,
        "task_id": request.task.id,
        "harness": request.harness,
        "model": request.model_label,
        "run_index": request.run_index,
        "status": result.status,
        "trace_dir": trace_dir,
    }


def _trace_event_to_dict(event: CanonicalTraceEvent) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "type": event.type,
            "ts": event.ts,
            "role": event.role,
            "text": event.text,
            "tool_name": event.tool_name,
            "input": event.input,
            "success": event.success,
            "output": event.output,
            "path": event.path,
            "error": event.error,
        }.items()
        if value is not None
    }


def _grader_result_to_dict(result: GraderResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "grader_type": result.grader_type,
        "name": result.name,
        "pass": result.passed,
    }
    if result.score is not None:
        data["score"] = result.score
    if result.details is not None:
        data["details"] = result.details
    if result.dimensions is not None:
        data["dimensions"] = [
            {
                "name": dim.name,
                "pass": dim.passed,
                "score": dim.score,
                "reason": dim.reason,
                "required": dim.required,
                "weight": dim.weight,
            }
            for dim in result.dimensions
        ]
    return data


def _run_result_to_dict(result: RunResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "harness": result.harness,
        "run_id": result.run_id,
        "run_index": result.run_index,
        "model": result.model,
        "status": result.status,
        "final_text": result.final_text,
        "artifacts": list(result.artifacts),
        "trace": [_trace_event_to_dict(event) for event in result.trace],
        "metrics": {
            "latency_sec": result.metrics.latency_sec,
            "input_tokens": result.metrics.input_tokens,
            "output_tokens": result.metrics.output_tokens,
            "cache_read_tokens": result.metrics.cache_read_tokens,
            "cache_write_tokens": result.metrics.cache_write_tokens,
            "total_tokens": result.metrics.total_tokens,
            "cost_usd": result.metrics.cost_usd,
            "cost_usd_no_cache": result.metrics.cost_usd_no_cache,
            "tool_calls": result.metrics.tool_calls,
            "turns": result.metrics.turns,
            "metrics_estimated": result.metrics.metrics_estimated,
        },
        "grader_results": [_grader_result_to_dict(grader) for grader in result.grader_results],
        "failure_origin": result.failure_origin,
        "infra_error_code": result.infra_error_code,
        "infra_error_details": result.infra_error_details,
    }
