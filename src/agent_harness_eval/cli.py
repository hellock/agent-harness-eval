"""CLI entry point for the agent harness evaluation framework."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config.eval_file import load_eval_yaml
from .config.providers import ModelSpec, ProviderConfig
from .config.runtime import RuntimeConfig, build_runtime_config
from .executor import create_executor, resolve_executor_backend
from .executor.docker import ensure_managed_harness_images, resolve_docker_image
from .graders.interface import expand_grader_specs, grader_name, run_graders
from .graders.judge_client import create_judge_llm
from .graders.specs import (
    FileExistsGrader,
    GraderResult,
    GraderSpec,
    JsonSchemaGrader,
    RegexGrader,
    RubricJudgeGrader,
    TestPassGrader,
    TestSuiteGrader,
    TrajectoryGrader,
)
from .task import Task
from .types import EvalConfig, RunResult, run_result_from_dict, run_result_to_dict


def _load_config(config_path: str | None) -> tuple[Path, Any, RuntimeConfig]:
    """Load .env + eval.yaml; return (project_root, eval_yaml_dict, RuntimeConfig).

    If config_path is None, looks for eval.yaml in the current directory.
    Fails immediately if the config file does not exist.
    """
    if config_path is None:
        config_file = Path.cwd() / "eval.yaml"
    else:
        config_file = Path(config_path)

    if not config_file.is_file():
        print(f"Error: config file not found: {config_file}", file=sys.stderr)
        example = config_file.parent / "eval.yaml.example"
        if example.exists():
            print(f"  Create one with:  cp {example} {config_file}", file=sys.stderr)
        sys.exit(1)

    project_root = config_file.parent
    load_dotenv(project_root / ".env")

    eval_yaml = load_eval_yaml(str(config_file))
    rc = build_runtime_config(project_root, eval_yaml)

    return project_root, eval_yaml, rc


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-harness-eval",
        description="Agent Harness Evaluation Framework",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Shared config argument — added to each subparser instead of the top-level
    # parser so users write `agent-harness-eval run --config ...` (idiomatic)
    # rather than `agent-harness-eval --config ... run`.
    _config_kwargs: dict[str, object] = {
        "type": str,
        "default": None,
        "help": "Path to eval.yaml config file (default: ./eval.yaml)",
    }

    # Run command
    run_parser = subparsers.add_parser("run", help="Run evaluation")
    run_parser.add_argument("--config", "-c", **_config_kwargs)  # type: ignore[arg-type]
    run_parser.add_argument(
        "--model",
        type=str,
        help="Model spec(s), comma-separated (provider:model shorthand)",
    )
    run_parser.add_argument(
        "--harness",
        type=str,
        action="append",
        help="Harnesses to run. Repeat the flag or pass a comma-separated list.",
    )
    run_parser.add_argument("--runs", type=int, help="Runs per task")
    run_parser.add_argument("--concurrency", type=int, help="Max concurrent tasks")
    run_parser.add_argument(
        "--executor",
        type=str,
        choices=["host", "docker"],
        help="Executor backend (overrides eval.yaml)",
    )
    run_parser.add_argument("--judge-model", type=str, help="Model for judge graders (provider:model)")
    run_parser.add_argument("--tasks-dir", type=str, help="Tasks directory")
    run_parser.add_argument("--category", type=str, help="Filter by category")
    run_parser.add_argument(
        "--task",
        "--task-id",
        type=str,
        action="append",
        help="Task IDs to run. Repeat the flag or pass a comma-separated list.",
    )
    run_parser.add_argument("--output", type=str, help="Output directory")
    run_parser.add_argument("--timeout", type=int, help="Per-task timeout in seconds")
    run_parser.add_argument("--reinstall", action="store_true", help="Reinstall harnesses")
    run_parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight probes (install / provider / harness). Use when you "
        "know the environment is good and want to save the LLM round-trip cost.",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate reports from results")
    report_parser.add_argument("--config", "-c", **_config_kwargs)  # type: ignore[arg-type]
    report_parser.add_argument("--input", type=str, required=True, help="Path to runs.jsonl")
    report_parser.add_argument("--output", type=str, help="Output directory")
    report_parser.add_argument(
        "--regrade",
        action="store_true",
        help="Recompute replayable grader results before generating reports.",
    )
    report_parser.add_argument(
        "--judge-model",
        type=str,
        help="Override judge model for `report --regrade` (provider:model).",
    )

    return parser


# ─── Model spec helpers ───


def _model_spec_from_yaml(eval_yaml: dict[str, Any]) -> ModelSpec:
    """Build a ModelSpec from the split provider + model YAML fields."""
    return ModelSpec(
        provider=str(eval_yaml["provider"]).strip(),
        model=str(eval_yaml["model"]).strip(),
    )


def _model_specs_from_yaml_matrix(entries: list[dict[str, Any]]) -> list[ModelSpec]:
    """Build ModelSpecs from the ``models:`` list-of-objects YAML field."""
    return [
        ModelSpec(
            provider=str(entry["provider"]).strip(),
            model=str(entry["model"]).strip(),
        )
        for entry in entries
    ]


def _judge_spec_from_yaml(value: dict[str, Any]) -> ModelSpec:
    """Build a ModelSpec from a judge_model YAML object."""
    return ModelSpec(
        provider=str(value["provider"]).strip(),
        model=str(value["model"]).strip(),
    )


def _parse_cli_model_specs(cli_value: str) -> list[ModelSpec]:
    """Parse comma-separated ``provider:model`` CLI shorthand."""
    from .config.providers import parse_model_spec

    return [parse_model_spec(s.strip()) for s in cli_value.split(",") if s.strip()]


def _infer_report_dirs(input_path: str, output_arg: str | None) -> tuple[str, str]:
    source_dir = os.path.dirname(input_path)
    if os.path.basename(input_path) == "runs.jsonl" and os.path.basename(source_dir) == "data":
        source_dir = os.path.dirname(source_dir)
    return source_dir, (output_arg or source_dir)


def _resolve_report_judge_model_spec(
    args: argparse.Namespace,
    eval_yaml: dict[str, Any],
) -> ModelSpec | None:
    from .config.providers import parse_model_spec

    if getattr(args, "judge_model", None) and not getattr(args, "regrade", False):
        raise ValueError("report --judge-model requires --regrade")

    if getattr(args, "regrade", False):
        if args.judge_model:
            return parse_model_spec(args.judge_model)
        if eval_yaml.get("judge_model"):
            return _judge_spec_from_yaml(eval_yaml["judge_model"])
        return None

    if eval_yaml.get("judge_model"):
        return _judge_spec_from_yaml(eval_yaml["judge_model"])
    return None


def _result_has_rubric_judge(task: Task) -> bool:
    return any(isinstance(spec, RubricJudgeGrader) for spec in expand_grader_specs(task))


def _task_with_graders(task: Task, graders: list[GraderSpec]) -> Task:
    return Task(
        id=task.id,
        category=task.category,
        description=task.description,
        user_query=task.user_query,
        graders=graders,
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


def _resolve_regrade_workspace(trace_dir: str) -> tuple[str | None, str]:
    kept_workspace_file = os.path.join(trace_dir, "kept_workspace.txt")
    if os.path.isfile(kept_workspace_file):
        kept_workspace = Path(kept_workspace_file).read_text(encoding="utf-8").strip()
        if kept_workspace and os.path.isdir(kept_workspace):
            return kept_workspace, "kept_workspace"

    run_output_dir = os.path.join(trace_dir, "raw", "run-output")
    if os.path.isdir(run_output_dir):
        return run_output_dir, "run_output"
    return None, "none"


def _is_replayable_grader(
    spec: GraderSpec,
    *,
    workspace_dir: str | None,
    workspace_source: str,
    judge_llm: Any | None,
) -> bool:
    match spec:
        case TrajectoryGrader():
            return True
        case RegexGrader(target="final_text"):
            return True
        case JsonSchemaGrader(target="final_text"):
            return True
        case FileExistsGrader():
            return workspace_dir is not None
        case RegexGrader():
            return workspace_dir is not None
        case JsonSchemaGrader():
            return workspace_dir is not None
        case RubricJudgeGrader(snapshot_paths=paths):
            if judge_llm is None:
                return False
            return not paths or workspace_dir is not None
        case TestPassGrader() | TestSuiteGrader():
            return workspace_source == "kept_workspace"
        case _:
            return False


def _original_grader_map(result: RunResult) -> dict[str, GraderResult]:
    return {grader.name: grader for grader in result.grader_results}


async def _regrade_result(
    result: RunResult,
    task: Task,
    judge_llm: Any | None,
    source_output_dir: str,
) -> RunResult:
    trace_dir = os.path.join(source_output_dir, "traces", result.run_id)
    workspace_dir, workspace_source = _resolve_regrade_workspace(trace_dir)
    expanded_specs = expand_grader_specs(task)
    replayable_specs = [
        spec
        for spec in expanded_specs
        if _is_replayable_grader(
            spec,
            workspace_dir=workspace_dir,
            workspace_source=workspace_source,
            judge_llm=judge_llm,
        )
    ]

    replayed_by_name: dict[str, GraderResult] = {}
    if replayable_specs:
        replay_task = _task_with_graders(task, replayable_specs)
        replayed = await run_graders(replay_task, result, judge_llm, workspace_dir)
        replayed_by_name = {grader.name: grader for grader in replayed}

    original_by_name = _original_grader_map(result)
    merged_graders: list[GraderResult] = []
    for spec in expanded_specs:
        name = grader_name(spec)
        if name in replayed_by_name:
            merged_graders.append(replayed_by_name[name])
            continue
        original = original_by_name.get(name)
        if original is not None:
            merged_graders.append(original)
            continue
        merged_graders.append(
            GraderResult(
                grader_type=spec.type,
                name=name,
                passed=False,
                score=0.0,
                details="Skipped regrade: original workspace-dependent result unavailable.",
            )
        )

    return RunResult(
        task_id=result.task_id,
        harness=result.harness,
        run_id=result.run_id,
        run_index=result.run_index,
        model=result.model,
        status=result.status,
        final_text=result.final_text,
        artifacts=list(result.artifacts),
        trace=list(result.trace),
        metrics=result.metrics,
        grader_results=merged_graders,
        failure_origin=result.failure_origin,
        infra_error_code=result.infra_error_code,
        infra_error_details=result.infra_error_details,
    )


async def _regrade_results(
    results: list[RunResult],
    tasks: list[Task],
    judge_llm: Any | None,
    source_output_dir: str,
) -> list[RunResult]:
    task_map = {task.id: task for task in tasks}
    regraded: list[RunResult] = []
    for result in results:
        task = task_map.get(result.task_id)
        if task is None:
            raise ValueError(f"Unknown task in results: {result.task_id}")
        regraded.append(await _regrade_result(result, task, judge_llm, source_output_dir))
    return regraded


def _write_results_jsonl(results: list[RunResult], output_dir: str) -> None:
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_path = data_dir / "runs.jsonl"
    with runs_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(run_result_to_dict(result)) + "\n")


# ─── Commands ───


async def _run_command(
    args: argparse.Namespace,
    eval_yaml: Any,
    rc: RuntimeConfig,
    project_root: Path,
) -> None:
    from .preflight import run_harness_preflight, write_preflight_artifacts
    from .runner import execute_eval
    from .task import load_tasks

    config = _build_run_eval_config(args, eval_yaml, rc)
    _print_run_banner(config)
    judge_provider = _validate_run_providers(config, rc)

    await _ensure_docker_images(config.harnesses, rc, project_root)
    tasks_dir = args.tasks_dir or str(project_root / "tasks")
    tasks = load_tasks(tasks_dir, config.task_filter)
    if not tasks:
        print("\nError: No tasks found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(tasks)} tasks")

    model_specs = config.models or [config.model_spec]
    config.output_dir = args.output or str(
        project_root / "results" / _build_default_run_dir_name(model_specs, config.harnesses, tasks)
    )
    print(f"Output: {config.output_dir}")
    print()

    for task in tasks:
        if not task.timeout_sec or task.timeout_sec > config.timeout_sec:
            task.timeout_sec = config.timeout_sec

    from .adapters import get_adapter_class

    executor = create_executor(rc)
    adapters = {h: get_adapter_class(h)(rc, executor) for h in config.harnesses}
    config.harnesses, adapters = await _run_preflight_phase(
        config,
        adapters,
        rc,
        run_harness_preflight,
        write_preflight_artifacts,
        skip_preflight=args.skip_preflight,
    )

    # Create judge LLM
    judge_llm = (
        create_judge_llm(config.judge_model_spec, judge_provider, rc)
        if judge_provider and config.judge_model_spec
        else None
    )

    results = await execute_eval(config, tasks, adapters, judge_llm, rc)

    # Generate reports
    from .reports.generate import generate_reports

    print("\n=== Generating Reports ===")
    generate_reports(results, tasks, config, judge_llm, runtime_config=rc)
    print(f"\nDone. Results saved to {config.output_dir}")


def _build_run_eval_config(
    args: argparse.Namespace,
    eval_yaml: Any,
    rc: RuntimeConfig,
) -> EvalConfig:
    # Resolve model specs — CLI shorthand overrides YAML split fields
    if args.model:
        model_specs = _parse_cli_model_specs(args.model)
    elif eval_yaml.get("models"):
        model_specs = _model_specs_from_yaml_matrix(eval_yaml["models"])
    else:
        model_specs = [_model_spec_from_yaml(eval_yaml)]

    # Judge model
    judge_model_str = args.judge_model
    if judge_model_str:
        from .config.providers import parse_model_spec

        judge_model_spec = parse_model_spec(judge_model_str)
    elif eval_yaml.get("judge_model"):
        judge_model_spec = _judge_spec_from_yaml(eval_yaml["judge_model"])
    else:
        judge_model_spec = model_specs[0]

    # Secondary judge
    secondary_judge_str = getattr(args, "secondary_judge_model", None)
    if secondary_judge_str:
        from .config.providers import parse_model_spec

        secondary_judge_spec = parse_model_spec(secondary_judge_str)
    elif eval_yaml.get("secondary_judge_model"):
        secondary_judge_spec = _judge_spec_from_yaml(eval_yaml["secondary_judge_model"])
    else:
        secondary_judge_spec = None

    configured_harnesses = eval_yaml.get("harnesses") or {}
    if args.harness:
        requested_harnesses = []
        for harness_arg in args.harness:
            requested_harnesses.extend(s.strip() for s in harness_arg.split(",") if s.strip())
    else:
        default_harness_list = ",".join("claude-code" if key == "claude_code" else key for key in configured_harnesses)
        if not default_harness_list:
            raise ValueError('No harnesses configured. Set "harnesses" in eval.yaml or pass --harness.')
        requested_harnesses = [s.strip() for s in default_harness_list.split(",") if s.strip()]

    runs_bust_cache_raw = eval_yaml.get("runs_bust_cache")
    runs_bust_cache = bool(runs_bust_cache_raw) if runs_bust_cache_raw is not None else True

    requested_task_ids = None
    if args.task:
        requested_task_ids = []
        for task_arg in args.task:
            requested_task_ids.extend(s.strip() for s in task_arg.split(",") if s.strip())

    return EvalConfig(
        model_spec=model_specs[0],
        models=model_specs,
        harnesses=requested_harnesses,
        runs_per_task=args.runs or int(eval_yaml["runs"]),
        max_concurrency=args.concurrency or int(eval_yaml["concurrency"]),
        judge_model_spec=judge_model_spec,
        secondary_judge_model_spec=secondary_judge_spec,
        providers=rc.providers,
        task_filter={
            "categories": [s.strip() for s in args.category.split(",")] if args.category else None,
            "ids": requested_task_ids,
        },
        output_dir="",
        timeout_sec=args.timeout or int(eval_yaml["timeout"]),
        runs_bust_cache=runs_bust_cache,
    )


def _print_run_banner(config: EvalConfig) -> None:
    models = config.models or [config.model_spec]
    is_multi_model = len(models) > 1
    print("=== Agent Harness Evaluation ===")
    print(f"Model{'s' if is_multi_model else ''}: {', '.join(model.label for model in models)}")
    if config.judge_model_spec is None:
        raise ValueError("run command requires judge_model_spec")
    print(f"Judge: {config.judge_model_spec.label}")
    print(f"Harnesses: {', '.join(config.harnesses)}")
    print(f"Runs per task: {config.runs_per_task}")


def _validate_run_providers(
    config: EvalConfig,
    rc: RuntimeConfig,
) -> ProviderConfig | None:
    providers = rc.providers
    if not providers:
        print("\nError: No API provider configured.", file=sys.stderr)
        sys.exit(1)

    # Check that the default provider exists for at least the primary model
    model_provider = providers.get(config.model_spec.provider)
    if not model_provider:
        print(
            f'\nError: No provider "{config.model_spec.provider}" configured for model {config.model_spec.label}.',
            file=sys.stderr,
        )
        sys.exit(1)

    if config.judge_model_spec is None:
        raise ValueError("run command requires judge_model_spec")
    judge_provider = providers.get(config.judge_model_spec.provider)
    if not judge_provider:
        print(f'Warning: No provider for judge model "{config.judge_model_spec.label}".')

    return judge_provider


async def _ensure_docker_images(
    harnesses: list[str],
    rc: RuntimeConfig,
    project_root: Path,
) -> None:
    if resolve_executor_backend(rc) != "docker":
        return

    selected_images = {harness: resolve_docker_image(harness, rc) for harness in harnesses}
    await ensure_managed_harness_images(
        project_root,
        harnesses,
        rc,
        selected_images=selected_images,
    )


async def _run_preflight_phase(
    config: EvalConfig,
    adapters: dict[str, Any],
    rc: RuntimeConfig,
    run_harness_preflight: Any,
    write_preflight_artifacts: Any,
    *,
    skip_preflight: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    print("\n=== Harness Preflight ===")
    if skip_preflight:
        print("  [SKIPPED] --skip-preflight is set; treating all configured harnesses as healthy.")
        return list(config.harnesses), dict(adapters)

    preflight = await run_harness_preflight(config, adapters, rc)
    write_preflight_artifacts(config.output_dir, preflight["results"])

    healthy = preflight["healthy_harnesses"]
    judge_ok = preflight.get("judge_ok", True)
    for result in preflight["results"]:
        status = "PASS" if result.status == "passed" else "FAIL"
        detail = f" ({result.details[:120]})" if result.details else ""
        print(f"  [{status}] {result.harness} / {result.model} @ {result.stage}:{result.code}{detail}")

    unhealthy = [harness for harness in config.harnesses if harness not in healthy]
    if unhealthy:
        print(f"Preflight filtered out harnesses: {', '.join(unhealthy)}")

    if not healthy:
        print("\nError: no harness passed preflight.", file=sys.stderr)
        sys.exit(1)

    if not judge_ok:
        print("\nError: judge model preflight failed.", file=sys.stderr)
        sys.exit(1)

    return healthy, {harness: adapter for harness, adapter in adapters.items() if harness in healthy}


def _report_command(
    args: argparse.Namespace,
    eval_yaml: Any,
    rc: RuntimeConfig,
    project_root: Path,
) -> None:
    from .task import load_tasks

    input_path = args.input
    source_output_dir, output_dir = _infer_report_dirs(input_path, args.output)

    with open(input_path) as f:
        results = [run_result_from_dict(json.loads(line)) for line in f if line.strip()]

    tasks = load_tasks(os.path.join(project_root, "tasks"))
    judge_model_spec = _resolve_report_judge_model_spec(args, eval_yaml)
    unique_models = sorted(set(r.model for r in results))
    from .config.providers import parse_model_spec

    model_specs = [parse_model_spec(m) for m in unique_models]
    result_task_ids = {result.task_id for result in results}
    relevant_tasks = [task for task in tasks if task.id in result_task_ids]

    providers = rc.providers if rc is not None else {}
    judge_llm = None
    if args.regrade:
        if judge_model_spec is None and any(_result_has_rubric_judge(task) for task in relevant_tasks):
            raise ValueError("report --regrade requires judge_model in config or --judge-model")
        if judge_model_spec is not None and any(_result_has_rubric_judge(task) for task in relevant_tasks):
            judge_provider = providers.get(judge_model_spec.provider)
            if judge_provider is None:
                raise ValueError(f'No provider "{judge_model_spec.provider}" configured for report regrade')
            judge_llm = create_judge_llm(judge_model_spec, judge_provider, rc)
        print("Regrading replayable graders from persisted results...")
        results = asyncio.run(_regrade_results(results, relevant_tasks, judge_llm, source_output_dir))

    config = EvalConfig(
        model_spec=model_specs[0] if model_specs else ModelSpec(provider="unknown", model="unknown"),
        models=model_specs,
        harnesses=sorted(set(r.harness for r in results)),
        runs_per_task=1,
        max_concurrency=1,
        judge_model_spec=judge_model_spec,
        providers=providers,
        output_dir=output_dir,
        timeout_sec=1800,
    )

    from .reports.generate import generate_reports

    _write_results_jsonl(results, output_dir)
    generate_reports(results, tasks, config, None, runtime_config=rc)
    print(f"Reports generated in {output_dir}")


def _build_default_run_dir_name(
    model_specs: list[ModelSpec],
    harnesses: list[str],
    tasks: list[Task],
) -> str:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_tag = _slugify(model_specs[0].label) if len(model_specs) == 1 else f"matrix-{len(model_specs)}m"
    harness_tag = _slugify(harnesses[0]) if len(harnesses) == 1 else f"{len(harnesses)}h"
    task_tag = _slugify(tasks[0].id) if len(tasks) == 1 else f"{len(tasks)}t"
    return f"run__{timestamp}__{model_tag}__{harness_tag}__{task_tag}"


def _slugify(value: str) -> str:
    import re

    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", value.lower()))[:48] or "default"


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    project_root, eval_yaml, rc = _load_config(args.config)

    if args.command == "run":
        if getattr(args, "executor", None):
            from dataclasses import replace

            rc = replace(rc, executor_backend=args.executor)
        asyncio.run(_run_command(args, eval_yaml, rc, project_root))
    elif args.command == "report":
        _report_command(args, eval_yaml, rc, project_root)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
