"""test_suite grader: runs N independent test cases and reports per-case pass/fail."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..executor import policy_for_grader
from ..utils.subprocess import SubprocessResult, run_subprocess
from .specs import GraderResult, TestSuiteGrader

logger = logging.getLogger(__name__)

# 30 seconds default timeout per case
_DEFAULT_CASE_TIMEOUT_MS = 30_000
_SETUP_COMMAND_TIMEOUT_MS = 120_000
_FAILED_OUTPUT_MAX_CHARS = 2000


async def _run_command(
    command: str,
    cwd: str | None,
    timeout_ms: int,
    *,
    executor: Any | None = None,
    execution_policy: Any | None = None,
    harness_name: str = "",
    grader_env: dict[str, str] | None = None,
) -> SubprocessResult:
    """Run a grader command through the executor if available, else directly."""
    if executor is not None and execution_policy is not None:
        policy = policy_for_grader(execution_policy, cwd=cwd)
        env = grader_env if grader_env is not None else dict(os.environ)
        return await executor.execute(
            harness_name or "_grader",
            policy,
            "bash",
            ["-c", command],
            env,
            timeout_ms=timeout_ms,
        )

    return await run_subprocess(
        "bash",
        ["-c", command],
        cwd=cwd,
        timeout_ms=timeout_ms,
    )


async def _run_batched_executor_commands(
    spec: TestSuiteGrader,
    cwd: str | None,
    *,
    executor: Any,
    execution_policy: Any,
    harness_name: str,
    grader_env: dict[str, str] | None,
) -> GraderResult:
    workspace_root = execution_policy.workspace_dir
    results_dir = tempfile.mkdtemp(prefix="aheval-test-suite-", dir=workspace_root)
    script = _build_batched_shell_script(spec, cwd, results_dir)
    policy = policy_for_grader(execution_policy, cwd=cwd)
    env = grader_env if grader_env is not None else dict(os.environ)
    total_timeout_ms = (
        len(spec.setup_commands) * _SETUP_COMMAND_TIMEOUT_MS + len(spec.cases) * _DEFAULT_CASE_TIMEOUT_MS + 5_000
    )
    try:
        batch_result = await executor.execute(
            harness_name or "_grader",
            policy,
            "bash",
            ["-lc", script],
            env,
            timeout_ms=total_timeout_ms,
        )
        if batch_result.timed_out:
            return GraderResult(
                grader_type="test_suite",
                name="test_suite",
                passed=False,
                score=0.0,
                details="test_suite batch runner timed out",
            )

        setup_entries = [
            _read_batched_entry(Path(results_dir), "setup", idx) for idx in range(len(spec.setup_commands))
        ]
        for entry in setup_entries:
            if entry is None:
                details = batch_result.stderr.strip() or batch_result.stdout.strip()
                return GraderResult(
                    grader_type="test_suite",
                    name="test_suite",
                    passed=False,
                    score=0.0,
                    details=f"test_suite batch runner produced incomplete results\n{details[:_FAILED_OUTPUT_MAX_CHARS]}",
                )
            timed_out = bool(entry["timed_out"])
            exit_code = entry["exit_code"]
            if timed_out or exit_code != 0:
                output = f"{entry['stdout']}{entry['stderr']}".strip()
                exit_info = "timeout" if timed_out else f"exit {exit_code}"
                return GraderResult(
                    grader_type="test_suite",
                    name="test_suite",
                    passed=False,
                    score=0.0,
                    details=(
                        f"setup_commands failed ({exit_info}): {entry['label']}\n{output[:_FAILED_OUTPUT_MAX_CHARS]}"
                    ),
                )

        case_entries = [_read_batched_entry(Path(results_dir), "case", idx) for idx in range(len(spec.cases))]
        if any(entry is None for entry in case_entries):
            details = batch_result.stderr.strip() or batch_result.stdout.strip()
            return GraderResult(
                grader_type="test_suite",
                name="test_suite",
                passed=False,
                score=0.0,
                details=f"test_suite batch runner produced incomplete results\n{details[:_FAILED_OUTPUT_MAX_CHARS]}",
            )

        passed_count = 0
        failed_names: list[str] = []
        for entry in case_entries:
            assert entry is not None
            timed_out = bool(entry["timed_out"])
            exit_code = entry["exit_code"]
            if exit_code == 0 and not timed_out:
                passed_count += 1
                continue
            output = f"{entry['stdout']}{entry['stderr']}".strip()
            exit_info = "timeout" if timed_out else f"exit {exit_code}"
            suffix = f" ({exit_info})"
            if output:
                suffix = f" ({exit_info}: {output[:_FAILED_OUTPUT_MAX_CHARS]})"
            failed_names.append(f"{entry['label']}{suffix}")

        total = len(spec.cases)
        score = passed_count / total
        passed = score >= spec.pass_threshold
        details = f"{passed_count}/{total} cases passed."
        if failed_names:
            details += " FAILED: " + "; ".join(failed_names)
        return GraderResult(
            grader_type="test_suite",
            name="test_suite",
            passed=passed,
            score=score,
            details=details,
        )
    finally:
        shutil.rmtree(results_dir, ignore_errors=True)


def _build_batched_shell_script(spec: TestSuiteGrader, cwd: str | None, results_dir: str) -> str:
    lines = [
        "set -u",
        f"RESULTS_DIR={_shell_quote(results_dir)}",
        'mkdir -p "$RESULTS_DIR"',
        "",
        "run_command() {",
        '  local command="$1"',
        '  local timeout_sec="$2"',
        '  local prefix="$3"',
        '  local stdout_file="${prefix}.stdout"',
        '  local stderr_file="${prefix}.stderr"',
        '  local exit_file="${prefix}.exit_code"',
        '  local timed_out_file="${prefix}.timed_out"',
        '  rm -f "$stdout_file" "$stderr_file" "$exit_file" "$timed_out_file"',
        '  bash -lc "$command" >"$stdout_file" 2>"$stderr_file" &',
        "  local cmd_pid=$!",
        "  (",
        '    sleep "$timeout_sec"',
        '    if kill -0 "$cmd_pid" 2>/dev/null; then',
        "      printf '1' > \"$timed_out_file\"",
        '      kill -TERM "$cmd_pid" 2>/dev/null || true',
        "      sleep 1",
        '      kill -KILL "$cmd_pid" 2>/dev/null || true',
        "    fi",
        "  ) &",
        "  local timer_pid=$!",
        '  wait "$cmd_pid"',
        "  local status=$?",
        '  kill "$timer_pid" 2>/dev/null || true',
        '  wait "$timer_pid" 2>/dev/null || true',
        '  if [ ! -f "$timed_out_file" ]; then',
        "    printf '0' > \"$timed_out_file\"",
        "  fi",
        '  printf \'%s\' "$status" > "$exit_file"',
        "}",
        "",
    ]

    setup_timeout_sec = str(_SETUP_COMMAND_TIMEOUT_MS // 1000)
    case_timeout_sec = str(_DEFAULT_CASE_TIMEOUT_MS // 1000)
    if cwd is not None:
        lines.append(f"cd {_shell_quote(cwd)}")
        lines.append("")

    for idx, command in enumerate(spec.setup_commands):
        prefix = os.path.join(results_dir, f"setup-{idx}")
        lines.extend(
            [
                f"printf '%s' {_shell_quote(command)} > {_shell_quote(prefix + '.label')}",
                f"run_command {_shell_quote(command)} {setup_timeout_sec} {_shell_quote(prefix)}",
                f'if [ "$(cat {_shell_quote(prefix + ".timed_out")})" = "1" ] || [ "$(cat {_shell_quote(prefix + ".exit_code")})" != "0" ]; then',
                "  exit 0",
                "fi",
                "",
            ]
        )

    for idx, case in enumerate(spec.cases):
        prefix = os.path.join(results_dir, f"case-{idx}")
        lines.extend(
            [
                f"printf '%s' {_shell_quote(case.name)} > {_shell_quote(prefix + '.label')}",
                f"run_command {_shell_quote(case.command)} {case_timeout_sec} {_shell_quote(prefix)}",
                "",
            ]
        )

    return "\n".join(lines)


def _read_batched_entry(results_dir: Path, kind: str, idx: int) -> dict[str, object] | None:
    prefix = results_dir / f"{kind}-{idx}"
    label_path = prefix.with_suffix(".label")
    exit_path = prefix.with_suffix(".exit_code")
    timed_out_path = prefix.with_suffix(".timed_out")
    stdout_path = prefix.with_suffix(".stdout")
    stderr_path = prefix.with_suffix(".stderr")

    required = [label_path, exit_path, timed_out_path, stdout_path, stderr_path]
    if any(not path.exists() for path in required):
        return None

    return {
        "label": label_path.read_text(encoding="utf-8"),
        "exit_code": int(exit_path.read_text(encoding="utf-8").strip()),
        "timed_out": timed_out_path.read_text(encoding="utf-8").strip() == "1",
        "stdout": stdout_path.read_text(encoding="utf-8"),
        "stderr": stderr_path.read_text(encoding="utf-8"),
    }


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


async def run_test_suite_grader(
    spec: TestSuiteGrader,
    workspace_dir: str | None,
    *,
    executor: Any | None = None,
    execution_policy: Any | None = None,
    harness_name: str = "",
    grader_env: dict[str, str] | None = None,
) -> GraderResult:
    """Run each case command independently and aggregate results."""
    cases = spec.cases
    if not cases:
        return GraderResult(
            grader_type="test_suite",
            name="test_suite",
            passed=False,
            score=0.0,
            details="No cases specified",
        )

    cwd = workspace_dir
    if spec.working_dir and workspace_dir:
        cwd = os.path.join(workspace_dir, spec.working_dir)

    if cwd and not os.path.isdir(cwd):
        return GraderResult(
            grader_type="test_suite",
            name="test_suite",
            passed=False,
            score=0.0,
            details=f"Working directory does not exist: {cwd}",
        )

    # Run setup_commands to rebuild the test environment before running cases
    if executor is not None and execution_policy is not None:
        return await _run_batched_executor_commands(
            spec,
            cwd,
            executor=executor,
            execution_policy=execution_policy,
            harness_name=harness_name,
            grader_env=grader_env,
        )

    for cmd in spec.setup_commands:
        setup_result = await _run_command(
            cmd,
            cwd,
            _SETUP_COMMAND_TIMEOUT_MS,
            executor=executor,
            execution_policy=execution_policy,
            harness_name=harness_name,
            grader_env=grader_env,
        )
        if setup_result.exit_code != 0 or setup_result.timed_out:
            output = (setup_result.stdout + setup_result.stderr).strip()
            exit_info = "timeout" if setup_result.timed_out else f"exit {setup_result.exit_code}"
            return GraderResult(
                grader_type="test_suite",
                name="test_suite",
                passed=False,
                score=0.0,
                details=f"setup_commands failed ({exit_info}): {cmd}\n{output[:_FAILED_OUTPUT_MAX_CHARS]}",
            )

    passed_count = 0
    total = len(cases)
    failed_names: list[str] = []

    for case in cases:
        proc_result = await _run_command(
            case.command,
            cwd,
            _DEFAULT_CASE_TIMEOUT_MS,
            executor=executor,
            execution_policy=execution_policy,
            harness_name=harness_name,
            grader_env=grader_env,
        )

        if proc_result.exit_code == 0 and not proc_result.timed_out:
            passed_count += 1
        else:
            output = (proc_result.stdout + proc_result.stderr).strip()
            exit_info = "timeout" if proc_result.timed_out else f"exit {proc_result.exit_code}"
            suffix = f" ({exit_info})"
            if output:
                suffix = f" ({exit_info}: {output[:_FAILED_OUTPUT_MAX_CHARS]})"
            failed_names.append(f"{case.name}{suffix}")

    score = passed_count / total
    passed = score >= spec.pass_threshold

    details = f"{passed_count}/{total} cases passed."
    if failed_names:
        details += " FAILED: " + "; ".join(failed_names)

    return GraderResult(
        grader_type="test_suite",
        name="test_suite",
        passed=passed,
        score=score,
        details=details,
    )
