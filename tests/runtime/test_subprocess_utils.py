from __future__ import annotations

import sys

import pytest

from agent_harness_eval.utils import subprocess as subprocess_utils
from agent_harness_eval.utils.subprocess import run_subprocess


@pytest.mark.asyncio
async def test_run_subprocess_captures_stdout_stderr_and_stdin() -> None:
    code = (
        "import os, sys; "
        "data = sys.stdin.read(); "
        'print(f\'OUT:{data}:{os.environ.get("CUSTOM_OK", "missing")}\'); '
        "print('ERR:warn', file=sys.stderr)"
    )

    result = await run_subprocess(
        sys.executable,
        ["-c", code],
        env={"CUSTOM_OK": "yes"},
        stdin="payload",
        timeout_ms=2000,
        filtered_env=False,
        inherit_env=False,
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert "OUT:payload:yes" in result.stdout
    assert "ERR:warn" in result.stderr


@pytest.mark.asyncio
async def test_run_subprocess_times_out() -> None:
    result = await run_subprocess(
        sys.executable,
        ["-c", "import time; time.sleep(2)"],
        timeout_ms=100,
        filtered_env=False,
        inherit_env=False,
    )

    assert result.timed_out is True
    assert result.exit_code is None


@pytest.mark.asyncio
async def test_run_subprocess_start_failure_returns_structured_error() -> None:
    result = await run_subprocess(
        "/definitely/not/a/real/binary",
        [],
        timeout_ms=1000,
        filtered_env=False,
        inherit_env=False,
    )

    assert result.timed_out is False
    assert result.exit_code == -1
    assert result.stdout == ""
    assert result.stderr


@pytest.mark.asyncio
async def test_run_subprocess_filtered_env_blocks_sensitive_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "blocked-key")
    monkeypatch.setenv("SAFE_VISIBLE", "nope")
    code = (
        "import os; "
        "print(os.environ.get('OPENAI_API_KEY', 'missing')); "
        "print(os.environ.get('HOME', 'missing')); "
        "print(os.environ.get('VISIBLE_EXTRA', 'missing'))"
    )

    result = await run_subprocess(
        sys.executable,
        ["-c", code],
        env={"VISIBLE_EXTRA": "ok"},
        timeout_ms=1000,
        filtered_env=True,
    )

    lines = [line.strip() for line in result.stdout.splitlines()]
    assert result.exit_code == 0
    assert lines[0] == "missing"
    assert lines[1] != "missing"
    assert lines[2] == "ok"


@pytest.mark.asyncio
async def test_run_subprocess_inherit_env_false_uses_only_explicit_env() -> None:
    code = "import os; print(os.environ.get('ONLY_EXPLICIT', 'missing')); print(os.environ.get('PATH', 'missing'))"

    result = await run_subprocess(
        sys.executable,
        ["-c", code],
        env={"ONLY_EXPLICIT": "present"},
        timeout_ms=1000,
        filtered_env=False,
        inherit_env=False,
    )

    lines = [line.strip() for line in result.stdout.splitlines()]
    assert result.exit_code == 0
    assert lines == ["present", "missing"]


def test_cleanup_active_subprocesses_kills_registered_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool]] = []

    class DummyProc:
        pid = 12345

    monkeypatch.setattr(
        subprocess_utils,
        "_kill_process_tree",
        lambda proc, use_pgroup: calls.append((proc.pid, use_pgroup)),
    )

    proc = DummyProc()
    subprocess_utils._ACTIVE_SUBPROCESSES[proc.pid] = (proc, True)
    try:
        subprocess_utils._cleanup_active_subprocesses()
    finally:
        subprocess_utils._ACTIVE_SUBPROCESSES.clear()

    assert calls == [(12345, True)]
    assert subprocess_utils._ACTIVE_SUBPROCESSES == {}
