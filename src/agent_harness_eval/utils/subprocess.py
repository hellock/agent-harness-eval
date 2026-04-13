"""Subprocess execution with timeout and environment filtering.

Design notes:
- Uses process groups (os.setsid) so SIGTERM/SIGKILL kills the entire
  child process tree, not just the leader. This prevents orphaned
  grandchild processes from accumulating.
- Graceful shutdown: SIGTERM → wait 5s → SIGKILL (sent to process group).
"""

from __future__ import annotations

import asyncio
import atexit
import os
import signal
import sys
import threading
from dataclasses import dataclass

from ..executor import filter_env


@dataclass
class SubprocessResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool


_ACTIVE_SUBPROCESSES: dict[int, tuple[asyncio.subprocess.Process, bool]] = {}
_PREVIOUS_SIGNAL_HANDLERS: dict[int, object] = {}
_SIGNAL_CLEANUP_INSTALLED = False
_ATEXIT_CLEANUP_REGISTERED = False


def _preexec_new_pgroup() -> None:
    """Called in child process to create a new process group.

    This allows us to kill the entire subprocess tree via os.killpg().
    Only used on Unix.
    """
    os.setsid()


def _register_active_subprocess(proc: asyncio.subprocess.Process, use_pgroup: bool) -> None:
    if proc.pid is not None:
        _ACTIVE_SUBPROCESSES[proc.pid] = (proc, use_pgroup)


def _unregister_active_subprocess(proc: asyncio.subprocess.Process) -> None:
    if proc.pid is not None:
        _ACTIVE_SUBPROCESSES.pop(proc.pid, None)


def _cleanup_active_subprocesses() -> None:
    for proc, use_pgroup in list(_ACTIVE_SUBPROCESSES.values()):
        _kill_process_tree(proc, use_pgroup)
    _ACTIVE_SUBPROCESSES.clear()


def _handle_termination_signal(signum: int, _frame: object) -> None:
    _cleanup_active_subprocesses()

    previous = _PREVIOUS_SIGNAL_HANDLERS.get(signum, signal.SIG_DFL)
    if previous is signal.SIG_IGN:
        return
    if callable(previous) and previous is not _handle_termination_signal:
        previous(signum, _frame)
        return
    raise SystemExit(128 + signum)


def _install_subprocess_cleanup_handlers() -> None:
    global _SIGNAL_CLEANUP_INSTALLED, _ATEXIT_CLEANUP_REGISTERED

    if not _ATEXIT_CLEANUP_REGISTERED:
        atexit.register(_cleanup_active_subprocesses)
        _ATEXIT_CLEANUP_REGISTERED = True

    if _SIGNAL_CLEANUP_INSTALLED:
        return
    if sys.platform == "win32":
        return
    if threading.current_thread() is not threading.main_thread():
        return

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous = signal.getsignal(signum)
        if previous is _handle_termination_signal:
            continue
        _PREVIOUS_SIGNAL_HANDLERS[signum] = previous
        signal.signal(signum, _handle_termination_signal)
    _SIGNAL_CLEANUP_INSTALLED = True


async def run_subprocess(
    command: str,
    args: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_ms: int,
    stdin: str | None = None,
    filtered_env: bool = True,
    inherit_env: bool = True,
) -> SubprocessResult:
    """Spawn a child process with a hard timeout and filtered environment.

    On Unix, the child runs in its own process group so that timeout
    cleanup kills the entire tree.
    """
    if filtered_env:
        proc_env = filter_env(dict(os.environ), env)
    elif inherit_env:
        proc_env = {**os.environ, **(env or {})}
    else:
        proc_env = dict(env or {})

    # Use process groups on Unix for clean tree cleanup
    use_pgroup = sys.platform != "win32"
    _install_subprocess_cleanup_handlers()

    try:
        proc = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=cwd,
            env=proc_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_preexec_new_pgroup if use_pgroup else None,
        )
    except Exception as e:
        return SubprocessResult(
            stdout="",
            stderr=str(e),
            exit_code=-1,
            timed_out=False,
        )

    _register_active_subprocess(proc, use_pgroup)
    timeout_sec = timeout_ms / 1000

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=stdin.encode() if stdin else None),
            timeout=timeout_sec,
        )
        return SubprocessResult(
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
            exit_code=proc.returncode,
            timed_out=False,
        )
    except asyncio.CancelledError:
        _terminate_process_tree(proc, use_pgroup)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            _kill_process_tree(proc, use_pgroup)
        raise
    except TimeoutError:
        # Kill the entire process group (child + all descendants)
        _terminate_process_tree(proc, use_pgroup)

        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            _kill_process_tree(proc, use_pgroup)

        # Collect whatever output we can
        stdout_data = b""
        stderr_data = b""
        try:
            if proc.stdout:
                stdout_data = await asyncio.wait_for(proc.stdout.read(), timeout=1)
        except Exception:
            pass
        try:
            if proc.stderr:
                stderr_data = await asyncio.wait_for(proc.stderr.read(), timeout=1)
        except Exception:
            pass

        return SubprocessResult(
            stdout=stdout_data.decode(errors="replace"),
            stderr=stderr_data.decode(errors="replace"),
            exit_code=None,
            timed_out=True,
        )
    finally:
        _unregister_active_subprocess(proc)


def _terminate_process_tree(proc: asyncio.subprocess.Process, use_pgroup: bool) -> None:
    """Send SIGTERM to the process group (or just the process on Windows)."""
    try:
        if use_pgroup and proc.pid:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _kill_process_tree(proc: asyncio.subprocess.Process, use_pgroup: bool) -> None:
    """Send SIGKILL to the process group (or just the process on Windows)."""
    try:
        if use_pgroup and proc.pid:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass
