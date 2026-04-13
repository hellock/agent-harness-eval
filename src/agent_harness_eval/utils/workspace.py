"""Temporary workspace creation and cleanup."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    root_dir: str
    input_dir: str
    workspace_seed_dir: str
    workspace_dir: str
    state_dir: str
    output_dir: str


def create_run_layout(
    run_id: str,
    *,
    workspace_files: list[dict[str, str]] | None = None,
) -> RunLayout:
    """Create the per-run filesystem layout.

    Contract:
    - input/      immutable task inputs for the run
    - runtime/    harness-private mutable state
    - output/     durable run outputs/artifacts
    """
    root_dir = Path(tempfile.mkdtemp(prefix=f"eval-{run_id}-"))
    input_dir = root_dir / "input"
    workspace_seed_dir = input_dir / "workspace"
    runtime_dir = root_dir / "runtime"
    workspace_dir = runtime_dir / "workspace"
    state_dir = runtime_dir / "state"
    output_dir = root_dir / "output"

    for path in (input_dir, workspace_seed_dir, runtime_dir, workspace_dir, state_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)

    if workspace_files:
        for file_info in workspace_files:
            rel_path = Path(file_info["path"])
            seed_path = workspace_seed_dir / rel_path
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            seed_path.write_text(file_info["content"])
            work_path = workspace_dir / rel_path
            work_path.parent.mkdir(parents=True, exist_ok=True)
            work_path.write_text(file_info["content"])

    return RunLayout(
        root_dir=str(root_dir),
        input_dir=str(input_dir),
        workspace_seed_dir=str(workspace_seed_dir),
        workspace_dir=str(workspace_dir),
        state_dir=str(state_dir),
        output_dir=str(output_dir),
    )


def create_workspace(
    run_id: str,
    *,
    workspace_files: list[dict[str, str]] | None = None,
) -> str:
    """Backward-compatible wrapper returning only the mutable workspace path."""
    layout = create_run_layout(run_id, workspace_files=workspace_files)
    return layout.workspace_dir


def remove_workspace(dir_path: str) -> None:
    """Thorough workspace cleanup.

    Accepts either the mutable workspace path or the run root path.
    """
    workspace = Path(dir_path)
    run_root = _resolve_run_root(workspace)

    # Step 1: Kill orphaned processes
    _kill_processes_in_dir(str(run_root))

    # Step 2: Restore permissions and remove
    try:
        subprocess.run(
            ["chmod", "-R", "u+w", str(run_root)],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass

    shutil.rmtree(run_root, ignore_errors=True)

    # Step 3: Log warning if side-effects detected
    side_effects = _detect_side_effects(dir_path)
    if side_effects:
        print(
            "[cleanup] Side-effects detected after workspace removal:\n" + "\n".join(f"  - {s}" for s in side_effects)
        )


def _resolve_run_root(path: Path) -> Path:
    if (path / "input").is_dir() and (path / "runtime").is_dir() and (path / "output").is_dir():
        return path

    if (
        path.name == "workspace"
        and path.parent.name == "runtime"
        and (path.parent.parent / "input").is_dir()
        and (path.parent.parent / "output").is_dir()
    ):
        return path.parent.parent

    return path


def _kill_processes_in_dir(dir_path: str) -> None:
    """Find and kill processes with open files in the given directory."""
    if os.name == "nt":
        return
    try:
        result = subprocess.run(
            ["lsof", "+D", dir_path, "-t"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [int(p) for p in result.stdout.split() if p.strip().isdigit() and int(p.strip()) != os.getpid()]
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        if pids:
            time.sleep(1)
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    except Exception:
        pass


def _detect_side_effects(workspace_dir: str) -> list[str]:
    """Detect common side-effects outside the workspace."""
    warnings: list[str] = []
    home = Path.home()

    suspicious_paths = [
        home / ".gitconfig",
        home / ".npmrc",
        home / ".bashrc",
        home / ".zshrc",
        home / ".ssh",
        home / ".claude",
        home / ".openclaw",
    ]

    now = time.time()
    recent_threshold = 60  # 1 minute

    for p in suspicious_paths:
        try:
            if now - p.stat().st_mtime < recent_threshold:
                warnings.append(f"{p} was modified in the last minute")
        except OSError:
            pass

    # Check for eval-related orphan temp dirs
    try:
        tmp_dir = Path(tempfile.gettempdir())
        workspace_base = Path(workspace_dir).name
        orphan_count = sum(
            1
            for entry in tmp_dir.iterdir()
            if entry.is_dir()
            and entry.name.startswith("eval-")
            and workspace_base not in entry.name
            and now - entry.stat().st_mtime < recent_threshold
        )
        if orphan_count > 10:
            warnings.append(f"{orphan_count} orphaned eval-* directories found in {tmp_dir}")
    except Exception:
        pass

    return warnings
