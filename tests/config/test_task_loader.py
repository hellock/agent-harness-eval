from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.task import Task, load_tasks


@pytest.fixture
def isolated_tasks_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_task_yaml(task_dir: Path, body: str) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(body, encoding="utf-8")


@pytest.mark.asyncio
async def test_load_tasks_rejects_deprecated_install_commands_field(
    isolated_tasks_dir: Path,
) -> None:
    task_dir = isolated_tasks_dir / "coding" / "deprecated-prepare"
    _write_task_yaml(
        task_dir,
        """
id: coding.99
category: coding
description: deprecated setup field
user_query: test
setup:
  install_commands:
    - echo hello
""".lstrip(),
    )

    with pytest.raises(ValueError, match=r"install_commands.*prepare_commands"):
        load_tasks(str(isolated_tasks_dir))


@pytest.mark.asyncio
async def test_load_tasks_rejects_unknown_setup_field(
    isolated_tasks_dir: Path,
) -> None:
    task_dir = isolated_tasks_dir / "coding" / "unknown-setup"
    _write_task_yaml(
        task_dir,
        """
id: coding.98
category: coding
description: unknown setup field
user_query: test
setup:
  prepare_commands:
    - echo hello
  unknown_flag: true
""".lstrip(),
    )

    with pytest.raises(ValueError, match="Unknown task setup field"):
        load_tasks(str(isolated_tasks_dir))


@pytest.mark.asyncio
async def test_load_tasks_preserves_task_asset_references(
    isolated_tasks_dir: Path,
) -> None:
    task_dir = isolated_tasks_dir / "coding" / "asset-refs"
    (task_dir / "workspace").mkdir(parents=True, exist_ok=True)
    (task_dir / "workspace" / "README.md").write_text("seed\n", encoding="utf-8")
    (task_dir / "memory").mkdir(parents=True, exist_ok=True)
    (task_dir / "memory" / "MEMORY.md").write_text("remember\n", encoding="utf-8")
    (task_dir / "history.jsonl").write_text(
        '{"role":"user","content":"hello"}\n',
        encoding="utf-8",
    )
    _write_task_yaml(
        task_dir,
        """
id: coding.97
category: coding
description: asset refs
user_query: test
setup:
  workspace_dir: workspace
  history_file: history.jsonl
  native_memory:
    memory_dir: memory
""".lstrip(),
    )

    tasks = load_tasks(str(isolated_tasks_dir))

    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, Task)
    assert task.task_dir == str(task_dir)
    assert task.workspace_dir == "workspace"
    assert task.workspace_files is None
    assert task.history_file == "history.jsonl"
    assert task.conversation_history is None
    assert task.native_memory is not None
    assert task.native_memory.memory_dir == "memory"
    assert task.native_memory.files is None


@pytest.mark.asyncio
async def test_materialize_task_expands_directory_backed_inputs(
    isolated_tasks_dir: Path,
) -> None:
    task_dir = isolated_tasks_dir / "coding" / "materialize"
    (task_dir / "workspace" / "docs").mkdir(parents=True, exist_ok=True)
    (task_dir / "workspace" / "docs" / "README.md").write_text("seed\n", encoding="utf-8")
    (task_dir / "memory").mkdir(parents=True, exist_ok=True)
    (task_dir / "memory" / "MEMORY.md").write_text("remember\n", encoding="utf-8")
    (task_dir / "history.yaml").write_text(
        "- role: user\n  content: hello\n- role: assistant\n  content: world\n",
        encoding="utf-8",
    )
    _write_task_yaml(
        task_dir,
        """
id: coding.96
category: coding
description: materialize
user_query: test
setup:
  workspace_dir: workspace
  history_file: history.yaml
  native_memory:
    memory_dir: memory
""".lstrip(),
    )

    [task] = load_tasks(str(isolated_tasks_dir))
    runtime_task = task.materialize()

    assert runtime_task.task_dir == str(task_dir)
    assert runtime_task.workspace_dir is None
    assert runtime_task.history_file is None
    assert runtime_task.workspace_files is not None
    assert runtime_task.workspace_files[0]["path"] == "docs/README.md"
    assert runtime_task.workspace_files[0]["content"] == "seed\n"
    assert runtime_task.conversation_history is not None
    assert [turn["role"] for turn in runtime_task.conversation_history] == ["user", "assistant"]
    assert runtime_task.native_memory is not None
    assert runtime_task.native_memory.memory_dir is None
    assert runtime_task.native_memory.files is not None
    assert runtime_task.native_memory.files[0]["path"] == "MEMORY.md"
    assert runtime_task.native_memory.files[0]["content"] == "remember\n"
