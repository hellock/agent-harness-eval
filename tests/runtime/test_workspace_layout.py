from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.utils.workspace import create_run_layout, remove_workspace


@pytest.fixture
def isolated_temp_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.asyncio
async def test_create_run_layout_materializes_input_and_runtime_workspace() -> None:
    layout = create_run_layout(
        "layout-test",
        workspace_files=[
            {"path": "docs/readme.txt", "content": "hello"},
            {"path": "src/app.ts", "content": "export const value = 1;\n"},
        ],
    )

    try:
        assert Path(layout.input_dir).is_dir()
        assert Path(layout.output_dir).is_dir()
        assert Path(layout.state_dir).is_dir()

        seed_readme = Path(layout.workspace_seed_dir) / "docs" / "readme.txt"
        runtime_readme = Path(layout.workspace_dir) / "docs" / "readme.txt"
        assert seed_readme.read_text() == "hello"
        assert runtime_readme.read_text() == "hello"

        runtime_readme.write_text("changed")
        assert seed_readme.read_text() == "hello"
    finally:
        remove_workspace(layout.workspace_dir)


@pytest.mark.asyncio
async def test_remove_workspace_removes_entire_run_root() -> None:
    layout = create_run_layout("remove-layout-test")
    root_dir = Path(layout.root_dir)

    remove_workspace(layout.workspace_dir)

    assert not root_dir.exists()
