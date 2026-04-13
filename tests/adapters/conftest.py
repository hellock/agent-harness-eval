from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def isolated_run_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-adapter-regression-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
