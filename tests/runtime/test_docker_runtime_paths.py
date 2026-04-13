from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.executor import (
    ExecutionPolicy,
    attach_run_layout_mounts,
    resolve_executor_backend,
)
from agent_harness_eval.executor.docker import (
    DockerExecutor,
    resolve_docker_image,
)
from agent_harness_eval.utils.workspace import RunLayout


@pytest.fixture
def isolated_run_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_runtime_config_preserves_symlink_state_dir_path(
    isolated_run_dir: Path,
) -> None:
    state_target = isolated_run_dir / "state-target"
    state_link = isolated_run_dir / "state-link"
    bin_path = state_target / "nanobot" / "bin" / "nanobot"
    state_target.mkdir()
    state_link.symlink_to(state_target, target_is_directory=True)
    bin_path.parent.mkdir(parents=True)
    bin_path.write_text("#!/usr/bin/env bash\n")

    rc = RuntimeConfig(
        project_root=isolated_run_dir,
        eval_state_dir=state_link,
    )

    assert rc.state_dir == state_link
    assert rc.harness_state_dir("nanobot") == state_link / "nanobot"
    assert rc.resolve_harness_bin("nanobot", "nanobot") == str(state_link / "nanobot" / "bin" / "nanobot")


def test_runtime_config_does_not_export_global_docker_image_by_default(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVAL_DOCKER_IMAGE", raising=False)

    rc = RuntimeConfig(
        project_root=isolated_run_dir,
        docker_image=None,
        harness_config={"nanobot": {"version": "latest"}},
    )
    subprocess_env = rc.subprocess_env

    assert "EVAL_DOCKER_IMAGE" not in os.environ
    assert "EVAL_DOCKER_IMAGE" not in subprocess_env
    assert resolve_docker_image("nanobot", rc) == "agent-harness-eval-nanobot:latest"


def test_get_docker_image_uses_harness_version_config_for_managed_nanobot(
    isolated_run_dir: Path,
) -> None:
    rc = RuntimeConfig(
        project_root=isolated_run_dir,
        harness_config={"nanobot": {"version": "0.1.5"}},
    )

    assert resolve_docker_image("nanobot", rc) == "agent-harness-eval-nanobot:0.1.5"


def test_runtime_config_overrides_global_sandbox_and_image_env(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVAL_DOCKER_IMAGE", "node:20")

    rc = RuntimeConfig(
        project_root=isolated_run_dir,
        executor_backend="docker",
        docker_image="python:3.12-slim",
        harness_config={"nanobot": {"version": "0.1.5"}},
    )

    assert resolve_executor_backend(rc) == "docker"
    assert resolve_docker_image("nanobot", rc) == "python:3.12-slim"


def test_resolve_executor_backend_rejects_unknown_backend(
    isolated_run_dir: Path,
) -> None:
    rc = RuntimeConfig(
        project_root=isolated_run_dir,
        executor_backend="auto",
    )

    with pytest.raises(ValueError, match="Unknown executor backend"):
        resolve_executor_backend(rc)


def test_wrap_docker_mounts_uv_runtime_when_present(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_home = isolated_run_dir / "home"
    uv_home = fake_home / ".local" / "share" / "uv"
    uv_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))

    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        harness_config={"nanobot": {"version": "latest"}},
    )
    executor = DockerExecutor(runtime_config)
    command = executor._wrap(
        "nanobot",
        ExecutionPolicy(workspace_dir=str(isolated_run_dir / "workspace")),
        "/tmp/nanobot",
        [],
        {},
    )

    assert any(arg == f"{uv_home}:{uv_home}:ro" for arg in command.args)


def test_wrap_docker_mounts_run_layout_state_input_and_output(
    isolated_run_dir: Path,
) -> None:
    layout = RunLayout(
        root_dir=str(isolated_run_dir / "run"),
        input_dir=str(isolated_run_dir / "run" / "input"),
        workspace_seed_dir=str(isolated_run_dir / "run" / "input" / "workspace"),
        workspace_dir=str(isolated_run_dir / "run" / "runtime" / "workspace"),
        state_dir=str(isolated_run_dir / "run" / "runtime" / "state"),
        output_dir=str(isolated_run_dir / "run" / "output"),
    )
    policy = ExecutionPolicy(workspace_dir=layout.workspace_dir)
    attach_run_layout_mounts(policy, layout)

    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        harness_config={"nanobot": {"version": "latest"}},
    )
    executor = DockerExecutor(runtime_config)
    command = executor._wrap(
        "nanobot",
        policy,
        "/tmp/nanobot",
        [],
        {},
    )

    assert any(arg == f"{layout.input_dir}:{layout.input_dir}:ro" for arg in command.args)
    assert any(arg == f"{layout.state_dir}:{layout.state_dir}:rw" for arg in command.args)
    assert any(arg == f"{layout.output_dir}:{layout.output_dir}:rw" for arg in command.args)
    assert ".harnesses" not in " ".join(command.args)


def test_wrap_docker_uses_cwd_without_narrowing_workspace_mount(
    isolated_run_dir: Path,
) -> None:
    workspace_dir = isolated_run_dir / "workspace"
    repo_dir = workspace_dir / "repo"
    repo_dir.mkdir(parents=True)

    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        harness_config={"nanobot": {"version": "latest"}},
    )
    executor = DockerExecutor(runtime_config)
    command = executor._wrap(
        "nanobot",
        ExecutionPolicy(workspace_dir=str(workspace_dir), cwd=str(repo_dir)),
        "python",
        ["-V"],
        {},
    )

    assert any(arg == f"{workspace_dir}:{workspace_dir}:rw" for arg in command.args)
    workdir_index = command.args.index("-w")
    assert command.args[workdir_index + 1] == str(repo_dir)


def test_wrap_docker_uses_network_none_for_strict_network_policy(
    isolated_run_dir: Path,
) -> None:
    workspace_dir = isolated_run_dir / "workspace"
    workspace_dir.mkdir()

    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        harness_config={"nanobot": {"version": "latest"}},
    )
    executor = DockerExecutor(runtime_config)
    command = executor._wrap(
        "nanobot",
        ExecutionPolicy(
            network=False,
            strict_network=True,
            workspace_dir=str(workspace_dir),
        ),
        "python",
        ["-V"],
        {},
    )

    assert "--network=none" in command.args


def test_wrap_docker_keeps_soft_network_boundary_without_strict_mode(
    isolated_run_dir: Path,
) -> None:
    workspace_dir = isolated_run_dir / "workspace"
    workspace_dir.mkdir()

    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        harness_config={"nanobot": {"version": "latest"}},
    )
    executor = DockerExecutor(runtime_config)
    command = executor._wrap(
        "nanobot",
        ExecutionPolicy(
            network=False,
            workspace_dir=str(workspace_dir),
        ),
        "python",
        ["-V"],
        {},
    )

    assert "--network=none" not in command.args


def test_wrap_docker_mounts_only_harness_state_for_non_managed_image(
    isolated_run_dir: Path,
) -> None:
    workspace_dir = isolated_run_dir / "workspace"
    workspace_dir.mkdir()

    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        docker_image="python:3.12-slim",
        harness_config={"nanobot": {"version": "latest"}},
    )
    executor = DockerExecutor(runtime_config)
    command = executor._wrap(
        "nanobot",
        ExecutionPolicy(workspace_dir=str(workspace_dir)),
        "python",
        ["-V"],
        {},
    )

    harness_state_dir = runtime_config.harness_state_dir("nanobot")
    assert any(arg == f"{harness_state_dir}:{harness_state_dir}:rw" for arg in command.args)
    assert f"{runtime_config.state_dir}:{runtime_config.state_dir}:rw" not in command.args
