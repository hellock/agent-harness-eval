from __future__ import annotations

from agent_harness_eval.executor import VolumeMount, attach_run_layout_mounts, filter_env, policy_from_task
from agent_harness_eval.task import Task, ToolBoundary
from agent_harness_eval.utils.workspace import create_run_layout, remove_workspace


def test_policy_from_task_maps_tool_boundary_flags() -> None:
    task = Task(
        id="task.policy",
        category="security",
        description="policy",
        user_query="answer",
        tool_boundary=ToolBoundary(internet="disabled", shell="enabled", file_write="disabled"),
        timeout_sec=45,
    )

    policy = policy_from_task(task, "/tmp/workspace", 45)

    assert policy.network is False
    assert policy.shell is True
    assert policy.file_write is False
    assert policy.workspace_dir == "/tmp/workspace"
    assert policy.timeout_sec == 45


def test_attach_run_layout_mounts_adds_input_state_and_output_mounts() -> None:
    layout = create_run_layout("executor-mounts")
    try:
        policy = attach_run_layout_mounts(
            policy_from_task(
                Task(
                    id="task.mounts",
                    category="coding",
                    description="mounts",
                    user_query="answer",
                    timeout_sec=30,
                ),
                layout.workspace_dir,
                30,
            ),
            layout,
        )

        mounts = set(policy.extra_mounts)
        assert VolumeMount(source=layout.input_dir, target=layout.input_dir, mode="ro") in mounts
        assert VolumeMount(source=layout.state_dir, target=layout.state_dir, mode="rw") in mounts
        assert VolumeMount(source=layout.output_dir, target=layout.output_dir, mode="rw") in mounts
    finally:
        remove_workspace(layout.workspace_dir)


def test_filter_env_drops_sensitive_values_and_honors_passthrough() -> None:
    env = {
        "PATH": "/usr/bin",
        "OPENAI_API_KEY": "secret",
        "SAFE_TOKEN": "drop-me",
        "CUSTOM_OK": "drop-me-too",
        "EVAL_CONTEXT": "keep",
        "ALLOW_ME": "keep-me",
    }

    filtered = filter_env(
        env,
        extra_vars={"EXTRA": "value"},
        passthrough=["ALLOW_ME"],
    )

    assert filtered["PATH"] == "/usr/bin"
    assert filtered["ALLOW_ME"] == "keep-me"
    assert filtered["EVAL_CONTEXT"] == "keep"
    assert filtered["EXTRA"] == "value"
    assert "OPENAI_API_KEY" not in filtered
    assert "SAFE_TOKEN" not in filtered
    assert "CUSTOM_OK" not in filtered


def test_filter_env_does_not_leak_eval_provider_api_keys() -> None:
    env = {
        "EVAL_PROVIDER_RELAY_API_KEY": "secret",
        "EVAL_PROVIDER_RELAY_BASE_URL": "https://relay.example/v1",
        "EVAL_CONTEXT": "keep",
    }

    filtered = filter_env(env)

    assert "EVAL_PROVIDER_RELAY_API_KEY" not in filtered
    assert filtered["EVAL_PROVIDER_RELAY_BASE_URL"] == "https://relay.example/v1"
    assert filtered["EVAL_CONTEXT"] == "keep"
