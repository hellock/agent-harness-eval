from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.cli import (
    _build_run_eval_config,
    _load_config,
    _regrade_results,
    _resolve_report_judge_model_spec,
    _run_preflight_phase,
)
from agent_harness_eval.config.eval_file import clear_cache
from agent_harness_eval.config.providers import ModelSpec, ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.graders.specs import GraderResult, RegexGrader, RubricJudgeGrader, TestPassGrader
from agent_harness_eval.task import Task
from agent_harness_eval.types import EvalConfig, RunMetrics, RunResult


def _make_args(**overrides: object) -> argparse.Namespace:
    data: dict[str, object] = {
        "model": None,
        "harness": None,
        "runs": None,
        "concurrency": None,
        "judge_model": None,
        "secondary_judge_model": None,
        "tasks_dir": None,
        "category": None,
        "task": None,
        "output": None,
        "timeout": None,
        "reinstall": False,
        "regrade": False,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


@pytest.fixture
def isolated_temp_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-cli-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_build_run_eval_config_cli_overrides_yaml_defaults() -> None:
    rc = RuntimeConfig(
        project_root=Path("/tmp/project"),
        providers={
            "anthropic": ProviderConfig(
                base_url="https://api.anthropic.com",
                api_key="anthropic-key",
                api_format="anthropic",
            ),
            "openai": ProviderConfig(
                base_url="https://api.openai.com/v1",
                api_key="openai-key",
                api_format="openai-chat-completions",
            ),
        },
    )
    eval_yaml = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "judge_model": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "secondary_judge_model": {"provider": "openai", "model": "gpt-5.4-mini"},
        "harnesses": {
            "openclaw": {"version": "2026.4.5"},
            "claude_code": {"version": "2.1.92"},
        },
        "runs": 1,
        "concurrency": 2,
        "timeout": 600,
    }
    args = _make_args(
        model="openai:gpt-5.4,anthropic:claude-sonnet-4-6",
        harness=["codex,openclaw"],
        runs=3,
        concurrency=5,
        judge_model="openai:gpt-5.4-mini",
        secondary_judge_model="anthropic:claude-sonnet-4-6",
        category="coding,security",
        task=["coding.01,security.02"],
        timeout=900,
    )

    config = _build_run_eval_config(args, eval_yaml, rc)

    assert [spec.label for spec in config.models] == [
        "openai:gpt-5.4",
        "anthropic:claude-sonnet-4-6",
    ]
    assert config.model_spec.label == "openai:gpt-5.4"
    assert config.harnesses == ["codex", "openclaw"]
    assert config.runs_per_task == 3
    assert config.max_concurrency == 5
    assert config.judge_model_spec == ModelSpec(provider="openai", model="gpt-5.4-mini")
    assert config.secondary_judge_model_spec == ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
    assert config.task_filter == {
        "categories": ["coding", "security"],
        "ids": ["coding.01", "security.02"],
    }
    assert config.timeout_sec == 900
    assert config.providers == rc.providers


def test_build_run_eval_config_accumulates_repeated_harness_flags() -> None:
    rc = RuntimeConfig(project_root=Path("/tmp/project"))
    eval_yaml = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "judge_model": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "harnesses": {
            "claude_code": {"version": "2.1.92"},
        },
        "runs": 1,
        "concurrency": 2,
        "timeout": 600,
    }

    config = _build_run_eval_config(
        _make_args(harness=["claude-code", "nanobot,openclaw", "zeroclaw"]),
        eval_yaml,
        rc,
    )

    assert config.harnesses == ["claude-code", "nanobot", "openclaw", "zeroclaw"]


def test_build_run_eval_config_accumulates_repeated_task_flags() -> None:
    rc = RuntimeConfig(project_root=Path("/tmp/project"))
    eval_yaml = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "judge_model": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "harnesses": {
            "claude_code": {"version": "2.1.92"},
        },
        "runs": 1,
        "concurrency": 2,
        "timeout": 600,
    }

    config = _build_run_eval_config(
        _make_args(task=["coding.01", "security.01,skills.03"]),
        eval_yaml,
        rc,
    )

    assert config.task_filter == {
        "categories": None,
        "ids": ["coding.01", "security.01", "skills.03"],
    }


def test_build_run_eval_config_uses_yaml_matrix_and_default_harness_names() -> None:
    rc = RuntimeConfig(project_root=Path("/tmp/project"))
    eval_yaml = {
        "models": [
            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            {"provider": "openai", "model": "gpt-5.4"},
        ],
        "judge_model": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "harnesses": {
            "claude_code": {"version": "2.1.92"},
            "openclaw": {"version": "2026.4.5"},
        },
        "runs": 2,
        "concurrency": 4,
        "timeout": 700,
    }

    config = _build_run_eval_config(_make_args(), eval_yaml, rc)

    assert [spec.label for spec in config.models] == [
        "anthropic:claude-sonnet-4-6",
        "openai:gpt-5.4",
    ]
    assert config.model_spec.label == "anthropic:claude-sonnet-4-6"
    assert config.harnesses == ["claude-code", "openclaw"]
    assert config.runs_per_task == 2
    assert config.max_concurrency == 4
    assert config.judge_model_spec == ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
    assert config.secondary_judge_model_spec is None
    assert config.timeout_sec == 700


def test_build_run_eval_config_requires_explicit_harnesses_when_none_configured() -> None:
    rc = RuntimeConfig(project_root=Path("/tmp/project"))
    eval_yaml = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "judge_model": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "harnesses": {},
        "runs": 1,
        "concurrency": 1,
        "timeout": 600,
    }

    with pytest.raises(ValueError, match="No harnesses configured"):
        _build_run_eval_config(_make_args(), eval_yaml, rc)


def test_load_config_exits_with_example_hint_when_config_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_path = tmp_path / "missing.yaml"
    (tmp_path / "eval.yaml.example").write_text(
        "provider: anthropic\nmodel: claude-sonnet-4-6\nexecutor: host\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        _load_config(str(missing_path))

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert f"Error: config file not found: {missing_path}" in stderr
    assert f"Create one with:  cp {tmp_path / 'eval.yaml.example'} {missing_path}" in stderr


def test_load_config_reads_project_root_and_runtime_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "provider: relay",
                "model: claude-sonnet-4-6",
                "executor: host",
                "providers:",
                "  relay:",
                '    base_url: "https://relay.example.com"',
                '    api_format: "anthropic"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("EVAL_PROVIDER_RELAY_API_KEY=relay-key\n", encoding="utf-8")

    clear_cache()
    monkeypatch.delenv("EVAL_PROVIDER_RELAY_API_KEY", raising=False)

    project_root, eval_yaml, runtime_config = _load_config(str(config_path))

    assert project_root == tmp_path
    assert eval_yaml["provider"] == "relay"
    assert runtime_config.project_root == tmp_path
    assert runtime_config.providers["relay"].api_key == "relay-key"


@pytest.mark.asyncio
async def test_run_preflight_phase_exits_when_judge_preflight_fails(tmp_path: Path) -> None:
    config = EvalConfig(
        model_spec=ModelSpec(provider="openai", model="gpt-5.4"),
        harnesses=["codex"],
        output_dir=str(tmp_path),
    )
    rc = RuntimeConfig(project_root=tmp_path)

    async def fake_run_preflight(_config, _adapters, _rc):
        return {
            "results": [],
            "healthy_harnesses": ["codex"],
            "judge_ok": False,
        }

    def fake_write_preflight_artifacts(_output_dir, _results):
        return None

    with pytest.raises(SystemExit) as exc_info:
        await _run_preflight_phase(
            config,
            {"codex": object()},
            rc,
            fake_run_preflight,
            fake_write_preflight_artifacts,
        )

    assert exc_info.value.code == 1


def test_resolve_report_judge_model_spec_uses_config_when_not_regrading() -> None:
    args = _make_args()
    eval_yaml = {"judge_model": {"provider": "anthropic", "model": "claude-sonnet-4-6"}}

    judge_spec = _resolve_report_judge_model_spec(args, eval_yaml)

    assert judge_spec == ModelSpec(provider="anthropic", model="claude-sonnet-4-6")


def test_resolve_report_judge_model_spec_rejects_override_without_regrade() -> None:
    args = _make_args(judge_model="openai:gpt-5.4")

    with pytest.raises(ValueError, match="requires --regrade"):
        _resolve_report_judge_model_spec(args, {"judge_model": {"provider": "anthropic", "model": "claude-sonnet-4-6"}})


@pytest.mark.asyncio
async def test_regrade_results_reruns_replayable_graders_and_preserves_workspace_dependent_results(
    isolated_temp_dir: Path,
) -> None:
    task = Task(
        id="reasoning.01",
        category="reasoning",
        description="task",
        user_query="answer",
        graders=[
            RegexGrader(pattern="updated"),
            TestPassGrader(command="pytest -q"),
            RubricJudgeGrader(rubric="Check whether the answer says updated."),
        ],
        timeout_sec=30,
    )
    result = RunResult(
        task_id="reasoning.01",
        harness="codex",
        run_id="run-1",
        run_index=0,
        model="openai:gpt-5.4",
        status="completed",
        final_text="updated answer",
        metrics=RunMetrics(),
        grader_results=[
            GraderResult(grader_type="regex", name="regex:old", passed=False, score=0.0),
            GraderResult(grader_type="test_pass", name="test_pass:pytest -q", passed=True, score=1.0),
            GraderResult(grader_type="rubric_judge", name="rubric_judge", passed=False, score=0.0),
        ],
    )

    class _Judge:
        async def generate(self, prompt: str) -> str:
            assert "updated answer" in prompt
            return json.dumps(
                {
                    "pass": True,
                    "score": 1.0,
                    "reason": "ok",
                    "dimensions": [],
                }
            )

    regraded = await _regrade_results([result], [task], _Judge(), str(isolated_temp_dir))

    grader_map = {grader.name: grader for grader in regraded[0].grader_results}
    assert grader_map["regex:updated"].passed is True
    assert grader_map["rubric_judge"].passed is True
    assert grader_map["test_pass:pytest -q"].passed is True
