from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_harness_eval.adapters.interface import HarnessAdapter, PreparedRun, VerifyInstallResult
from agent_harness_eval.config.providers import ModelSpec, ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.executor import ExecutionPolicy, Executor, policy_from_task
from agent_harness_eval.preflight import run_harness_preflight, write_preflight_artifacts
from agent_harness_eval.task import Task
from agent_harness_eval.types import CanonicalTraceEvent, EvalConfig, RunMetrics, RunResult
from agent_harness_eval.utils.subprocess import SubprocessResult
from agent_harness_eval.utils.workspace import create_run_layout, remove_workspace

_TEST_PROVIDERS = {
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_format="openai-chat-completions",
    ),
}


@pytest.fixture
def isolated_temp_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-preflight-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class DummyExecutor(Executor):
    name = "dummy"

    async def execute(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
        timeout_ms: int,
    ) -> SubprocessResult:
        raise AssertionError("DummyExecutor.execute should not be called in preflight unit tests")


class ScriptedDockerExecutor(Executor):
    name = "docker"

    def __init__(self, runtime_config: RuntimeConfig, results: dict[str, SubprocessResult]):
        super().__init__(runtime_config)
        self._results = results
        self.calls: list[str] = []

    async def execute(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
        timeout_ms: int,
    ) -> SubprocessResult:
        self.calls.append(inner_command)
        return self._results.get(
            inner_command,
            SubprocessResult(stdout="", stderr=f"{inner_command}: not found", exit_code=127, timed_out=False),
        )


class DummyResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(self._payload)

    def json(self) -> dict:
        return self._payload


class DummyAsyncClient:
    def __init__(self, *, timeout: object, response: DummyResponse | Exception, recorder: list[dict]):
        self.timeout = timeout
        self._response = response
        self._recorder = recorder

    async def __aenter__(self) -> DummyAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, json: dict, headers: dict) -> DummyResponse:
        self._recorder.append({"url": url, "json": json, "headers": headers, "timeout": self.timeout})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class ScriptedAdapter(HarnessAdapter):
    name = "scripted"

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        executor: Executor,
        run_behaviors: list[Callable[[PreparedRun, str], RunResult]],
        *,
        emits_paired_trace_events: bool = False,
        install_ok: bool = True,
    ):
        super().__init__(runtime_config, executor)
        self._run_behaviors = list(run_behaviors)
        self._run_calls = 0
        self._install_ok = install_ok
        self.emits_paired_trace_events = emits_paired_trace_events

    @property
    def run_calls(self) -> int:
        return self._run_calls

    async def verify_install(self) -> VerifyInstallResult:
        return VerifyInstallResult(
            ok=self._install_ok,
            version="1.0.0" if self._install_ok else None,
            error=None if self._install_ok else "binary not found",
        )

    def prepare(self, task: Task, run_id: str) -> PreparedRun:
        layout = create_run_layout(run_id, workspace_files=task.workspace_files)
        policy = policy_from_task(task, layout.workspace_dir, task.timeout_sec)
        return PreparedRun(task=task, layout=layout, execution_policy=policy)

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        behavior = self._run_behaviors[min(self._run_calls, len(self._run_behaviors) - 1)]
        self._run_calls += 1
        return behavior(prepared, model)

    def cleanup(self, prepared: PreparedRun) -> None:
        remove_workspace(prepared.workspace_dir)


class InstallCheckingAdapter(HarnessAdapter):
    name = "custom"

    def prepare(self, task: Task, run_id: str) -> PreparedRun:
        raise AssertionError("prepare should not run when install check fails")

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        raise AssertionError("run should not run when install check fails")

    def cleanup(self, prepared: PreparedRun) -> None:
        return None


class InstallCheckingScriptedAdapter(HarnessAdapter):
    name = "scripted"

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        executor: Executor,
        run_behavior: Callable[[PreparedRun, str], RunResult],
    ) -> None:
        super().__init__(runtime_config, executor)
        self._run_behavior = run_behavior

    def prepare(self, task: Task, run_id: str) -> PreparedRun:
        layout = create_run_layout(run_id, workspace_files=task.workspace_files)
        policy = policy_from_task(task, layout.workspace_dir, task.timeout_sec)
        return PreparedRun(task=task, layout=layout, execution_policy=policy)

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        return self._run_behavior(prepared, model)

    def cleanup(self, prepared: PreparedRun) -> None:
        remove_workspace(prepared.workspace_dir)


@pytest.fixture(autouse=True)
def _mock_provider_probe_http(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []
    response = DummyResponse(200, {"ok": True})
    monkeypatch.setattr(
        "agent_harness_eval.preflight.httpx.AsyncClient",
        lambda timeout: DummyAsyncClient(timeout=timeout, response=response, recorder=calls),
    )
    return calls


def _make_eval_config(
    *,
    harnesses: list[str],
    models: list[ModelSpec] | None = None,
    judge_model_spec: ModelSpec | None = None,
    secondary_judge_model_spec: ModelSpec | None = None,
) -> EvalConfig:
    selected_models = models or [ModelSpec(provider="openai", model="gpt-5.4")]
    return EvalConfig(
        model_spec=selected_models[0],
        models=selected_models,
        harnesses=harnesses,
        judge_model_spec=judge_model_spec,
        secondary_judge_model_spec=secondary_judge_model_spec,
        timeout_sec=30,
    )


def _completed_result(task_id: str, final_text: str, *, trace_events: int = 1, tool_calls: int = 0) -> RunResult:
    trace = [CanonicalTraceEvent(type="task_completed", ts=f"2026-01-01T00:00:0{idx}Z") for idx in range(trace_events)]
    return RunResult(
        task_id=task_id,
        harness="scripted",
        run_id="",
        run_index=0,
        model="openai:gpt-5.4",
        status="completed",
        final_text=final_text,
        trace=trace,
        metrics=RunMetrics(tool_calls=tool_calls),
    )


def _failed_result(task_id: str) -> RunResult:
    return RunResult(
        task_id=task_id,
        harness="scripted",
        run_id="",
        run_index=0,
        model="openai:gpt-5.4",
        status="failed",
        final_text="",
        trace=[CanonicalTraceEvent(type="task_failed", ts="2026-01-01T00:00:00Z", error="api error")],
        metrics=RunMetrics(),
    )


# ─── Layer 1: Static checks ───


@pytest.mark.asyncio
async def test_preflight_fails_missing_adapter() -> None:
    config = _make_eval_config(harnesses=["missing"])
    rc = RuntimeConfig(project_root=Path.cwd(), providers=_TEST_PROVIDERS)

    payload = await run_harness_preflight(config, {}, rc)

    assert payload["healthy_harnesses"] == []
    result = payload["results"][0]
    assert result.stage == "config"
    assert result.code == "missing_adapter"


@pytest.mark.asyncio
async def test_preflight_fails_provider_config_error() -> None:
    """Harness with supported_api_formats that doesn't match the provider."""
    rc = RuntimeConfig(
        project_root=Path.cwd(),
        providers={
            "openai": ProviderConfig(
                base_url="https://api.openai.com/v1",
                api_key="test-key",
                api_format="openai-chat-completions",
            ),
        },
    )
    adapter = ScriptedAdapter(rc, DummyExecutor(rc), [])
    adapter.supported_api_formats = ["anthropic"]
    config = _make_eval_config(harnesses=["scripted"])

    payload = await run_harness_preflight(config, {"scripted": adapter}, rc)

    assert payload["healthy_harnesses"] == []
    result = payload["results"][0]
    assert result.stage == "config"
    assert result.code == "provider_config_error"


@pytest.mark.asyncio
async def test_preflight_fails_install_check() -> None:
    rc = RuntimeConfig(project_root=Path.cwd(), providers=_TEST_PROVIDERS)
    adapter = ScriptedAdapter(rc, DummyExecutor(rc), [], install_ok=False)
    config = _make_eval_config(harnesses=["scripted"])

    payload = await run_harness_preflight(config, {"scripted": adapter}, rc)

    assert payload["healthy_harnesses"] == []
    result = payload["results"][0]
    assert result.stage == "install"
    assert result.code == "install_check_failed"


@pytest.mark.asyncio
async def test_write_preflight_artifacts_persists_install_stdout_and_stderr(
    isolated_temp_dir: Path,
) -> None:
    rc = RuntimeConfig(project_root=Path.cwd(), providers=_TEST_PROVIDERS)
    adapter = ScriptedAdapter(rc, DummyExecutor(rc), [], install_ok=False)
    config = _make_eval_config(harnesses=["scripted"])

    payload = await run_harness_preflight(config, {"scripted": adapter}, rc)
    write_preflight_artifacts(str(isolated_temp_dir), payload["results"])

    preflight_json = json.loads((isolated_temp_dir / "data" / "preflight.json").read_text(encoding="utf-8"))
    entry = preflight_json[0]
    assert entry["stage"] == "install"
    assert entry["artifacts"] is None

    payload["results"][0].install_stdout = "install stdout\n"
    payload["results"][0].install_stderr = "install stderr\n"
    write_preflight_artifacts(str(isolated_temp_dir), payload["results"])

    preflight_json = json.loads((isolated_temp_dir / "data" / "preflight.json").read_text(encoding="utf-8"))
    entry = preflight_json[0]
    assert len(entry["artifacts"]) == 2
    stdout_path = isolated_temp_dir / entry["artifacts"][0]["path"]
    stderr_path = isolated_temp_dir / entry["artifacts"][1]["path"]
    assert stdout_path.read_text(encoding="utf-8") == "install stdout\n"
    assert stderr_path.read_text(encoding="utf-8") == "install stderr\n"


@pytest.mark.asyncio
async def test_preflight_fails_install_check_when_custom_docker_image_lacks_verifier_runtime() -> None:
    rc = RuntimeConfig(
        project_root=Path.cwd(),
        executor_backend="docker",
        docker_image="python:3.12-slim",
        providers=_TEST_PROVIDERS,
    )
    executor = ScriptedDockerExecutor(
        rc,
        {
            "custom": SubprocessResult(stdout="custom 1.0.0\n", stderr="", exit_code=0, timed_out=False),
            "bash": SubprocessResult(stdout="", stderr="bash: not found", exit_code=127, timed_out=False),
        },
    )
    adapter = InstallCheckingAdapter(rc, executor)
    config = _make_eval_config(harnesses=["custom"], models=[ModelSpec(provider="openai", model="gpt-5.4")])

    payload = await run_harness_preflight(config, {"custom": adapter}, rc)

    assert payload["healthy_harnesses"] == []
    result = payload["results"][0]
    assert result.stage == "install"
    assert result.code == "install_check_failed"
    assert 'must include "bash"' in (result.details or "")
    assert executor.calls == ["custom", "bash"]


@pytest.mark.asyncio
async def test_preflight_allows_custom_docker_image_without_python_runtime_requirement() -> None:
    def success(prepared: PreparedRun, model: str) -> RunResult:
        return _completed_result(prepared.task.id, "ok")

    rc = RuntimeConfig(
        project_root=Path.cwd(),
        executor_backend="docker",
        docker_image="debian:stable-slim",
        providers=_TEST_PROVIDERS,
    )
    executor = ScriptedDockerExecutor(
        rc,
        {
            "scripted": SubprocessResult(stdout="scripted 1.0.0\n", stderr="", exit_code=0, timed_out=False),
            "bash": SubprocessResult(stdout="GNU bash 5.2\n", stderr="", exit_code=0, timed_out=False),
        },
    )
    adapter = InstallCheckingScriptedAdapter(rc, executor, success)
    config = _make_eval_config(harnesses=["scripted"])

    payload = await run_harness_preflight(config, {"scripted": adapter}, rc)

    assert payload["healthy_harnesses"] == ["scripted"]
    assert executor.calls == ["scripted", "bash"]


@pytest.mark.asyncio
async def test_preflight_recovers_after_custom_docker_install_failure() -> None:
    def success(prepared: PreparedRun, model: str) -> RunResult:
        return _completed_result(prepared.task.id, "ok")

    failed_rc = RuntimeConfig(
        project_root=Path.cwd(),
        executor_backend="docker",
        docker_image="python:3.12-slim",
        providers=_TEST_PROVIDERS,
    )
    failed_executor = ScriptedDockerExecutor(
        failed_rc,
        {
            "custom": SubprocessResult(stdout="custom 1.0.0\n", stderr="", exit_code=0, timed_out=False),
            "bash": SubprocessResult(stdout="", stderr="bash: not found", exit_code=127, timed_out=False),
        },
    )
    failed_adapter = InstallCheckingAdapter(failed_rc, failed_executor)

    failed_payload = await run_harness_preflight(
        _make_eval_config(harnesses=["custom"], models=[ModelSpec(provider="openai", model="gpt-5.4")]),
        {"custom": failed_adapter},
        failed_rc,
    )
    assert failed_payload["healthy_harnesses"] == []

    passing_rc = RuntimeConfig(
        project_root=Path.cwd(),
        executor_backend="docker",
        docker_image="debian:stable-slim",
        providers=_TEST_PROVIDERS,
    )
    passing_executor = ScriptedDockerExecutor(
        passing_rc,
        {
            "scripted": SubprocessResult(stdout="scripted 1.0.0\n", stderr="", exit_code=0, timed_out=False),
            "bash": SubprocessResult(stdout="GNU bash 5.2\n", stderr="", exit_code=0, timed_out=False),
        },
    )
    passing_adapter = InstallCheckingScriptedAdapter(passing_rc, passing_executor, success)

    passing_payload = await asyncio.wait_for(
        run_harness_preflight(_make_eval_config(harnesses=["scripted"]), {"scripted": passing_adapter}, passing_rc),
        timeout=2,
    )

    assert passing_payload["healthy_harnesses"] == ["scripted"]
    assert passing_executor.calls == ["scripted", "bash"]


@pytest.mark.asyncio
async def test_preflight_fails_missing_judge_provider_config() -> None:
    rc = RuntimeConfig(project_root=Path.cwd(), providers=_TEST_PROVIDERS)
    adapter = ScriptedAdapter(
        rc, DummyExecutor(rc), [lambda prepared, model: _completed_result(prepared.task.id, "ok")]
    )
    config = _make_eval_config(
        harnesses=["scripted"],
        judge_model_spec=ModelSpec(provider="anthropic", model="claude-sonnet-4-6"),
    )

    payload = await run_harness_preflight(config, {"scripted": adapter}, rc)

    assert payload["judge_ok"] is False
    judge_result = next(r for r in payload["results"] if r.harness == "judge")
    assert judge_result.stage == "config"
    assert judge_result.code == "provider_config_error"


# ─── Layer 2: Probe ───


@pytest.mark.asyncio
async def test_preflight_passes_on_completed_run_with_trace() -> None:
    def success(prepared: PreparedRun, model: str) -> RunResult:
        return _completed_result(prepared.task.id, "some response text")

    rc = RuntimeConfig(project_root=Path.cwd(), providers=_TEST_PROVIDERS)
    adapter = ScriptedAdapter(rc, DummyExecutor(rc), [success])
    config = _make_eval_config(harnesses=["scripted"])

    payload = await run_harness_preflight(config, {"scripted": adapter}, rc)

    assert payload["healthy_harnesses"] == ["scripted"]
    result = payload["results"][0]
    assert result.status == "passed"
    assert result.stage == "probe"


@pytest.mark.asyncio
async def test_preflight_retries_transient_failure_then_passes() -> None:
    def transient_fail(prepared: PreparedRun, model: str) -> RunResult:
        return _failed_result(prepared.task.id)

    def success(prepared: PreparedRun, model: str) -> RunResult:
        return _completed_result(prepared.task.id, "ok", tool_calls=2)

    rc = RuntimeConfig(project_root=Path.cwd(), preflight_max_attempts=2, providers=_TEST_PROVIDERS)
    adapter = ScriptedAdapter(rc, DummyExecutor(rc), [transient_fail, success])
    config = _make_eval_config(harnesses=["scripted"])

    payload = await run_harness_preflight(config, {"scripted": adapter}, rc)

    assert adapter.run_calls == 2
    assert payload["healthy_harnesses"] == ["scripted"]
    result = payload["results"][0]
    assert result.status == "passed"
    assert result.warnings == ["count-only-tool-trace: metrics.tool_calls=2 but no trace tool_call events"]


@pytest.mark.asyncio
async def test_preflight_skips_probe_when_static_check_fails() -> None:
    """Config error should prevent any LLM calls."""
    rc = RuntimeConfig(project_root=Path.cwd(), providers={})  # no providers
    adapter = ScriptedAdapter(rc, DummyExecutor(rc), [])
    config = _make_eval_config(harnesses=["scripted"])

    payload = await run_harness_preflight(config, {"scripted": adapter}, rc)

    assert adapter.run_calls == 0  # never reached probe
    assert payload["healthy_harnesses"] == []
    result = payload["results"][0]
    assert result.stage == "config"


@pytest.mark.asyncio
async def test_preflight_skips_harness_probe_when_provider_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    response = DummyResponse(401, text="unauthorized")
    monkeypatch.setattr(
        "agent_harness_eval.preflight.httpx.AsyncClient",
        lambda timeout: DummyAsyncClient(timeout=timeout, response=response, recorder=calls),
    )

    rc = RuntimeConfig(project_root=Path.cwd(), providers=_TEST_PROVIDERS)
    adapter = ScriptedAdapter(
        rc, DummyExecutor(rc), [lambda prepared, model: _completed_result(prepared.task.id, "ok")]
    )
    config = _make_eval_config(harnesses=["scripted"])

    payload = await run_harness_preflight(config, {"scripted": adapter}, rc)

    assert adapter.run_calls == 0
    assert len(calls) == 1
    assert payload["healthy_harnesses"] == []
    result = payload["results"][0]
    assert result.stage == "provider"
    assert result.code == "provider_api_error"
    assert result.failure_origin == "provider"


@pytest.mark.asyncio
async def test_preflight_deduplicates_provider_probe_across_harnesses(
    _mock_provider_probe_http: list[dict],
) -> None:
    def success(prepared: PreparedRun, model: str) -> RunResult:
        return _completed_result(prepared.task.id, "ok")

    rc = RuntimeConfig(project_root=Path.cwd(), providers=_TEST_PROVIDERS)
    adapter_a = ScriptedAdapter(rc, DummyExecutor(rc), [success])
    adapter_b = ScriptedAdapter(rc, DummyExecutor(rc), [success])
    config = _make_eval_config(harnesses=["scripted-a", "scripted-b"])

    payload = await run_harness_preflight(
        config,
        {"scripted-a": adapter_a, "scripted-b": adapter_b},
        rc,
    )

    assert len(_mock_provider_probe_http) == 1
    assert adapter_a.run_calls == 1
    assert adapter_b.run_calls == 1
    assert sorted(payload["healthy_harnesses"]) == ["scripted-a", "scripted-b"]


@pytest.mark.asyncio
async def test_preflight_deduplicates_provider_probe_with_judge_model(
    _mock_provider_probe_http: list[dict],
) -> None:
    def success(prepared: PreparedRun, model: str) -> RunResult:
        return _completed_result(prepared.task.id, "ok")

    rc = RuntimeConfig(project_root=Path.cwd(), providers=_TEST_PROVIDERS)
    adapter = ScriptedAdapter(rc, DummyExecutor(rc), [success])
    config = _make_eval_config(
        harnesses=["scripted"],
        judge_model_spec=ModelSpec(provider="openai", model="gpt-5.4"),
    )

    payload = await run_harness_preflight(config, {"scripted": adapter}, rc)

    assert payload["judge_ok"] is True
    assert len(_mock_provider_probe_http) == 1
    assert payload["healthy_harnesses"] == ["scripted"]
