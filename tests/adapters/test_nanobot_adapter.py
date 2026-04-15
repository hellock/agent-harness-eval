from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agent_harness_eval.adapters.interface import PreparedRun
from agent_harness_eval.adapters.nanobot import (
    NanobotAdapter,
    _build_nanobot_runtime_config,
    _normalize_session_ts,
    _prime_nanobot_workspace,
    _read_nanobot_session,
    _write_nanobot_eval_wrapper,
)
from agent_harness_eval.config.providers import ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.executor import ExecutionPolicy, Executor
from agent_harness_eval.task import Task, ToolBoundary
from agent_harness_eval.types import CanonicalTraceEvent
from agent_harness_eval.utils.subprocess import SubprocessResult
from agent_harness_eval.utils.workspace import RunLayout

from ._regression_helpers import RecordingExecutor


@pytest.fixture
def isolated_run_dir() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agent-harness-eval-test-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _make_layout(isolated_run_dir: Path) -> tuple[Path, Path, RunLayout]:
    workspace_dir = isolated_run_dir / "workspace"
    runtime_dir = isolated_run_dir / "runtime"
    workspace_dir.mkdir()
    runtime_dir.mkdir()
    layout = RunLayout(
        root_dir=str(isolated_run_dir),
        input_dir=str(isolated_run_dir / "input"),
        workspace_seed_dir=str(isolated_run_dir / "input" / "workspace"),
        workspace_dir=str(workspace_dir),
        state_dir=str(runtime_dir),
        output_dir=str(isolated_run_dir / "output"),
    )
    Path(layout.input_dir).mkdir(parents=True, exist_ok=True)
    Path(layout.workspace_seed_dir).mkdir(parents=True, exist_ok=True)
    Path(layout.output_dir).mkdir(parents=True, exist_ok=True)
    return workspace_dir, runtime_dir, layout


class _RecordingExecutor(Executor):
    """Test executor that records execute() calls and returns a canned result."""

    def __init__(self, runtime_config: RuntimeConfig, result: SubprocessResult):
        super().__init__(runtime_config)
        self._result = result
        self.calls: list[dict] = []

    async def execute(self, harness, policy, inner_command, inner_args, inner_env, timeout_ms):
        self.calls.append(
            {
                "harness": harness,
                "command": inner_command,
                "args": list(inner_args),
                "env": dict(inner_env),
            }
        )
        return self._result


@pytest.mark.asyncio
async def test_nanobot_run_returns_failed_on_non_zero_subprocess_exit(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_dir, runtime_dir, layout = _make_layout(isolated_run_dir)
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_KEY", "test-key")
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_BASE_URL", "http://relay")
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_FORMAT", "anthropic")

    task = Task(
        id="security.01",
        category="security",
        description="docker sandbox failure regression",
        user_query="Read PRECHECK.md and return the marker.",
        tool_boundary=ToolBoundary(internet="disabled", shell="enabled", file_write="disabled"),
        timeout_sec=30,
    )
    prepared = PreparedRun(
        task=task,
        layout=layout,
        env={"_EVAL_SESSION_ID": "test-session", "_EVAL_RUNTIME_DIR": str(runtime_dir)},
        execution_policy=ExecutionPolicy(workspace_dir=str(workspace_dir)),
    )
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        executor_backend="docker",
        providers={"relay": ProviderConfig(base_url="http://relay", api_key="test-key", api_format="anthropic")},
    )
    executor = _RecordingExecutor(
        runtime_config,
        SubprocessResult(
            stdout="",
            stderr="permission denied while trying to connect to the docker API at unix:///var/run/docker.sock",
            exit_code=-1,
            timed_out=False,
        ),
    )

    result = await NanobotAdapter(runtime_config, executor).run(prepared, "relay:claude-sonnet-4-6")

    assert result.status == "failed"
    assert result.failure_origin == "sandbox"
    assert result.infra_error_code == "sandbox_permission_error"
    assert len(result.trace) == 1
    assert result.trace[0].type == "task_failed"
    assert result.trace[0].error is not None
    assert "permission denied" in result.trace[0].error.lower()


@pytest.mark.asyncio
async def test_nanobot_run_writes_json_runtime_config_from_provider_env(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_dir, runtime_dir, layout = _make_layout(isolated_run_dir)
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_KEY", "test-key")
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_BASE_URL", "http://relay")
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_FORMAT", "anthropic")

    task = Task(
        id="security.01",
        category="security",
        description="runtime config generation",
        user_query="Read PRECHECK.md and return the marker.",
        tool_boundary=ToolBoundary(internet="disabled", shell="enabled", file_write="disabled"),
        timeout_sec=30,
    )
    prepared = PreparedRun(
        task=task,
        layout=layout,
        env={"_EVAL_SESSION_ID": "test-session", "_EVAL_RUNTIME_DIR": str(runtime_dir)},
        execution_policy=ExecutionPolicy(workspace_dir=str(workspace_dir)),
    )
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        executor_backend="docker",
        providers={"relay": ProviderConfig(base_url="http://relay", api_key="test-key", api_format="anthropic")},
    )
    executor = _RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="MARKER-OK", stderr="", exit_code=0, timed_out=False),
    )

    result = await NanobotAdapter(runtime_config, executor).run(prepared, "relay:claude-sonnet-4-6")

    assert result.status == "completed"
    assert len(executor.calls) == 1
    call = executor.calls[0]
    assert call["command"] == "python"
    assert "-c" in call["args"]
    config_index = call["args"].index("-c") + 1
    assert Path(call["args"][config_index]) == runtime_dir / "config.json"

    config_data = json.loads((runtime_dir / "config.json").read_text(encoding="utf-8"))
    assert config_data["providers"]["anthropic"]["apiKey"] == "test-key"
    assert config_data["providers"]["anthropic"]["apiBase"] == "http://relay"
    assert config_data["agents"]["defaults"]["model"] == "claude-sonnet-4-6"
    assert config_data["agents"]["defaults"]["provider"] == "anthropic"


def test_write_nanobot_eval_wrapper_embeds_state_dir_flag(
    isolated_run_dir: Path,
) -> None:
    wrapper_path = Path(_write_nanobot_eval_wrapper(str(isolated_run_dir)))

    wrapper_text = wrapper_path.read_text(encoding="utf-8")
    assert wrapper_path.is_file()
    assert 'parser.add_argument("--state-dir", required=True)' in wrapper_text
    assert "session_manager = SessionManager(state_dir)" in wrapper_text
    assert 'CronService(state_dir / "cron" / "jobs.json")' in wrapper_text


def test_read_nanobot_session_extracts_trace_and_usage(
    isolated_run_dir: Path,
) -> None:
    session_dir = isolated_run_dir / "runtime" / "sessions"
    session_dir.mkdir(parents=True)
    session_path = session_dir / "eval-session.jsonl"
    session_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "_type": "metadata",
                        "metadata": {
                            "last_usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 4,
                                "cache_read_input_tokens": 3,
                                "cache_creation_input_tokens": 2,
                                "total_tokens": 19,
                            }
                        },
                    }
                ),
                json.dumps({"role": "user", "content": "read PRECHECK.md", "timestamp": "2026-04-11T00:00:00+00:00"}),
                json.dumps(
                    {
                        "role": "assistant",
                        "content": "I'll read it now.",
                        "tool_calls": [
                            {
                                "id": "tool-1",
                                "function": {"name": "read_file", "arguments": '{"path":"/tmp/PRECHECK.md"}'},
                            }
                        ],
                        "timestamp": "2026-04-11T00:00:01+00:00",
                    }
                ),
                json.dumps(
                    {
                        "role": "tool",
                        "tool_call_id": "tool-1",
                        "name": "read_file",
                        "content": "1| MARKER=MARKER-123",
                        "timestamp": "2026-04-11T00:00:02+00:00",
                    }
                ),
                json.dumps(
                    {
                        "role": "assistant",
                        "content": "PREFLIGHT_OK MARKER-123",
                        "timestamp": "2026-04-11T00:00:03+00:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    session_data = _read_nanobot_session(str(isolated_run_dir / "runtime"), "eval-session")

    assert [event.type for event in session_data["trace"]] == [
        "message",
        "message",
        "tool_call_started",
        "tool_call_completed",
        "message",
    ]
    assert session_data["trace"][2].tool_name == "read_file"
    assert session_data["trace"][2].input == {"path": "/tmp/PRECHECK.md"}
    assert session_data["trace"][3].success is True
    assert session_data["usage"] == {
        "input": 10,
        "output": 4,
        "cache_read": 3,
        "cache_write": 2,
        "total_tokens": 19,
        "tool_calls": 1,
        "turns": 2,
    }


def test_prime_nanobot_workspace_creates_required_templates(
    isolated_run_dir: Path,
) -> None:
    workspace_dir = isolated_run_dir / "workspace"
    workspace_dir.mkdir(parents=True)

    _prime_nanobot_workspace(workspace_dir)

    assert (workspace_dir / "SOUL.md").exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "TOOLS.md").exists()
    assert (workspace_dir / "USER.md").exists()
    assert (workspace_dir / "HEARTBEAT.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()
    assert (workspace_dir / "memory" / "history.jsonl").exists()
    assert (workspace_dir / "skills").is_dir()
    assert (workspace_dir / ".git").is_dir()


def test_build_nanobot_runtime_config_uses_custom_provider_for_openai_compatible_relays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVAL_PROVIDER_MYRELAY_API_KEY", "relay-key")
    monkeypatch.setenv("EVAL_PROVIDER_MYRELAY_BASE_URL", "https://relay.example.com/v1")
    monkeypatch.setenv("EVAL_PROVIDER_MYRELAY_API_FORMAT", "openai-chat-completions")

    from agent_harness_eval.config.providers import ModelSpec

    config = _build_nanobot_runtime_config(
        ModelSpec(provider="myrelay", model="gpt-5.4"),
        "/tmp/workspace",
        ProviderConfig(
            base_url="https://relay.example.com/v1",
            api_key="relay-key",
            api_format="openai-chat-completions",
        ),
    )

    assert config["providers"]["custom"]["apiKey"] == "relay-key"
    assert config["providers"]["custom"]["apiBase"] == "https://relay.example.com/v1"
    assert config["agents"]["defaults"]["provider"] == "custom"
    assert config["agents"]["defaults"]["model"] == "gpt-5.4"


@pytest.mark.asyncio
async def test_nanobot_run_builds_wrapper_runtime_config_and_workspace_contract(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="openai-key",
                api_format="openai-responses",
                extra_headers={"x-eval": "1"},
            )
        },
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="NANOBOT_OK", stderr="", exit_code=0, timed_out=False),
    )
    adapter = NanobotAdapter(runtime_config, executor)
    task = Task(
        id="nanobot.regression.01",
        category="coding",
        description="nanobot runtime contract",
        user_query="Reply with NANOBOT_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "nanobot-regression")

    monkeypatch.setattr("agent_harness_eval.adapters.nanobot.resolve_executor_backend", lambda runtime_cfg: "docker")
    monkeypatch.setattr(
        "agent_harness_eval.adapters.nanobot._read_nanobot_session",
        lambda runtime_dir, session_id: {
            "trace": [
                CanonicalTraceEvent(
                    type="message",
                    role="assistant",
                    text="NANOBOT_OK",
                    ts="2026-01-01T00:00:00+00:00",
                )
            ],
            "usage": {
                "input": 9,
                "output": 3,
                "cache_read": 1,
                "cache_write": 0,
                "total_tokens": 13,
                "tool_calls": 0,
                "turns": 1,
            },
        },
    )

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "completed"
        assert result.final_text == "NANOBOT_OK"
        assert len(executor.calls) == 1
        call = executor.calls[0]
        assert call["inner_command"] == "python"
        args = call["inner_args"]
        assert isinstance(args, list)
        assert args[0].endswith("nanobot_eval_agent.py")
        assert args[args.index("-s") + 1] == prepared.env["_EVAL_SESSION_ID"]
        assert args[args.index("-w") + 1] == prepared.workspace_dir
        assert args[args.index("-c") + 1].endswith("config.json")
        assert args[args.index("--state-dir") + 1] == prepared.env["_EVAL_RUNTIME_DIR"]
        assert "--no-markdown" in args

        runtime_dir = Path(prepared.env["_EVAL_RUNTIME_DIR"])
        config_data = json.loads((runtime_dir / "config.json").read_text(encoding="utf-8"))
        assert config_data["providers"]["custom"]["apiKey"] == "openai-key"
        assert config_data["providers"]["custom"]["apiBase"] == "https://relay.example.com/v1"
        assert config_data["providers"]["custom"]["extraHeaders"] == {"x-eval": "1"}
        assert config_data["agents"]["defaults"]["provider"] == "custom"
        assert config_data["agents"]["defaults"]["model"] == "gpt-5.4"

        workspace_dir = Path(prepared.workspace_dir)
        assert workspace_dir.joinpath("SOUL.md").is_file()
        assert workspace_dir.joinpath("AGENTS.md").is_file()
        assert workspace_dir.joinpath("memory", "MEMORY.md").is_file()
        assert workspace_dir.joinpath("skills").is_dir()
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_nanobot_run_returns_failed_result_when_session_parse_fails(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="openai-key",
                api_format="openai-responses",
            )
        },
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="NANOBOT_BAD", stderr="session unreadable", exit_code=0, timed_out=False),
    )
    adapter = NanobotAdapter(runtime_config, executor)
    task = Task(
        id="nanobot.regression.parse-failure",
        category="coding",
        description="nanobot parse failure",
        user_query="Reply with NANOBOT_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "nanobot-parse-failure")

    monkeypatch.setattr("agent_harness_eval.adapters.nanobot.resolve_executor_backend", lambda runtime_cfg: "docker")
    monkeypatch.setattr(
        "agent_harness_eval.adapters.nanobot._read_nanobot_session",
        lambda runtime_dir, session_id: (_ for _ in ()).throw(ValueError("bad session trace")),
    )

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "failed"
        assert result.final_text == "NANOBOT_BAD"
        assert result.failure_origin == "adapter"
        assert result.infra_error_code == "adapter_output_error"
        assert result.trace and result.trace[0].type == "task_failed"
        assert "Nanobot output parse error" in (result.trace[0].error or "")
    finally:
        adapter.cleanup(prepared)


def test_normalize_session_ts_forces_utc_on_naive_input() -> None:
    """Nanobot's session.jsonl stores naive datetimes; they must be tagged UTC.

    Leaking naive ts into CanonicalTraceEvent causes every downstream
    ``fromisoformat`` + UTC arithmetic to crash with "can't subtract
    offset-naive and offset-aware datetimes". The normalizer (now a thin
    wrapper around ``utils.timestamps.to_canonical_ts``) parses, tags naive
    input as UTC, **converts non-UTC offsets to UTC** (canonical contract),
    truncates sub-ms precision, and falls back to now() on unparseable input.
    """
    naive = _normalize_session_ts("2026-04-15T07:21:59.336779")
    assert naive == "2026-04-15T07:21:59.336+00:00"

    z_form = _normalize_session_ts("2026-04-15T07:21:59.336779Z")
    assert z_form == "2026-04-15T07:21:59.336+00:00"

    # Non-UTC offset is converted to UTC (07:21:59 +05:30 → 01:51:59 UTC).
    explicit_offset = _normalize_session_ts("2026-04-15T07:21:59.336779+05:30")
    assert explicit_offset == "2026-04-15T01:51:59.336+00:00"

    # Unparseable / missing → fall through to now(UTC); numeric is treated
    # as epoch seconds by the canonical helper, so 12345 is no longer
    # "garbage" — it parses as 1970-01-01T03:25:45 UTC.
    for bad_input in ("", None, "not-a-timestamp"):
        fallback = _normalize_session_ts(bad_input)
        assert fallback.endswith("+00:00")


def test_read_nanobot_session_normalizes_naive_session_timestamps(
    isolated_run_dir: Path,
) -> None:
    """Regression: nanobot used to pass naive session.jsonl ts through verbatim.

    End-to-end check that a session log with naive ``timestamp`` fields yields
    only tz-aware ts on the returned trace events — matching the convention
    every other adapter already follows.
    """
    session_dir = isolated_run_dir / "runtime" / "sessions"
    session_dir.mkdir(parents=True)
    session_path = session_dir / "naive-session.jsonl"
    session_path.write_text(
        "\n".join(
            [
                # naive (no offset) — this is the real-world nanobot signature
                json.dumps({"role": "user", "content": "hi", "timestamp": "2026-04-15T07:21:59.336779"}),
                json.dumps(
                    {
                        "role": "assistant",
                        "content": "hello",
                        "timestamp": "2026-04-15T07:21:59.337100",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    session_data = _read_nanobot_session(str(isolated_run_dir / "runtime"), "naive-session")

    assert len(session_data["trace"]) == 2
    for event in session_data["trace"]:
        # Every emitted ts must be tz-aware — no naive strings leak out
        assert event.ts is not None
        assert event.ts.endswith("+00:00")


@pytest.mark.asyncio
async def test_nanobot_timeout_surfaces_session_recovery_failure_as_infra_details(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout + unreadable session.jsonl must not look like a clean timeout.

    Previously ``except Exception: pass`` swallowed recovery failures silently,
    making "timeout with no recoverable state" and "timeout + corrupted
    session log" indistinguishable in downstream reports. The fix narrows
    the exception list and stamps the failure reason into
    ``infra_error_details`` so the two cases stay separable.
    """
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="openai-key",
                api_format="openai-responses",
            )
        },
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=True),
    )
    adapter = NanobotAdapter(runtime_config, executor)
    task = Task(
        id="nanobot.regression.timeout-recovery-error",
        category="skills",
        description="nanobot timeout + session recovery failure",
        user_query="Reply with NANOBOT_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "nanobot-timeout-recovery-error")

    def _raise_corrupt(runtime_dir: str, session_id: str) -> dict:
        raise ValueError("session jsonl line 3 not valid json")

    monkeypatch.setattr(
        "agent_harness_eval.adapters.nanobot._read_nanobot_session",
        _raise_corrupt,
    )

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "timed_out"
        # Empty recovery: trace stays empty, usage stays at zeros
        assert result.trace == []
        assert result.metrics.total_tokens == 0
        # But the *reason* recovery failed must be preserved, not swallowed
        assert result.infra_error_details is not None
        assert "session.jsonl recovery failed" in result.infra_error_details
        assert "ValueError" in result.infra_error_details
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_nanobot_run_recovers_trace_and_usage_on_timeout(
    isolated_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = RuntimeConfig(
        project_root=isolated_run_dir,
        providers={
            "relay": ProviderConfig(
                base_url="https://relay.example.com/v1",
                api_key="openai-key",
                api_format="openai-responses",
            )
        },
    )
    executor = RecordingExecutor(
        runtime_config,
        SubprocessResult(stdout="", stderr="", exit_code=0, timed_out=True),
    )
    adapter = NanobotAdapter(runtime_config, executor)
    task = Task(
        id="nanobot.regression.timeout-recovery",
        category="skills",
        description="nanobot timeout recovery",
        user_query="Reply with NANOBOT_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "nanobot-timeout-recovery")

    monkeypatch.setattr(
        "agent_harness_eval.adapters.nanobot._read_nanobot_session",
        lambda runtime_dir, session_id: {
            "trace": [
                CanonicalTraceEvent(
                    type="message",
                    role="assistant",
                    text="Recovered timeout answer",
                    ts="2026-01-01T00:00:00+00:00",
                )
            ],
            "usage": {
                "input": 15,
                "output": 7,
                "cache_read": 2,
                "cache_write": 1,
                "total_tokens": 25,
                "tool_calls": 3,
                "turns": 2,
            },
        },
    )

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "timed_out"
        assert result.final_text == "Recovered timeout answer"
        assert result.trace and result.trace[0].type == "message"
        assert result.metrics.input_tokens == 15
        assert result.metrics.output_tokens == 7
        assert result.metrics.cache_read_tokens == 2
        assert result.metrics.cache_write_tokens == 1
        assert result.metrics.total_tokens == 25
        assert result.metrics.tool_calls == 3
        assert result.metrics.turns == 2
    finally:
        adapter.cleanup(prepared)
