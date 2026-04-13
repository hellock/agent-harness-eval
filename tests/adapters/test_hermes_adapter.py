from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness_eval.adapters.hermes import (
    HermesAdapter,
    _extract_response_text,
    _extract_session_id,
    _read_hermes_session,
)
from agent_harness_eval.config.providers import ProviderConfig
from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.task import Task
from agent_harness_eval.types import CanonicalTraceEvent
from agent_harness_eval.utils.subprocess import SubprocessResult

from ._regression_helpers import RecordingExecutor


@pytest.mark.asyncio
async def test_hermes_run_builds_expected_shell_script_and_env(
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
        SubprocessResult(
            stdout="╭─ ⚕ Hermes ─────╮\nHERMES_OK\nsession_id: sess-123\n╰────────────────╯\n",
            stderr="",
            exit_code=0,
            timed_out=False,
        ),
    )
    adapter = HermesAdapter(runtime_config, executor)
    task = Task(
        id="hermes.regression.01",
        category="coding",
        description="hermes cli contract",
        user_query="Reply with HERMES_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "hermes-regression")

    monkeypatch.setattr(
        "agent_harness_eval.adapters.hermes._read_hermes_session",
        lambda runtime_dir, session_id: {
            "trace": [
                CanonicalTraceEvent(type="message", role="assistant", text="HERMES_OK", ts="2026-01-01T00:00:00+00:00")
            ],
            "usage": {
                "input": 10,
                "output": 5,
                "cache_read": 2,
                "cache_write": 1,
                "total_tokens": 18,
                "tool_calls": 1,
                "turns": 1,
            },
        },
    )

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "completed"
        assert result.final_text == "HERMES_OK"
        assert len(executor.calls) == 1
        call = executor.calls[0]
        assert call["inner_command"] == "sh"
        args = call["inner_args"]
        assert isinstance(args, list)
        assert "-c" in args
        script = args[1]
        assert "config set model.provider custom" in script
        assert "config set model.default gpt-5.4" in script
        assert "config set model.api_key_env OPENAI_API_KEY" in script
        assert "config set model.base_url https://relay.example.com/v1" in script
        assert "chat --quiet --query" in script
        env = call["inner_env"]
        assert isinstance(env, dict)
        assert env["OPENAI_API_KEY"] == "openai-key"
        assert env["HOME"] == prepared.env["_EVAL_RUNTIME_DIR"]
        assert env["HERMES_HOME"] == prepared.env["_EVAL_RUNTIME_DIR"]
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_hermes_run_returns_failed_result_when_session_parse_fails(
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
        SubprocessResult(
            stdout="╭─ ⚕ Hermes ─────╮\nHERMES_BAD\nsession_id: sess-123\n╰────────────────╯\n",
            stderr="session db unreadable",
            exit_code=0,
            timed_out=False,
        ),
    )
    adapter = HermesAdapter(runtime_config, executor)
    task = Task(
        id="hermes.regression.parse-failure",
        category="coding",
        description="hermes parse failure",
        user_query="Reply with HERMES_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "hermes-parse-failure")

    monkeypatch.setattr(
        "agent_harness_eval.adapters.hermes._read_hermes_session",
        lambda runtime_dir, session_id: (_ for _ in ()).throw(ValueError("bad state db")),
    )

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "failed"
        assert "HERMES_BAD" in result.final_text
        assert result.failure_origin == "adapter"
        assert result.infra_error_code == "adapter_output_error"
        assert result.trace and result.trace[0].type == "task_failed"
        assert "Hermes session parse error" in (result.trace[0].error or "")
    finally:
        adapter.cleanup(prepared)


@pytest.mark.asyncio
async def test_hermes_run_returns_timed_out_result_without_session_parse(
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
    adapter = HermesAdapter(runtime_config, executor)
    task = Task(
        id="hermes.regression.timeout",
        category="coding",
        description="hermes timeout result",
        user_query="Reply with HERMES_OK.",
        workspace_files=[{"path": "README.md", "content": "seed\n"}],
        timeout_sec=30,
    )
    prepared = adapter.prepare(task, "hermes-timeout")

    monkeypatch.setattr(
        "agent_harness_eval.adapters.hermes._read_hermes_session",
        lambda runtime_dir, session_id: (_ for _ in ()).throw(
            AssertionError("session parse should not run on timeout")
        ),
    )

    try:
        result = await adapter.run(prepared, "relay:gpt-5.4")

        assert result.status == "timed_out"
        assert result.final_text == ""
        assert result.trace == []
        assert result.metrics.latency_sec == 30
    finally:
        adapter.cleanup(prepared)


def test_extract_hermes_response_text_strips_box_and_session_metadata() -> None:
    stdout = """
╭─ ⚕ Hermes ─────╮
Hello from Hermes
session_id: abc123
╰────────────────╯
""".strip()

    assert _extract_response_text(stdout) == "Hello from Hermes"


def test_extract_hermes_response_text_strips_tool_progress_lines() -> None:
    stdout = """
╭─ ⚕ Hermes ─────╮
┊ ⚡ preparing mcp_todo…
┊ 📋 plan      5 task(s)  0.0s
┊ 🔎 find      cache  0.6s
The task is complete.
session_id: abc123
╰────────────────╯
""".strip()

    assert _extract_response_text(stdout) == "The task is complete."


def test_extract_hermes_response_text_strips_emoji_progress_lines() -> None:
    stdout = """
╭─ ⚕ Hermes ─────╮
⚡ warming up
✅ done processing
Here is the actual answer.
session_id: abc123
╰────────────────╯
""".strip()

    assert _extract_response_text(stdout) == "Here is the actual answer."


def test_extract_hermes_session_id() -> None:
    stdout = "Hello from Hermes\n\nsession_id: 20260411_112755_bad165\n"
    assert _extract_session_id(stdout) == "20260411_112755_bad165"


def test_extract_hermes_session_id_missing() -> None:
    assert _extract_session_id("no session here") is None
    assert _extract_session_id("") is None
    assert _extract_session_id(None) is None


def test_read_hermes_session_missing_db(tmp_path: Path) -> None:
    result = _read_hermes_session(str(tmp_path), "some-session-id")
    assert result["trace"] == []
    assert result["usage"]["tool_calls"] == 0
    assert result["usage"]["input"] == 0


def test_read_hermes_session_no_session_id_no_db(tmp_path: Path) -> None:
    result = _read_hermes_session(str(tmp_path), None)
    assert result["trace"] == []


def test_read_hermes_session_no_session_id_falls_back_to_latest(tmp_path: Path) -> None:
    import sqlite3
    import time

    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (6);

        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT NOT NULL,
            started_at REAL NOT NULL, input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0, cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0, tool_call_count INTEGER DEFAULT 0,
            message_count INTEGER DEFAULT 0
        );

        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT, tool_call_id TEXT, tool_calls TEXT,
            tool_name TEXT, timestamp REAL NOT NULL,
            token_count INTEGER, finish_reason TEXT,
            reasoning TEXT, reasoning_details TEXT,
            codex_reasoning_items TEXT
        );
    """)

    now = time.time()
    conn.execute(
        "INSERT INTO sessions (id, source, started_at, input_tokens, output_tokens) VALUES (?, ?, ?, ?, ?)",
        ("old_session", "tool", now - 100, 100, 50),
    )
    conn.execute(
        "INSERT INTO sessions (id, source, started_at, input_tokens, output_tokens) VALUES (?, ?, ?, ?, ?)",
        ("latest_session", "tool", now, 500, 300),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("latest_session", "assistant", "Hello from latest", now + 1),
    )
    conn.commit()
    conn.close()

    result = _read_hermes_session(str(tmp_path), None)
    assert result["usage"]["input"] == 500
    assert result["usage"]["output"] == 300
    assert len(result["trace"]) == 1
    assert result["trace"][0].text == "Hello from latest"


def test_read_hermes_session_extracts_messages(tmp_path: Path) -> None:
    import json
    import sqlite3
    import time

    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (6);

        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            user_id TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            title TEXT
        );

        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_count INTEGER,
            finish_reason TEXT,
            reasoning TEXT,
            reasoning_details TEXT,
            codex_reasoning_items TEXT
        );
    """)

    session_id = "test_session_001"
    now = time.time()
    conn.execute(
        """INSERT INTO sessions (id, source, started_at, input_tokens, output_tokens,
           cache_read_tokens, cache_write_tokens, tool_call_count, message_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, "tool", now, 1500, 800, 200, 100, 2, 4),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, "user", "Read the file data.csv", now + 1),
    )

    tool_calls = json.dumps(
        [
            {
                "id": "call_001",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "data.csv"})},
            }
        ]
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, tool_calls, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, "assistant", "", tool_calls, now + 2),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, tool_call_id, tool_name, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, "tool", "col1,col2\n1,2\n3,4", "call_001", "read_file", now + 3),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, "assistant", "The file contains 2 rows of data.", now + 4),
    )
    conn.commit()
    conn.close()

    result = _read_hermes_session(str(tmp_path), session_id)

    assert result["usage"]["input"] == 1500
    assert result["usage"]["output"] == 800
    assert result["usage"]["cache_read"] == 200
    assert result["usage"]["cache_write"] == 100
    assert result["usage"]["tool_calls"] == 2
    assert result["usage"]["total_tokens"] == 2300
    assert result["usage"]["turns"] == 2

    trace = result["trace"]
    assert len(trace) == 4
    assert trace[0].type == "message"
    assert trace[0].role == "user"
    assert trace[1].type == "tool_call_started"
    assert trace[1].tool_name == "read_file"
    assert trace[1].input == {"path": "data.csv"}
    assert trace[2].type == "tool_call_completed"
    assert trace[2].tool_name == "read_file"
    assert trace[2].success is True
    assert "col1,col2" in trace[2].output
    assert trace[3].type == "message"
    assert trace[3].role == "assistant"
    assert "2 rows" in trace[3].text


def test_read_hermes_session_orphaned_tool_calls(tmp_path: Path) -> None:
    import json
    import sqlite3
    import time

    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (6);

        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT NOT NULL,
            started_at REAL NOT NULL, input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0, cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0, tool_call_count INTEGER DEFAULT 0,
            message_count INTEGER DEFAULT 0
        );

        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT, tool_call_id TEXT, tool_calls TEXT,
            tool_name TEXT, timestamp REAL NOT NULL,
            token_count INTEGER, finish_reason TEXT,
            reasoning TEXT, reasoning_details TEXT,
            codex_reasoning_items TEXT
        );
    """)

    session_id = "orphan_test"
    now = time.time()
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
        (session_id, "tool", now),
    )
    tool_calls = json.dumps([{"id": "call_orphan", "function": {"name": "exec", "arguments": '{"command": "ls"}'}}])
    conn.execute(
        "INSERT INTO messages (session_id, role, content, tool_calls, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, "assistant", "", tool_calls, now + 1),
    )
    conn.commit()
    conn.close()

    result = _read_hermes_session(str(tmp_path), session_id)
    trace = result["trace"]

    assert len(trace) == 2
    assert trace[0].type == "tool_call_started"
    assert trace[0].tool_name == "exec"
    assert trace[1].type == "tool_call_completed"
    assert trace[1].tool_name == "exec"
    assert trace[1].success is False
    assert trace[1].output == "<no result recorded>"
