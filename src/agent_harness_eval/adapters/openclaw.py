"""OpenClaw adapter.

Uses `--profile agent-eval` to run a fully isolated OpenClaw instance.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar

from ..config.providers import ProviderConfig, parse_model_spec
from ..constants import STDERR_PREVIEW_MAX_CHARS, TOOL_OUTPUT_MAX_CHARS
from ..executor import attach_run_layout_mounts, filter_env, policy_from_task
from ..task import Task
from ..types import CanonicalTraceEvent, RunMetrics, RunResult
from ..utils.conversation import format_task_message
from ..utils.cost import calculate_cost
from ..utils.failure_origin import detect_failure_origin_from_error
from ..utils.subprocess import SubprocessResult
from ..utils.workspace import create_run_layout, remove_workspace
from . import register_adapter
from .interface import (
    HarnessAdapter,
    NativeMemoryFile,
    PreparedRun,
    detect_subprocess_failure,
)


@register_adapter
class OpenClawAdapter(HarnessAdapter):
    name = "openclaw"
    managed_docker_image = True
    _profile_name = "agent-eval"
    supports_native_memory = True
    supports_conversation_history_replay = True
    emits_paired_trace_events = True

    required_env_vars: ClassVar[list[list[str]]] = [["ANTHROPIC_API_KEY"], ["OPENAI_API_KEY"]]

    def install_memory(self, prepared: PreparedRun, files: list[NativeMemoryFile]) -> None:
        if not files:
            return
        _install_openclaw_native_memory_files(prepared.workspace_dir, files)

    def prepare(self, task: Task, run_id: str) -> PreparedRun:
        """File-level setup only — no harness CLI calls.

        Writes workspace files, bootstraps the eval profile config, and
        pre-generates auth files. All of this is file I/O that works
        regardless of whether run() executes locally or in Docker
        (the state dir is volume-mounted in docker mode).
        """
        layout = create_run_layout(run_id, workspace_files=task.workspace_files)
        workspace_dir = layout.workspace_dir
        execution_policy = policy_from_task(task, workspace_dir, task.timeout_sec)
        attach_run_layout_mounts(execution_policy, layout)
        state_dir = layout.state_dir

        agent_id, session_id = _make_ephemeral_ids()

        _bootstrap_eval_profile(state_dir)

        return PreparedRun(
            task=task,
            layout=layout,
            env={
                "_EVAL_AGENT_ID": agent_id,
                "_EVAL_SESSION_ID": session_id,
                "_OPENCLAW_STATE_DIR": state_dir,
            },
            execution_policy=execution_policy,
        )

    async def run(self, prepared: PreparedRun, model: str) -> RunResult:
        """Agent registration + execution in a single executor call.

        Both steps run in the same environment (host or docker container),
        so paths and configs are consistent.
        """
        task = prepared.task
        workspace_dir = prepared.workspace_dir
        execution_policy = prepared.execution_policy
        model_spec = parse_model_spec(model)
        agent_id = prepared.env["_EVAL_AGENT_ID"]
        session_id = prepared.env["_EVAL_SESSION_ID"]
        state_dir = prepared.env["_OPENCLAW_STATE_DIR"]
        start_time = asyncio.get_running_loop().time()
        timeout_ms = task.timeout_sec * 1000

        message_text = format_task_message(task)

        provider_config = self.resolve_provider(model_spec)
        # Derive the provider name for openclaw's internal registration
        from .interface import _harness_config_key

        yaml_key = _harness_config_key(self.name)
        harness_cfg = (
            self.runtime_config.harness_config.get(yaml_key) or self.runtime_config.harness_config.get(self.name) or {}
        )
        provider_name = str(harness_cfg.get("provider") or model_spec.provider).strip()
        qualified_model = _qualify_openclaw_model(model_spec.model, provider_name)
        key_env_var = _resolve_api_key_env_var(provider_name, provider_config)

        # Ensure model is configured in the profile (file I/O)
        adapter_env = self.runtime_config.subprocess_env
        _ensure_openclaw_model_configured(
            state_dir,
            qualified_model,
            provider_name,
            provider_config,
        )

        auth_key_names = _read_auth_key_names(state_dir, key_env_var)
        base_env = _openclaw_base_env(state_dir)
        inner_env = filter_env(
            adapter_env,
            {**prepared.env, **base_env},
            [
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "OPENCLAW_STATE_DIR",
                "OPENCLAW_CONFIG_PATH",
                "HOME",
                *auth_key_names,
            ],
        )
        openclaw_bin = self.resolve_binary()
        # Build a compound shell script that does register + run in one go.
        # This ensures both steps share the same filesystem view (critical
        # for docker mode where the state dir is volume-mounted).
        api_format = provider_config.api_format
        register_cmd = _build_register_command(
            openclaw_bin,
            self._profile_name,
            agent_id,
            workspace_dir,
            qualified_model,
            provider_name,
            key_env_var,
            state_dir,
            api_format=api_format,
        )
        agent_cmd = _build_agent_command(
            openclaw_bin,
            self._profile_name,
            agent_id,
            session_id,
            message_text,
            task.timeout_sec,
        )

        # Use sh -c to run both commands sequentially
        compound_script = f"{register_cmd} && {agent_cmd}"
        inner_args = ["-c", compound_script]

        wrapped = self.executor.wrap_command(
            self.name,
            execution_policy,
            "sh",
            inner_args,
            inner_env,
        )
        session_dir = os.path.join(state_dir, "agents", agent_id, "sessions")
        session_path = os.path.join(session_dir, f"{session_id}.jsonl")
        helper_result = await _run_openclaw_subprocess_until_json(
            wrapped.command,
            wrapped.args,
            cwd=wrapped.cwd,
            env=wrapped.env,
            timeout_ms=timeout_ms,
            session_dir=session_dir,
            session_path=session_path,
        )
        result = SubprocessResult(
            stdout=helper_result["stdout"],
            stderr=helper_result["stderr"],
            exit_code=helper_result["exit_code"],
            timed_out=helper_result["timed_out"],
        )

        latency_sec = asyncio.get_running_loop().time() - start_time

        if result.timed_out:
            session_data = _read_openclaw_session_with_usage(state_dir, agent_id, session_id)
            trace = session_data["trace"]
            final_text = _extract_openclaw_final_text(trace)
            if not any(event.type in {"task_completed", "task_failed"} for event in trace):
                trace.append(
                    CanonicalTraceEvent(
                        type="task_failed",
                        error="OpenClaw timed out",
                        ts=datetime.now(UTC).isoformat(),
                    )
                )
            usage = session_data["usage"]
            tool_calls = len([e for e in trace if e.type == "tool_call_started"])
            turns = usage["calls"] or len([e for e in trace if e.type == "message"])
            return self._make_result(
                task,
                model,
                "timed_out",
                final_text,
                trace,
                RunMetrics(
                    latency_sec=task.timeout_sec,
                    input_tokens=usage["input"],
                    output_tokens=usage["output"],
                    cache_read_tokens=usage["cache_read"],
                    cache_write_tokens=usage["cache_write"],
                    total_tokens=usage["total_tokens"],
                    tool_calls=tool_calls,
                    turns=turns,
                ),
            )
        subprocess_failure = detect_subprocess_failure(result, command_label="OpenClaw")
        if subprocess_failure:
            return self._make_result(
                task,
                model,
                "failed",
                result.stdout or "",
                [
                    CanonicalTraceEvent(
                        type="task_failed",
                        error=subprocess_failure.error,
                        ts=datetime.now(UTC).isoformat(),
                    )
                ],
                RunMetrics(latency_sec=latency_sec),
                failure_origin=subprocess_failure.failure_origin,
                infra_error_code=subprocess_failure.infra_error_code,
            )

        session_data = _read_openclaw_session_with_usage(state_dir, agent_id, session_id)
        session_trace = session_data["trace"]
        session_final_text = _extract_openclaw_final_text(session_trace)
        if session_final_text:
            if not any(event.type == "task_completed" for event in session_trace):
                session_trace.append(
                    CanonicalTraceEvent(
                        type="task_completed",
                        ts=datetime.now(UTC).isoformat(),
                    )
                )
            usage = session_data["usage"]
            input_tokens = usage["input"]
            output_tokens = usage["output"]
            cache_read_tokens = usage["cache_read"]
            cache_write_tokens = usage["cache_write"]
            total_tokens = usage["total_tokens"] or (
                input_tokens + output_tokens + cache_read_tokens + cache_write_tokens
            )
            tool_calls = len([e for e in session_trace if e.type == "tool_call_started"])
            turns = usage["calls"] or max(1, len([e for e in session_trace if e.type == "message"]))
            return self._make_result(
                task,
                model,
                "completed",
                session_final_text,
                session_trace,
                RunMetrics(
                    latency_sec=latency_sec,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    total_tokens=total_tokens,
                    cost_usd=calculate_cost(
                        model,
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_write_tokens,
                        pricing=self.pricing_override(),
                    ),
                    tool_calls=tool_calls,
                    turns=turns,
                ),
            )

        try:
            try:
                json_str = _extract_json_from_output(result.stdout)
            except ValueError:
                json_str = _extract_json_from_output(result.stderr)

            output = json.loads(json_str)
            payloads = output.get("payloads", [])
            meta = output.get("meta", {})
            agent_meta = meta.get("agentMeta", {})

            final_text = "\n".join(filter(None, (p.get("text") or p.get("body", "") for p in payloads)))

            cumulative_usage = agent_meta.get("usage", {})
            last_usage = agent_meta.get("lastCallUsage", {})
            last_call_tokens = last_usage.get("total") or last_usage.get("input", 0)
            cumulative_total = cumulative_usage.get("input", 0) + cumulative_usage.get("output", 0)
            estimated_turns = max(1, round(cumulative_total / last_call_tokens)) if last_call_tokens > 0 else 1
            estimated_tool_calls = max(0, estimated_turns - 1)

            session_id_from_meta = agent_meta.get("sessionId")
            session_data = (
                _read_openclaw_session_with_usage(state_dir, agent_id, session_id_from_meta)
                if session_id_from_meta
                else {
                    "trace": [],
                    "usage": {
                        "input": 0,
                        "output": 0,
                        "cache_read": 0,
                        "cache_write": 0,
                        "total_tokens": 0,
                        "calls": 0,
                    },
                }
            )

            trace = session_data["trace"]
            if final_text and not trace:
                trace.append(
                    CanonicalTraceEvent(
                        type="message",
                        role="assistant",
                        text=final_text,
                        ts=datetime.now(UTC).isoformat(),
                    )
                )
            trace.append(
                CanonicalTraceEvent(
                    type="task_completed",
                    ts=datetime.now(UTC).isoformat(),
                )
            )

            usage = session_data["usage"]
            input_tokens = usage["input"] or cumulative_usage.get("input") or last_usage.get("input", 0)
            output_tokens = usage["output"] or cumulative_usage.get("output") or last_usage.get("output", 0)
            cache_read_tokens = (
                usage["cache_read"] or cumulative_usage.get("cacheRead") or last_usage.get("cacheRead", 0)
            )
            cache_write_tokens = (
                usage["cache_write"] or cumulative_usage.get("cacheWrite") or last_usage.get("cacheWrite", 0)
            )
            total_tokens = usage["total_tokens"] or (
                input_tokens + output_tokens + cache_read_tokens + cache_write_tokens
            )

            trace_tool_calls = len([e for e in trace if e.type == "tool_call_started"])
            tool_calls = trace_tool_calls if trace_tool_calls > 0 else estimated_tool_calls
            turns = usage["calls"] if usage["calls"] > 0 else estimated_turns

            return self._make_result(
                task,
                model,
                "completed",
                final_text,
                trace,
                RunMetrics(
                    latency_sec=latency_sec,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    total_tokens=total_tokens,
                    cost_usd=calculate_cost(
                        model,
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_write_tokens,
                        pricing=self.pricing_override(),
                    ),
                    tool_calls=tool_calls,
                    turns=turns,
                ),
            )
        except Exception as parse_error:
            exception_msg = f"{type(parse_error).__name__}: {parse_error}"
            stderr_tail = (result.stderr or "")[:STDERR_PREVIEW_MAX_CHARS]
            error = f"OpenClaw output parse error: {exception_msg}"
            if stderr_tail:
                error += f"\nstderr: {stderr_tail}"

            failure = detect_failure_origin_from_error(error)
            return self._make_result(
                task,
                model,
                "failed",
                result.stdout or "",
                [
                    CanonicalTraceEvent(
                        type="task_failed",
                        error=error[:800],
                        ts=datetime.now(UTC).isoformat(),
                    )
                ],
                RunMetrics(latency_sec=latency_sec),
                failure_origin=failure.get("failure_origin"),
                infra_error_code=failure.get("infra_error_code"),
            )

    def cleanup(self, prepared: PreparedRun) -> None:
        agent_id = prepared.env.get("_EVAL_AGENT_ID")
        state_dir = prepared.env.get("_OPENCLAW_STATE_DIR")
        if agent_id:
            try:
                if state_dir:
                    _prune_openclaw_agent_from_profile(state_dir, agent_id)
            except Exception:
                pass
            import shutil

            if state_dir:
                shutil.rmtree(_resolve_openclaw_agent_runtime_dir(state_dir, agent_id), ignore_errors=True)
        self.executor.restore_workspace(prepared.workspace_dir)
        remove_workspace(prepared.workspace_dir)


def _make_ephemeral_ids() -> tuple[str, str]:
    return (
        f"run-{uuid.uuid4().hex[:12]}",
        f"eval-{uuid.uuid4().hex[:16]}",
    )


def _resolve_openclaw_agent_runtime_dir(state_dir: str, agent_id: str) -> str:
    return os.path.join(state_dir, "agents", agent_id)


def _install_openclaw_native_memory_files(workspace_dir: str, files: list[NativeMemoryFile]) -> None:
    for file in files:
        rel_path = _normalize_relative_memory_path(file.path)
        target_path = os.path.join(workspace_dir, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w") as f:
            f.write(file.content)


def _normalize_relative_memory_path(input_path: str) -> str:
    trimmed = input_path.lstrip("/")
    if trimmed == "MEMORY.md" or trimmed.startswith("memory/"):
        return trimmed
    return f"memory/{trimmed}"


def _qualify_openclaw_model(model: str, provider_name: str) -> str:
    if "/" in model:
        return model
    return f"{provider_name}/{model}"


def _resolve_api_key_env_var(provider_name: str, provider: ProviderConfig) -> str:
    upper = provider_name.upper()
    if upper in {"ANTHROPIC", "OPENAI", "OPENROUTER"}:
        return f"{upper}_API_KEY"
    return f"EVAL_PROVIDER_{upper}_API_KEY"


def _openclaw_base_env(state_dir: str) -> dict[str, str]:
    return {
        "OPENCLAW_STATE_DIR": state_dir,
        "OPENCLAW_CONFIG_PATH": os.path.join(state_dir, "openclaw.json"),
        "HOME": state_dir,
        "USERPROFILE": state_dir,
        "XDG_CONFIG_HOME": state_dir,
        "XDG_DATA_HOME": state_dir,
        "XDG_CACHE_HOME": os.path.join(state_dir, ".cache"),
    }


def _read_auth_key_names(state_dir: str, key_env_var: str) -> list[str]:
    names: set[str] = set()
    eval_agents_dir = os.path.join(state_dir, "agents")
    try:
        for agent_dir in os.listdir(eval_agents_dir):
            auth_path = os.path.join(eval_agents_dir, agent_dir, "agent", "auth-profiles.json")
            try:
                with open(auth_path) as f:
                    data = json.load(f)
                for profile in (data.get("profiles") or {}).values():
                    key_ref = profile.get("keyRef") if isinstance(profile, dict) else None
                    if isinstance(key_ref, dict) and key_ref.get("source") == "env" and key_ref.get("id"):
                        names.add(key_ref["id"])
            except Exception:
                pass
    except Exception:
        pass

    names.add(key_env_var)
    return list(names)


def _bootstrap_eval_profile(
    state_dir: str,
) -> None:
    profile_dir = state_dir
    cfg_path = os.path.join(profile_dir, "openclaw.json")

    os.makedirs(profile_dir, exist_ok=True)
    if os.path.exists(cfg_path):
        return

    eval_cfg = {
        "models": {"mode": "merge", "providers": {}},
        "agents": {
            "defaults": {
                "compaction": {"mode": "safeguard"},
                "memorySearch": {"enabled": False},
                "skipBootstrap": True,
            },
            "list": [],
        },
        "plugins": {
            "allow": [],
            "load": {"paths": []},
            "slots": {"memory": "none"},
            "entries": {"memory-core": {"enabled": False}},
        },
    }
    with open(cfg_path, "w") as f:
        json.dump(eval_cfg, f, indent=2)


def _shell_quote(s: str) -> str:
    """Quote a string for safe use in sh -c."""
    return "'" + s.replace("'", "'\\''") + "'"


def _build_register_command(
    openclaw_bin: str,
    profile_name: str,
    agent_id: str,
    workspace_dir: str,
    model: str,
    provider_name: str,
    key_env_var: str,
    state_dir: str,
    api_format: str = "anthropic",
) -> str:
    """Build a shell command string that registers an OpenClaw agent.

    This creates the agent directory + auth profile using the CLI,
    then re-applies the model config (because `agents add` rewrites
    openclaw.json, potentially dropping the model entry).
    """
    # Node prefix for .mjs files
    node_prefix = "node " if openclaw_bin.endswith((".mjs", ".js")) else ""
    bin_q = _shell_quote(openclaw_bin)

    # Delete any stale agent first, then create
    delete_cmd = f"{node_prefix}{bin_q} --profile {_shell_quote(profile_name)} --no-color agents delete {_shell_quote(agent_id)} 2>/dev/null || true"
    add_cmd = (
        f"{node_prefix}{bin_q} --profile {_shell_quote(profile_name)} --no-color "
        f"agents add {_shell_quote(agent_id)} --non-interactive "
        f"--workspace {_shell_quote(workspace_dir)} --model {_shell_quote(model)}"
    )

    # Write auth-profiles.json after agent creation
    auth_json = json.dumps(
        {
            "version": 1,
            "profiles": {
                f"{provider_name}:default": {
                    "type": "api_key",
                    "provider": provider_name,
                    "keyRef": {"source": "env", "provider": "default", "id": key_env_var},
                }
            },
            "lastGood": {provider_name: f"{provider_name}:default"},
        }
    )
    auth_dir = os.path.join(state_dir, "agents", agent_id, "agent")
    auth_path = os.path.join(auth_dir, "auth-profiles.json")
    write_auth = f"mkdir -p {_shell_quote(auth_dir)} && echo {_shell_quote(auth_json)} > {_shell_quote(auth_path)}"

    # Re-apply model config after `agents add` rewrites openclaw.json.
    # Use node (always available) to patch the JSON in-place.
    cfg_path = os.path.join(state_dir, "openclaw.json")
    openclaw_api = "anthropic-messages" if api_format == "anthropic" else "openai-completions"
    patch_script = (
        f"const fs=require('fs');"
        f"const p={_shell_quote(cfg_path)};"
        f"const c=JSON.parse(fs.readFileSync(p,'utf8'));"
        f"const prov=c.models&&c.models.providers&&c.models.providers[{_shell_quote(provider_name)}];"
        f"if(prov){{"
        f"prov.models=prov.models||[];"
        f"if(!prov.models.some(m=>m.id==={_shell_quote(model)}))"
        f"prov.models.push({{id:{_shell_quote(model)},name:{_shell_quote(model + ' (eval)')},api:{_shell_quote(openclaw_api)},input:['text'],contextWindow:200000,maxTokens:8192}});"
        f"}}"
        f"c.agents=c.agents||{{}};c.agents.defaults=c.agents.defaults||{{}};"
        f"c.agents.defaults.model={{primary:{_shell_quote(model)}}};"
        f"c.agents.defaults.memorySearch={{enabled:false}};"
        f"c.agents.defaults.skipBootstrap=true;"
        f"c.plugins=c.plugins||{{}};c.plugins.load=c.plugins.load||{{}};c.plugins.load.paths=[];"
        f"c.plugins.slots=c.plugins.slots||{{}};c.plugins.slots.memory='none';"
        f"c.plugins.entries=c.plugins.entries||{{}};c.plugins.entries['memory-core']={{enabled:false}};"
        f"fs.writeFileSync(p,JSON.stringify(c,null,2));"
    )
    ensure_model = f"node -e {_shell_quote(patch_script)}"

    # Chain: delete old → add new → wait for dir → write auth → ensure model
    wait_loop = f"for i in $(seq 1 30); do [ -d {_shell_quote(os.path.join(state_dir, 'agents', agent_id))} ] && break; sleep 0.2; done"

    return f"{{ {delete_cmd} ; {add_cmd} ; {wait_loop} ; {write_auth} ; {ensure_model} ; }}"


def _build_agent_command(
    openclaw_bin: str,
    profile_name: str,
    agent_id: str,
    session_id: str,
    message: str,
    timeout_sec: int,
) -> str:
    """Build a shell command string that runs the OpenClaw agent."""
    node_prefix = "node " if openclaw_bin.endswith((".mjs", ".js")) else ""
    bin_q = _shell_quote(openclaw_bin)
    return (
        f"{node_prefix}{bin_q} --profile {_shell_quote(profile_name)} --no-color "
        f"agent --local --json "
        f"--agent {_shell_quote(agent_id)} "
        f"--session-id {_shell_quote(session_id)} "
        f"--message {_shell_quote(message)} "
        f"--timeout {timeout_sec}"
    )


def _ensure_openclaw_model_configured(
    state_dir: str,
    model: str,
    provider_name: str,
    provider: ProviderConfig,
) -> None:
    cfg_path = os.path.join(state_dir, "openclaw.json")
    _bootstrap_eval_profile(state_dir)

    with open(cfg_path) as f:
        cfg = json.load(f)
    _apply_openclaw_model_to_profile(cfg, model, provider_name, provider)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def _apply_openclaw_model_to_profile(
    profile: dict,
    model: str,
    provider_name: str,
    provider_config: ProviderConfig,
) -> None:
    provider = profile.get("models", {}).get("providers", {}).get(provider_name)
    if not provider:
        provider = {
            "baseUrl": provider_config.base_url or None,
            "api": ("anthropic-messages" if provider_config.api_format == "anthropic" else "openai-completions"),
            "models": [],
        }
        profile.setdefault("models", {}).setdefault("providers", {})[provider_name] = provider

    models = provider.get("models", [])
    if not isinstance(models, list):
        models = []
    if not any(entry.get("id") == model for entry in models):
        models.append(
            {
                "id": model,
                "name": f"{model} (eval)",
                "api": provider.get("api"),
                "input": ["text"],
                "contextWindow": 200000,
                "maxTokens": 8192,
            }
        )
    provider["models"] = models
    profile.setdefault("agents", {}).setdefault("defaults", {})
    profile["agents"]["defaults"]["model"] = {"primary": _qualify_openclaw_model(model, provider_name)}
    profile["agents"]["defaults"]["memorySearch"] = {"enabled": False}
    profile["agents"]["defaults"]["skipBootstrap"] = True
    profile.setdefault("plugins", {})
    profile["plugins"].setdefault("load", {})
    profile["plugins"]["load"]["paths"] = []
    profile["plugins"].setdefault("slots", {})
    profile["plugins"]["slots"]["memory"] = "none"
    profile["plugins"].setdefault("entries", {})
    profile["plugins"]["entries"]["memory-core"] = {
        **profile["plugins"]["entries"].get("memory-core", {}),
        "enabled": False,
    }


def _prune_openclaw_agent_from_profile(state_dir: str, agent_id: str) -> None:
    cfg_path = os.path.join(state_dir, "openclaw.json")
    try:
        with open(cfg_path) as f:
            profile = json.load(f)
        agents_list = profile.get("agents", {}).get("list", [])
        if isinstance(agents_list, list):
            profile["agents"]["list"] = [e for e in agents_list if isinstance(e, dict) and e.get("id") != agent_id]
        with open(cfg_path, "w") as f:
            json.dump(profile, f, indent=2)
    except Exception:
        pass


def _extract_json_from_output(stdout: str) -> str:
    trimmed = stdout.strip()
    if trimmed.startswith("{"):
        return trimmed
    lines = trimmed.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("{"):
            return "\n".join(lines[i:])
    match = re.search(r"\{[\s\S]*\}$", trimmed)
    if match:
        return match.group(0)
    raise ValueError("No JSON found in output")


async def _run_openclaw_subprocess_until_json(
    command: str,
    args: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_ms: int,
    session_dir: str | None = None,
    session_path: str | None = None,
) -> dict[str, Any]:
    """Run OpenClaw subprocess and terminate once JSON or session output is ready."""
    proc = await asyncio.create_subprocess_exec(
        command,
        *args,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout = ""
    stderr = ""
    timed_out = False
    json_seen = False
    session_completed = False
    completed_early = False

    async def read_stream(stream: asyncio.StreamReader, is_stderr: bool = False) -> None:
        nonlocal stdout, stderr
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode(errors="replace")
            if is_stderr:
                stderr += text
            else:
                stdout += text

    stdout_task = asyncio.create_task(read_stream(proc.stdout, False))
    stderr_task = asyncio.create_task(read_stream(proc.stderr, True))
    wait_task = asyncio.create_task(proc.wait())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (timeout_ms / 1000)

    while True:
        if wait_task.done():
            break
        try:
            _extract_json_from_output(stdout)
            json_seen = True
        except ValueError:
            try:
                _extract_json_from_output(stderr)
                json_seen = True
            except ValueError:
                pass

        if not json_seen and session_dir:
            session_completed = bool(_read_openclaw_session_terminal_text(session_dir, session_path))

        if json_seen or session_completed:
            completed_early = True
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            break

        if loop.time() >= deadline:
            timed_out = True
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            break

        await asyncio.sleep(0.2)

    try:
        await asyncio.wait_for(wait_task, timeout=5)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await asyncio.gather(wait_task, return_exceptions=True)

    await _finalize_openclaw_stream_tasks(stdout_task, stderr_task)

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": 0 if completed_early and not timed_out else proc.returncode,
        "timed_out": timed_out,
        "session_completed": session_completed,
    }


async def _finalize_openclaw_stream_tasks(*tasks: asyncio.Task[None]) -> None:
    pending = [task for task in tasks if not task.done()]
    if pending:
        _, still_pending = await asyncio.wait(pending, timeout=1)
        for task in still_pending:
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _looks_like_tool_failure(text: str) -> bool:
    if not text:
        return False
    if re.search(r"\(Command exited with code [1-9]\d*\)", text):
        return True
    if re.match(r'^\s*\{\s*"status"\s*:\s*"error"', text):
        return True
    return bool(re.search(r"EISDIR|ENOENT|EACCES|permission denied", text, re.IGNORECASE))


def _read_openclaw_session_with_usage(state_dir: str, agent_id: str, session_id: str) -> dict[str, Any]:
    trace: list[CanonicalTraceEvent] = []
    usage = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 0,
        "calls": 0,
    }
    try:
        session_dir = os.path.join(state_dir, "agents", agent_id, "sessions")
        session_path = os.path.join(session_dir, f"{session_id}.jsonl")
        content = _read_openclaw_session_content(session_dir, session_path)
        if not content:
            return {"trace": trace, "usage": usage}

        pending_tools: list[dict[str, str]] = []

        for line in content.split("\n"):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") != "message" or not event.get("message"):
                continue

            msg = event["message"]
            role = msg.get("role", "")
            ts = (
                datetime.fromtimestamp(event["timestamp"] / 1000, tz=UTC).isoformat()
                if isinstance(event.get("timestamp"), (int, float))
                else datetime.now(UTC).isoformat()
            )
            content_blocks = msg.get("content")

            if role == "assistant" and msg.get("usage"):
                u = msg["usage"]
                usage["input"] += int(u.get("input", 0))
                usage["output"] += int(u.get("output", 0))
                usage["cache_read"] += int(u.get("cacheRead", 0))
                usage["cache_write"] += int(u.get("cacheWrite", 0))
                usage["total_tokens"] += int(u.get("totalTokens", 0))
                usage["calls"] += 1

            if not isinstance(content_blocks, list):
                continue

            if role in ("toolResult", "tool"):
                for block in content_blocks:
                    if block.get("type") not in ("text", "toolResult", "tool_result"):
                        continue
                    text = block.get("text") or block.get("content", "")
                    if not isinstance(text, str):
                        text = ""
                    pending = pending_tools.pop(0) if pending_tools else None
                    tool_name = pending["tool_name"] if pending else "unknown"
                    explicit_error = block.get("isError")
                    success = not explicit_error if explicit_error is True else not _looks_like_tool_failure(text)
                    trace.append(
                        CanonicalTraceEvent(
                            type="tool_call_completed",
                            tool_name=tool_name,
                            success=success,
                            output=text[:TOOL_OUTPUT_MAX_CHARS] if text else None,
                            ts=ts,
                        )
                    )
                continue

            for block in content_blocks:
                if block.get("type") == "text" and block.get("text"):
                    trace.append(
                        CanonicalTraceEvent(
                            type="message",
                            role="assistant" if role == "assistant" else "user",
                            text=block["text"],
                            ts=ts,
                        )
                    )
                elif block.get("type") in ("toolCall", "tool_use"):
                    tool_name = block.get("name") or block.get("toolName", "unknown")
                    trace.append(
                        CanonicalTraceEvent(
                            type="tool_call_started",
                            tool_name=tool_name,
                            input=block.get("input") or block.get("arguments"),
                            ts=ts,
                        )
                    )
                    pending_tools.append({"tool_name": tool_name, "ts": ts})
                elif block.get("type") in ("toolResult", "tool_result"):
                    pending = pending_tools.pop(0) if pending_tools else None
                    tool_name = pending["tool_name"] if pending else block.get("toolName", "unknown")
                    text = block.get("text") or block.get("content", "")
                    if not isinstance(text, str):
                        text = ""
                    success = not block.get("isError", False) and not _looks_like_tool_failure(text)
                    trace.append(
                        CanonicalTraceEvent(
                            type="tool_call_completed",
                            tool_name=tool_name,
                            success=success,
                            output=text[:TOOL_OUTPUT_MAX_CHARS] if text else None,
                            ts=ts,
                        )
                    )

        # Remaining pending tools
        for pending in pending_tools:
            trace.append(
                CanonicalTraceEvent(
                    type="tool_call_completed",
                    tool_name=pending["tool_name"],
                    success=False,
                    output="<no result recorded>",
                    ts=pending["ts"],
                )
            )
    except Exception:
        pass

    return {"trace": trace, "usage": usage}


def _extract_openclaw_final_text(trace: list[CanonicalTraceEvent]) -> str:
    for event in reversed(trace):
        if event.type == "message" and event.role == "assistant" and event.text:
            return event.text
    return ""


def _read_openclaw_session_final_text(session_dir: str, session_path: str | None = None) -> str:
    try:
        content = _read_openclaw_session_content(session_dir, session_path or "")
        for line in reversed(content.splitlines()):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") != "message":
                continue
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            content_blocks = message.get("content")
            if not isinstance(content_blocks, list):
                continue
            for block in content_blocks:
                if block.get("type") == "text" and block.get("text"):
                    return str(block["text"])
    except Exception:
        return ""
    return ""


def _read_openclaw_session_terminal_text(session_dir: str, session_path: str | None = None) -> str:
    try:
        content = _read_openclaw_session_content(session_dir, session_path or "")
        for line in reversed(content.splitlines()):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") != "message":
                continue
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue

            content_blocks = message.get("content")
            if not isinstance(content_blocks, list):
                continue
            if any(block.get("type") == "toolCall" for block in content_blocks if isinstance(block, dict)):
                continue
            if message.get("stopReason") == "toolUse":
                continue

            text_parts = [
                str(block.get("text"))
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
            ]
            if text_parts:
                return "".join(text_parts)
    except Exception:
        return ""
    return ""


def _read_openclaw_session_content(session_dir: str, session_path: str) -> str:
    if os.path.exists(session_path):
        with open(session_path) as f:
            return f.read()

    candidates: list[tuple[float, str]] = []
    try:
        for name in os.listdir(session_dir):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(session_dir, name)
            if not os.path.isfile(path):
                continue
            candidates.append((os.path.getmtime(path), path))
    except FileNotFoundError:
        return ""

    if not candidates:
        return ""

    candidates.sort(reverse=True)
    with open(candidates[0][1]) as f:
        return f.read()
