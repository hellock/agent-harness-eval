from __future__ import annotations

from agent_harness_eval.config.runtime import RuntimeConfig
from agent_harness_eval.executor import ExecutionPolicy, Executor, WrappedCommand
from agent_harness_eval.utils.subprocess import SubprocessResult


class RecordingExecutor(Executor):
    name = "recording"

    def __init__(self, runtime_config: RuntimeConfig, result: SubprocessResult):
        super().__init__(runtime_config)
        self.result = result
        self.calls: list[dict[str, object]] = []
        self.wrapped_calls: list[dict[str, object]] = []

    async def execute(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
        timeout_ms: int,
    ) -> SubprocessResult:
        self.calls.append(
            {
                "harness": harness,
                "policy": policy,
                "inner_command": inner_command,
                "inner_args": list(inner_args),
                "inner_env": dict(inner_env),
                "timeout_ms": timeout_ms,
            }
        )
        return self.result

    def wrap_command(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
    ) -> WrappedCommand:
        self.wrapped_calls.append(
            {
                "harness": harness,
                "policy": policy,
                "inner_command": inner_command,
                "inner_args": list(inner_args),
                "inner_env": dict(inner_env),
            }
        )
        return WrappedCommand(
            command=inner_command,
            args=list(inner_args),
            env=dict(inner_env),
            cwd=policy.cwd or policy.workspace_dir,
        )


class SequentialRecordingExecutor(Executor):
    name = "recording-sequential"

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        results: list[SubprocessResult],
        side_effects: list[object] | None = None,
    ):
        super().__init__(runtime_config)
        self.results = list(results)
        self.side_effects = list(side_effects or [])
        self.calls: list[dict[str, object]] = []
        self.wrapped_calls: list[dict[str, object]] = []

    async def execute(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
        timeout_ms: int,
    ) -> SubprocessResult:
        self.calls.append(
            {
                "harness": harness,
                "policy": policy,
                "inner_command": inner_command,
                "inner_args": list(inner_args),
                "inner_env": dict(inner_env),
                "timeout_ms": timeout_ms,
            }
        )
        if not self.results:
            raise AssertionError("No more canned subprocess results")
        if self.side_effects:
            side_effect = self.side_effects.pop(0)
            if side_effect is not None:
                side_effect()
        return self.results.pop(0)

    def wrap_command(
        self,
        harness: str,
        policy: ExecutionPolicy,
        inner_command: str,
        inner_args: list[str],
        inner_env: dict[str, str],
    ) -> WrappedCommand:
        self.wrapped_calls.append(
            {
                "harness": harness,
                "policy": policy,
                "inner_command": inner_command,
                "inner_args": list(inner_args),
                "inner_env": dict(inner_env),
            }
        )
        return WrappedCommand(
            command=inner_command,
            args=list(inner_args),
            env=dict(inner_env),
            cwd=policy.cwd or policy.workspace_dir,
        )


def arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]
