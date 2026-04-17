"""Task-domain models and task input materialization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .graders.specs import GraderSpec, parse_grader_spec

TaskCategory = Literal[
    "reasoning",
    "implementation",
    "coding",
    "security",
    "skills",
    "memory",
    "robustness",
]

_TASK_SETUP_FIELDS = {
    "workspace_dir",
    "workspace_files",
    "history_file",
    "conversation_history",
    "native_memory",
    "memory_state",
    "prepare_commands",
    "tool_boundary",
}


WorkspaceFile = dict[str, str]
ConversationTurn = dict[str, str]


@dataclass(slots=True)
class NativeMemoryConfig:
    memory_dir: str | None = None
    files: list[WorkspaceFile] | None = None


@dataclass(slots=True)
class ToolBoundary:
    internet: Literal["enabled", "disabled"] = "enabled"
    shell: Literal["enabled", "disabled"] = "enabled"
    file_write: Literal["enabled", "disabled"] = "enabled"

    def is_disabled(self, constraint: Literal["internet", "shell", "file_write"]) -> bool:
        return getattr(self, constraint) == "disabled"


class Task:
    __slots__ = (
        "category",
        "conversation_history",
        "description",
        "graders",
        "history_file",
        "id",
        "memory_state",
        "native_memory",
        "prepare_commands",
        "task_dir",
        "timeout_sec",
        "tool_boundary",
        "user_query",
        "workspace_dir",
        "workspace_files",
    )

    def __init__(
        self,
        *,
        id: str,
        category: TaskCategory,
        description: str,
        user_query: str,
        graders: list[GraderSpec] | None = None,
        timeout_sec: int = 1800,
        task_dir: str | None = None,
        workspace_dir: str | None = None,
        workspace_files: list[WorkspaceFile] | None = None,
        history_file: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        native_memory: NativeMemoryConfig | None = None,
        memory_state: dict[str, Any] | None = None,
        prepare_commands: list[str] | None = None,
        tool_boundary: ToolBoundary | None = None,
    ) -> None:
        self.id = id
        self.category = category
        self.description = description
        self.user_query = user_query
        self.graders = list(graders or [])
        self.timeout_sec = timeout_sec
        self.task_dir = task_dir
        self.workspace_dir = workspace_dir
        self.workspace_files = list(workspace_files) if workspace_files is not None else None
        self.history_file = history_file
        self.conversation_history = list(conversation_history) if conversation_history is not None else None
        self.native_memory = native_memory
        self.memory_state = memory_state
        self.prepare_commands = list(prepare_commands or [])
        self.tool_boundary = tool_boundary or ToolBoundary()

    @classmethod
    def from_dir(cls, task_dir: str | Path) -> Task:
        task_dir_path = Path(task_dir).resolve()
        task_yaml_path = task_dir_path / "task.yaml"
        with task_yaml_path.open(encoding="utf-8") as handle:
            task_data = yaml.safe_load(handle) or {}
        if not isinstance(task_data, dict):
            raise ValueError(f"Task YAML must parse to a mapping: {task_yaml_path}")
        return _task_from_dict(task_data, task_dir=task_dir_path)

    def resolve_path(self, relative_path: str) -> Path:
        if not self.task_dir:
            raise ValueError(f"Task {self.id} has no task_dir; cannot resolve {relative_path!r}")
        return Path(self.task_dir, relative_path)

    def materialize(self) -> Task:
        workspace_files = self.workspace_files
        workspace_dir = self.workspace_dir
        if workspace_dir is not None:
            workspace_files = _load_workspace_files_from_dir(self.resolve_path(workspace_dir))
            workspace_dir = None

        conversation_history = self.conversation_history
        history_file = self.history_file
        if history_file is not None:
            conversation_history = _load_conversation_history(self.resolve_path(history_file))
            history_file = None

        native_memory = self.native_memory
        if native_memory is not None and native_memory.memory_dir is not None:
            native_memory = NativeMemoryConfig(
                memory_dir=None,
                files=_load_workspace_files_from_dir(self.resolve_path(native_memory.memory_dir)),
            )

        return Task(
            id=self.id,
            category=self.category,
            description=self.description,
            user_query=self.user_query,
            graders=self.graders,
            timeout_sec=self.timeout_sec,
            task_dir=self.task_dir,
            workspace_dir=workspace_dir,
            workspace_files=workspace_files,
            history_file=history_file,
            conversation_history=conversation_history,
            native_memory=native_memory,
            memory_state=self.memory_state,
            prepare_commands=self.prepare_commands,
            tool_boundary=self.tool_boundary,
        )


def _parse_workspace_file(data: dict[str, Any]) -> WorkspaceFile:
    return {"path": str(data["path"]), "content": str(data["content"])}


def _task_from_dict(data: dict[str, Any], *, task_dir: str | Path) -> Task:
    setup = dict(data.get("setup") or {})
    unknown_setup = sorted(set(setup) - _TASK_SETUP_FIELDS)
    if "install_commands" in setup:
        raise ValueError("Task setup field 'install_commands' was renamed to 'prepare_commands'")
    if unknown_setup:
        raise ValueError(f"Unknown task setup field(s): {', '.join(unknown_setup)}")

    workspace_files = None
    if setup.get("workspace_files"):
        workspace_files = [_parse_workspace_file(item) for item in setup["workspace_files"]]

    conversation_history = None
    if setup.get("conversation_history"):
        conversation_history = _parse_conversation_history(setup["conversation_history"])

    native_memory = None
    if setup.get("native_memory") is not None:
        native_memory = _parse_native_memory_config(setup["native_memory"])

    graders_data = data.get("graders") or []
    graders = [parse_grader_spec(item) for item in graders_data]

    return Task(
        id=str(data["id"]),
        category=data["category"],
        description=str(data["description"]),
        user_query=str(data["user_query"]),
        graders=graders,
        timeout_sec=int(data.get("timeout_sec", 1800)),
        task_dir=str(Path(task_dir).resolve()),
        workspace_dir=setup.get("workspace_dir"),
        workspace_files=workspace_files,
        history_file=setup.get("history_file"),
        conversation_history=conversation_history,
        native_memory=native_memory,
        memory_state=data.get("memory_state"),
        prepare_commands=[str(item) for item in setup.get("prepare_commands") or []],
        tool_boundary=_parse_tool_boundary(setup.get("tool_boundary")),
    )


def _parse_conversation_turn(data: dict[str, Any]) -> ConversationTurn:
    return {"role": data["role"], "content": str(data["content"])}


def _parse_conversation_history(data: list[dict[str, Any]]) -> list[ConversationTurn]:
    return [_parse_conversation_turn(item) for item in data]


def _parse_native_memory_config(data: dict[str, Any]) -> NativeMemoryConfig:
    files = data.get("files")
    return NativeMemoryConfig(
        memory_dir=data.get("memory_dir"),
        files=[_parse_workspace_file(item) for item in files] if files else None,
    )


def _parse_tool_boundary(data: dict[str, Any] | None) -> ToolBoundary:
    if data is None:
        return ToolBoundary()
    return ToolBoundary(
        internet=data.get("internet", "enabled"),
        shell=data.get("shell", "enabled"),
        file_write=data.get("file_write", "enabled"),
    )


def _load_workspace_files_from_dir(root_dir: Path) -> list[WorkspaceFile]:
    if not root_dir.is_dir():
        raise ValueError(f"Expected directory: {root_dir}")

    files: list[WorkspaceFile] = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        files.append(
            {
                "path": path.relative_to(root_dir).as_posix(),
                "content": path.read_text(encoding="utf-8"),
            }
        )
    return files


def _load_conversation_history(path: Path) -> list[ConversationTurn]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        turns: list[ConversationTurn] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            turns.append(_parse_conversation_turn(json.loads(stripped)))
        return turns

    if suffix in {".yaml", ".yml", ".json"}:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(parsed, list):
            raise ValueError(f"Conversation history must be a list: {path}")
        return _parse_conversation_history(parsed)

    raise ValueError(f"Unsupported conversation history format: {path}")


# ─── Task loading ───


def load_tasks(
    tasks_dir: str,
    task_filter: dict[str, Any] | None = None,
) -> list[Task]:
    """Scan a tasks directory and return matching Task objects."""
    tasks: list[Task] = []

    if not os.path.isdir(tasks_dir):
        return tasks

    for category in sorted(os.listdir(tasks_dir)):
        category_dir = os.path.join(tasks_dir, category)
        if not os.path.isdir(category_dir):
            continue

        for entry in sorted(os.listdir(category_dir)):
            task_dir = os.path.join(category_dir, entry)
            if not os.path.isdir(task_dir):
                continue
            task_yaml = os.path.join(task_dir, "task.yaml")
            if not os.path.isfile(task_yaml):
                continue

            tasks.append(Task.from_dir(task_dir))

    if task_filter:
        categories = task_filter.get("categories")
        if categories:
            tasks = [t for t in tasks if t.category in categories]
        ids = task_filter.get("ids")
        if ids:
            tasks = [t for t in tasks if t.id in ids]

    tasks.sort(key=lambda t: t.id)
    return tasks
