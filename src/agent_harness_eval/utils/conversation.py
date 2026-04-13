"""Message formatting utilities."""

from __future__ import annotations

from ..task import Task


def format_task_message(task: Task) -> str:
    """Build the canonical message text for an agent invocation."""
    history = task.conversation_history
    if not history:
        return task.user_query

    formatted_history = "\n\n".join(f"[{message['role']}]: {message['content']}" for message in history)
    return f"Previous conversation:\n{formatted_history}\n\nCurrent request: {task.user_query}"
