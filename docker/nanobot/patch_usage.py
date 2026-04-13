"""Patch nanobot 0.1.5 to save usage metadata to session files.

Nanobot 0.1.5 ships without usage metadata support in sessions — the
`_build_usage_metadata` helper is missing and `last_usage` is never
written to the session JSONL. This patch injects the helper and wires
it into both the system-message and user-message code paths.

Also creates a missing `tool_hints.py` stub that nanobot imports but
doesn't ship.

Usage:
    python patch_usage.py <site-packages-dir>
"""

import sys
from pathlib import Path

HELPER_SNIPPET = """\
    def _build_usage_metadata(self) -> dict[str, int]:
        usage: dict[str, int] = {}
        for key, value in self._last_usage.items():
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                usage[key] = number
        return usage
"""

# ── System-message path patch ──

OLD_SYSTEM = (
    "            self._save_turn(session, all_msgs, 1 + len(history))\n"
    "            self._clear_runtime_checkpoint(session)\n"
    "            self.sessions.save(session)\n"
    "            self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))\n"
    "            return OutboundMessage(channel=channel, chat_id=chat_id,\n"
    '                                  content=final_content or "Background task completed.")'
)

NEW_SYSTEM = (
    "            self._save_turn(session, all_msgs, 1 + len(history))\n"
    "            self._clear_runtime_checkpoint(session)\n"
    "            usage_metadata = self._build_usage_metadata()\n"
    "            if usage_metadata:\n"
    '                session.metadata["last_usage"] = usage_metadata\n'
    "            else:\n"
    '                session.metadata.pop("last_usage", None)\n'
    "            self.sessions.save(session)\n"
    "            self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))\n"
    "            return OutboundMessage(\n"
    "                channel=channel,\n"
    "                chat_id=chat_id,\n"
    '                content=final_content or "Background task completed.",\n'
    '                metadata={"usage": usage_metadata} if usage_metadata else {},\n'
    "            )"
)

# ── User-message path patch ──

OLD_USER = (
    "        self._save_turn(session, all_msgs, 1 + len(history))\n"
    "        self._clear_runtime_checkpoint(session)\n"
    "        self.sessions.save(session)\n"
    "        self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))\n"
    "\n"
    '        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:\n'
    "            return None\n"
    "\n"
    '        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content\n'
    '        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)\n'
    "\n"
    "        meta = dict(msg.metadata or {})\n"
    "        if on_stream is not None:\n"
    '            meta["_streamed"] = True\n'
    "        return OutboundMessage(\n"
    "            channel=msg.channel, chat_id=msg.chat_id, content=final_content,\n"
    "            metadata=meta,\n"
    "        )"
)

NEW_USER = (
    "        self._save_turn(session, all_msgs, 1 + len(history))\n"
    "        self._clear_runtime_checkpoint(session)\n"
    "        usage_metadata = self._build_usage_metadata()\n"
    "        if usage_metadata:\n"
    '            session.metadata["last_usage"] = usage_metadata\n'
    "        else:\n"
    '            session.metadata.pop("last_usage", None)\n'
    "        self.sessions.save(session)\n"
    "        self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))\n"
    "\n"
    '        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:\n'
    "            return None\n"
    "\n"
    '        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content\n'
    '        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)\n'
    "\n"
    "        meta = dict(msg.metadata or {})\n"
    "        if on_stream is not None:\n"
    '            meta["_streamed"] = True\n'
    "        if usage_metadata:\n"
    '            meta["usage"] = usage_metadata\n'
    "        return OutboundMessage(\n"
    "            channel=msg.channel, chat_id=msg.chat_id, content=final_content,\n"
    "            metadata=meta,\n"
    "        )"
)

TOOL_HINTS_STUB = '''\
"""Stub for missing nanobot 0.1.5 tool_hints module."""

from typing import Any, Iterable


def format_tool_hints(tool_calls: Iterable[Any]) -> str:
    if not tool_calls:
        return ""
    names: list[str] = []
    for tc in tool_calls:
        name = getattr(tc, "name", None)
        if name is None and hasattr(tc, "function"):
            name = getattr(getattr(tc, "function", None), "name", None)
        if name is None and isinstance(tc, dict):
            name = tc.get("name") or (tc.get("function") or {}).get("name")
        if name:
            names.append(str(name))
    if not names:
        return ""
    return f"calling {names[0]}" if len(names) == 1 else f"calling {len(names)} tools"
'''


def find_loop_py(site_packages: Path) -> Path:
    loop = site_packages / "nanobot" / "agent" / "loop.py"
    if loop.exists():
        return loop
    raise FileNotFoundError(f"nanobot/agent/loop.py not found under {site_packages}")


def patch_loop(loop_path: Path) -> bool:
    text = loop_path.read_text()
    changed = False

    # Inject _build_usage_metadata helper
    if "def _build_usage_metadata(self)" not in text:
        marker = (
            "    def _clear_runtime_checkpoint(self, session: Session) -> None:\n"
            "        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:\n"
            "            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)\n"
        )
        if marker not in text:
            print(f"WARNING: could not find insertion point for helper in {loop_path}")
            return False
        text = text.replace(marker, marker + "\n" + HELPER_SNIPPET)
        changed = True

    # Patch system-message path
    if OLD_SYSTEM in text:
        text = text.replace(OLD_SYSTEM, NEW_SYSTEM)
        changed = True

    # Patch user-message path
    if OLD_USER in text:
        text = text.replace(OLD_USER, NEW_USER)
        changed = True

    if changed:
        loop_path.write_text(text)
    return changed


def ensure_tool_hints(site_packages: Path) -> bool:
    stub_path = site_packages / "nanobot" / "utils" / "tool_hints.py"
    if stub_path.exists():
        return False
    stub_path.parent.mkdir(parents=True, exist_ok=True)
    stub_path.write_text(TOOL_HINTS_STUB)
    return True


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python patch_usage.py <site-packages-dir>")
        sys.exit(1)

    site_packages = Path(sys.argv[1])
    loop_path = find_loop_py(site_packages)

    if patch_loop(loop_path):
        print(f"[nanobot] patched usage metadata in {loop_path}")
    else:
        print(f"[nanobot] usage patch already present in {loop_path}")

    if ensure_tool_hints(site_packages):
        print("[nanobot] wrote missing tool_hints stub")
    else:
        print("[nanobot] tool_hints stub already present")


if __name__ == "__main__":
    main()
