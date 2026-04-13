"""Adapter registry for harness auto-discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import HarnessAdapter

_REGISTRY: dict[str, type[HarnessAdapter]] = {}


def register_adapter(cls: type[HarnessAdapter]) -> type[HarnessAdapter]:
    """Class decorator that registers a harness adapter by its ``name``."""
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter_class(name: str) -> type[HarnessAdapter]:
    """Return the adapter class for *name*, or raise ``KeyError``."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown harness adapter: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_registered_adapters() -> dict[str, type[HarnessAdapter]]:
    """Return a snapshot of all registered adapters."""
    return dict(_REGISTRY)


# Import all adapter modules to trigger registration.
from . import claude_code as _claude_code  # noqa: E402, F401
from . import codex as _codex  # noqa: E402, F401
from . import hermes as _hermes  # noqa: E402, F401
from . import nanobot as _nanobot  # noqa: E402, F401
from . import openclaw as _openclaw  # noqa: E402, F401
from . import zeroclaw as _zeroclaw  # noqa: E402, F401
