"""Parse model specification strings and resolve provider configurations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

ApiFormat = Literal["openai-chat-completions", "openai-responses", "anthropic"]


@dataclass
class ModelSpec:
    provider: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


_ENDPOINT_SUFFIX: dict[ApiFormat, str] = {
    "anthropic": "/v1/messages",
    "openai-chat-completions": "/v1/chat/completions",
    "openai-responses": "/v1/responses",
}


@dataclass
class ProviderConfig:
    base_url: str
    api_key: str
    api_format: ApiFormat = "openai-chat-completions"
    extra_headers: dict[str, str] | None = None

    @property
    def is_openai_compat(self) -> bool:
        return self.api_format in ("openai-chat-completions", "openai-responses")

    def endpoint_url(self, api_format: ApiFormat | None = None) -> str:
        """Build the full endpoint URL for this provider."""
        fmt = api_format or self.api_format
        suffix = _ENDPOINT_SUFFIX[fmt]
        base = self.base_url.rstrip("/")
        if base.endswith("/v1") and suffix.startswith("/v1"):
            return base + suffix[3:]
        return base + suffix


def parse_model_spec(spec: str) -> ModelSpec:
    """Parse a ``provider:model`` string into a ModelSpec."""
    trimmed = spec.strip()
    if not trimmed:
        raise ValueError("Empty model spec")

    colon_idx = trimmed.find(":")
    if colon_idx <= 0 or colon_idx == len(trimmed) - 1:
        raise ValueError(f'Invalid model spec "{spec}". Expected explicit "provider:model" format.')

    provider = trimmed[:colon_idx].strip()
    model = trimmed[colon_idx + 1 :].strip()
    if not provider or not model:
        raise ValueError(f'Invalid model spec "{spec}". Expected explicit "provider:model" format.')
    return ModelSpec(provider=provider, model=model)


def resolve_providers(
    env: dict[str, str] | None = None,
    configured_providers: dict[str, dict[str, Any]] | None = None,
) -> dict[str, ProviderConfig]:
    """Resolve provider configs from real environment vars plus eval.yaml."""
    if env is None:
        env = dict(os.environ)
    if configured_providers is None:
        configured_providers = {}

    providers: dict[str, ProviderConfig] = {}

    if env.get("ANTHROPIC_API_KEY"):
        anthropic_format = _normalize_api_format(
            env.get("ANTHROPIC_API_FORMAT", "anthropic"),
            source="ANTHROPIC_API_FORMAT",
        )
        providers["anthropic"] = ProviderConfig(
            base_url=_normalize_provider_base_url(
                env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                api_format=anthropic_format,
                source="ANTHROPIC_BASE_URL",
            ),
            api_key=env["ANTHROPIC_API_KEY"],
            api_format=anthropic_format,
        )

    if env.get("OPENAI_API_KEY"):
        providers["openai"] = ProviderConfig(
            base_url=_normalize_provider_base_url(
                env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_format="openai-chat-completions",
                source="OPENAI_BASE_URL",
            ),
            api_key=env["OPENAI_API_KEY"],
            api_format="openai-chat-completions",
        )

    if env.get("OPENROUTER_API_KEY"):
        providers["openrouter"] = ProviderConfig(
            base_url=_normalize_provider_base_url(
                env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                api_format="openai-chat-completions",
                source="OPENROUTER_BASE_URL",
            ),
            api_key=env["OPENROUTER_API_KEY"],
            api_format="openai-chat-completions",
        )

    for name, config in configured_providers.items():
        api_key = _resolve_configured_provider_api_key(name, env, providers)
        if not api_key:
            continue

        existing = providers.get(name)
        base_url = str(config.get("base_url") or (existing.base_url if existing else "")).strip()
        if not base_url:
            raise ValueError(f"Provider {name!r} requires base_url in eval.yaml or a standard provider default.")

        api_format = _normalize_api_format(
            str(config.get("api_format") or (existing.api_format if existing else "openai-chat-completions")),
            source=f"providers.{name}.api_format",
        )
        headers = config.get("headers") or None
        if headers is not None and not isinstance(headers, dict):
            raise ValueError(f"Provider {name!r} headers must be a mapping.")

        providers[name] = ProviderConfig(
            base_url=_normalize_provider_base_url(
                base_url,
                api_format=api_format,
                source=f"providers.{name}.base_url",
            ),
            api_key=api_key,
            api_format=api_format,
            extra_headers={str(k): str(v) for k, v in headers.items()} if headers else None,
        )

    return providers


def _normalize_api_format(api_format: str, *, source: str) -> str:
    value = api_format.strip()
    if value not in ("openai-chat-completions", "openai-responses", "anthropic"):
        raise ValueError(f"Invalid api_format {value!r} from {source}.")
    return value


def _normalize_provider_base_url(
    base_url: str,
    *,
    api_format: ApiFormat,
    source: str,
) -> str:
    """Validate and clean up a provider base_url."""
    value = base_url.strip()
    if not value:
        raise ValueError(f"{source} must not be empty.")

    trimmed = value.rstrip("/")
    path = urlsplit(trimmed).path.rstrip("/")

    _ensure_not_endpoint_url(path, source=source, api_format=api_format)

    return trimmed


def _ensure_not_endpoint_url(path: str, *, source: str, api_format: ApiFormat) -> None:
    endpoint_suffixes = (
        ("/messages",) if api_format == "anthropic" else ("/chat/completions", "/responses", "/messages")
    )
    if any(path.endswith(suffix) for suffix in endpoint_suffixes):
        raise ValueError(
            f"{source} must be a provider root URL, not a concrete {api_format} endpoint path: {path or '/'}"
        )


def _resolve_configured_provider_api_key(
    name: str,
    env: dict[str, str],
    resolved_providers: dict[str, ProviderConfig],
) -> str | None:
    existing = resolved_providers.get(name)
    if existing is not None:
        return existing.api_key

    env_key = f"EVAL_PROVIDER_{name.upper()}_API_KEY"
    api_key = env.get(env_key)
    return api_key or None
