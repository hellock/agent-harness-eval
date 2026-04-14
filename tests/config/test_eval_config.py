from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_harness_eval.config.eval_file import clear_cache, load_eval_yaml
from agent_harness_eval.config.runtime import build_runtime_config


def test_eval_yaml_does_not_mutate_env_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "provider: anthropic",
                "model: claude-sonnet-4-6",
                "executor: host",
            ]
        )
        + "\n"
    )

    clear_cache()

    load_eval_yaml(str(tmp_path / "eval.yaml"))

    assert "EVAL_STATE_DIR" not in os.environ


def test_runtime_config_does_not_project_eval_metadata_to_subprocess_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "provider: anthropic",
                "model: claude-sonnet-4-6",
                "executor: host",
                "providers:",
                "  relay:",
                '    base_url: "http://relay.example"',
                '    api_format: "anthropic"',
            ]
        )
        + "\n"
    )

    clear_cache()
    monkeypatch.delenv("EVAL_PROVIDER_RELAY_BASE_URL", raising=False)
    monkeypatch.delenv("EVAL_PROVIDER_RELAY_API_FORMAT", raising=False)

    config = load_eval_yaml(str(tmp_path / "eval.yaml"))
    runtime_config = build_runtime_config(tmp_path, config)
    subprocess_env = runtime_config.subprocess_env

    assert "EVAL_STATE_DIR" not in subprocess_env
    assert "EVAL_PROVIDER_RELAY_BASE_URL" not in subprocess_env
    assert "EVAL_PROVIDER_RELAY_API_FORMAT" not in subprocess_env


def test_runtime_config_resolves_yaml_provider_from_env_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "provider: relay",
                "model: claude-sonnet-4-6",
                "executor: host",
                "providers:",
                "  relay:",
                '    base_url: "http://relay.example"',
                '    api_format: "anthropic"',
                "    headers:",
                '      x-relay-org: "research"',
                "harnesses:",
                "  nanobot:",
                '    version: "0.1.5"',
            ]
        )
        + "\n"
    )

    clear_cache()
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_KEY", "relay-key")

    config = load_eval_yaml(str(tmp_path / "eval.yaml"))
    runtime_config = build_runtime_config(tmp_path, config)
    provider = runtime_config.providers["relay"]

    assert provider.base_url == "http://relay.example"
    assert provider.api_key == "relay-key"
    assert provider.api_format == "anthropic"
    assert provider.extra_headers == {"x-relay-org": "research"}


def test_runtime_config_preserves_base_url_without_auto_v1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """base_url is used as-is (trailing slash stripped), no auto /v1 append."""
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "provider: relay",
                "model: gpt-5.4",
                "executor: host",
                "providers:",
                "  relay:",
                '    base_url: "https://relay.example.com/"',
                '    api_format: "openai-responses"',
                "harnesses:",
                "  codex:",
                '    version: "0.118.0"',
            ]
        )
        + "\n"
    )

    clear_cache()
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_KEY", "relay-key")

    config = load_eval_yaml(str(eval_yaml))
    runtime_config = build_runtime_config(tmp_path, config)

    provider = runtime_config.providers["relay"]
    assert provider.base_url == "https://relay.example.com"
    assert provider.api_format == "openai-responses"


@pytest.mark.parametrize(
    ("api_format", "base_url"),
    [
        ("anthropic", "https://relay.example.com/v1/messages"),
        ("openai-chat-completions", "https://relay.example.com/v1/chat/completions"),
        ("openai-responses", "https://relay.example.com/v1/responses"),
    ],
)
def test_runtime_config_rejects_provider_endpoint_urls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    api_format: str,
    base_url: str,
) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "provider: relay",
                "model: claude-sonnet-4-6",
                "executor: host",
                "providers:",
                "  relay:",
                f'    base_url: "{base_url}"',
                f'    api_format: "{api_format}"',
                "harnesses:",
                "  nanobot:",
                '    version: "0.1.5"',
            ]
        )
        + "\n"
    )

    clear_cache()
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_KEY", "relay-key")

    config = load_eval_yaml(str(eval_yaml))
    with pytest.raises(ValueError, match="provider root URL"):
        build_runtime_config(tmp_path, config)


def test_runtime_config_rejects_invalid_yaml_provider_api_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "provider: relay",
                "model: claude-sonnet-4-6",
                "executor: host",
                "providers:",
                "  relay:",
                '    base_url: "http://relay.example"',
                '    api_format: "bad-format"',
                "harnesses:",
                "  nanobot:",
                '    version: "0.1.5"',
            ]
        )
        + "\n"
    )

    clear_cache()
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_KEY", "relay-key")

    config = load_eval_yaml(str(tmp_path / "eval.yaml"))
    with pytest.raises(ValueError, match="Invalid api_format"):
        build_runtime_config(tmp_path, config)


def test_eval_yaml_requires_explicit_executor(tmp_path: Path) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "provider: anthropic",
                "model: claude-sonnet-4-6",
            ]
        )
        + "\n"
    )

    clear_cache()

    with pytest.raises(ValueError, match='requires a non-empty "executor" field'):
        load_eval_yaml(str(eval_yaml))


def test_eval_yaml_rejects_invalid_models_matrix_entry(
    tmp_path: Path,
) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "executor: host",
                "models:",
                "  - provider: anthropic",
                "    model: ''",  # empty model in matrix entry
            ]
        )
        + "\n"
    )

    clear_cache()
    with pytest.raises(ValueError, match="model"):
        load_eval_yaml(str(tmp_path / "eval.yaml"))


def test_eval_yaml_accepts_models_matrix(
    tmp_path: Path,
) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "executor: host",
                "models:",
                "  - provider: anthropic",
                "    model: claude-sonnet-4-6",
                "  - provider: openai",
                "    model: gpt-5.4",
            ]
        )
        + "\n"
    )

    clear_cache()
    config = load_eval_yaml(str(tmp_path / "eval.yaml"))
    assert len(config["models"]) == 2
    assert config["models"][0]["provider"] == "anthropic"
    assert config["models"][1]["model"] == "gpt-5.4"


def test_provider_config_parses_max_concurrency_from_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """providers.<name>.max_concurrency feeds ProviderConfig.max_concurrency."""
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "provider: relay",
                "model: claude-sonnet-4-6",
                "executor: host",
                "providers:",
                "  relay:",
                '    base_url: "http://relay.example"',
                '    api_format: "anthropic"',
                "    max_concurrency: 5",
            ]
        )
        + "\n"
    )
    clear_cache()
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_KEY", "relay-key")

    config = load_eval_yaml(str(eval_yaml))
    rc = build_runtime_config(tmp_path, config)
    provider = rc.providers["relay"]

    assert provider.max_concurrency == 5


def test_provider_config_defaults_max_concurrency_to_zero_unbounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "provider: relay",
                "model: claude-sonnet-4-6",
                "executor: host",
                "providers:",
                "  relay:",
                '    base_url: "http://relay.example"',
                '    api_format: "anthropic"',
            ]
        )
        + "\n"
    )
    clear_cache()
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_KEY", "relay-key")

    config = load_eval_yaml(str(eval_yaml))
    rc = build_runtime_config(tmp_path, config)

    assert rc.providers["relay"].max_concurrency == 0


def test_provider_config_rejects_negative_max_concurrency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "provider: relay",
                "model: claude-sonnet-4-6",
                "executor: host",
                "providers:",
                "  relay:",
                '    base_url: "http://relay.example"',
                '    api_format: "anthropic"',
                "    max_concurrency: -1",
            ]
        )
        + "\n"
    )
    clear_cache()
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_KEY", "relay-key")

    config = load_eval_yaml(str(eval_yaml))
    with pytest.raises(ValueError, match="max_concurrency must be >= 0"):
        build_runtime_config(tmp_path, config)


@pytest.mark.asyncio
async def test_provider_slot_bounds_concurrent_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """provider_slot hands out a Semaphore capped at max_concurrency; lazy-singleton."""
    import asyncio

    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "\n".join(
            [
                "provider: relay",
                "model: claude-sonnet-4-6",
                "executor: host",
                "providers:",
                "  relay:",
                '    base_url: "http://relay.example"',
                '    api_format: "anthropic"',
                "    max_concurrency: 2",
            ]
        )
        + "\n"
    )
    clear_cache()
    monkeypatch.setenv("EVAL_PROVIDER_RELAY_API_KEY", "relay-key")

    config = load_eval_yaml(str(eval_yaml))
    rc = build_runtime_config(tmp_path, config)

    inside = 0
    peak = 0

    async def worker() -> None:
        nonlocal inside, peak
        async with rc.provider_slot("relay"):
            inside += 1
            peak = max(peak, inside)
            await asyncio.sleep(0.02)
            inside -= 1

    await asyncio.gather(*(worker() for _ in range(6)))

    assert peak == 2, f"expected semaphore to cap at 2, saw peak={peak}"
    # Second access returns the same semaphore instance (lazy singleton, not per-call).
    assert rc.provider_slot("relay") is rc.provider_slot("relay")
