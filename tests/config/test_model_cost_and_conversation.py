from __future__ import annotations

import pytest

from agent_harness_eval.config.providers import parse_model_spec, resolve_providers
from agent_harness_eval.task import Task
from agent_harness_eval.utils.conversation import format_task_message
from agent_harness_eval.utils.cost import (
    ModelPricing,
    calculate_cost,
    calculate_cost_no_cache,
    get_model_pricing,
)


def test_parse_model_spec_requires_explicit_provider_model_format() -> None:
    parsed = parse_model_spec("openai:gpt-5.4")

    assert parsed.provider == "openai"
    assert parsed.model == "gpt-5.4"
    assert parsed.label == "openai:gpt-5.4"

    with pytest.raises(ValueError, match="Expected explicit"):
        parse_model_spec("gpt-5.4")

    with pytest.raises(ValueError, match="Empty model spec"):
        parse_model_spec("   ")


def test_resolve_providers_merges_env_defaults_and_eval_overrides() -> None:
    providers = resolve_providers(
        env={
            "OPENAI_API_KEY": "openai-key",
            "EVAL_PROVIDER_RELAY_API_KEY": "relay-key",
        },
        configured_providers={
            "relay": {
                "base_url": "https://relay.example/v1",
                "api_format": "openai-responses",
                "headers": {"X-Test": "1"},
            }
        },
    )

    assert providers["openai"].base_url == "https://api.openai.com/v1"
    assert providers["openai"].api_key == "openai-key"
    assert providers["relay"].api_key == "relay-key"
    assert providers["relay"].api_format == "openai-responses"
    assert providers["relay"].extra_headers == {"X-Test": "1"}
    assert providers["relay"].is_openai_compat is True


def test_cost_helpers_handle_prefixes_overrides_and_cache_pricing() -> None:
    override = ModelPricing(input=10, output=20, cache_read=1, cache_write=2)

    assert get_model_pricing("openai:gpt-5.4").input == 2.50
    assert get_model_pricing("custom:model", override).input == 10

    cached_cost = calculate_cost(
        "openai:gpt-5.4",
        input_tokens=1_000,
        output_tokens=2_000,
        cache_read_tokens=400,
        cache_write_tokens=100,
        pricing=override,
    )
    no_cache_cost = calculate_cost_no_cache(
        "openai:gpt-5.4",
        input_tokens=1_000,
        output_tokens=2_000,
        cache_read_tokens=400,
        cache_write_tokens=100,
        pricing=override,
    )

    assert cached_cost == pytest.approx(0.0466)
    assert no_cache_cost == pytest.approx(0.055)
    assert cached_cost < no_cache_cost


def test_format_task_message_includes_history_when_present() -> None:
    task = Task(
        id="task.history",
        category="reasoning",
        description="desc",
        user_query="Solve the problem",
        conversation_history=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        timeout_sec=30,
    )

    message = format_task_message(task)

    assert message.startswith("Previous conversation:")
    assert "[user]: Hi" in message
    assert "[assistant]: Hello" in message
    assert message.endswith("Current request: Solve the problem")
