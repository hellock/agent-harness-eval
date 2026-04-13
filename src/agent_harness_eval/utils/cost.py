"""Token cost calculation utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input: float
    output: float
    cache_read: float
    cache_write: float


DEFAULT_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4-6": ModelPricing(input=3, output=15, cache_read=0.30, cache_write=3.75),
    "claude-opus-4-6": ModelPricing(input=15, output=75, cache_read=1.50, cache_write=18.75),
    "claude-haiku-4-5": ModelPricing(input=0.80, output=4, cache_read=0.08, cache_write=1),
    "gpt-5.4": ModelPricing(input=2.50, output=15, cache_read=0.25, cache_write=0),
    "gpt-5.4-mini": ModelPricing(input=0.75, output=4.50, cache_read=0.075, cache_write=0),
    "gpt-5.4-nano": ModelPricing(input=0.20, output=1.25, cache_read=0.02, cache_write=0),
}


def get_model_pricing(
    model: str,
    override: ModelPricing | None = None,
) -> ModelPricing:
    """Resolve pricing for a model.

    Priority: explicit override > built-in defaults.
    """
    if override is not None:
        return override

    # Strip provider prefix: "anthropic:claude-sonnet-4-6" → "claude-sonnet-4-6"
    model_name = model.split(":")[-1] if ":" in model else model
    return DEFAULT_PRICING.get(model_name, DEFAULT_PRICING["claude-sonnet-4-6"])


def calculate_cost_no_cache(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    *,
    pricing: ModelPricing | None = None,
) -> float:
    """Calculate cost ignoring cache pricing — every prompt token billed at input rate."""
    p = get_model_pricing(model, pricing)
    m = 1_000_000
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    return (prompt_tokens / m) * p.input + (output_tokens / m) * p.output


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    *,
    pricing: ModelPricing | None = None,
) -> float:
    """Calculate cost in USD from token usage."""
    p = get_model_pricing(model, pricing)
    m = 1_000_000
    fresh_input_tokens = max(0, input_tokens - cache_read_tokens)
    return (
        (fresh_input_tokens / m) * p.input
        + (output_tokens / m) * p.output
        + (cache_read_tokens / m) * p.cache_read
        + (cache_write_tokens / m) * p.cache_write
    )
