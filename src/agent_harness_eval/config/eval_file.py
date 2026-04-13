"""Load eval.yaml configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_cache: dict[str, dict[str, Any]] = {}


def load_eval_yaml(config_path: str) -> dict[str, Any]:
    """Load eval.yaml from the given file path.

    Raises FileNotFoundError if the file does not exist.
    """
    config_path = str(Path(config_path).resolve())
    if config_path in _cache:
        return _cache[config_path]

    with open(config_path) as f:
        parsed = yaml.safe_load(f)

    config_data = _deep_merge(_default_config_data(), parsed) if parsed else _default_config_data()
    _validate_model_fields(config_data)
    _cache[config_path] = config_data
    return config_data


def clear_cache() -> None:
    """Clear the config cache. Use in tests to ensure fresh config loads."""
    _cache.clear()


def _default_config_data() -> dict[str, Any]:
    return {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "concurrency": 1,
        "runs": 1,
        "timeout": 600,
        "providers": {},
        "harnesses": {},
    }


def _validate_model_fields(config: dict[str, Any]) -> None:
    """Validate that model fields use the split provider + model format."""

    def _check_split(field_name: str, provider: Any, model: Any) -> None:
        if not provider or not isinstance(provider, str) or not provider.strip():
            raise ValueError(f'eval.yaml {field_name} requires a non-empty "provider" field.')
        if not model or not isinstance(model, str) or not model.strip():
            raise ValueError(f'eval.yaml {field_name} requires a non-empty "model" field.')

    def _check_model_obj(field_name: str, obj: Any) -> None:
        if not isinstance(obj, dict):
            raise ValueError(
                f'eval.yaml {field_name} must be an object with "provider" and "model" keys, got {type(obj).__name__}.'
            )
        _check_split(field_name, obj.get("provider"), obj.get("model"))

    if config.get("models"):
        for i, entry in enumerate(config["models"]):
            _check_model_obj(f"models[{i}]", entry)
    else:
        _check_split("top-level", config.get("provider"), config.get("model"))

    for field in ("judge_model", "secondary_judge_model"):
        value = config.get(field)
        if value:
            _check_model_obj(field, value)

    executor = config.get("executor")
    if not executor or not isinstance(executor, str) or not executor.strip():
        raise ValueError('eval.yaml requires a non-empty "executor" field.')


def _deep_merge(target: dict, source: dict) -> dict:
    result = dict(target)
    for key in source:
        if (
            isinstance(source.get(key), dict)
            and not isinstance(source.get(key), list)
            and isinstance(target.get(key), dict)
            and not isinstance(target.get(key), list)
        ):
            result[key] = _deep_merge(target[key], source[key])
        elif source[key] is not None and source[key] != "":
            result[key] = source[key]
    return result
