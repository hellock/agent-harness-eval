"""Extract JSON from LLM output that may contain markdown fences or extra text."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from LLM output."""
    # Try markdown fenced JSON block first
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", text)
    json_str = fence_match.group(1).strip() if fence_match else text.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find a JSON object in the text
        object_match = re.search(r"\{[\s\S]*\}", json_str)
        if object_match:
            try:
                return json.loads(object_match.group(0))
            except json.JSONDecodeError:
                return None
        return None
