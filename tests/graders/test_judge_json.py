from __future__ import annotations

from agent_harness_eval.graders.judge_json import extract_json


def test_extract_json_prefers_json_from_markdown_fence() -> None:
    text = """
Some explanation first.

```json
{"pass": true, "score": 0.9, "reason": "good"}
```

Trailing text.
"""

    assert extract_json(text) == {"pass": True, "score": 0.9, "reason": "good"}


def test_extract_json_extracts_inline_object_with_surrounding_text() -> None:
    text = 'Judge response: {"pass": false, "score": 0.2, "reason": "missing file"} Thanks.'

    assert extract_json(text) == {"pass": False, "score": 0.2, "reason": "missing file"}


def test_extract_json_returns_none_for_invalid_fenced_json() -> None:
    text = """```json
{"pass": true, "score": }
```"""

    assert extract_json(text) is None


def test_extract_json_returns_none_when_multiple_objects_make_parse_ambiguous() -> None:
    text = 'First {"pass": true} then {"pass": false}'

    assert extract_json(text) is None


def test_extract_json_returns_none_when_no_json_present() -> None:
    assert extract_json("No structured payload here.") is None
