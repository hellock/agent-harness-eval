"""Tests for utils.failure_origin classification.

``detect_failure_origin_from_error`` is one of the most cross-cutting
functions in the codebase: every adapter feeds its subprocess error
output into it, every post-mortem grader result-classifier calls it,
and downstream reports key off the returned ``failure_origin`` enum.

The classifier is brittle text-matching (lowercased substring + a regex
for HTTP status codes). These tests lock down every branch so future
edits to the rule list can't silently shift a whole class of errors
into the wrong bucket.
"""

from __future__ import annotations

import pytest

from agent_harness_eval.utils.failure_origin import detect_failure_origin_from_error

# --- Provider branch ---------------------------------------------------------

_PROVIDER_KEYWORDS = [
    "ValidationException thrown by bedrock",
    "invalid beta flag: prompt-caching",
    "Anthropic API error: model not found",
    "401 Unauthorized — bad key",
    "HTTP 403 Forbidden",
    "HTTP 429 Too Many Requests",
    "HTTP 500 internal server error",
    "HTTP 503 service unavailable",
    "HTTP 599 custom",
    "Authentication required",
    "Invalid config in provider section",
    "Provider error: upstream refused",
    "Upstream provider timeout",
    "All providers/models failed. Attempts...",
    "System memory overloaded",
    "new_api_error wrapper returned",
    "Bedrock runtime exception",
    "provider api error on retry",
]


@pytest.mark.parametrize("error", _PROVIDER_KEYWORDS)
def test_provider_branch_classifies_known_keywords(error: str) -> None:
    result = detect_failure_origin_from_error(error)
    assert result == {
        "failure_origin": "provider",
        "infra_error_code": "provider_api_error",
    }


def test_provider_branch_case_insensitive() -> None:
    """Classifier lowercases input — mixed-case inputs still match."""
    assert detect_failure_origin_from_error("PROVIDER API error in stream")["failure_origin"] == "provider"
    assert detect_failure_origin_from_error("Http 429 Too Many")["failure_origin"] == "provider"


# HTTP status regex has word-boundary anchors — must not eat substrings of
# longer number tokens.
def test_http_status_regex_word_boundaries() -> None:
    # "1429" should NOT match (no word boundary before 4).
    assert detect_failure_origin_from_error("processed 1429 items")["failure_origin"] != "provider"
    # "429" as a standalone token SHOULD match.
    assert detect_failure_origin_from_error("got status 429 back")["failure_origin"] == "provider"
    # 400 is NOT in the rule set (only 401/403/429/5xx); classifier should
    # not pick it up as provider. Documents current contract.
    assert detect_failure_origin_from_error("request 400 bad request")["failure_origin"] != "provider"


# --- Sandbox branch ----------------------------------------------------------

_SANDBOX_KEYWORDS = [
    "EACCES: permission denied",
    "Permission denied writing to /etc",
    "sandbox violation: tried to access /proc",
    "docker: failed to start container",
    "write failed: Read-only file system",
]


@pytest.mark.parametrize("error", _SANDBOX_KEYWORDS)
def test_sandbox_branch_classifies_known_keywords(error: str) -> None:
    result = detect_failure_origin_from_error(error)
    assert result == {
        "failure_origin": "sandbox",
        "infra_error_code": "sandbox_permission_error",
    }


# --- Adapter branch ----------------------------------------------------------

_ADAPTER_KEYWORDS = [
    "No result event in stream",
    "Failed to parse JSONL",
    "parse error at line 12",
    "Session file not found at /tmp/...",
    "adapter output unusable",
]


@pytest.mark.parametrize("error", _ADAPTER_KEYWORDS)
def test_adapter_branch_classifies_known_keywords(error: str) -> None:
    result = detect_failure_origin_from_error(error)
    assert result == {
        "failure_origin": "adapter",
        "infra_error_code": "adapter_output_error",
    }


# --- Unknown fallback --------------------------------------------------------

_UNKNOWN_ERRORS = [
    "segfault",
    "something unexpected happened",
    "",
    "exit code 137",  # killed — but no keyword matches
]


@pytest.mark.parametrize("error", _UNKNOWN_ERRORS)
def test_unknown_fallback_for_unclassified_errors(error: str) -> None:
    result = detect_failure_origin_from_error(error)
    assert result == {
        "failure_origin": "unknown",
        "infra_error_code": None,
    }


# --- Precedence --------------------------------------------------------------


def test_precedence_provider_wins_over_sandbox() -> None:
    """When an error contains both provider and sandbox keywords, provider
    classification wins because it's checked first. Docs the ordering so
    a future refactor can't silently reshuffle."""
    error = "HTTP 429 inside docker container"  # has 429 AND "docker"
    assert detect_failure_origin_from_error(error)["failure_origin"] == "provider"


def test_precedence_provider_wins_over_adapter() -> None:
    error = "Failed to parse HTTP 401 response"  # "parse" AND 401
    assert detect_failure_origin_from_error(error)["failure_origin"] == "provider"


def test_precedence_sandbox_wins_over_adapter() -> None:
    error = "Permission denied trying to parse config"  # "permission denied" AND "parse"
    assert detect_failure_origin_from_error(error)["failure_origin"] == "sandbox"


# --- Substring-match traps ---------------------------------------------------


def test_substring_match_adapter_is_broad() -> None:
    """The ``adapter`` branch matches the bare substring "adapter" AND
    "parse" — so any error mentioning either word is classified as
    adapter. Document this so callers know not to pass ambiguous context
    strings (e.g. the command label "Codex adapter") through the raw
    detector.
    """
    # Bare "adapter" in arbitrary context classifies adapter:
    result = detect_failure_origin_from_error("the adapter worked fine but X")
    assert result["failure_origin"] == "adapter"
    # "parse" is also broad — any phrase containing it trips adapter:
    result = detect_failure_origin_from_error("could not parse the spiderweb")
    assert result["failure_origin"] == "adapter"
    # NOT a false-positive trap but documents limitation: "sparse" contains
    # "parse" as substring. Classifier sees it as adapter error.
    assert detect_failure_origin_from_error("sparse matrix overflow")["failure_origin"] == "adapter"


def test_empty_string_is_unknown_not_provider() -> None:
    """Defensive: when no error text exists (e.g. subprocess killed with no
    output), classifier must not spuriously hit any branch.
    """
    assert detect_failure_origin_from_error("")["failure_origin"] == "unknown"
