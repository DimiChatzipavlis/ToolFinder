"""Regression tests for resilient JSON extraction, including resource bounds."""

from __future__ import annotations

import json

import pytest

from toolfinder import utils


def test_parses_plain_json() -> None:
    assert utils.extract_and_parse_json('{"a": 1}') == {"a": 1}


def test_parses_markdown_fenced_json() -> None:
    text = 'Here you go:\n```json\n{"tool": "x", "arguments": {"y": 2}}\n```\ndone'
    assert utils.extract_and_parse_json(text)["tool"] == "x"


def test_recovers_python_literal_dict() -> None:
    assert utils.extract_and_parse_json("{'a': 'b'}") == {"a": "b"}


def test_rejects_oversized_input_fast() -> None:
    huge = "x" * (utils._MAX_INPUT_CHARS + 1)
    with pytest.raises(utils.LLMOutputParsingError, match="parsing limit"):
        utils.extract_and_parse_json(huge)


def test_candidate_scan_is_bounded() -> None:
    """A valid object hidden after more than the candidate budget of decoy
    braces is treated as unparseable instead of triggering a quadratic scan."""
    decoys = "{bad " * (utils._MAX_PARSE_CANDIDATES + 10)
    text = decoys + json.dumps({"late": True})
    with pytest.raises(utils.LLMOutputParsingError):
        utils.extract_and_parse_json(text)


def test_object_within_candidate_budget_is_found() -> None:
    text = "{bad {bad " + json.dumps({"found": 1})
    assert utils.extract_and_parse_json(text) == {"found": 1}
