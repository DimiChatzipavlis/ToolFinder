"""Utility helpers for resilient JSON extraction from model output."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


_MARKDOWN_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_JSON_OBJECT_START_RE = re.compile(r"\{")

# Resource bounds for the recovery scan. Each candidate brace triggers a fresh
# decode of the remaining text, so an adversarial output packed with braces
# costs O(n^2) without a cap. Real model outputs put the JSON object within the
# first few braces; anything beyond these limits is treated as unparseable.
_MAX_INPUT_CHARS = 100_000
_MAX_PARSE_CANDIDATES = 32


class LLMOutputParsingError(RuntimeError):
    """Raised when no valid JSON object can be parsed from model output."""

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def _count_nesting_depth(text: str) -> int:
    """ALGY-1 FIX: Count maximum nesting depth to prevent RecursionError in ast.literal_eval."""
    max_depth = 0
    current_depth = 0
    for char in text:
        if char in "{[(":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char in "}])":
            current_depth = max(0, current_depth - 1)
    return max_depth


def _parse_json_object(candidate: str, raw_text: str, decoder: json.JSONDecoder) -> dict[str, Any]:
    stripped = candidate.strip().strip("`").strip()
    try:
        parsed, _ = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        # ALGY-1 FIX: Check nesting depth before ast.literal_eval to prevent DoS
        nesting_depth = _count_nesting_depth(stripped)
        if nesting_depth > 100:
            raise LLMOutputParsingError(
                f"Nesting depth {nesting_depth} exceeds limit of 100; refusing to parse.",
                raw_text,
            )
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise LLMOutputParsingError(
                f"Failed to parse JSON or Python dict: {exc}. Raw: {stripped}",
                raw_text,
            ) from exc

    if not isinstance(parsed, dict):
        raise LLMOutputParsingError("LLM output did not contain a JSON object.", raw_text)
    return parsed


def extract_and_parse_json(text: str) -> dict[str, Any]:
    """Extract and parse the first valid JSON-like object from free-form text.

    Args:
        text: Raw LLM output that may include markdown fences or extra prose.

    Returns:
        Parsed JSON object as a Python dictionary.

    Edge cases:
        Attempts strict JSON first, then `ast.literal_eval` recovery.
        Rejects payloads exceeding nesting-depth safety limits.
    """
    decoder = json.JSONDecoder()
    raw_text = text.strip()
    if len(raw_text) > _MAX_INPUT_CHARS:
        raise LLMOutputParsingError(
            f"Model output of {len(raw_text)} chars exceeds the {_MAX_INPUT_CHARS}-char "
            "parsing limit; refusing to scan.",
            raw_text[:1000],
        )
    last_error: LLMOutputParsingError | None = None
    attempts = 0

    for match in _MARKDOWN_JSON_BLOCK_RE.finditer(raw_text):
        candidate = match.group(1).strip().strip("`").strip()
        if not candidate.startswith("{"):
            continue
        attempts += 1
        if attempts > _MAX_PARSE_CANDIDATES:
            break
        try:
            return _parse_json_object(candidate, raw_text, decoder)
        except LLMOutputParsingError as exc:
            last_error = exc
            continue

    for match in _JSON_OBJECT_START_RE.finditer(raw_text):
        attempts += 1
        if attempts > _MAX_PARSE_CANDIDATES:
            break
        candidate = raw_text[match.start() :].strip().strip("`").strip()
        try:
            return _parse_json_object(candidate, raw_text, decoder)
        except LLMOutputParsingError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise LLMOutputParsingError("Failed to extract a valid JSON object from LLM output.", raw_text)