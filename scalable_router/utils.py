from __future__ import annotations

import ast
import json
import re
from typing import Any


_MARKDOWN_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_JSON_OBJECT_START_RE = re.compile(r"\{")


class LLMOutputParsingError(RuntimeError):
    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def _parse_json_object(candidate: str, raw_text: str, decoder: json.JSONDecoder) -> dict[str, Any]:
    stripped = candidate.strip().strip("`").strip()
    try:
        parsed, _ = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
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
    decoder = json.JSONDecoder()
    raw_text = text.strip()
    last_error: LLMOutputParsingError | None = None

    for match in _MARKDOWN_JSON_BLOCK_RE.finditer(raw_text):
        candidate = match.group(1).strip().strip("`").strip()
        if not candidate.startswith("{"):
            continue
        try:
            return _parse_json_object(candidate, raw_text, decoder)
        except LLMOutputParsingError as exc:
            last_error = exc
            continue

    for match in _JSON_OBJECT_START_RE.finditer(raw_text):
        candidate = raw_text[match.start() :].strip().strip("`").strip()
        try:
            return _parse_json_object(candidate, raw_text, decoder)
        except LLMOutputParsingError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise LLMOutputParsingError("Failed to extract a valid JSON object from LLM output.", raw_text)