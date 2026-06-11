"""Schema-to-text representations used as retrieval documents.

One function per representation so the choice is an explicit, ablatable
experimental condition rather than an implementation accident. `raw` is the
default for the main tables: every system sees the identical document text.
"""

from __future__ import annotations

import copy
import json
from typing import Any


def _strip_property_descriptions(schema: dict[str, Any]) -> dict[str, Any]:
    minified = copy.deepcopy(schema)
    properties = minified.get("inputSchema", {}).get("properties", {})
    for property_schema in properties.values():
        if isinstance(property_schema, dict):
            property_schema.pop("description", None)
    return minified


def represent_raw(schema: dict[str, Any]) -> str:
    """Canonical JSON of the full schema (name, description, inputSchema)."""
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def represent_minified(schema: dict[str, Any]) -> str:
    """Canonical JSON with property-level descriptions stripped (runtime style)."""
    return json.dumps(_strip_property_descriptions(schema), sort_keys=True, separators=(",", ":"))


def represent_name_desc(schema: dict[str, Any]) -> str:
    """Tool name and top-level description only."""
    return f"{schema.get('name', '')}. {schema.get('description', '')}".strip()


def represent_desc_only(schema: dict[str, Any]) -> str:
    """Top-level description only (no tool name signal)."""
    return str(schema.get("description", ""))


REPRESENTATIONS = {
    "raw": represent_raw,
    "minified": represent_minified,
    "name_desc": represent_name_desc,
    "desc_only": represent_desc_only,
}


def lexicalize(text: str) -> str:
    """Normalize identifiers for lexical systems (BM25/TF-IDF word units).

    snake_case and path-ish separators become spaces so 'add_issue_comment'
    matches the words 'add issue comment'. Dense encoders receive the original
    text; their subword tokenizers handle separators natively.
    """
    return text.replace("_", " ").replace("/", " ")
