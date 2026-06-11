"""Build a multi-server tool catalog from real OpenAPI specs (apis.guru).

Converts operations from well-known public APIs into MCP-style tool schemas
({name, description, inputSchema}) so the unseen-server regime can rank against
hundreds of real distractor tools from ~20+ providers instead of one server.

Provenance: every tool records its source API and the apis.guru spec URL.
These are real production API operations, not synthetic schemas; they stand in
for MCP servers ("server" = API provider).

Usage:
    python experiments/dataset/build_multiserver_catalog.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402

LIST_URL = "https://api.apis.guru/v2/list.json"
MAX_TOOLS_PER_SERVER = 25
MIN_DESCRIPTION_CHARS = 30
MAX_DESCRIPTION_CHARS = 400
TARGET_SERVERS = 24

# Recognizable providers spanning messaging, music, payments, devops, docs,
# storage, productivity, monitoring - chosen for domain diversity, matched
# against apis.guru keys by prefix.
PREFERRED_PROVIDERS = [
    "slack.com",
    "spotify.com",
    "twilio.com",
    "stripe.com",
    "sendgrid.com",
    "mailchimp.com",
    "trello.com",
    "asana.com",
    "atlassian.com",
    "digitalocean.com",
    "datadoghq.com",
    "pagerduty.com",
    "zoom.us",
    "dropbox.com",
    "box.com",
    "docusign.net",
    "twitter.com",
    "medium.com",
    "godaddy.com",
    "circleci.com",
    "travis-ci.com",
    "netlify.com",
    "vercel.com",
    "cloudflare.com",
    "fastly.com",
    "launchdarkly.com",
    "linode.com",
    "vultr.com",
    "openweathermap.org",
    "nytimes.com",
    "spoonacular.com",
    "calendly.com",
    "intercom.io",
    "zendesk.com",
    "shopify.com",
    "squareup.com",
    "plaid.com",
    "lob.com",
    "postmarkapp.com",
    "statuspage.io",
]


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags common in specs
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_DESCRIPTION_CHARS]


def parameter_schema(parameters: list[dict], body_schema: dict | None) -> dict:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in parameters or []:
        name = param.get("name")
        if not name or param.get("in") not in {"query", "path"}:
            continue
        schema = param.get("schema") or {}
        properties[name] = {
            "type": schema.get("type", param.get("type", "string")),
            "description": clean_text(str(param.get("description", ""))) or f"The {name} parameter",
        }
        if param.get("required"):
            required.append(name)
    if body_schema and isinstance(body_schema.get("properties"), dict):
        for name, prop in list(body_schema["properties"].items())[:8]:
            if name in properties or not isinstance(prop, dict):
                continue
            properties[name] = {
                "type": prop.get("type", "string"),
                "description": clean_text(str(prop.get("description", ""))) or f"The {name} field",
            }
        for name in body_schema.get("required", []):
            if name in properties and name not in required:
                required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def extract_body_schema(operation: dict, spec: dict) -> dict | None:
    request_body = operation.get("requestBody")
    if not isinstance(request_body, dict):
        return None
    content = request_body.get("content", {})
    for media_type in ("application/json", "application/x-www-form-urlencoded"):
        schema = content.get(media_type, {}).get("schema")
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref_name = schema["$ref"].rsplit("/", 1)[-1]
                schema = spec.get("components", {}).get("schemas", {}).get(ref_name, {})
            return schema if isinstance(schema, dict) else None
    return None


def convert_spec(server: str, spec: dict, source_url: str) -> list[dict]:
    tools: list[dict] = []
    for path, methods in (spec.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        shared_parameters = methods.get("parameters", [])
        for method, operation in methods.items():
            if method.lower() not in {"get", "post", "put", "delete", "patch"} or not isinstance(operation, dict):
                continue
            if operation.get("deprecated"):
                continue
            summary = clean_text(str(operation.get("summary", "")))
            details = clean_text(str(operation.get("description", "")))
            description = summary if summary else details
            if details and details.lower() != summary.lower():
                description = clean_text(f"{summary}. {details}") if summary else details
            if len(description) < MIN_DESCRIPTION_CHARS:
                continue
            name = operation.get("operationId") or f"{method}_{path}"
            name = slugify(str(name))
            if not name:
                continue
            parameters = list(shared_parameters) + list(operation.get("parameters") or [])
            schema = {
                "name": name,
                "description": description,
                "inputSchema": parameter_schema(parameters, extract_body_schema(operation, spec)),
            }
            tools.append(
                {
                    "server": server,
                    "tool": name,
                    "schema": schema,
                    "provenance": {"source": "apis.guru", "spec_url": source_url, "path": path, "method": method},
                }
            )
            if len(tools) >= MAX_TOOLS_PER_SERVER:
                return tools
    return tools


def main() -> None:
    paths.ensure_dirs()
    catalog_dir = paths.DATA_DIR / "catalogs"
    catalog_dir.mkdir(exist_ok=True)

    print("[fetch] apis.guru index")
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        index = client.get(LIST_URL).json()

        chosen: dict[str, dict] = {}
        for provider in PREFERRED_PROVIDERS:
            if len(chosen) >= TARGET_SERVERS:
                break
            for key, entry in index.items():
                if not key.startswith(provider):
                    continue
                preferred_version = entry.get("preferred")
                version_entry = entry.get("versions", {}).get(preferred_version, {})
                spec_url = version_entry.get("swaggerUrl") or version_entry.get("openapiVer")
                if spec_url:
                    chosen[provider] = {"key": key, "url": spec_url}
                break

        print(f"[fetch] downloading {len(chosen)} specs")
        all_tools: dict[str, dict] = {}
        per_server: dict[str, int] = {}
        for provider, info in chosen.items():
            server = slugify(provider.split(".")[0])
            try:
                spec = client.get(info["url"]).json()
            except Exception as exc:  # noqa: BLE001 - network robustness
                print(f"  [skip] {provider}: {type(exc).__name__}")
                continue
            tools = convert_spec(server, spec, info["url"])
            if len(tools) < 5:
                print(f"  [skip] {provider}: only {len(tools)} usable operations")
                continue
            for entry in tools:
                all_tools[f"{server}/{entry['tool']}"] = entry
            per_server[server] = len(tools)
            print(f"  [ok] {server}: {len(tools)} tools")

    out_path = catalog_dir / "multiserver_catalog.json"
    out_path.write_text(json.dumps(all_tools, indent=1, sort_keys=True), encoding="utf-8")
    print(
        f"\nwrote {out_path}: {len(all_tools)} tools across {len(per_server)} servers "
        f"(min {min(per_server.values())}, max {max(per_server.values())} per server)"
    )


if __name__ == "__main__":
    main()
