from __future__ import annotations

import asyncio

from toolfinder.openapi_adapter import OpenAPIClient

SPEC = {
    "openapi": "3.0.0",
    "servers": [{"url": "https://api.example.com/v1"}],
    "components": {
        "schemas": {
            "Pet": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "tag": {"type": "string"}},
                "required": ["name"],
            }
        }
    },
    "paths": {
        "/pets/{petId}": {
            "get": {
                "operationId": "getPet",
                "summary": "Get a pet",
                "parameters": [
                    {"name": "petId", "in": "path", "required": True, "schema": {"type": "integer"}},
                    {"name": "verbose", "in": "query", "schema": {"type": "boolean"}},
                ],
            }
        },
        "/pets": {
            "post": {
                "operationId": "createPet",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}},
                },
            }
        },
    },
}


def test_openapi_spec_becomes_tools() -> None:
    async def run() -> None:
        client = OpenAPIClient("petstore", spec=SPEC)
        try:
            tools = await client.initialize_and_get_tools()
            names = {t["tool_name"] for t in tools}
            assert names == {"getPet", "createPet"}
            get = next(t for t in tools if t["tool_name"] == "getPet")
            props = get["inputSchema"]["properties"]
            assert props["petId"]["type"] == "integer"
            assert props["verbose"]["type"] == "boolean"
            assert get["inputSchema"]["required"] == ["petId"]
        finally:
            await client.close()

    asyncio.run(run())


def test_requestbody_ref_is_resolved() -> None:
    async def run() -> None:
        client = OpenAPIClient("petstore", spec=SPEC)
        try:
            tools = await client.initialize_and_get_tools()
            create = next(t for t in tools if t["tool_name"] == "createPet")
            body_schema = create["inputSchema"]["properties"]["body"]
            assert body_schema["properties"]["name"]["type"] == "string"  # resolved from $ref
            assert body_schema["required"] == ["name"]
            assert create["inputSchema"]["required"] == ["body"]
        finally:
            await client.close()

    asyncio.run(run())


def test_build_request_path_query_and_bearer_auth(monkeypatch) -> None:
    monkeypatch.setenv("TOK", "secret123")

    async def run() -> None:
        client = OpenAPIClient("petstore", spec=SPEC, auth={"type": "bearer", "token_env": "TOK"})
        try:
            await client.initialize_and_get_tools()
            method, url, params, headers, body = client._build_request("getPet", {"petId": 7, "verbose": True})
            assert method == "GET"
            assert url == "/pets/7"
            assert params == {"verbose": True}
            assert headers["Authorization"] == "Bearer secret123"
            assert body is None
            # POST with a JSON body, no auth configured -> no Authorization header
            client._auth = {}
            method, url, params, headers, body = client._build_request("createPet", {"body": {"name": "Rex"}})
            assert (method, url) == ("POST", "/pets")
            assert body == {"name": "Rex"}
            assert "Authorization" not in headers
        finally:
            await client.close()

    asyncio.run(run())


def test_unknown_operation_returns_error() -> None:
    async def run() -> None:
        client = OpenAPIClient("petstore", spec=SPEC)
        try:
            await client.initialize_and_get_tools()
            result = await client.call_tool("does_not_exist", {})
            assert "error" in result
        finally:
            await client.close()

    asyncio.run(run())
