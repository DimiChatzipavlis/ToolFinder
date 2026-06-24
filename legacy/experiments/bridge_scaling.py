"""Crossover study: when does the ToolFinder bridge pay off, and which pattern?

The filesystem task (create -> edit -> read) is held fixed; the agent's *catalog*
is grown from 14 (filesystem only) to N by padding with real distractor tools
drawn from a 574-tool multi-server pool. This isolates the one variable that
matters — how many tools the agent must choose among.

Two measurement axes:

FREE (no API calls):
  - per-turn tool-schema token weight: baseline (all N schemas) vs bridge (1-2
    meta-tools, constant in N).
  - router selection recall@1/@3: does find_tools still surface the correct
    filesystem tool when it is buried among N-14 distractors? This is the
    retrieval value proposition, measured at scale for free.

API (GPT-5.4, a few N): end-to-end total tokens + task success for three arms:
  baseline (all N tools bound) / find_call (find_tools+call_tool) /
  single (one do_filesystem(intent, arguments) tool, routes + executes in one hop).

Usage:
    python experiments/bridge_scaling.py --free-only          # no API spend
    python experiments/bridge_scaling.py --api-sizes 14 60 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import paths  # noqa: E402
from experiments.bridge_ab import (  # noqa: E402
    BASELINE_SYSTEM, TASK_TEMPLATE, TOOLFINDER_SYSTEM,
    load_dotenv, make_client_factory, mcp_to_openai, reset_sandbox,
    result_text, run_arm, tokens, verify,
)
from toolfinder import UniversalMCPRouter, to_openai_tools  # noqa: E402
from toolfinder.mcp_adapter import DynamicMCPClient  # noqa: E402

FREE_GRID = [14, 30, 60, 120, 250, 400]
DISTRACTOR_SEED = 7

# Filesystem intents -> the set of acceptable correct tools (some are genuinely
# ambiguous, e.g. create vs overwrite both map to write_file/edit_file).
PROBES = [
    ("create a new text file with some content", {"write_file"}),
    ("overwrite the entire contents of a file", {"write_file", "edit_file"}),
    ("make a small edit to an existing file", {"edit_file", "write_file"}),
    ("read the contents of a text file", {"read_text_file", "read_file", "read_media_file"}),
    ("list the files in a directory", {"list_directory", "list_directory_with_sizes", "directory_tree"}),
    ("create a new folder", {"create_directory"}),
    ("move or rename a file", {"move_file"}),
    ("search for files by name", {"search_files"}),
]

SINGLE_SYSTEM = (
    "You have ONE tool: do_filesystem(intent, arguments). For each step, call it with a short "
    "natural-language `intent` describing the filesystem action and an `arguments` object "
    "(e.g. path, content). Use absolute paths inside the allowed directory. After the file "
    "reads 'hello world' and you have read it back, reply DONE."
)


def _dedupe_required(node: Any) -> Any:
    """Recursively dedupe JSON-Schema `required` arrays and drop entries absent
    from `properties` — both rejected by the OpenAI function-schema validator
    (real apis.guru-derived schemas contain e.g. required=['accountId','accountId'])."""
    if isinstance(node, dict):
        out = {k: _dedupe_required(v) for k, v in node.items()}
        if isinstance(out.get("required"), list):
            props = out.get("properties")
            seen: set[str] = set()
            cleaned: list[str] = []
            for r in out["required"]:
                if not isinstance(r, str) or r in seen:
                    continue
                if isinstance(props, dict) and r not in props:
                    continue
                seen.add(r)
                cleaned.append(r)
            out["required"] = cleaned
        return out
    if isinstance(node, list):
        return [_dedupe_required(x) for x in node]
    return node


def clean_parameters(schema: Any) -> dict:
    """Return an OpenAI-valid object schema for a tool's parameters."""
    s = _dedupe_required(schema)
    if not isinstance(s, dict) or s.get("type") != "object":
        props = s.get("properties") if isinstance(s, dict) else {}
        return {"type": "object", "properties": props if isinstance(props, dict) else {}}
    if not isinstance(s.get("properties"), dict):
        s = {**s, "properties": {}}
    return s


def sanitize(name: str, used: set[str]) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:60] or "tool"
    candidate = clean
    i = 1
    while candidate in used:
        candidate = f"{clean}_{i}"[:64]
        i += 1
    used.add(candidate)
    return candidate


def build_catalog(fs_tools: list[dict], pool: list[dict], n: int, seed: int = DISTRACTOR_SEED) -> list[dict]:
    """14 real filesystem tools + (n-14) real distractor tools, names unique/valid."""
    used: set[str] = set()
    catalog: list[dict] = []
    for t in fs_tools:
        catalog.append({"tool_name": sanitize(t["tool_name"], used),
                        "description": t.get("description", ""), "inputSchema": clean_parameters(t.get("inputSchema", {}))})
    rng = random.Random(seed)
    distractors = [p for p in pool if p["schema"]["name"] not in {t["tool_name"] for t in fs_tools}]
    rng.shuffle(distractors)
    for p in distractors[: max(0, n - len(fs_tools))]:
        s = p["schema"]
        catalog.append({"tool_name": sanitize(s["name"], used),
                        "description": s.get("description", ""), "inputSchema": clean_parameters(s.get("inputSchema", {}))})
    return catalog


def make_router(catalog: list[dict]) -> UniversalMCPRouter:
    router = UniversalMCPRouter(model_name="sentence-transformers/all-MiniLM-L6-v2")
    router.ingest_server("catalog", catalog)
    return router


def free_measurements(fs_tools, pool) -> dict:
    single_schema = [{"type": "function", "function": {"name": "do_filesystem", "description": "x" * 80,
                      "parameters": {"type": "object", "properties": {"intent": {"type": "string"},
                                     "arguments": {"type": "object"}}, "required": ["intent", "arguments"]}}}]
    findcall_schema = [
        {"type": "function", "function": {"name": "find_tools", "description": "x" * 80,
         "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "call_tool", "description": "x" * 80,
         "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"},
                        "arguments": {"type": "object"}}, "required": ["tool_name", "arguments"]}}},
    ]
    points = []
    for n in FREE_GRID:
        catalog = build_catalog(fs_tools, pool, n)
        baseline_tokens = tokens([mcp_to_openai(t) for t in catalog])
        router = make_router(catalog)
        r1 = r3 = 0
        for intent, acceptable in PROBES:
            ranked = [r.tool_name for r in router.route_top_k(intent, k=3)]
            if ranked and ranked[0] in acceptable:
                r1 += 1
            if any(t in acceptable for t in ranked[:3]):
                r3 += 1
        points.append({
            "n_tools": len(catalog),
            "baseline_schema_tokens": baseline_tokens,
            "findcall_schema_tokens": tokens(findcall_schema),
            "single_schema_tokens": tokens(single_schema),
            "router_recall@1": round(r1 / len(PROBES), 3),
            "router_recall@3": round(r3 / len(PROBES), 3),
        })
        print(f"[free] N={len(catalog):>4} | baseline schema={baseline_tokens:>6} tok | "
              f"bridge={tokens(single_schema)}/{tokens(findcall_schema)} tok | "
              f"router R@1={points[-1]['router_recall@1']} R@3={points[-1]['router_recall@3']}")
    return {"probes": len(PROBES), "points": points}


async def api_arm(make_client, model, arm, catalog, fs_client, fs_names, router, root, temperature) -> dict:
    task = TASK_TEMPLATE.format(root=root.as_posix())

    async def baseline_exec(name, args):
        if name in fs_names:
            try:
                return result_text(await fs_client.call_tool(name, args))
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: {exc}"
        return f"ERROR: tool '{name}' is not available."

    async def tf_exec(name, args):
        if name == "find_tools":
            return json.dumps(to_openai_tools(router.route_top_k(str(args.get("query", "")), k=3)))
        if name == "call_tool":
            tn = str(args.get("tool_name", ""))
            if tn in fs_names:
                try:
                    return result_text(await fs_client.call_tool(tn, args.get("arguments") or {}))
                except Exception as exc:  # noqa: BLE001
                    return f"ERROR: {exc}"
            return f"ERROR: tool '{tn}' is not available."
        return f"ERROR: unknown tool {name}"

    async def single_exec(name, args):
        intent = str(args.get("intent", ""))
        arguments = args.get("arguments") or {}
        top = router.route_top_k(intent, k=1)
        if not top:
            return "ERROR: no matching tool"
        chosen = top[0].tool_name
        if chosen in fs_names:
            try:
                return result_text(await fs_client.call_tool(chosen, arguments))
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: {exc}"
        return f"ERROR: routed to unavailable tool '{chosen}'"

    if arm == "baseline":
        tools = [mcp_to_openai(t) for t in catalog]
        system, executor = BASELINE_SYSTEM, baseline_exec
    elif arm == "find_call":
        tools = [
            {"type": "function", "function": {"name": "find_tools", "description": "Find the tools relevant to an action; returns their schemas.",
             "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "call_tool", "description": "Execute one tool by name with arguments.",
             "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["tool_name", "arguments"]}}},
        ]
        system, executor = TOOLFINDER_SYSTEM, tf_exec
    else:  # single
        tools = [{"type": "function", "function": {"name": "do_filesystem", "description": "Perform a filesystem action described in natural language.",
                  "parameters": {"type": "object", "properties": {"intent": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["intent", "arguments"]}}}]
        system, executor = SINGLE_SYSTEM, single_exec

    reset_sandbox(root)
    metrics = await run_arm(make_client, model, system, tools, executor, task, temperature)
    metrics["task_success"] = verify(root)
    metrics["arm"] = arm
    return metrics


async def main_async(args) -> None:
    paths.ensure_dirs()
    pool = list(json.loads((paths.DATA_DIR / "corpus_multiserver.json").read_text(encoding="utf-8")).values())
    root = Path(tempfile.mkdtemp(prefix="toolfinder-scale-"))
    fs_client = DynamicMCPClient(
        server_name="filesystem", command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(root)],
        startup_timeout_s=90.0, request_timeout_s=45.0,
    )
    try:
        fs_tools = await fs_client.initialize_and_get_tools()
        fs_names = {t["tool_name"] for t in fs_tools}
        print(f"[setup] {len(fs_tools)} filesystem tools; {len(pool)} distractor pool; sandbox={root}")

        free = free_measurements(fs_tools, pool)
        (paths.RESULTS_DIR / "bridge_scaling_free.json").write_text(json.dumps(free, indent=1), encoding="utf-8")
        print(f"wrote {paths.RESULTS_DIR / 'bridge_scaling_free.json'}")

        if args.free_only:
            return

        make_client = make_client_factory(args.backend, args.base_url)
        temperature = 0 if args.backend == "ollama" else None
        api_path = paths.RESULTS_DIR / "bridge_scaling_gpt.json"
        # Resume: keep completed (N, arm) cells from a prior run with the same model.
        api_results = {"backend": args.backend, "model": args.model, "sizes": {}}
        if api_path.exists():
            prev = json.loads(api_path.read_text(encoding="utf-8"))
            if prev.get("model") == args.model:
                api_results = prev
                api_results.setdefault("sizes", {})
        for n in args.api_sizes:
            catalog = build_catalog(fs_tools, pool, n)
            router = make_router(catalog)
            cell = api_results["sizes"].setdefault(str(n), {})
            for arm in ("baseline", "find_call", "single"):
                if arm in cell and "error" not in cell[arm]:
                    print(f"[skip] N={n} arm={arm} already done (resume)")
                    continue
                print(f"[api] N={n} arm={arm} ({args.model})...")
                try:
                    m = await api_arm(make_client, args.model, arm, catalog, fs_client, fs_names, router, root, temperature)
                    print(f"      total_tokens={m['total_tokens']} turns={m['model_turns']} success={m['task_success']}")
                except Exception as exc:  # noqa: BLE001 - isolate a bad cell, keep the run going
                    m = {"arm": arm, "error": str(exc)[:300]}
                    print(f"      [error] {exc}")
                cell[arm] = m
                api_path.write_text(json.dumps(api_results, indent=1), encoding="utf-8")  # incremental save
        print(f"wrote {api_path}")
    finally:
        await fs_client.close()
        for p in root.glob("*"):
            if p.is_file():
                p.unlink()
        root.rmdir()


def main() -> None:
    load_dotenv(paths.EXPERIMENTS_DIR / ".env")
    load_dotenv(paths.REPO_ROOT / ".env")
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["ollama", "openai"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--free-only", action="store_true")
    parser.add_argument("--api-sizes", nargs="+", type=int, default=[14, 60, 120])
    args = parser.parse_args()
    if args.model is None:
        args.model = (os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or os.environ.get("AGENT_MODEL")
                      or ("llama3.2" if args.backend == "ollama" else "gpt-4.1-mini"))
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
