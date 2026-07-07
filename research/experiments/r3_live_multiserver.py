"""R3 — LIVE multi-server validation with repeats and error bars.

The production-credibility study: several REAL downstream MCP servers under one
router, REAL cross-server tasks, objective verification (on-disk content checks
and direct queries against the memory server's graph — never the agent's own
claim of success), N repeats per cell, and Wilson 95% CIs on success rates.

Servers (all real, spawned via npx): filesystem, memory, everything.
Tasks (each verified objectively, with a unique per-trial tag):
  fs_write_read      — create a file with exact content, read it back  (disk check)
  memory_store       — store an entity in the knowledge graph           (harness queries search_nodes)
  cross_fs_memory    — file on disk AND a memory entity about it        (both checks)
Arms:
  baseline  — bind ALL downstream tools directly (the simple approach)
  gateway   — bind only find_tools + call_tool (ToolFinder)

Also watched: whether any downstream emits `tools/list_changed` live (E2's push
path) — reported honestly either way.

Run (key from repo-root .env):
  python research/experiments/r3_live_multiserver.py
  python research/experiments/r3_live_multiserver.py --repeats 10 --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))  # research/

from experiments import paths  # noqa: E402
from experiments.bridge_ab import load_dotenv, make_client_factory, mcp_to_openai, result_text, run_arm  # noqa: E402
from experiments.bridge_scaling import clean_parameters  # noqa: E402
from toolfinder import UniversalMCPRouter, to_openai_tools  # noqa: E402
from toolfinder.dynamic_faiss_router import RouterHyperparameters  # noqa: E402
from toolfinder.mcp_adapter import DynamicMCPClient  # noqa: E402

TRUE_ROOT = HERE.parents[1]
BIENCODER = "sentence-transformers/all-MiniLM-L6-v2"

GATEWAY_TOOLS = [
    {"type": "function", "function": {"name": "find_tools",
     "description": "Find the tools relevant to an action across all servers; returns their schemas.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "call_tool",
     "description": "Execute one tool by name with arguments (dispatched to the server that owns it).",
     "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"}, "arguments": {"type": "object"}},
                    "required": ["tool_name", "arguments"]}}},
]

SYSTEM_BASELINE = (
    "You are an agent with tools spanning a filesystem, a knowledge-graph memory, and a demo server. "
    "Use the tools to complete the task exactly. Use absolute paths inside the allowed directory. "
    "Take one tool action at a time. Reply DONE when finished."
)
SYSTEM_GATEWAY = (
    "You reach every tool through a gateway with exactly two tools: find_tools(query) returns the "
    "relevant tool schemas, call_tool(tool_name, arguments) executes one. For each step, first call "
    "find_tools, then call_tool. Use absolute paths inside the allowed directory. Reply DONE when finished."
)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---- tasks (prompt + OBJECTIVE verification) --------------------------------

def make_tasks(root: Path):
    async def verify_fs(clients, tag: str) -> bool:
        target = root / f"note_{tag}.txt"
        return target.exists() and target.read_text(encoding="utf-8").strip() == f"hello {tag}"

    async def verify_memory(clients, tag: str) -> bool:
        memory = clients.get("memory")
        if memory is None:
            return False
        try:  # ask the memory server directly — never trust the agent's claim
            found = await memory.call_tool("search_nodes", {"query": tag})
            return tag in json.dumps(found)
        except Exception:  # noqa: BLE001
            return False

    async def verify_cross(clients, tag: str) -> bool:
        return await verify_fs(clients, tag) and await verify_memory(clients, tag)

    return [
        {"id": "fs_write_read",
         "prompt": lambda tag: (f"Create a text file {root.as_posix()}/note_{tag}.txt whose contents are exactly: "
                                f"hello {tag} — then read it back and reply DONE."),
         "verify": verify_fs},
        {"id": "memory_store",
         "prompt": lambda tag: (f"Store this in the knowledge-graph memory: an entity named '{tag}' of type "
                                f"'project' with the observation 'codename {tag}'. Then reply DONE."),
         "verify": verify_memory},
        {"id": "cross_fs_memory",
         "prompt": lambda tag: (f"Two steps: 1) create a text file {root.as_posix()}/note_{tag}.txt with contents "
                                f"exactly: hello {tag} — 2) store in the knowledge-graph memory an entity named "
                                f"'{tag}' of type 'file-note' with the observation 'file created'. Then reply DONE."),
         "verify": verify_cross},
    ]


async def main_async(args) -> None:
    paths.ensure_dirs()
    root = Path(tempfile.mkdtemp(prefix="toolfinder-r3-"))
    specs = [
        ("filesystem", ["-y", "@modelcontextprotocol/server-filesystem", str(root)]),
        ("memory", ["-y", "@modelcontextprotocol/server-memory"]),
        ("everything", ["-y", "@modelcontextprotocol/server-everything"]),
    ]
    clients: dict[str, DynamicMCPClient] = {}
    tool_to_server: dict[str, str] = {}
    all_tools: list[dict] = []
    notifications: list[tuple[str, str]] = []
    # Mirror the shipped bridge defaults post-R3-fix: the 36-tool union crosses
    # the auto-rerank threshold (25) with the stock encoder, so rerank is ON.
    router = UniversalMCPRouter(model_name=BIENCODER,
                                config=RouterHyperparameters(rerank=not args.no_rerank))

    try:
        for name, srv_args in specs:  # spawn what runs; report what doesn't (fault isolation)
            client = DynamicMCPClient(server_name=name, command="npx", args=srv_args,
                                      startup_timeout_s=120.0, request_timeout_s=60.0)
            try:
                tools = await client.initialize_and_get_tools()
            except Exception as exc:  # noqa: BLE001
                print(f"[setup] {name} FAILED to start (skipped): {str(exc)[:120]}")
                await client.close()
                continue
            client.on_notification = (lambda n: lambda method, params: notifications.append((n, method)))(name)
            clients[name] = client
            router.ingest_server(name, tools)
            for t in tools:
                tool_to_server.setdefault(t["tool_name"], name)
                all_tools.append(t)
            print(f"[setup] {name}: {len(tools)} tools")
        if "filesystem" not in clients or "memory" not in clients:
            raise SystemExit("R3 needs at least filesystem + memory running")
        print(f"[setup] union: {len(all_tools)} tools from {len(clients)} live servers; model={args.model}")

        baseline_tools = [mcp_to_openai({**t, "inputSchema": clean_parameters(t.get("inputSchema", {}))})
                          for t in all_tools]

        async def dispatch(tool_name: str, arguments: dict) -> str:
            owner = tool_to_server.get(tool_name)
            if owner is None:
                return f"ERROR: tool '{tool_name}' is not available."
            try:
                return result_text(await clients[owner].call_tool(tool_name, arguments or {}))
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: {exc}"

        async def baseline_exec(name: str, arguments: dict) -> str:
            return await dispatch(name, arguments)

        async def gateway_exec(name: str, arguments: dict) -> str:
            if name == "find_tools":
                # best-effort discovery, matching the fixed bridge behavior
                matches = router.route_top_k(str(arguments.get("query", "")), k=3, min_score=-1.0)
                return json.dumps(to_openai_tools(matches))
            if name == "call_tool":
                return await dispatch(str(arguments.get("tool_name", "")), arguments.get("arguments") or {})
            return f"ERROR: unknown tool {name}"

        arms = {
            "baseline": (SYSTEM_BASELINE, baseline_tools, baseline_exec),
            "gateway": (SYSTEM_GATEWAY, GATEWAY_TOOLS, gateway_exec),
        }
        tasks = make_tasks(root)
        if args.tasks:
            tasks = [t for t in tasks if t["id"] in set(args.tasks)]
        if args.arms:
            arms = {name: spec for name, spec in arms.items() if name in set(args.arms)}
        make_client = make_client_factory("openai", None)
        rng = random.Random(11)
        out_path = paths.RESULTS_DIR / "r3_live_multiserver.json"
        results: dict = {"model": args.model, "repeats": args.repeats,
                         "servers": {n: True for n in clients}, "union_tools": len(all_tools), "cells": {}}
        if out_path.exists():  # resume/merge: keep other cells from a prior run
            previous = json.loads(out_path.read_text(encoding="utf-8"))
            if previous.get("model") == args.model:
                results["cells"] = previous.get("cells", {})
                results["prior_notes"] = previous.get("prior_notes", {})

        for task in tasks:
            for arm_name, (system, tools, executor) in arms.items():
                successes, tokens_list, turns_list = 0, [], []
                for rep in range(args.repeats):
                    tag = f"{task['id'][:6]}{rep}{rng.randrange(1000, 9999)}"
                    for stale in root.glob("note_*.txt"):
                        stale.unlink()
                    try:
                        metrics = await run_arm(make_client, args.model, system, tools, executor,
                                                task["prompt"](tag), None)
                        ok = await task["verify"](clients, tag)
                        successes += int(ok)
                        tokens_list.append(metrics["total_tokens"])
                        turns_list.append(metrics["model_turns"])
                    except Exception as exc:  # noqa: BLE001 - a failed trial counts as failure, run continues
                        print(f"    [trial-error] {task['id']}/{arm_name}/r{rep}: {str(exc)[:100]}")
                low, high = wilson(successes, args.repeats)
                cell = {"success": f"{successes}/{args.repeats}", "rate": round(successes / args.repeats, 2),
                        "wilson95": [round(low, 2), round(high, 2)],
                        "tokens_mean": round(statistics.mean(tokens_list)) if tokens_list else None,
                        "tokens_std": round(statistics.stdev(tokens_list)) if len(tokens_list) > 1 else 0,
                        "turns_mean": round(statistics.mean(turns_list), 1) if turns_list else None}
                results["cells"][f"{task['id']}|{arm_name}"] = cell
                print(f"  {task['id']:<16} {arm_name:<9} success={cell['success']} "
                      f"CI95=[{low:.2f},{high:.2f}] tokens={cell['tokens_mean']}±{cell['tokens_std']} "
                      f"turns={cell['turns_mean']}")
                out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

        results["list_changed_notifications"] = len([1 for _, m in notifications
                                                     if "list_changed" in m])
        out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        _report(results, args)
        print(f"\nwrote {out_path}")
    finally:
        for client in clients.values():
            await client.close()
        router.teardown()
        for p in root.glob("*"):
            if p.is_file():
                p.unlink()
        root.rmdir()


def _report(results: dict, args) -> None:
    print("\n" + "=" * 80)
    print(f"R3 — LIVE MULTI-SERVER, {args.repeats} repeats/cell ({results['model']}, "
          f"{results['union_tools']} tools from {len(results['servers'])} real servers)")
    print("=" * 80)
    print(f"{'task':<18} | {'arm':<9} | {'success':>8} | {'wilson95':>13} | {'tokens':>13} | {'turns':>5}")
    for key, c in results["cells"].items():
        task, arm = key.split("|")
        print(f"{task:<18} | {arm:<9} | {c['success']:>8} | {str(c['wilson95']):>13} | "
              f"{c['tokens_mean']}±{c['tokens_std']:>5} | {c['turns_mean']:>5}")
    n_notif = results.get("list_changed_notifications", 0)
    print(f"\ntools/list_changed notifications observed live: {n_notif} "
          f"({'E2 push path exercised for real' if n_notif else 'none emitted by these servers — the E2 push path remains fake-validated only; per-server refresh() is the guaranteed path'})")
    print("Honest scope: gpt-4.1-mini only, 3 task families, small n (Wilson CIs are wide by design).")


def main() -> None:
    load_dotenv(TRUE_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tasks", nargs="*", default=None, help="subset of task ids to (re)run")
    parser.add_argument("--arms", nargs="*", default=None, help="subset of arms to (re)run")
    parser.add_argument("--no-rerank", action="store_true", help="disable rerank (replicates the pre-fix run)")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
