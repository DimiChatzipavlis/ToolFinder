"""Step 3 / M1 (measured): validate the cache model and test top-k (k=5) selection.

Two things the Step 1 *model* could not give us, now measured live against the API:

1. **Real prompt-cache behavior.** We log the API's reported
   `usage.prompt_tokens_details.cached_tokens` per turn for the `baseline` arm
   (the one with a big static tool block). That tells us how much of the
   repeated prefix is actually served from cache — validating (or correcting)
   the modeled cached-vs-uncached numbers in `bridge_cache_aware.json`.

2. **Top-k (k=5) selection.** A `find5` arm exposes `find_tools` returning the
   top **5** tools (vs the k=3 in the original `find_call`), letting the model
   pick among 5. We measure end-task success — the professor's point that you
   don't need to surface exactly one tool; surfacing five and letting the LLM
   choose is fine, maybe preferable.

Scoped to stay well under budget: baseline at N in {14,60,120}, find5 at {60,120}.
Incremental save after every cell; MAX_TURNS bounds runaway loops.

Run (key + model come from the repo-root .env: API_KEY + AGENT_MODEL):
    python legacy/experiments/bridge_cache_measured.py
    python legacy/experiments/bridge_cache_measured.py --sizes 60 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Imports: the experiments package lives one level up (legacy/experiments -> legacy).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import paths  # noqa: E402
from experiments.bridge_ab import (  # noqa: E402
    BASELINE_SYSTEM,
    MAX_TURNS,
    TASK_TEMPLATE,
    TOOLFINDER_SYSTEM,
    load_dotenv,
    make_client_factory,
    mcp_to_openai,
    result_text,
    reset_sandbox,
    verify,
)
from experiments.bridge_scaling import build_catalog, make_router  # noqa: E402
from toolfinder import to_openai_tools  # noqa: E402
from toolfinder.mcp_adapter import DynamicMCPClient  # noqa: E402

# The TRUE repository root is two levels above this file (…/ToolFinder), not
# paths.REPO_ROOT (which now points at legacy/ after the repo restructure).
TRUE_ROOT = Path(__file__).resolve().parents[2]

BASELINE_SIZES = [14, 60, 120]
FIND5_SIZES = [60, 120]
CACHE_READ_RATES = {"openai_0.50": 0.50, "openai_0.25": 0.25, "anthropic_0.10": 0.10}


def _cached_tokens(usage) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    return int(getattr(details, "cached_tokens", 0) or 0) if details else 0


async def run_arm_measured(make_client, model, system, tools, executor, task, temperature=None) -> dict:
    """Agent loop that additionally records per-turn cached prompt tokens."""
    client = make_client()
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    m = {
        "schema_tokens": None,
        "n_bound_tools": len(tools),
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "first_call_prompt_tokens": None,
        "model_turns": 0,
        "tool_calls": 0,
        "per_turn": [],
    }
    create_kwargs: dict = {"model": model, "tools": tools, "tool_choice": "auto"}
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    try:
        for _ in range(MAX_TURNS):
            resp = await client.chat.completions.create(messages=messages, **create_kwargs)
            m["model_turns"] += 1
            usage = getattr(resp, "usage", None)
            if usage:
                pt = usage.prompt_tokens or 0
                ct = _cached_tokens(usage)
                m["prompt_tokens"] += pt
                m["cached_prompt_tokens"] += ct
                m["completion_tokens"] += usage.completion_tokens or 0
                m["total_tokens"] += usage.total_tokens or 0
                m["per_turn"].append({"prompt": pt, "cached": ct})
                if m["first_call_prompt_tokens"] is None:
                    m["first_call_prompt_tokens"] = pt
            msg = resp.choices[0].message
            assistant: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            messages.append(assistant)
            if not msg.tool_calls:
                break
            for tc in msg.tool_calls:
                m["tool_calls"] += 1
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                content = await executor(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content[:3000]})
    finally:
        await client.close()
    return m


def _make_executors(fs_client, fs_names, router, k_shortlist):
    async def baseline_exec(name, args):
        if name in fs_names:
            try:
                return result_text(await fs_client.call_tool(name, args))
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: {exc}"
        return f"ERROR: tool '{name}' is not available."

    async def find_exec(name, args):
        if name == "find_tools":
            return json.dumps(to_openai_tools(router.route_top_k(str(args.get("query", "")), k=k_shortlist)))
        if name == "call_tool":
            tn = str(args.get("tool_name", ""))
            if tn in fs_names:
                try:
                    return result_text(await fs_client.call_tool(tn, args.get("arguments") or {}))
                except Exception as exc:  # noqa: BLE001
                    return f"ERROR: {exc}"
            return f"ERROR: tool '{tn}' is not available."
        return f"ERROR: unknown tool {name}"

    return baseline_exec, find_exec


def _billable_input(prompt_tokens: int, cached_tokens: int, read_rate: float) -> float:
    """Measured billable input under a cache-read rate: uncached at full price,
    cached served at `read_rate` (write premium ignored — negligible here)."""
    return (prompt_tokens - cached_tokens) + cached_tokens * read_rate


async def main_async(args) -> None:
    paths.ensure_dirs()
    pool = list(json.loads((paths.DATA_DIR / "corpus_multiserver.json").read_text(encoding="utf-8")).values())
    root = Path(tempfile.mkdtemp(prefix="toolfinder-cache-"))
    fs_client = DynamicMCPClient(
        server_name="filesystem", command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(root)],
        startup_timeout_s=90.0, request_timeout_s=45.0,
    )
    make_client = make_client_factory("openai", args.base_url)
    out_path = paths.RESULTS_DIR / "bridge_cache_measured.json"
    results = {"model": args.model, "task": "create->edit->read", "cells": {}}
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("model") == args.model:
            results = prev
            results.setdefault("cells", {})

    try:
        fs_tools = await fs_client.initialize_and_get_tools()
        fs_names = {t["tool_name"] for t in fs_tools}
        task = TASK_TEMPLATE.format(root=root.as_posix())
        print(f"[setup] {len(fs_tools)} filesystem tools; {len(pool)} distractor pool; model={args.model}")

        plan: list[tuple[str, int]] = [("baseline", n) for n in args.sizes if n in BASELINE_SIZES]
        plan += [("find5", n) for n in args.sizes if n in FIND5_SIZES]

        for arm, n in plan:
            key = f"{arm}@{n}"
            if key in results["cells"] and "error" not in results["cells"][key]:
                print(f"[skip] {key} already done (resume)")
                continue
            catalog = build_catalog(fs_tools, pool, n)
            router = make_router(catalog)
            baseline_exec, find_exec = _make_executors(fs_client, fs_names, router, k_shortlist=5)
            reset_sandbox(root)
            print(f"[api] {key} ...")
            try:
                if arm == "baseline":
                    tools = [mcp_to_openai(t) for t in catalog]
                    m = await run_arm_measured(make_client, args.model, BASELINE_SYSTEM, tools, baseline_exec, task)
                else:  # find5
                    tools = [
                        {"type": "function", "function": {"name": "find_tools",
                         "description": "Find the tools relevant to an action; returns their schemas.",
                         "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
                        {"type": "function", "function": {"name": "call_tool",
                         "description": "Execute one tool by name with arguments.",
                         "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["tool_name", "arguments"]}}},
                    ]
                    m = await run_arm_measured(make_client, args.model, TOOLFINDER_SYSTEM, tools, find_exec, task)
                m["task_success"] = verify(root)
                m["n_tools"] = n
                m["arm"] = arm
                m["shortlist_k"] = 5 if arm == "find5" else None
                pt, ct = m["prompt_tokens"], m["cached_prompt_tokens"]
                m["cached_fraction"] = round(ct / pt, 4) if pt else 0.0
                m["billable_input"] = {r: round(_billable_input(pt, ct, rate), 1) for r, rate in CACHE_READ_RATES.items()}
                print(f"      turns={m['model_turns']} prompt={pt} cached={ct} "
                      f"({m['cached_fraction']:.0%}) success={m['task_success']}")
            except Exception as exc:  # noqa: BLE001
                m = {"arm": arm, "n_tools": n, "error": str(exc)[:300]}
                print(f"      [error] {exc}")
            results["cells"][key] = m
            out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    finally:
        await fs_client.close()
        for p in root.glob("*"):
            if p.is_file():
                p.unlink()
        root.rmdir()

    _report(results)
    print(f"\nwrote {out_path}")


def _report(results: dict) -> None:
    cells = results.get("cells", {})
    print("\n" + "=" * 78)
    print(f"MEASURED CACHE + TOP-5  ({results.get('model')})")
    print("=" * 78)
    print(f"{'cell':<14}{'turns':>6}{'prompt':>9}{'cached':>9}{'cached%':>9}{'success':>9}")
    for key in sorted(cells):
        c = cells[key]
        if "error" in c:
            print(f"{key:<14}  ERROR: {c['error'][:50]}")
            continue
        print(f"{key:<14}{c['model_turns']:>6}{c['prompt_tokens']:>9}{c['cached_prompt_tokens']:>9}"
              f"{c['cached_fraction']:>8.0%}{str(c['task_success']):>9}")
    print("\nMeasured billable input (cached served at the listed read rate):")
    for key in sorted(cells):
        c = cells[key]
        if "error" in c:
            continue
        bi = c["billable_input"]
        print(f"  {key:<14} " + "  ".join(f"{r}={bi[r]:,.0f}" for r in bi))
    print("\nNotes: cached% is the API-reported share of input served from the prompt cache.")
    print("Compare baseline cells against the modeled numbers in bridge_cache_aware.json;")
    print("find5 shows whether end-task success holds when the model picks from 5 tools.")


def main() -> None:
    # Load the repo-root .env FIRST so its API_KEY/AGENT_MODEL win (setdefault).
    load_dotenv(TRUE_ROOT / ".env")
    load_dotenv(paths.EXPERIMENTS_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="defaults to AGENT_MODEL/OPENAI_MODEL from .env")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=[14, 60, 120])
    args = parser.parse_args()
    if args.model is None:
        args.model = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or os.environ.get("AGENT_MODEL")
    if not args.model:
        raise SystemExit("no model configured (set AGENT_MODEL in the repo-root .env)")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
