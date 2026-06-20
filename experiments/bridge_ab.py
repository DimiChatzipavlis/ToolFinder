"""A/B test: a plain agent vs a ToolFinder-bridged agent over a filesystem MCP server.

Both agents run the identical task (create -> edit -> read a file) against the same
real `@modelcontextprotocol/server-filesystem` instance. The only difference is
what tools the LLM sees:

  baseline    : all ~14 filesystem tools bound directly.
  toolfinder  : just two meta-tools — find_tools(query) (the dense router, returns
                the top-k relevant filesystem schemas) and call_tool(name, args)
                (proxy execution). The model discovers, then calls.

We measure, per arm: the token-weight of the bound tool schemas (deterministic
context cost), the API-reported prompt/completion/total tokens over the whole
task, the number of model round-trips, and task success (verified on disk).

Backend-agnostic via the OpenAI SDK: point it at OpenAI or at a local Ollama
(`/v1`) server. The numbers below were produced locally on Ollama; rerun with
`--backend openai` for GPT.

Usage:
    # local proof (no API key, uses the running Ollama):
    python experiments/bridge_ab.py --backend ollama --model llama3.2 --repeats 3
    # GPT (set OPENAI_API_KEY first):
    python experiments/bridge_ab.py --backend openai --model gpt-4.1-mini --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import paths  # noqa: E402
from toolfinder import UniversalMCPRouter, to_openai_tools  # noqa: E402
from toolfinder.mcp_adapter import DynamicMCPClient  # noqa: E402

ENC = tiktoken.get_encoding("o200k_base")
MAX_TURNS = 14
TASK_TEMPLATE = (
    "You are operating inside the directory {root} (the only directory you may touch). "
    "Do these three steps in order, using the tools: "
    "1) create a text file {root}/notes.txt whose contents are exactly: hello "
    "2) edit that file so its contents become exactly: hello world "
    "3) read the file back and report its final contents. "
    "When the file says 'hello world' and you have read it, reply DONE."
)

BASELINE_SYSTEM = (
    "You are a filesystem agent. Use the provided tools to complete the task. "
    "Always pass absolute paths inside the allowed directory. Take one tool action at a time."
)
TOOLFINDER_SYSTEM = (
    "You are a filesystem agent with a router. You have exactly two tools:\n"
    "- find_tools(query): returns the filesystem tools relevant to a described action.\n"
    "- call_tool(tool_name, arguments): executes one filesystem tool.\n"
    "For each action: first call find_tools with a short description, then call_tool with the "
    "chosen tool_name and its arguments. Always use absolute paths inside the allowed directory."
)


def tokens(obj: Any) -> int:
    return len(ENC.encode(json.dumps(obj, ensure_ascii=False)))


def mcp_to_openai(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["tool_name"],
            "description": (tool.get("description") or "")[:1024],
            "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
        },
    }


def result_text(result: dict) -> str:
    content = result.get("content") if isinstance(result, dict) else None
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if isinstance(c, dict)]
        return "\n".join(p for p in parts if p)[:2000]
    return json.dumps(result)[:2000]


async def run_arm(make_client, model, system, tools, executor, task, temperature=None) -> dict:
    """Generic agent loop. Returns token/turn metrics."""
    client = make_client()
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    metrics = {
        "schema_tokens": tokens(tools),
        "n_bound_tools": len(tools),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "first_call_prompt_tokens": None,
        "model_turns": 0,
        "tool_calls": 0,
    }
    create_kwargs: dict = {"model": model, "tools": tools, "tool_choice": "auto"}
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    for _ in range(MAX_TURNS):
        resp = await client.chat.completions.create(messages=messages, **create_kwargs)
        metrics["model_turns"] += 1
        usage = getattr(resp, "usage", None)
        if usage:
            metrics["prompt_tokens"] += usage.prompt_tokens or 0
            metrics["completion_tokens"] += usage.completion_tokens or 0
            metrics["total_tokens"] += usage.total_tokens or 0
            if metrics["first_call_prompt_tokens"] is None:
                metrics["first_call_prompt_tokens"] = usage.prompt_tokens or 0
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
            metrics["tool_calls"] += 1
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            content = await executor(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content[:3000]})
    await client.close()
    return metrics


async def one_trial(make_client, model, fs_client, fs_tools, router, root: Path, temperature=None) -> dict:
    task = TASK_TEMPLATE.format(root=root.as_posix())

    # ---- baseline: all filesystem tools bound directly ----
    baseline_tools = [mcp_to_openai(t) for t in fs_tools]

    async def baseline_exec(name: str, args: dict) -> str:
        try:
            return result_text(await fs_client.call_tool(name, args))
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    reset_sandbox(root)
    baseline = await run_arm(make_client, model, BASELINE_SYSTEM, baseline_tools, baseline_exec, task, temperature)
    baseline["task_success"] = verify(root)

    # ---- toolfinder: two meta-tools ----
    tf_tools = [
        {"type": "function", "function": {
            "name": "find_tools",
            "description": "Find the filesystem tools relevant to an action. Returns their schemas.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "call_tool",
            "description": "Execute one filesystem tool by name with its arguments.",
            "parameters": {"type": "object", "properties": {
                "tool_name": {"type": "string"},
                "arguments": {"type": "object"}}, "required": ["tool_name", "arguments"]}}},
    ]

    async def tf_exec(name: str, args: dict) -> str:
        if name == "find_tools":
            results = router.route_top_k(str(args.get("query", "")), k=3)
            return json.dumps(to_openai_tools(results))
        if name == "call_tool":
            try:
                return result_text(await fs_client.call_tool(str(args.get("tool_name", "")), args.get("arguments") or {}))
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: {exc}"
        return f"ERROR: unknown tool {name}"

    reset_sandbox(root)
    toolfinder = await run_arm(make_client, model, TOOLFINDER_SYSTEM, tf_tools, tf_exec, task, temperature)
    toolfinder["task_success"] = verify(root)

    return {"baseline": baseline, "toolfinder": toolfinder}


def reset_sandbox(root: Path) -> None:
    for p in root.glob("*"):
        if p.is_file():
            p.unlink()


def verify(root: Path) -> bool:
    target = root / "notes.txt"
    return target.exists() and target.read_text(encoding="utf-8").strip() == "hello world"


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Does not overwrite existing env vars
    and never logs values."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def make_client_factory(backend: str, base_url: str | None):
    from openai import AsyncOpenAI

    if backend == "ollama":
        return lambda: AsyncOpenAI(base_url=base_url or "http://localhost:11434/v1", api_key="ollama")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    if not key:
        raise SystemExit("API key not found (checked OPENAI_API_KEY / API_KEY in .env and environment).")
    return lambda: AsyncOpenAI(api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL"))


async def selftest(fs_client, fs_tools, router, root: Path) -> None:
    """Drive the task directly (no LLM) to prove the harness can complete it and
    that the router routes the three steps correctly — isolates harness
    correctness from model capability."""
    names = {t["tool_name"] for t in fs_tools}
    write = "write_file" if "write_file" in names else next(n for n in names if "write" in n)
    read = "read_text_file" if "read_text_file" in names else next(n for n in names if "read" in n)
    target = (root / "notes.txt").as_posix()

    await fs_client.call_tool(write, {"path": target, "content": "hello"})
    await fs_client.call_tool(write, {"path": target, "content": "hello world"})
    back = result_text(await fs_client.call_tool(read, {"path": target}))
    print(f"[selftest] wrote+overwrote+read; file reads: {back!r} | verify()={verify(root)}")

    for intent in ("create a file with some text", "change the contents of a file", "read a file's contents"):
        top = router.route_top_k(intent, k=3)
        print(f"[selftest] route {intent!r:38s} -> {[r.tool_name for r in top]}")


async def main_async(args) -> None:
    paths.ensure_dirs()
    root = Path(tempfile.mkdtemp(prefix="toolfinder-bridge-"))
    fs_client = DynamicMCPClient(
        server_name="filesystem", command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(root)],
        startup_timeout_s=90.0, request_timeout_s=45.0,
    )
    try:
        fs_tools = await fs_client.initialize_and_get_tools()
        print(f"[setup] filesystem server exposes {len(fs_tools)} tools; sandbox={root}")
        router = UniversalMCPRouter(model_name="sentence-transformers/all-MiniLM-L6-v2")
        router.ingest_server("filesystem", fs_tools)

        if args.selftest:
            await selftest(fs_client, fs_tools, router, root)
            return

        make_client = make_client_factory(args.backend, args.base_url)
        temperature = 0 if args.backend == "ollama" else None
        trials = []
        for i in range(args.repeats):
            print(f"[trial {i + 1}/{args.repeats}] running both arms ({args.model})...")
            trials.append(await one_trial(make_client, args.model, fs_client, fs_tools, router, root, temperature))
    finally:
        await fs_client.close()
        for p in root.glob("*"):
            if p.is_file():
                p.unlink()
        root.rmdir()

    summary = summarize(trials, args, len(fs_tools))
    out = paths.RESULTS_DIR / "bridge_ab.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_summary(summary)
    print(f"\nwrote {out}")


def summarize(trials, args, n_fs_tools) -> dict:
    def agg(arm, key):
        vals = [t[arm][key] for t in trials if t[arm][key] is not None]
        return round(statistics.mean(vals), 1) if vals else None

    arms = {}
    for arm in ("baseline", "toolfinder"):
        arms[arm] = {
            "n_bound_tools": trials[0][arm]["n_bound_tools"],
            "schema_tokens": trials[0][arm]["schema_tokens"],
            "first_call_prompt_tokens": agg(arm, "first_call_prompt_tokens"),
            "total_prompt_tokens": agg(arm, "prompt_tokens"),
            "total_completion_tokens": agg(arm, "completion_tokens"),
            "total_tokens": agg(arm, "total_tokens"),
            "model_turns": agg(arm, "model_turns"),
            "tool_calls": agg(arm, "tool_calls"),
            "task_success_rate": round(sum(t[arm]["task_success"] for t in trials) / len(trials), 2),
        }
    return {
        "backend": args.backend, "model": args.model, "repeats": args.repeats,
        "n_filesystem_tools": n_fs_tools, "arms": arms,
    }


def print_summary(s) -> None:
    print("\n" + "=" * 72)
    print(f"BRIDGE A/B — {s['model']} via {s['backend']} | {s['n_filesystem_tools']} filesystem tools | {s['repeats']} trials")
    print("=" * 72)
    b, t = s["arms"]["baseline"], s["arms"]["toolfinder"]
    rows = [
        ("tools bound to the LLM", b["n_bound_tools"], t["n_bound_tools"]),
        ("tool-schema tokens (context weight)", b["schema_tokens"], t["schema_tokens"]),
        ("1st-call prompt tokens", b["first_call_prompt_tokens"], t["first_call_prompt_tokens"]),
        ("total prompt tokens", b["total_prompt_tokens"], t["total_prompt_tokens"]),
        ("total tokens", b["total_tokens"], t["total_tokens"]),
        ("model round-trips", b["model_turns"], t["model_turns"]),
        ("task success rate", b["task_success_rate"], t["task_success_rate"]),
    ]
    print(f"{'metric':40s} {'baseline':>14s} {'toolfinder':>14s}")
    for name, bv, tv in rows:
        print(f"{name:40s} {str(bv):>14s} {str(tv):>14s}")


def main() -> None:
    load_dotenv(paths.EXPERIMENTS_DIR / ".env")
    load_dotenv(paths.REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["ollama", "openai"], default="openai")
    parser.add_argument("--model", default=None, help="overrides OPENAI_MODEL/MODEL from .env")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--selftest", action="store_true", help="drive the task without an LLM to verify harness correctness")
    args = parser.parse_args()
    if args.model is None:
        args.model = (
            os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or os.environ.get("AGENT_MODEL")
            or ("llama3.2" if args.backend == "ollama" else "gpt-4.1-mini")
        )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
