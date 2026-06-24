"""Step 4 / M1 final: does a smaller operating context *improve* selection?

Cost is not the only claimed benefit of the bridge — narrowing the tool surface
should also help the model *choose correctly* ("lost in the middle"). On the easy
create->edit->read task every arm hit 100%, so that effect is invisible. This
experiment makes it visible by isolating a single decision and stressing it.

Method: for each of 8 filesystem intent probes (intent -> set of acceptable
tools), bind all N tools and force exactly one tool call (`tool_choice=required`),
then check whether the chosen tool is acceptable. Sweep N and compare:

  - a WEAK model (distraction-prone) vs a STRONG model (control),
  - against the router's recall@1 (the bridge's selection, model-independent).

If the weak model's accuracy falls as N grows while the router's holds, that is
the measured "context reduction helps selection" effect — strongest for weak
models, which matches the honest story that for strong models the bridge's value
is cost, not accuracy. A null result (no degradation) is reported as such.

Run (key from repo-root .env; strong model = AGENT_MODEL):
    python legacy/experiments/bridge_selection_accuracy.py
    python legacy/experiments/bridge_selection_accuracy.py --weak gpt-4.1-mini --sizes 14 60 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import paths  # noqa: E402
from experiments.bridge_ab import load_dotenv, make_client_factory, mcp_to_openai  # noqa: E402
from experiments.bridge_scaling import PROBES, build_catalog, make_router  # noqa: E402
from toolfinder.mcp_adapter import DynamicMCPClient  # noqa: E402

TRUE_ROOT = Path(__file__).resolve().parents[2]
SELECT_SYSTEM = (
    "You are a filesystem agent. The user describes ONE action. Call the single "
    "available tool that best performs it. Choose exactly one tool."
)


async def selection_probe(client, model, tools, intent) -> tuple[str | None, int, int]:
    """One forced tool call; return (chosen_tool_name, prompt_tokens, completion_tokens)."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SELECT_SYSTEM}, {"role": "user", "content": intent}],
        tools=tools,
        tool_choice="required",
    )
    msg = resp.choices[0].message
    usage = getattr(resp, "usage", None)
    pt = (usage.prompt_tokens or 0) if usage else 0
    ct = (usage.completion_tokens or 0) if usage else 0
    name = msg.tool_calls[0].function.name if msg.tool_calls else None
    return name, pt, ct


def router_recall_at_1(catalog) -> float:
    router = make_router(catalog)
    correct = 0
    for intent, acceptable in PROBES:
        top = router.route_top_k(intent, k=1)
        if top and top[0].tool_name in acceptable:
            correct += 1
    return round(correct / len(PROBES), 3)


async def main_async(args) -> None:
    paths.ensure_dirs()
    pool = list(json.loads((paths.DATA_DIR / "corpus_multiserver.json").read_text(encoding="utf-8")).values())
    root = Path(tempfile.mkdtemp(prefix="toolfinder-select-"))
    fs_client = DynamicMCPClient(
        server_name="filesystem", command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(root)],
        startup_timeout_s=90.0, request_timeout_s=45.0,
    )
    make_client = make_client_factory("openai", args.base_url)
    models = [("weak", args.weak), ("strong", args.strong)]
    out_path = paths.RESULTS_DIR / "bridge_selection_accuracy.json"
    results = {"models": {tag: name for tag, name in models}, "n_probes": len(PROBES), "cells": {}, "router_recall@1": {}}

    try:
        fs_tools = await fs_client.initialize_and_get_tools()
        print(f"[setup] {len(fs_tools)} filesystem tools; {len(pool)} distractor pool")
        print(f"[setup] weak={args.weak}  strong={args.strong}  sizes={args.sizes}")

        catalogs = {n: build_catalog(fs_tools, pool, n) for n in args.sizes}
        for n in args.sizes:
            results["router_recall@1"][str(n)] = router_recall_at_1(catalogs[n])

        tot_prompt = tot_completion = 0
        for tag, model in models:
            client = make_client()
            try:
                for n in args.sizes:
                    tools = [mcp_to_openai(t) for t in catalogs[n]]
                    correct = 0
                    per_probe = []
                    for intent, acceptable in PROBES:
                        try:
                            name, pt, ct = await selection_probe(client, model, tools, intent)
                            tot_prompt += pt
                            tot_completion += ct
                            ok = name in acceptable
                            correct += int(ok)
                            per_probe.append({"intent": intent, "chosen": name, "ok": ok})
                        except Exception as exc:  # noqa: BLE001
                            per_probe.append({"intent": intent, "error": str(exc)[:160]})
                    acc = round(correct / len(PROBES), 3)
                    results["cells"][f"{tag}@{n}"] = {
                        "model": model, "n_tools": n, "accuracy": acc,
                        "correct": correct, "of": len(PROBES), "probes": per_probe,
                    }
                    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
                    print(f"[{tag}] {model} N={n:>4}  selection acc = {correct}/{len(PROBES)} ({acc:.0%})")
            finally:
                await client.close()
        results["tokens"] = {"prompt": tot_prompt, "completion": tot_completion}
        out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    finally:
        await fs_client.close()
        for p in root.glob("*"):
            if p.is_file():
                p.unlink()
        root.rmdir()

    _report(results, args)
    print(f"\nwrote {out_path}")


def _report(results: dict, args) -> None:
    sizes = args.sizes
    rr = results["router_recall@1"]
    print("\n" + "=" * 70)
    print("SELECTION ACCURACY vs CATALOG SIZE  (8 forced single-tool probes)")
    print("=" * 70)
    header = f"{'N':>5} | {'router R@1':>11} | {'weak (' + args.weak + ')':>22} | {'strong (' + args.strong + ')':>24}"
    print(header)
    for n in sizes:
        w = results["cells"].get(f"weak@{n}", {})
        s = results["cells"].get(f"strong@{n}", {})
        wv = f"{w.get('correct', '?')}/{w.get('of', '?')} ({w.get('accuracy', 0):.0%})" if w else "-"
        sv = f"{s.get('correct', '?')}/{s.get('of', '?')} ({s.get('accuracy', 0):.0%})" if s else "-"
        print(f"{n:>5} | {rr.get(str(n), '?'):>11} | {wv:>22} | {sv:>24}")
    print("\nReading it: the router R@1 is the bridge's selection (independent of the")
    print("agent model). If the WEAK model's accuracy drops as N grows while the router")
    print("holds, that quantifies 'less context -> better selection' — the bridge's")
    print("accuracy value, which is largest for weak models. Flat columns = null result")
    print("(strong models are robust to catalog size; for them the bridge's value is cost).")


def main() -> None:
    load_dotenv(TRUE_ROOT / ".env")
    load_dotenv(paths.EXPERIMENTS_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak", default="gpt-4.1-mini", help="distraction-prone model")
    parser.add_argument("--strong", default=None, help="control model; defaults to AGENT_MODEL")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=[14, 60, 120])
    args = parser.parse_args()
    if args.strong is None:
        args.strong = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or os.environ.get("AGENT_MODEL")
    if not args.strong:
        raise SystemExit("no strong model configured (set AGENT_MODEL in the repo-root .env)")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
