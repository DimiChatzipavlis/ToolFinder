"""Strict, visible A/B: the ToolFinder gateway vs a plain LLM that binds all tools.

Same model, same real filesystem task (create -> edit -> read, verified on disk),
the catalog padded with real distractor tools to size N. The only thing that
changes is how the model is given tools:

  baseline   : bind all N tool schemas directly      (the simple LLM-orchestrated way)
  find_call  : bind only find_tools + call_tool       (the gateway, schema-aware)
  single     : bind only route_and_call               (the gateway, one hop)

For each (N, arm) we run `--repeats` trials and report task-success rate, mean
total tokens, and mean model turns — so we can see whether the gateway actually
does better, and where (cost vs accuracy), rather than asserting it.

Run (uses API_KEY from the repo-root .env):
  python research/experiments/eval_gateway_vs_baseline.py
  python research/experiments/eval_gateway_vs_baseline.py --model gpt-4.1-mini --sizes 15 60 120 --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # research/

from experiments import paths  # noqa: E402
from experiments.bridge_ab import load_dotenv, make_client_factory  # noqa: E402
from experiments.bridge_scaling import api_arm, build_catalog, make_router  # noqa: E402
from toolfinder.mcp_adapter import DynamicMCPClient  # noqa: E402

TRUE_ROOT = Path(__file__).resolve().parents[2]
ARMS = ("baseline", "find_call", "single")
LABEL = {"baseline": "baseline (all tools)", "find_call": "gateway find+call", "single": "gateway route_and_call"}


async def main_async(args) -> None:
    paths.ensure_dirs()
    pool = list(json.loads((paths.DATA_DIR / "corpus_multiserver.json").read_text(encoding="utf-8")).values())
    root = Path(tempfile.mkdtemp(prefix="toolfinder-eval-"))
    fs_client = DynamicMCPClient(server_name="filesystem", command="npx",
                                 args=["-y", "@modelcontextprotocol/server-filesystem", str(root)],
                                 startup_timeout_s=90.0, request_timeout_s=45.0)
    make_client = make_client_factory("openai", None)
    results: dict = {"model": args.model, "repeats": args.repeats, "sizes": {}}

    try:
        fs_tools = await fs_client.initialize_and_get_tools()
        fs_names = {t["tool_name"] for t in fs_tools}
        print(f"[setup] model={args.model} repeats={args.repeats} sizes={args.sizes}; "
              f"{len(fs_tools)} filesystem tools + distractors -> N")

        for n in args.sizes:
            catalog = build_catalog(fs_tools, pool, n)
            router = make_router(catalog)
            results["sizes"][str(n)] = {}
            for arm in ARMS:
                succ, toks, turns = 0, [], []
                for rep in range(args.repeats):
                    m = await api_arm(make_client, args.model, arm, catalog, fs_client, fs_names, router, root, None)
                    if "error" in m:
                        print(f"  [error] N={n} {arm} rep{rep}: {m['error'][:120]}")
                        continue
                    succ += int(m["task_success"])
                    toks.append(m["total_tokens"])
                    turns.append(m["model_turns"])
                cell = {"success": f"{succ}/{args.repeats}",
                        "mean_total_tokens": round(statistics.mean(toks)) if toks else None,
                        "mean_turns": round(statistics.mean(turns), 1) if turns else None}
                results["sizes"][str(n)][arm] = cell
                print(f"  N={n:>4} {LABEL[arm]:24s} success={cell['success']} "
                      f"tokens={cell['mean_total_tokens']} turns={cell['mean_turns']}")
            out = paths.RESULTS_DIR / "eval_gateway_vs_baseline.json"
            out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    finally:
        await fs_client.close()
        for p in root.glob("*"):
            if p.is_file():
                p.unlink()
        root.rmdir()

    _report(results, args)


def _report(results: dict, args) -> None:
    print("\n" + "=" * 84)
    print(f"GATEWAY vs SIMPLE LLM ORCHESTRATION  ({results['model']}, {results['repeats']} trials/cell)")
    print("=" * 84)
    print(f"{'N':>5} | {'metric':14s} | {'baseline':>12} | {'gateway f+c':>12} | {'gateway r&c':>12}")
    for n in args.sizes:
        c = results["sizes"][str(n)]
        b, f, s = c["baseline"], c["find_call"], c["single"]
        print(f"{n:>5} | {'success':14s} | {b['success']:>12} | {f['success']:>12} | {s['success']:>12}")
        print(f"{'':>5} | {'tokens (mean)':14s} | {str(b['mean_total_tokens']):>12} | "
              f"{str(f['mean_total_tokens']):>12} | {str(s['mean_total_tokens']):>12}")
        # token ratio baseline / gateway (find+call)
        if b["mean_total_tokens"] and f["mean_total_tokens"]:
            print(f"{'':>5} | {'tok ratio b/f':14s} | {'1.0x':>12} | "
                  f"{b['mean_total_tokens'] / f['mean_total_tokens']:>11.1f}x | "
                  f"{(b['mean_total_tokens'] / s['mean_total_tokens']) if s['mean_total_tokens'] else float('nan'):>11.1f}x")
        print(f"{'':>5} | {'':14s} | {'':>12} | {'':>12} | {'':>12}")
    print("Read it: if success is comparable but gateway tokens are far lower (esp. as N grows),")
    print("the gateway wins on cost. If baseline success DROPS at large N while the gateway holds,")
    print("the gateway also wins on accuracy (the model stops drowning in tools). Single trial counts")
    print(f"are n={args.repeats}; treat small success differences as noise.")


def main() -> None:
    load_dotenv(TRUE_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--sizes", nargs="+", type=int, default=[15, 60, 120])
    parser.add_argument("--repeats", type=int, default=3)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
