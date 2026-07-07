"""Scaled-up, confusable selection benchmark — does the gateway beat
bind-everything on ACCURACY when the catalog gets large and tools get similar?

Catalog: the 574-tool multi-server pool (24 real APIs — github, asana, box,
plaid, …; genuinely confusable). Queries: regime4 held-out-server test set.
For each catalog size N and each query we force the model to pick ONE tool and
check it against the gold label:

  baseline : bind ALL N tools, model picks                 (simple LLM orchestration)
  gateway  : router → top-5 shortlist (rerank on), model picks  (ToolFinder)

We also log router recall@1/@5 (model-independent) and whether binding N tools is
even possible (large N can exceed the API's tool-count / context limits — itself
a reason to wrap behind a gateway). Weak model (gpt-4.1-mini) on purpose: that is
where "lost among many tools" actually bites.

Run (uses API_KEY from the repo-root .env):
  python research/experiments/eval_selection_at_scale.py
  python research/experiments/eval_selection_at_scale.py --sizes 60 250 574 --queries 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # research/

from experiments import paths  # noqa: E402
from experiments.bridge_ab import load_dotenv, make_client_factory  # noqa: E402
from experiments.bridge_scaling import clean_parameters, sanitize  # noqa: E402
from experiments.bridge_selection_accuracy import selection_probe  # noqa: E402
from toolfinder import UniversalMCPRouter  # noqa: E402
from toolfinder.dynamic_faiss_router import RouterHyperparameters  # noqa: E402

TRUE_ROOT = Path(__file__).resolve().parents[2]
BIENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DATA = paths.DATA_DIR


def oai_tool(name: str, description: str, input_schema: dict) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": (description or "")[:1024], "parameters": clean_parameters(input_schema)}}


def load_data(n_queries: int):
    corpus = json.loads((DATA / "corpus_multiserver.json").read_text(encoding="utf-8"))
    import csv
    rows = list(csv.DictReader((DATA / "queries_multiserver.csv").open(encoding="utf-8")))
    test_ids = set(json.loads((DATA / "splits" / "regime4_multiserver.json").read_text(encoding="utf-8"))["test"])
    queries = [(r["anchor"], r["tool"]) for r in rows if r["query_id"] in test_ids and r["tool"] in corpus]
    return corpus, queries[:n_queries]


def build_catalog(corpus: dict, gold_keys: list[str], n: int, rng: random.Random):
    used: set[str] = set()
    keys = list(dict.fromkeys(gold_keys))                      # the golds first (always present)
    others = [k for k in corpus if k not in set(keys)]
    rng.shuffle(others)
    keys = (keys + others)[:max(n, len(keys))]
    catalog, sani = [], {}
    for k in keys:
        s = corpus[k]["schema"]
        name = sanitize(k, used)
        sani[k] = name
        catalog.append({"tool_name": name, "description": s.get("description", ""),
                        "inputSchema": clean_parameters(s.get("inputSchema", {}))})
    return catalog, sani


async def main_async(args) -> None:
    corpus, queries = load_data(args.queries)
    gold_keys = [g for _, g in queries]
    print(f"[setup] pool={len(corpus)} tools; {len(queries)} confusable test queries; model={args.model}")
    make_client = make_client_factory("openai", None)
    client = make_client()
    results: dict = {"model": args.model, "n_queries": len(queries), "sizes": {}}

    try:
        for n in args.sizes:
            rng = random.Random(7)
            catalog, sani = build_catalog(corpus, gold_keys, n, rng)
            baseline_tools = [oai_tool(c["tool_name"], c["description"], c["inputSchema"]) for c in catalog]
            router = UniversalMCPRouter(model_name=BIENCODER,
                                        config=RouterHyperparameters(min_cosine_similarity=-1.0, rerank=True, rerank_pool=20))
            router.ingest_server("pool", catalog)

            base_ok = base_err = gate_ok = r1 = r5 = 0
            for anchor, gold in queries:
                gold_name = sani[gold]
                # gateway: router -> top-5 -> model picks from the shortlist
                shortlist = router.route_top_k(anchor, k=5)
                names5 = [r.tool_name for r in shortlist]
                r1 += int(bool(names5) and names5[0] == gold_name)
                r5 += int(gold_name in names5)
                gw_tools = [oai_tool(r.tool_name, r.schema.get("description", ""), r.schema.get("inputSchema", {})) for r in shortlist]
                try:
                    pick, _, _ = await selection_probe(client, args.model, gw_tools, anchor)
                    gate_ok += int(pick == gold_name)
                except Exception:  # noqa: BLE001
                    pass
                # baseline: bind ALL N tools, model picks
                try:
                    pick, _, _ = await selection_probe(client, args.model, baseline_tools, anchor)
                    base_ok += int(pick == gold_name)
                except Exception as exc:  # noqa: BLE001 - large N may exceed API tool/context limits
                    base_err += 1
                    if base_err == 1:
                        print(f"  [baseline N={n}] bind error (counts as failure): {str(exc)[:90]}")
            q = len(queries)
            cell = {
                "baseline_acc": round(base_ok / q, 3), "baseline_bind_errors": base_err,
                "gateway_acc": round(gate_ok / q, 3),
                "router_recall@1": round(r1 / q, 3), "router_recall@5": round(r5 / q, 3),
            }
            results["sizes"][str(n)] = cell
            router.teardown()
            print(f"  N={n:>4}  baseline={cell['baseline_acc']:.2f} (errors {base_err})  "
                  f"gateway={cell['gateway_acc']:.2f}  router R@1={cell['router_recall@1']:.2f} R@5={cell['router_recall@5']:.2f}")
            (paths.RESULTS_DIR / "eval_selection_at_scale.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    finally:
        await client.close()

    print("\n" + "=" * 78)
    print(f"SELECTION ACCURACY AT SCALE  ({args.model}, {len(queries)} confusable queries)")
    print("=" * 78)
    print(f"{'N tools':>8} | {'baseline (bind all)':>20} | {'gateway (top-5)':>16} | {'router R@1':>10} | {'R@5':>6}")
    for n in args.sizes:
        c = results["sizes"][str(n)]
        berr = f" !{c['baseline_bind_errors']}err" if c["baseline_bind_errors"] else ""
        print(f"{n:>8} | {c['baseline_acc']:>19.2f}{berr:>0} | {c['gateway_acc']:>16.2f} | "
              f"{c['router_recall@1']:>10.2f} | {c['router_recall@5']:>6.2f}")
    print("\nIf baseline accuracy falls as N grows (or it can't bind N at all) while the")
    print("gateway holds, ToolFinder wins on ACCURACY at scale — not just tokens.")


def main() -> None:
    load_dotenv(TRUE_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--sizes", nargs="+", type=int, default=[60, 250, 574])
    parser.add_argument("--queries", type=int, default=24)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
