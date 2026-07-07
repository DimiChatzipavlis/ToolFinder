"""P0 — which encoder configuration restores routing quality at scale?

The measured bottleneck (eval_selection_at_scale.py): with the stock zero-shot
MiniLM the router's recall@1 on confusable held-out-server queries degrades
0.88 -> 0.46 going 60 -> 574 tools — and that was *with* the cross-encoder
reranker already on. So rerank alone does not fix scale; the encoder is the
lever. This eval measures the grid, purely locally ($0 API):

    encoder ∈ {stock MiniLM, fine-tuned (GitHub-only), fine-tuned (multi-server)}
    × rerank ∈ {off, on}
    × N ∈ {60, 250, 574}

on all 60 regime4 test queries (their gold tools live on servers *never seen in
training* — the honest generalization setting). Metrics: recall@1/@5 and MRR@5.
Deterministic (fixed encoders, fixed seed for distractor sampling): re-runs
reproduce exactly.

Run:  python legacy/experiments/eval_encoder_at_scale.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))  # legacy/

from experiments import paths  # noqa: E402
from experiments.eval_selection_at_scale import build_catalog, load_data  # noqa: E402
from toolfinder import UniversalMCPRouter  # noqa: E402
from toolfinder.dynamic_faiss_router import RouterHyperparameters  # noqa: E402

ARTIFACTS = HERE / "artifacts"
ENCODERS = [
    ("stock", "sentence-transformers/all-MiniLM-L6-v2"),
    ("ft-github", str(ARTIFACTS / "biencoder" / "minilm" / "seed42" / "final")),
    ("ft-multisrv", str(ARTIFACTS / "biencoder_multiserver" / "minilm" / "seed42" / "final")),
]
SIZES = [60, 250, 574]
K = 5


def metrics(ranks: list[int], n: int) -> dict[str, float]:
    return {
        "recall@1": round(sum(r == 0 for r in ranks) / n, 3),
        "recall@5": round(sum(0 <= r < K for r in ranks) / n, 3),
        "mrr@5": round(sum(1.0 / (r + 1) for r in ranks if 0 <= r < K) / n, 3),
    }


def main() -> None:
    corpus, queries = load_data(60)
    gold_keys = [g for _, g in queries]
    print(f"[setup] pool={len(corpus)} tools; {len(queries)} held-out-server queries; grid="
          f"{[e[0] for e in ENCODERS]} x rerank(off/on) x N={SIZES}\n")

    results: dict = {"n_queries": len(queries), "grid": {}}
    for n in SIZES:
        catalog, sani = build_catalog(corpus, gold_keys, n, random.Random(7))
        golds = [sani[g] for g in gold_keys]
        for enc_label, model_name in ENCODERS:
            if "/" not in model_name.replace("\\", "/") or (Path(model_name).is_absolute() and not Path(model_name).exists()):
                print(f"  [skip] {enc_label}: artifact missing at {model_name}")
                continue
            for rerank in (False, True):
                label = f"{enc_label}|rerank={'on' if rerank else 'off'}|N={n}"
                config = RouterHyperparameters(min_cosine_similarity=-1.0, rerank=rerank, rerank_pool=20)
                router = UniversalMCPRouter(model_name=model_name, config=config)
                router.ingest_server("pool", catalog)
                ranks = []
                for (anchor, _), gold_name in zip(queries, golds):
                    names = [r.tool_name for r in router.route_top_k(anchor, k=K)]
                    ranks.append(names.index(gold_name) if gold_name in names else -1)
                router.teardown()
                m = metrics(ranks, len(queries))
                results["grid"][label] = m
                print(f"  {label:34s} R@1={m['recall@1']:.3f}  R@5={m['recall@5']:.3f}  MRR@5={m['mrr@5']:.3f}")
        (paths.RESULTS_DIR / "eval_encoder_at_scale.json").write_text(
            json.dumps(results, indent=1), encoding="utf-8")

    print("\n" + "=" * 78)
    print("P0 GRID — recall@1 by encoder x rerank x catalog size (60 held-out queries)")
    print("=" * 78)
    header = f"{'encoder':<12} {'rerank':<7}" + "".join(f"{('N=' + str(n)):>10}" for n in SIZES)
    print(header)
    for enc_label, _ in ENCODERS:
        for rr in ("off", "on"):
            row = f"{enc_label:<12} {rr:<7}"
            for n in SIZES:
                m = results["grid"].get(f"{enc_label}|rerank={rr}|N={n}")
                row += f"{(m['recall@1'] if m else float('nan')):>10.3f}"
            print(row)
    print("\nwrote", paths.RESULTS_DIR / "eval_encoder_at_scale.json")
    print("Read it: the winning row is the P0 recommendation (what to point TOOLFINDER_MODEL at,")
    print("and whether rerank should be on) for large confusable catalogs.")


if __name__ == "__main__":
    main()
