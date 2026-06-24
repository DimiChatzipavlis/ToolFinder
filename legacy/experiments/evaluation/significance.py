"""Paired significance tests: fine-tuned bi-encoder vs BM25, per regime.

Three seeds with means is not statistics; this closes the gap with a paired
bootstrap over test queries (the queries are the random unit, so pairing the
two systems on identical queries removes query-difficulty variance).

For each regime and each fine-tuned seed: resample query indices with
replacement 10,000 times, compute the Recall@1 difference (ft - bm25) per
resample, and report the mean difference, its 95% percentile CI, and the
two-sided bootstrap p-value. Reads per-query ranks from
results/diagnostics/main_eval_per_query.json (no models re-run).

Usage:
    python experiments/evaluation/significance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402

N_RESAMPLES = 10_000
BASELINE = "bm25"
SYSTEM_PREFIX = "ft_minilm_seed"


def hits(per_query: dict[str, int | None], query_ids: list[str]) -> np.ndarray:
    return np.array([1.0 if per_query.get(qid) == 1 else 0.0 for qid in query_ids])


def paired_bootstrap(system_hits: np.ndarray, baseline_hits: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(system_hits)
    indices = rng.integers(0, n, size=(N_RESAMPLES, n))
    deltas = system_hits[indices].mean(axis=1) - baseline_hits[indices].mean(axis=1)
    observed = float(system_hits.mean() - baseline_hits.mean())
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    p_two_sided = 2.0 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))
    return {
        "delta_r1": round(observed, 4),
        "ci95": [round(float(lower), 4), round(float(upper), 4)],
        "p_value": round(max(p_two_sided, 1.0 / N_RESAMPLES), 5),
        "n_queries": n,
    }


def main() -> None:
    paths.ensure_dirs()
    diagnostics = json.loads(
        (paths.DIAGNOSTICS_DIR / "main_eval_per_query.json").read_text(encoding="utf-8")
    )

    output: dict = {"baseline": BASELINE, "n_resamples": N_RESAMPLES, "regimes": {}}
    for regime, block in diagnostics["regimes"].items():
        if BASELINE not in block:
            continue
        baseline_per_query = block[BASELINE]["per_query"]
        query_ids = sorted(baseline_per_query)
        baseline_hits = hits(baseline_per_query, query_ids)

        regime_block: dict = {}
        for system_name, payload in sorted(block.items()):
            if not system_name.startswith(SYSTEM_PREFIX):
                continue
            system_hits = hits(payload["per_query"], query_ids)
            regime_block[system_name] = paired_bootstrap(system_hits, baseline_hits)

        output["regimes"][regime] = regime_block
        for system_name, stats in regime_block.items():
            print(
                f"[{regime}] {system_name} vs {BASELINE}: "
                f"dR@1={stats['delta_r1']:+.4f} CI95={stats['ci95']} p={stats['p_value']}"
            )

    out_path = paths.RESULTS_DIR / "significance.json"
    out_path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
