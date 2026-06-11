"""Ablation: which schema-to-text representation should be embedded?

Conditions: raw canonical JSON, minified (property descriptions stripped),
name+description only, description only. Evaluated on both regimes for the
lexical systems and frozen/fine-tuned encoders.

Caveat recorded in the report: the fine-tuned bi-encoder was trained on `raw`
documents, so for it this ablation measures inference-time representation
robustness, not per-representation training quality.

Usage:
    python experiments/ablation_representation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402
from experiments.baselines import Bm25Ranker, EncoderRanker, TfidfRanker  # noqa: E402
from experiments.evaluation import metrics  # noqa: E402
from experiments.evaluation.evaluate import REGIMES  # noqa: E402
from experiments.evaluation.ood import best_finetuned_artifact  # noqa: E402
from experiments.representation import REPRESENTATIONS  # noqa: E402


def main() -> None:
    paths.ensure_dirs()
    queries = pd.read_csv(paths.QUERIES_CSV).set_index("query_id")
    corpus = json.loads(paths.CORPUS_JSON.read_text(encoding="utf-8"))
    corpus_tools = sorted(corpus)
    splits = {
        regime: json.loads((paths.SPLITS_DIR / f"{regime}.json").read_text(encoding="utf-8"))
        for regime in REGIMES
    }
    ft_name, ft_path = best_finetuned_artifact()

    output: dict = {"finetuned_system": ft_name, "representations": {}}
    for representation_name, render in REPRESENTATIONS.items():
        corpus_texts = [render(corpus[tool]["schema"]) for tool in corpus_tools]
        systems = [
            Bm25Ranker(corpus_tools, corpus_texts),
            TfidfRanker(corpus_tools, corpus_texts, analyzer="char_wb", ngram_range=(3, 5)),
            EncoderRanker("frozen_mpnet", "sentence-transformers/all-mpnet-base-v2", corpus_tools, corpus_texts),
            EncoderRanker(ft_name, ft_path, corpus_tools, corpus_texts),
        ]
        block: dict = {}
        for regime in REGIMES:
            test_ids = splits[regime]["test"]
            anchors = queries.loc[test_ids, "anchor"].tolist()
            truths = queries.loc[test_ids, "tool"].tolist()
            regime_block = {}
            for system in systems:
                rankings = (
                    system.rank_batch(anchors)
                    if hasattr(system, "rank_batch")
                    else [system.rank(anchor) for anchor in anchors]
                )
                ranks = metrics.ranks_from_rankings(rankings, truths)
                regime_block[system.name] = {
                    "recall@1": round(float(metrics.recall_at_k(ranks, 1).mean()), 4),
                    "recall@3": round(float(metrics.recall_at_k(ranks, 3).mean()), 4),
                    "mrr": round(float(metrics.reciprocal_rank(ranks).mean()), 4),
                }
            block[regime] = regime_block
        output["representations"][representation_name] = block
        print(f"[{representation_name}] done")

    out_path = paths.RESULTS_DIR / "ablation_representation.json"
    out_path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
