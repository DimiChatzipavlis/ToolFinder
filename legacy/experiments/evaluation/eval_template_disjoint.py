"""Evaluate the template-disjoint control (regime 1b).

Regime 1's scenario-grouped split still shares surface templates between train
and test, so its near-perfect scores upper-bound in-grammar generalization.
This script measures the controlled variant: models retrained on the
doubly-disjoint regime-1b split (templates AND scenarios unseen at test time),
compared against the same training-free baselines on the same test rows — plus
the 1-NN train-anchor leakage probe, which quantifies how much surface overlap
each split leaves (it scores ~0.96 under random splits).

Also evaluates the 1-NN probe on regime 1 for the side-by-side leakage story.

Usage (after training: python experiments/models/biencoder.py --models minilm
       --split-name regime1b_template_disjoint --artifact-root biencoder_r1b
       --results-name biencoder_training_r1b.json):
    python experiments/evaluation/eval_template_disjoint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402
from experiments.baselines import (  # noqa: E402
    Bm25Ranker,
    EncoderRanker,
    NearestTrainAnchorRanker,
    RandomRanker,
    TfidfRanker,
)
from experiments.evaluation import metrics  # noqa: E402
from experiments.evaluation.evaluate import aggregate_seed_groups  # noqa: E402
from experiments.representation import represent_raw  # noqa: E402


def evaluate_split(split_name: str, systems_extra: list, queries, corpus_tools, corpus_texts) -> dict:
    split = json.loads((paths.SPLITS_DIR / f"{split_name}.json").read_text(encoding="utf-8"))
    test_ids = split["test"]
    anchors = queries.loc[test_ids, "anchor"].tolist()
    truths = queries.loc[test_ids, "tool"].tolist()
    train_anchors = queries.loc[split["train"], "anchor"].tolist()
    train_labels = queries.loc[split["train"], "tool"].tolist()

    systems = [
        RandomRanker(corpus_tools, seed=0),
        NearestTrainAnchorRanker(corpus_tools, train_anchors, train_labels),
        Bm25Ranker(corpus_tools, corpus_texts),
        TfidfRanker(corpus_tools, corpus_texts, analyzer="char_wb", ngram_range=(3, 5)),
        *systems_extra,
    ]

    block: dict[str, dict] = {}
    for system in systems:
        rankings = (
            system.rank_batch(anchors)
            if hasattr(system, "rank_batch")
            else [system.rank(anchor) for anchor in anchors]
        )
        block[system.name] = metrics.summarize(rankings, truths)
        print(
            f"  [{split_name}] {system.name:24s} "
            f"R@1={block[system.name]['recall@1']['mean']:.4f} "
            f"MRR={block[system.name]['mrr']['mean']:.4f}"
        )
    block.update(aggregate_seed_groups(block))
    return {"n_test": len(test_ids), "systems": block}


def main() -> None:
    paths.ensure_dirs()
    queries = pd.read_csv(paths.QUERIES_CSV).set_index("query_id")
    corpus = json.loads(paths.CORPUS_JSON.read_text(encoding="utf-8"))
    corpus_tools = sorted(corpus)
    corpus_texts = [represent_raw(corpus[tool]["schema"]) for tool in corpus_tools]

    r1b_systems = []
    root = paths.ARTIFACTS_DIR / "biencoder_r1b" / "minilm"
    if root.exists():
        for seed_dir in sorted(root.iterdir()):
            final = seed_dir / "final"
            if (final / "config.json").exists():
                name = f"ft_minilm_r1b_{seed_dir.name}"
                print(f"[load] {name}")
                r1b_systems.append(EncoderRanker(name, str(final), corpus_tools, corpus_texts))

    frozen = EncoderRanker(
        "frozen_minilm", "sentence-transformers/all-MiniLM-L6-v2", corpus_tools, corpus_texts
    )

    output = {
        "purpose": "template-disjoint control for regime-1 saturation",
        "regime1b_template_disjoint": evaluate_split(
            "regime1b_template_disjoint", [frozen, *r1b_systems], queries, corpus_tools, corpus_texts
        ),
        "regime1_leakage_probe": evaluate_split(
            "regime1_unseen_queries", [], queries, corpus_tools, corpus_texts
        ),
    }

    out_path = paths.RESULTS_DIR / "template_disjoint_eval.json"
    out_path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
