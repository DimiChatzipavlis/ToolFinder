"""Open-set rejection analysis: can the similarity score tell ID from OOD?

In-distribution = regime-1 test queries (unseen phrasings of catalog tools).
Out-of-distribution = chitchat, out-of-catalog, adversarial near-miss sets.

For each scoring rule (top-1 cosine, top1-top2 margin) and each system:
  - AUROC of ID vs OOD separation (pooled and per OOD subset)
  - FPR@95TPR: fraction of OOD accepted when the threshold keeps 95% of ID
  - Risk-coverage sweep over the threshold tau: coverage and selective risk on
    ID queries, plus per-subset OOD acceptance, at every operating point.

This turns the router's `min_cosine_similarity` from a magic constant into a
measured operating point.

Usage:
    python experiments/evaluation/ood.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402
from experiments.baselines import EncoderRanker  # noqa: E402
from experiments.representation import represent_raw  # noqa: E402


def best_finetuned_artifact() -> tuple[str, str]:
    """Pick the trained bi-encoder with the highest validation MRR."""
    records = json.loads((paths.RESULTS_DIR / "biencoder_training.json").read_text(encoding="utf-8"))
    best = max(records, key=lambda r: r["final_val_mrr@10"])
    name = f"ft_{best['model_key']}_seed{best['seed']}"
    return name, str(paths.REPO_ROOT / best["artifact"])


def load_ood_frame() -> pd.DataFrame:
    frames = [
        pd.read_csv(paths.OOD_DIR / "chitchat.csv"),
        pd.read_csv(paths.OOD_DIR / "out_of_catalog.csv"),
        pd.read_csv(paths.OOD_DIR / "adversarial_near_miss.csv"),
    ]
    return pd.concat(frames, ignore_index=True)


def score_queries(system: EncoderRanker, queries: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (top1_score, margin, top1_index) for each query."""
    scores = system.scores_batch(queries)
    order = np.argsort(-scores, axis=1, kind="stable")
    top1 = scores[np.arange(len(queries)), order[:, 0]]
    top2 = scores[np.arange(len(queries)), order[:, 1]]
    return top1, top1 - top2, order[:, 0]


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    return float(roc_auc_score(labels, np.concatenate([id_scores, ood_scores])))


def fpr_at_95_tpr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    threshold = float(np.quantile(id_scores, 0.05))
    return float((ood_scores >= threshold).mean())


def main() -> None:
    paths.ensure_dirs()
    queries = pd.read_csv(paths.QUERIES_CSV).set_index("query_id")
    split = json.loads(
        (paths.SPLITS_DIR / "regime1_unseen_queries.json").read_text(encoding="utf-8")
    )
    corpus = json.loads(paths.CORPUS_JSON.read_text(encoding="utf-8"))
    corpus_tools = sorted(corpus)
    corpus_texts = [represent_raw(corpus[tool]["schema"]) for tool in corpus_tools]

    id_df = queries.loc[split["test"]]
    ood_df = load_ood_frame()

    ft_name, ft_path = best_finetuned_artifact()
    systems = {
        ft_name: ft_path,
        "frozen_mpnet": "sentence-transformers/all-mpnet-base-v2",
    }

    output: dict = {"id_set": "regime1_test", "n_id": len(id_df), "n_ood": len(ood_df), "systems": {}}
    for system_name, model_path in systems.items():
        print(f"[score] {system_name}")
        system = EncoderRanker(system_name, model_path, corpus_tools, corpus_texts)

        id_top1, id_margin, id_pred = score_queries(system, id_df["anchor"].tolist())
        ood_top1, ood_margin, _ = score_queries(system, ood_df["query"].tolist())
        id_correct = np.array(
            [corpus_tools[pred] == truth for pred, truth in zip(id_pred, id_df["tool"])]
        )

        block: dict = {"id_top1_accuracy": round(float(id_correct.mean()), 4), "scores": {}}
        for score_name, id_scores, ood_scores in (
            ("max_sim", id_top1, ood_top1),
            ("margin", id_margin, ood_margin),
        ):
            per_subset = {}
            for subset, subset_df in ood_df.groupby("subset"):
                mask = ood_df["subset"] == subset
                per_subset[subset] = {
                    "auroc": round(auroc(id_scores, ood_scores[mask.to_numpy()]), 4),
                    "fpr@95tpr": round(fpr_at_95_tpr(id_scores, ood_scores[mask.to_numpy()]), 4),
                    "n": int(len(subset_df)),
                }

            taus = np.round(np.linspace(0.0, 1.0, 101), 3)
            sweep = []
            for tau in taus:
                answered = id_scores >= tau
                coverage = float(answered.mean())
                selective_risk = float((~id_correct[answered]).mean()) if answered.any() else 0.0
                point = {
                    "tau": float(tau),
                    "coverage": round(coverage, 4),
                    "selective_risk": round(selective_risk, 4),
                }
                for subset in ood_df["subset"].unique():
                    mask = (ood_df["subset"] == subset).to_numpy()
                    point[f"accept_{subset}"] = round(float((ood_scores[mask] >= tau).mean()), 4)
                sweep.append(point)

            block["scores"][score_name] = {
                "auroc_pooled": round(auroc(id_scores, ood_scores), 4),
                "fpr@95tpr_pooled": round(fpr_at_95_tpr(id_scores, ood_scores), 4),
                "per_subset": per_subset,
                "sweep": sweep,
                "id_score_mean": round(float(id_scores.mean()), 4),
                "ood_score_mean": round(float(ood_scores.mean()), 4),
            }
        output["systems"][system_name] = block

        for score_name in ("max_sim", "margin"):
            stats = block["scores"][score_name]
            print(
                f"  {score_name:8s} AUROC={stats['auroc_pooled']:.4f} "
                f"FPR@95TPR={stats['fpr@95tpr_pooled']:.4f}"
            )

    out_path = paths.RESULTS_DIR / "ood_eval.json"
    out_path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
