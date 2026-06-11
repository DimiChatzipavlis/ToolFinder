"""Calibration analysis of cross-encoder confidence scores.

The reranker's sigmoid score on its top candidate is the natural confidence
signal for execution gating ("only auto-execute above c"). That only works if
the score is calibrated: among decisions scored 0.9, ~90% should be correct.

Measures, on regime-1 (val for fitting, test for reporting):
  - reliability diagram (10 bins) and Expected Calibration Error (ECE)
  - before and after temperature scaling fitted on validation by NLL grid search

Outputs results/calibration.json and figures/fig_calibration.png.

Usage:
    python experiments/evaluation/calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402
from experiments.baselines import EncoderRanker  # noqa: E402
from experiments.evaluation.ood import best_finetuned_artifact  # noqa: E402
from experiments.representation import represent_raw  # noqa: E402

RERANK_DEPTH = 10
N_BINS = 10


def top1_logits_and_correctness(
    crossencoder,
    biencoder: EncoderRanker,
    corpus_texts_by_tool: dict[str, str],
    anchors: list[str],
    truths: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-encoder logit of the reranked top-1 candidate + correctness flag."""
    base_rankings = biencoder.rank_batch(anchors)
    pairs = [
        (anchor, corpus_texts_by_tool[tool])
        for anchor, ranking in zip(anchors, base_rankings)
        for tool in ranking[:RERANK_DEPTH]
    ]
    logits = np.asarray(
        crossencoder.predict(pairs, batch_size=128, show_progress_bar=False, apply_softmax=False)
    ).reshape(len(anchors), RERANK_DEPTH)

    top_logits = np.empty(len(anchors))
    correct = np.empty(len(anchors), dtype=bool)
    for i, (ranking, truth) in enumerate(zip(base_rankings, truths)):
        best = int(np.argmax(logits[i]))
        top_logits[i] = logits[i, best]
        correct[i] = ranking[best] == truth
    return top_logits, correct


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(logits: np.ndarray, correct: np.ndarray) -> float:
    """NLL grid search for the scaling temperature on validation decisions."""
    temperatures = np.linspace(0.25, 10.0, 196)
    best_t, best_nll = 1.0, np.inf
    labels = correct.astype(float)
    for temperature in temperatures:
        probs = np.clip(sigmoid(logits / temperature), 1e-6, 1 - 1e-6)
        nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        if nll < best_nll:
            best_nll, best_t = nll, float(temperature)
    return best_t


def ece_and_bins(probs: np.ndarray, correct: np.ndarray) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, N_BINS + 1)
    bins: list[dict] = []
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probs >= low) & (probs < high if high < 1 else probs <= high)
        if not mask.any():
            bins.append({"bin": [round(low, 1), round(high, 1)], "n": 0})
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += mask.mean() * abs(confidence - accuracy)
        bins.append(
            {
                "bin": [round(low, 1), round(high, 1)],
                "n": int(mask.sum()),
                "confidence": round(confidence, 4),
                "accuracy": round(accuracy, 4),
            }
        )
    return round(float(ece), 4), bins


def main() -> None:
    from sentence_transformers.cross_encoder import CrossEncoder

    paths.ensure_dirs()
    ce_path = paths.ARTIFACTS_DIR / "crossencoder" / "seed42" / "final"
    if not (ce_path / "config.json").exists():
        raise SystemExit("train the cross-encoder first (experiments/models/crossencoder.py)")

    queries = pd.read_csv(paths.QUERIES_CSV).set_index("query_id")
    split = json.loads(
        (paths.SPLITS_DIR / "regime1_unseen_queries.json").read_text(encoding="utf-8")
    )
    corpus = json.loads(paths.CORPUS_JSON.read_text(encoding="utf-8"))
    corpus_tools = sorted(corpus)
    corpus_texts = [represent_raw(corpus[tool]["schema"]) for tool in corpus_tools]
    corpus_texts_by_tool = dict(zip(corpus_tools, corpus_texts))

    _, ft_path = best_finetuned_artifact()
    biencoder = EncoderRanker("base", ft_path, corpus_tools, corpus_texts)
    crossencoder = CrossEncoder(str(ce_path))

    val_logits, val_correct = top1_logits_and_correctness(
        crossencoder, biencoder, corpus_texts_by_tool,
        queries.loc[split["val"], "anchor"].tolist(), queries.loc[split["val"], "tool"].tolist(),
    )
    test_logits, test_correct = top1_logits_and_correctness(
        crossencoder, biencoder, corpus_texts_by_tool,
        queries.loc[split["test"], "anchor"].tolist(), queries.loc[split["test"], "tool"].tolist(),
    )

    temperature = fit_temperature(val_logits, val_correct)
    raw_probs = sigmoid(test_logits)
    scaled_probs = sigmoid(test_logits / temperature)
    raw_ece, raw_bins = ece_and_bins(raw_probs, test_correct)
    scaled_ece, scaled_bins = ece_and_bins(scaled_probs, test_correct)

    output = {
        "temperature_fitted_on_val": temperature,
        "test_top1_accuracy": round(float(test_correct.mean()), 4),
        "raw": {"ece": raw_ece, "bins": raw_bins},
        "temperature_scaled": {"ece": scaled_ece, "bins": scaled_bins},
    }
    out_path = paths.RESULTS_DIR / "calibration.json"
    out_path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(f"T={temperature:.2f} | ECE raw={raw_ece:.4f} -> scaled={scaled_ece:.4f}")
    print(f"wrote {out_path}")

    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    for label, bins, ece in (("raw", raw_bins, raw_ece), (f"T={temperature:.2f}", scaled_bins, scaled_ece)):
        points = [(b["confidence"], b["accuracy"]) for b in bins if b.get("n", 0) > 0]
        axis.plot(*zip(*points), "o-", label=f"{label} (ECE {ece:.3f})")
    axis.set_xlabel("mean confidence (sigmoid CE score, top-1)")
    axis.set_ylabel("accuracy")
    axis.set_title("Cross-encoder calibration, regime-1 test")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.savefig(paths.FIGURES_DIR / "fig_calibration.png", dpi=150, bbox_inches="tight")
    print(f"wrote {paths.FIGURES_DIR / 'fig_calibration.png'}")


if __name__ == "__main__":
    main()
