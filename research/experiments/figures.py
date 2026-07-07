"""Generate every figure in the report from experiments/results/*.json.

Each figure function is independent and skips gracefully when its input is
missing, so this can be re-run at any pipeline stage:

    python experiments/figures.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import paths  # noqa: E402
from experiments.representation import represent_raw  # noqa: E402

DPI = 150


def _save(figure: plt.Figure, name: str) -> None:
    out = paths.FIGURES_DIR / name
    figure.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out}")


def fig_leakage_audit() -> None:
    """Random row split vs scenario split: similarity of test anchors to training set."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import train_test_split

    queries = pd.read_csv(paths.QUERIES_CSV)
    v1 = queries[queries["dataset"] == "v1"].reset_index(drop=True)
    split = json.loads(
        (paths.SPLITS_DIR / "regime1_unseen_queries.json").read_text(encoding="utf-8")
    )

    def max_train_similarity(train_texts: list[str], test_texts: list[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        train_matrix = vectorizer.fit_transform(train_texts)
        test_matrix = vectorizer.transform(test_texts)
        return cosine_similarity(test_matrix, train_matrix).max(axis=1)

    random_train, random_test = train_test_split(v1, test_size=0.2, random_state=42)
    random_sims = max_train_similarity(random_train["anchor"].tolist(), random_test["anchor"].tolist())

    by_id = v1.set_index("query_id")
    grouped_sims = max_train_similarity(
        by_id.loc[split["train"], "anchor"].tolist(),
        by_id.loc[split["test"], "anchor"].tolist(),
    )

    figure, axis = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    axis.hist(random_sims, bins=bins, alpha=0.6, label=f"random row split (mean {random_sims.mean():.2f})")
    axis.hist(grouped_sims, bins=bins, alpha=0.6, label=f"scenario-grouped split (mean {grouped_sims.mean():.2f})")
    axis.set_xlabel("max char-ngram cosine similarity of test anchor to any training anchor")
    axis.set_ylabel("test queries")
    axis.set_title("Leakage audit: how close is the test set to the training set?")
    axis.legend()
    _save(figure, "fig_leakage_audit.png")


def fig_main_results() -> None:
    results_path = paths.RESULTS_DIR / "main_eval.json"
    if not results_path.exists():
        print("[skip] main_eval.json missing")
        return
    results = json.loads(results_path.read_text(encoding="utf-8"))

    preferred_order = [
        "random(seed=0)",
        "bm25",
        "tfidf_word",
        "tfidf_char",
        "frozen_minilm",
        "frozen_bge",
        "frozen_mpnet",
        "ft_minilm (avg over seeds)",
        "ft_bge (avg over seeds)",
        "ft_mpnet (avg over seeds)",
        "hybrid_bm25+ft_minilm_seed42",
        "ft_minilm+ce_rerank (avg over seeds)",
    ]
    regimes = list(results["regimes"])
    figure, axes = plt.subplots(1, len(regimes), figsize=(4.8 * len(regimes) + 1, 4.5), sharey=True)
    for axis, regime in zip(np.atleast_1d(axes), regimes):
        block = results["regimes"][regime]
        names = [name for name in preferred_order if name in block]
        means, lows, highs = [], [], []
        for name in names:
            entry = block[name]["recall@1"]
            means.append(entry["mean"])
            if "ci95" in entry:
                lows.append(entry["mean"] - entry["ci95"][0])
                highs.append(entry["ci95"][1] - entry["mean"])
            else:
                lows.append(entry.get("std", 0.0))
                highs.append(entry.get("std", 0.0))
        positions = np.arange(len(names))
        axis.bar(positions, means, yerr=[lows, highs], capsize=3, color="#4878a8")
        axis.set_xticks(positions)
        axis.set_xticklabels([n.replace(" (avg over seeds)", "*") for n in names], rotation=45, ha="right", fontsize=8)
        axis.set_title(regime.replace("_", " "))
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Recall@1")
    figure.suptitle("Recall@1 by system (whiskers: 95% bootstrap CI; * seed std over 3 seeds)")
    _save(figure, "fig_main_results.png")


def fig_confusion() -> None:
    diagnostics_path = paths.DIAGNOSTICS_DIR / "main_eval_per_query.json"
    if not diagnostics_path.exists():
        print("[skip] diagnostics/main_eval_per_query.json missing")
        return
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    block = diagnostics["regimes"]["regime1_unseen_queries"]
    # Render Model A (the headline MiniLM bi-encoder) so the figure matches the
    # "Model A" caption and §4.3 table; fall back to any seed-42 fine-tune.
    preferred = "ft_minilm_seed42"
    candidates = [name for name in block if name.startswith("ft_") and "_seed42" in name]
    if not candidates:
        print("[skip] no seed-42 fine-tuned system in diagnostics")
        return
    system = preferred if preferred in candidates else sorted(candidates)[0]

    queries = pd.read_csv(paths.QUERIES_CSV).set_index("query_id")
    top1 = block[system]["top1"]
    tools = sorted(queries.loc[list(top1)]["tool"].unique())
    tool_index = {tool: i for i, tool in enumerate(tools)}
    matrix = np.zeros((len(tools), len(tools)))
    for query_id, predicted in top1.items():
        truth = queries.loc[query_id, "tool"]
        if predicted in tool_index:
            matrix[tool_index[truth], tool_index[predicted]] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(tools)))
    axis.set_xticklabels(tools, rotation=90, fontsize=7)
    axis.set_yticks(range(len(tools)))
    axis.set_yticklabels(tools, fontsize=7)
    axis.set_xlabel("predicted tool")
    axis.set_ylabel("true tool")
    axis.set_title(f"Top-1 confusion, regime 1 ({system})")
    figure.colorbar(image, shrink=0.8)
    _save(figure, "fig_confusion_regime1.png")


def fig_embedding_map() -> None:
    """t-SNE of test-query embeddings, frozen vs fine-tuned, colored by tool."""
    from sklearn.manifold import TSNE

    from experiments.baselines import EncoderRanker
    from experiments.evaluation.ood import best_finetuned_artifact

    training_path = paths.RESULTS_DIR / "biencoder_training.json"
    if not training_path.exists():
        print("[skip] biencoder_training.json missing")
        return

    queries = pd.read_csv(paths.QUERIES_CSV)
    corpus = json.loads(paths.CORPUS_JSON.read_text(encoding="utf-8"))
    corpus_tools = sorted(corpus)
    corpus_texts = [represent_raw(corpus[tool]["schema"]) for tool in corpus_tools]

    sample = queries.groupby("tool", group_keys=False).apply(
        lambda g: g.sample(min(20, len(g)), random_state=0), include_groups=False
    )
    anchors = sample["anchor"].tolist()
    tools = sample["tool"] if "tool" in sample else queries.loc[sample.index, "tool"]
    tool_codes = pd.Categorical(tools).codes

    ft_name, ft_path = best_finetuned_artifact()
    panels = [
        ("frozen MiniLM", "sentence-transformers/all-MiniLM-L6-v2"),
        (f"fine-tuned ({ft_name})", ft_path),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for axis, (title, model_path) in zip(axes, panels):
        ranker = EncoderRanker("viz", model_path, corpus_tools, corpus_texts)
        embeddings = ranker._encode(anchors)
        coordinates = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(embeddings)
        axis.scatter(coordinates[:, 0], coordinates[:, 1], c=tool_codes, cmap="tab20", s=8)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("t-SNE of query embeddings, colored by true tool (30 tools, 20 queries each)")
    _save(figure, "fig_embedding_tsne.png")


def fig_threshold() -> None:
    ood_path = paths.RESULTS_DIR / "ood_eval.json"
    if not ood_path.exists():
        print("[skip] ood_eval.json missing")
        return
    ood = json.loads(ood_path.read_text(encoding="utf-8"))
    system_name = next(name for name in ood["systems"] if name.startswith("ft_"))
    sweep = ood["systems"][system_name]["scores"]["max_sim"]["sweep"]
    taus = [point["tau"] for point in sweep]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    left.plot(taus, [p["coverage"] for p in sweep], label="ID coverage (answered)")
    left.plot(taus, [p["selective_risk"] for p in sweep], label="selective risk (errors among answered)")
    left.axvline(0.15, color="gray", linestyle="--", linewidth=1, label="runtime default tau=0.15")
    left.set_xlabel("threshold tau on top-1 cosine")
    left.set_title(f"Risk-coverage on ID queries ({system_name})")
    left.legend(fontsize=8)
    left.grid(alpha=0.3)

    for subset in ("chitchat", "out_of_catalog", "adversarial_near_miss"):
        right.plot(taus, [p[f"accept_{subset}"] for p in sweep], label=f"accepted {subset}")
    right.axvline(0.15, color="gray", linestyle="--", linewidth=1)
    right.set_xlabel("threshold tau on top-1 cosine")
    right.set_title("OOD acceptance rate (should be 0)")
    right.legend(fontsize=8)
    right.grid(alpha=0.3)
    _save(figure, "fig_threshold_sweep.png")


def fig_scaling() -> None:
    bench_path = paths.RESULTS_DIR / "scaling_bench.json"
    if not bench_path.exists():
        print("[skip] scaling_bench.json missing")
        return
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    tiers = bench["tiers"]
    sizes = [tier["n"] for tier in tiers]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label, style in (
        ("flat", "Flat (exact)", "o-"),
        ("hnsw_m32_ef64", "HNSW M=32 ef=64", "s--"),
        ("hnsw_m16_ef64", "HNSW M=16 ef=64", "^--"),
    ):
        latencies = [tier["indexes"][key]["latency"]["p50_ms"] for tier in tiers]
        left.loglog(sizes, latencies, style, label=label)
    encoder = bench.get("encoder_latency", {})
    if "cpu" in encoder:
        left.axhline(encoder["cpu"]["p50_ms"], color="red", linestyle=":", linewidth=1,
                     label=f"query encoding, CPU p50 ({encoder['cpu']['p50_ms']:.0f} ms)")
    left.set_xlabel("catalog size N")
    left.set_ylabel("search latency p50 (ms, single thread)")
    left.set_title("Search latency vs catalog size")
    left.legend(fontsize=8)
    left.grid(alpha=0.3, which="both")

    largest_schema_tier = [tier for tier in tiers if tier["tier"] == "schema"][-1]
    ef_values, recalls, latencies = [], [], []
    for key, entry in largest_schema_tier["indexes"].items():
        if key.startswith("hnsw_m32"):
            ef_values.append(int(key.split("ef")[-1]))
            recalls.append(entry["recall@10_vs_exact"])
            latencies.append(entry["latency"]["p50_ms"])
    order = np.argsort(ef_values)
    right.plot(np.array(ef_values)[order], np.array(recalls)[order], "o-")
    for i in order:
        right.annotate(f"{latencies[i]:.3f} ms", (ef_values[i], recalls[i]), fontsize=7,
                       textcoords="offset points", xytext=(5, -10))
    right.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    right.set_xlabel("efSearch (M=32)")
    right.set_ylabel("recall@10 vs exact")
    right.set_title(f"HNSW recall/latency at N={largest_schema_tier['n']}")
    right.grid(alpha=0.3)
    _save(figure, "fig_scaling.png")


def fig_loss_curves() -> None:
    training_path = paths.RESULTS_DIR / "biencoder_training.json"
    if not training_path.exists():
        print("[skip] biencoder_training.json missing")
        return
    records = json.loads(training_path.read_text(encoding="utf-8"))
    seed42 = [record for record in records if record["seed"] == 42]
    if not seed42:
        print("[skip] no seed-42 runs yet")
        return

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    for record in seed42:
        steps = [point["step"] for point in record["loss_history"]]
        losses = [point["loss"] for point in record["loss_history"]]
        left.plot(steps, losses, label=record["model_key"])
        epochs = [point["epoch"] for point in record["val_history"]]
        mrrs = [point["val_mrr@10"] for point in record["val_history"]]
        right.plot(epochs, mrrs, "o-", label=record["model_key"])
    left.set_xlabel("training step")
    left.set_ylabel("MNRL training loss")
    left.set_title("Training loss (seed 42)")
    left.legend()
    left.grid(alpha=0.3)
    right.set_xlabel("epoch")
    right.set_ylabel("validation MRR@10")
    right.set_title("Validation MRR@10 per epoch (seed 42)")
    right.legend()
    right.grid(alpha=0.3)
    _save(figure, "fig_loss_curves.png")


def main() -> None:
    paths.ensure_dirs()
    fig_leakage_audit()
    fig_main_results()
    fig_confusion()
    fig_loss_curves()
    fig_threshold()
    fig_scaling()
    fig_embedding_map()


if __name__ == "__main__":
    main()
