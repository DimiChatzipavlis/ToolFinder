"""Main evaluation: every system, both regimes, one protocol.

Systems: random floor, BM25, TF-IDF (word/char), frozen encoders (zero-shot),
fine-tuned bi-encoders (per seed), and optionally cross-encoder reranking over
the best bi-encoder's candidates.

Protocol: rank the full 30-tool corpus for each test query; score with
R@{1,3,5}, MRR, NDCG@10 and 95% bootstrap CIs. Trained systems additionally get
mean +/- std across seeds. Per-query ranks are saved so figures and the OOD
analysis reuse the same runs.

Usage:
    python experiments/evaluation/evaluate.py            # all available systems
    python experiments/evaluation/evaluate.py --skip-encoders   # lexical only (fast)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402
from experiments.baselines import Bm25Ranker, EncoderRanker, RandomRanker, TfidfRanker  # noqa: E402
from experiments.evaluation import metrics  # noqa: E402
from experiments.representation import represent_raw  # noqa: E402

FROZEN_MODELS = {
    "frozen_minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "frozen_bge": "BAAI/bge-small-en-v1.5",
    "frozen_mpnet": "sentence-transformers/all-mpnet-base-v2",
}

REGIMES = ("regime1_unseen_queries", "regime2_unseen_tools")


def load_data() -> tuple[pd.DataFrame, list[str], list[str], dict]:
    queries = pd.read_csv(paths.QUERIES_CSV).set_index("query_id")
    corpus = json.loads(paths.CORPUS_JSON.read_text(encoding="utf-8"))
    corpus_tools = sorted(corpus)
    corpus_texts = [represent_raw(corpus[tool]["schema"]) for tool in corpus_tools]
    splits = {
        regime: json.loads((paths.SPLITS_DIR / f"{regime}.json").read_text(encoding="utf-8"))
        for regime in REGIMES
    }
    return queries, corpus_tools, corpus_texts, splits


def evaluate_system(
    system,
    queries: pd.DataFrame,
    test_ids: list[str],
) -> dict:
    anchors = queries.loc[test_ids, "anchor"].tolist()
    truths = queries.loc[test_ids, "tool"].tolist()

    started = time.perf_counter()
    if hasattr(system, "rank_batch"):
        rankings = system.rank_batch(anchors)
    else:
        rankings = [system.rank(anchor) for anchor in anchors]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    summary = metrics.summarize(rankings, truths)
    summary["latency_ms_per_query"] = round(elapsed_ms / len(anchors), 3)
    ranks = metrics.ranks_from_rankings(rankings, truths)
    summary["per_query"] = {
        qid: (int(rank) if np.isfinite(rank) else None)
        for qid, rank in zip(test_ids, ranks)
    }
    summary["top1"] = {qid: ranking[0] for qid, ranking in zip(test_ids, rankings)}
    return summary


def discover_finetuned() -> dict[str, str]:
    """Map system name -> artifact path for every trained bi-encoder run."""
    found: dict[str, str] = {}
    root = paths.ARTIFACTS_DIR / "biencoder"
    if not root.exists():
        return found
    for model_dir in sorted(root.iterdir()):
        for seed_dir in sorted(model_dir.iterdir()):
            final = seed_dir / "final"
            if (final / "config.json").exists():
                found[f"ft_{model_dir.name}_{seed_dir.name}"] = str(final)
    return found


def discover_crossencoders() -> dict[str, str]:
    """Map seed name ('seed42') -> artifact path for trained cross-encoders."""
    found: dict[str, str] = {}
    root = paths.ARTIFACTS_DIR / "crossencoder"
    if not root.exists():
        return found
    for seed_dir in sorted(root.iterdir()):
        final = seed_dir / "final"
        if (final / "config.json").exists():
            found[seed_dir.name] = str(final)
    return found


def aggregate_seed_groups(results: dict[str, dict]) -> dict[str, dict]:
    """Mean +/- std across seeds for systems named <base>_seed<N>."""
    groups: dict[str, list[str]] = {}
    for name in results:
        if "_seed" in name:
            groups.setdefault(name.rsplit("_seed", 1)[0], []).append(name)

    aggregated: dict[str, dict] = {}
    for base, members in groups.items():
        if len(members) < 2:
            continue
        block: dict = {"seeds": len(members), "members": sorted(members)}
        for metric_name in ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@10"):
            values = [results[m][metric_name]["mean"] for m in members]
            block[metric_name] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
            }
        aggregated[f"{base} (avg over seeds)"] = block
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-encoders", action="store_true", help="lexical/random systems only")
    args = parser.parse_args()

    paths.ensure_dirs()
    queries, corpus_tools, corpus_texts, splits = load_data()

    systems: list = [
        RandomRanker(corpus_tools, seed=0),
        Bm25Ranker(corpus_tools, corpus_texts),
        TfidfRanker(corpus_tools, corpus_texts, analyzer="word", ngram_range=(1, 2)),
        TfidfRanker(corpus_tools, corpus_texts, analyzer="char_wb", ngram_range=(3, 5)),
    ]
    if not args.skip_encoders:
        for name, model_path in FROZEN_MODELS.items():
            print(f"[load] {name}")
            systems.append(EncoderRanker(name, model_path, corpus_tools, corpus_texts))

        finetuned = discover_finetuned()
        encoders_by_name: dict[str, EncoderRanker] = {}
        for name, artifact in finetuned.items():
            print(f"[load] {name}")
            encoder = EncoderRanker(name, artifact, corpus_tools, corpus_texts)
            encoders_by_name[name] = encoder
            systems.append(encoder)

        # Retrieve-then-rerank: pair each cross-encoder seed with the matching
        # ft_minilm seed so seed averaging covers the whole pipeline.
        from experiments.models.reranker import CrossEncoderReranker

        corpus_texts_by_tool = dict(zip(corpus_tools, corpus_texts))
        for seed_name, ce_artifact in discover_crossencoders().items():
            base_name = f"ft_minilm_{seed_name}"
            if base_name not in encoders_by_name:
                continue
            name = f"ft_minilm+ce_rerank_{seed_name}"
            print(f"[load] {name}")
            systems.append(
                CrossEncoderReranker(
                    name,
                    encoders_by_name[base_name],
                    ce_artifact,
                    corpus_texts_by_tool,
                )
            )

    output: dict = {"corpus_size": len(corpus_tools), "regimes": {}}
    for regime in REGIMES:
        test_ids = splits[regime]["test"]
        print(f"\n=== {regime} ({len(test_ids)} test queries, corpus={len(corpus_tools)}) ===")
        regime_results: dict[str, dict] = {}
        for system in systems:
            summary = evaluate_system(system, queries, test_ids)
            regime_results[system.name] = summary
            print(
                f"  {system.name:32s} R@1={summary['recall@1']['mean']:.4f} "
                f"R@3={summary['recall@3']['mean']:.4f} MRR={summary['mrr']['mean']:.4f}"
            )
        regime_results.update(aggregate_seed_groups(regime_results))
        output["regimes"][regime] = regime_results

    out_path = paths.RESULTS_DIR / "main_eval.json"
    out_path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
