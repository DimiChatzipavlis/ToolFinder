"""Main evaluation: every system, every regime, one protocol.

Systems: random floor, BM25, TF-IDF (word/char), hybrid BM25+dense fusion,
frozen encoders (zero-shot), fine-tuned bi-encoders (per seed), and
cross-encoder reranking over the matching bi-encoder seed.

Regimes are defined by split files in experiments/data/splits/; each split may
carry its own corpus (`corpus_file`) and extra query files (`queries_files`),
so the unseen-server regime ranks against the 574-tool merged corpus while
regimes 1-2 use the 30-tool GitHub corpus. Systems are (re)built per regime.

Protocol: rank the full regime corpus for each test query; score with
R@{1,3,5}, MRR, NDCG@10, 95% bootstrap CIs; mean +/- std across seeds for
trained systems. Per-query ranks and top-1 predictions are saved for reuse by
figures and the OOD analysis.

Usage:
    python experiments/evaluation/evaluate.py                  # all regimes
    python experiments/evaluation/evaluate.py --regimes regime3_unseen_servers
    python experiments/evaluation/evaluate.py --skip-encoders  # lexical only
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

DEFAULT_REGIMES = (
    "regime1_unseen_queries",
    "regime2_unseen_tools",
    "regime3_unseen_servers",
)


class HybridRrfRanker:
    """Reciprocal-rank fusion of BM25 and a dense ranker (k=60)."""

    def __init__(self, name: str, bm25: Bm25Ranker, dense: EncoderRanker, rrf_k: int = 60) -> None:
        self.name = name
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def rank_batch(self, queries: list[str]) -> list[list[str]]:
        dense_rankings = self.dense.rank_batch(queries)
        fused: list[list[str]] = []
        for query, dense_ranking in zip(queries, dense_rankings):
            bm25_ranking = self.bm25.rank(query)
            scores: dict[str, float] = {}
            for ranking in (bm25_ranking, dense_ranking):
                for position, tool in enumerate(ranking, start=1):
                    scores[tool] = scores.get(tool, 0.0) + 1.0 / (self.rrf_k + position)
            fused.append(sorted(scores, key=lambda tool: -scores[tool]))
        return fused

    def rank(self, query: str) -> list[str]:
        return self.rank_batch([query])[0]


def load_regime(regime: str) -> tuple[dict, pd.DataFrame, list[str], list[str]]:
    split = json.loads((paths.SPLITS_DIR / f"{regime}.json").read_text(encoding="utf-8"))

    corpus_file = split.get("corpus_file", "corpus.json")
    corpus = json.loads((paths.DATA_DIR / corpus_file).read_text(encoding="utf-8"))
    corpus_tools = sorted(corpus)
    corpus_texts = [represent_raw(corpus[tool]["schema"]) for tool in corpus_tools]

    frames = [pd.read_csv(paths.QUERIES_CSV)]
    for extra in split.get("queries_files", []):
        frames.append(pd.read_csv(paths.DATA_DIR / extra))
    queries = pd.concat(frames, ignore_index=True).set_index("query_id")
    return split, queries, corpus_tools, corpus_texts


def classification_block(predictions: list[str], truths: list[str]) -> dict:
    """Top-1 selection viewed as classification: accuracy + macro P/R/F1."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        truths, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(truths, predictions)), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
    }


def evaluate_system(system, queries: pd.DataFrame, test_ids: list[str]) -> dict:
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
    summary["classification"] = classification_block(
        [ranking[0] for ranking in rankings], truths
    )
    ranks = metrics.ranks_from_rankings(rankings, truths)
    summary["per_query"] = {
        qid: (int(rank) if np.isfinite(rank) else None)
        for qid, rank in zip(test_ids, ranks)
    }
    summary["top1"] = {qid: ranking[0] for qid, ranking in zip(test_ids, rankings)}
    return summary


def discover_finetuned() -> dict[str, str]:
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
    found: dict[str, str] = {}
    root = paths.ARTIFACTS_DIR / "crossencoder"
    if not root.exists():
        return found
    for seed_dir in sorted(root.iterdir()):
        final = seed_dir / "final"
        if (final / "config.json").exists():
            found[seed_dir.name] = str(final)
    return found


def _release_memory() -> None:
    """Free model memory between systems so one model is resident at a time.

    The previous implementation kept all ~16 encoder systems alive
    simultaneously, which exhausted RAM/VRAM on consumer hardware and killed
    the process mid-evaluation without a traceback.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def encoder_system_factories(
    corpus_tools: list[str],
    corpus_texts: list[str],
    bm25: Bm25Ranker,
) -> list[tuple[str, "callable"]]:
    """(name, builder) pairs for every memory-heavy system, built lazily."""
    factories: list[tuple[str, callable]] = []

    for name, model_path in FROZEN_MODELS.items():
        factories.append(
            (name, lambda n=name, p=model_path: EncoderRanker(n, p, corpus_tools, corpus_texts))
        )

    finetuned = discover_finetuned()
    for name, artifact in finetuned.items():
        factories.append(
            (name, lambda n=name, p=artifact: EncoderRanker(n, p, corpus_tools, corpus_texts))
        )

    if "ft_minilm_seed42" in finetuned:
        def build_hybrid(path=finetuned["ft_minilm_seed42"]):
            dense = EncoderRanker("ft_minilm_seed42", path, corpus_tools, corpus_texts)
            return HybridRrfRanker("hybrid_bm25+ft_minilm_seed42", bm25, dense)

        factories.append(("hybrid_bm25+ft_minilm_seed42", build_hybrid))

    corpus_texts_by_tool = dict(zip(corpus_tools, corpus_texts))
    for seed_name, ce_artifact in discover_crossencoders().items():
        base_name = f"ft_minilm_{seed_name}"
        if base_name not in finetuned:
            continue

        def build_reranker(base_path=finetuned[base_name], ce_path=ce_artifact, seed=seed_name):
            from experiments.models.reranker import CrossEncoderReranker

            base = EncoderRanker(f"ft_minilm_{seed}", base_path, corpus_tools, corpus_texts)
            return CrossEncoderReranker(
                f"ft_minilm+ce_rerank_{seed}", base, ce_path, corpus_texts_by_tool
            )

        factories.append((f"ft_minilm+ce_rerank_{seed_name}", build_reranker))
    return factories


def aggregate_seed_groups(results: dict[str, dict]) -> dict[str, dict]:
    groups: dict[str, list[str]] = {}
    for name in results:
        if "_seed" in name:
            groups.setdefault(name.rsplit("_seed", 1)[0], []).append(name)

    aggregated: dict[str, dict] = {}
    for base, members in groups.items():
        members = [m for m in members if "recall@1" in results[m]]
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
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
    args = parser.parse_args()

    paths.ensure_dirs()
    out_path = paths.RESULTS_DIR / "main_eval.json"
    output: dict = {"regimes": {}}
    if out_path.exists():
        output = json.loads(out_path.read_text(encoding="utf-8"))
        output.setdefault("regimes", {})

    for regime in args.regimes:
        split_path = paths.SPLITS_DIR / f"{regime}.json"
        if not split_path.exists():
            print(f"[skip] {regime}: split file missing")
            continue
        split, queries, corpus_tools, corpus_texts = load_regime(regime)
        test_ids = split["test"]
        print(f"\n=== {regime} ({len(test_ids)} test queries, corpus={len(corpus_tools)}) ===")

        regime_results: dict[str, dict] = {}

        def run_one(system) -> None:
            summary = evaluate_system(system, queries, test_ids)
            regime_results[system.name] = summary
            print(
                f"  {system.name:36s} R@1={summary['recall@1']['mean']:.4f} "
                f"R@3={summary['recall@3']['mean']:.4f} MRR={summary['mrr']['mean']:.4f}"
            )

        bm25 = Bm25Ranker(corpus_tools, corpus_texts)
        for system in (
            RandomRanker(corpus_tools, seed=0),
            bm25,
            TfidfRanker(corpus_tools, corpus_texts, analyzer="word", ngram_range=(1, 2)),
            TfidfRanker(corpus_tools, corpus_texts, analyzer="char_wb", ngram_range=(3, 5)),
        ):
            run_one(system)

        if not args.skip_encoders:
            # One model resident at a time: build, evaluate, free. A failure in
            # one system is recorded and must not abort the remaining systems.
            for name, build in encoder_system_factories(corpus_tools, corpus_texts, bm25):
                print(f"  [load] {name}")
                try:
                    system = build()
                    run_one(system)
                except Exception as exc:  # noqa: BLE001 - isolate per-system failures
                    print(f"  [error] {name}: {exc}")
                    regime_results[name] = {"error": str(exc)}
                finally:
                    system = None
                    _release_memory()

        regime_results.update(aggregate_seed_groups(regime_results))
        output["regimes"][regime] = regime_results
        output["corpus_size_" + regime] = len(corpus_tools)
        out_path.write_text(json.dumps(output, indent=1), encoding="utf-8")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
