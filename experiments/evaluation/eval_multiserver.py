"""R1 result: does training on diverse servers beat GitHub-only training when
the test servers are completely unseen?

Evaluates on the regime-4 test split (60 queries over 7 held-out servers),
ranking against the full 574-tool multi-server corpus. Compares:
  - BM25 (lexical floor)
  - frozen MiniLM (no fine-tuning)
  - FT GitHub-only (the headline model, trained on 15 GitHub tools)
  - FT multi-server (trained on 12 other servers, server-disjoint from the test)

Usage: python experiments/evaluation/eval_multiserver.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402
from experiments.baselines import Bm25Ranker, EncoderRanker  # noqa: E402
from experiments.evaluation import metrics  # noqa: E402
from experiments.representation import represent_raw  # noqa: E402

SYSTEMS = [
    ("frozen_minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    ("ft_github_only", str(paths.ARTIFACTS_DIR / "biencoder" / "minilm" / "seed42" / "final")),
    ("ft_multiserver", str(paths.ARTIFACTS_DIR / "biencoder_multiserver" / "minilm" / "seed42" / "final")),
]


def main() -> None:
    paths.ensure_dirs()
    split = json.loads((paths.SPLITS_DIR / "regime4_multiserver.json").read_text(encoding="utf-8"))
    corpus = json.loads((paths.DATA_DIR / "corpus_multiserver.json").read_text(encoding="utf-8"))
    queries = pd.read_csv(paths.DATA_DIR / "queries_multiserver.csv").set_index("query_id")

    corpus_tools = sorted(corpus)
    corpus_texts = [represent_raw(corpus[t]["schema"]) for t in corpus_tools]
    test = queries.loc[split["test"]]
    anchors, truths = test["anchor"].tolist(), test["tool"].tolist()

    output = {
        "test_servers": split["test_servers"],
        "n_test_queries": len(test),
        "corpus_size": len(corpus_tools),
        "systems": {},
    }

    def record(name, rankings):
        ranks = metrics.ranks_from_rankings(rankings, truths)
        block = {
            "recall@1": round(float(metrics.recall_at_k(ranks, 1).mean()), 4),
            "recall@3": round(float(metrics.recall_at_k(ranks, 3).mean()), 4),
            "recall@5": round(float(metrics.recall_at_k(ranks, 5).mean()), 4),
            "mrr": round(float(metrics.reciprocal_rank(ranks).mean()), 4),
        }
        output["systems"][name] = block
        print(f"  {name:18s} R@1={block['recall@1']:.3f} R@3={block['recall@3']:.3f} MRR={block['mrr']:.3f}")

    print(f"=== regime 4: {len(test)} queries over unseen servers {split['test_servers']} (corpus={len(corpus_tools)}) ===")
    bm25 = Bm25Ranker(corpus_tools, corpus_texts)
    record("bm25", [bm25.rank(a) for a in anchors])
    for name, path in SYSTEMS:
        if not (Path(path).exists() or "/" not in path):
            print(f"  [skip] {name}: artifact missing ({path})")
            continue
        record(name, EncoderRanker(name, path, corpus_tools, corpus_texts).rank_batch(anchors))

    out = paths.RESULTS_DIR / "multiserver_eval.json"
    out.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
