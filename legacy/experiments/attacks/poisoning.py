"""Tool-description poisoning attack and mitigation measurements.

Threat model: a hostile MCP server publishes a tool whose description embeds
bait text engineered to attract unrelated queries, hijacking routing. The
attacker is assumed to know the query *distribution* (here: validation
anchors) but not the test queries themselves.

Attack: a benign-looking decoy tool ("workspace_notes_sync") gets K validation
anchors appended to its description, K in {0, 1, 5, 10, 20}. The poisoned
corpus is the 30 real tools + the decoy.

Measured per system (BM25, frozen MPNet, fine-tuned bi-encoder):
  hijack@1 / hijack@3 - fraction of regime-1 test queries where the decoy
  outranks the true tool into the top position / top 3.

Mitigations measured:
  1. Description length cap (300 chars) applied at ingest.
  2. Embedding outlier detection: decoy's cosine distance to the corpus
     centroid, expressed as a z-score against the legitimate tools.
  3. Cross-encoder reranking of the bi-encoder's top 10 as a second factor.

Usage:
    python experiments/attacks/poisoning.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402
from experiments.baselines import Bm25Ranker, EncoderRanker  # noqa: E402
from experiments.evaluation.ood import best_finetuned_artifact  # noqa: E402
from experiments.representation import represent_raw  # noqa: E402

DECOY_BASE = {
    "name": "workspace_notes_sync",
    "description": "Synchronize personal workspace notes across devices.",
    "inputSchema": {
        "type": "object",
        "properties": {"note_id": {"type": "string", "description": "The note identifier"}},
        "required": ["note_id"],
    },
}
ATTACK_STRENGTHS = (0, 1, 5, 10, 20)
LENGTH_CAP = 300
RERANK_DEPTH = 10


def build_poisoned_corpus(bait_anchors: list[str], cap: int | None) -> dict:
    decoy = json.loads(json.dumps(DECOY_BASE))
    if bait_anchors:
        decoy["description"] = decoy["description"] + " " + " ".join(bait_anchors)
    if cap is not None:
        decoy["description"] = decoy["description"][:cap]
    return decoy


def hijack_rates(rankings: list[list[str]], decoy_name: str) -> tuple[float, float]:
    top1 = float(np.mean([ranking[0] == decoy_name for ranking in rankings]))
    top3 = float(np.mean([decoy_name in ranking[:3] for ranking in rankings]))
    return round(top1, 4), round(top3, 4)


def centroid_z_score(corpus_embeddings: np.ndarray, decoy_embedding: np.ndarray) -> float:
    centroid = corpus_embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    legit_sims = corpus_embeddings @ centroid
    decoy_sim = float(decoy_embedding @ centroid)
    return round(float((legit_sims.mean() - decoy_sim) / (legit_sims.std() + 1e-9)), 2)


def main() -> None:
    paths.ensure_dirs()
    queries = pd.read_csv(paths.QUERIES_CSV).set_index("query_id")
    split = json.loads(
        (paths.SPLITS_DIR / "regime1_unseen_queries.json").read_text(encoding="utf-8")
    )
    corpus = json.loads(paths.CORPUS_JSON.read_text(encoding="utf-8"))
    corpus_tools = sorted(corpus)

    rng = np.random.default_rng(42)
    val_anchors = queries.loc[split["val"], "anchor"].tolist()
    test_anchors = queries.loc[split["test"], "anchor"].tolist()

    ft_name, ft_path = best_finetuned_artifact()

    ce_artifact = None
    ce_root = paths.ARTIFACTS_DIR / "crossencoder" / "seed42" / "final"
    if (ce_root / "config.json").exists():
        ce_artifact = str(ce_root)

    output: dict = {
        "decoy": DECOY_BASE["name"],
        "id_set": "regime1_test",
        "n_queries": len(test_anchors),
        "finetuned_system": ft_name,
        "attacks": [],
    }

    for strength in ATTACK_STRENGTHS:
        bait = list(rng.choice(val_anchors, size=strength, replace=False)) if strength else []
        entry: dict = {"k_bait_anchors": strength, "systems": {}}

        for capped in (False, True):
            decoy_schema = build_poisoned_corpus(bait, LENGTH_CAP if capped else None)
            tools = corpus_tools + [decoy_schema["name"]]
            texts = [represent_raw(corpus[tool]["schema"]) for tool in corpus_tools] + [
                represent_raw(decoy_schema)
            ]
            suffix = "+length_cap" if capped else ""

            bm25 = Bm25Ranker(tools, texts)
            bm25_rankings = [bm25.rank(anchor) for anchor in test_anchors]
            entry["systems"][f"bm25{suffix}"] = hijack_rates(bm25_rankings, decoy_schema["name"])

            for system_name, model_path in (
                ("frozen_mpnet", "sentence-transformers/all-mpnet-base-v2"),
                (ft_name, ft_path),
            ):
                encoder = EncoderRanker(system_name, model_path, tools, texts)
                rankings = encoder.rank_batch(test_anchors)
                entry["systems"][f"{system_name}{suffix}"] = hijack_rates(rankings, decoy_schema["name"])

                if not capped:
                    entry["systems"][f"{system_name}_centroid_z"] = centroid_z_score(
                        encoder.corpus_embeddings[:-1], encoder.corpus_embeddings[-1]
                    )

                if not capped and ce_artifact and system_name == ft_name:
                    from experiments.models.reranker import CrossEncoderReranker

                    reranker = CrossEncoderReranker(
                        "ce_rerank",
                        encoder,
                        ce_artifact,
                        dict(zip(tools, texts)),
                        rerank_depth=RERANK_DEPTH,
                    )
                    reranked = reranker.rank_batch(test_anchors)
                    entry["systems"][f"{system_name}+ce_rerank"] = hijack_rates(
                        reranked, decoy_schema["name"]
                    )

        output["attacks"].append(entry)
        print(f"k={strength}: " + json.dumps(entry["systems"]))

    out_path = paths.RESULTS_DIR / "poisoning.json"
    out_path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
