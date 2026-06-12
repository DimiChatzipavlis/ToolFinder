"""Recover scenario groups in the query datasets and emit the annotated corpus.

The original anchors were generated as `<template prefix> + <scenario clause>`
products: for most tools, ~5-7 underlying scenarios were each rendered through
~10 surface templates, with the scenario clause kept verbatim (e.g. "add a
GitHub Actions workflow file" appears in 10 anchors that differ only in prefix).
A random row split therefore puts paraphrases of the same scenario in both train
and test, which leaks the answer key: a 1-NN lookup over train anchors scores
96% Recall@1 on such a split without reading a single schema.

Recovery algorithm, per (dataset, tool) group:
  1. Normalize anchors and take the last-4-word tail. Verbatim scenario clauses
     make same-scenario tails identical, while template prefixes differ.
  2. Merge tail groups whose tails are near-duplicates (char-ngram cosine >= 0.7)
     to absorb boundary artifacts from short scenario clauses.
Over-merging scenarios is safe (splits only become more conservative);
under-merging is what reintroduces leakage, hence the fuzzy second pass.

Outputs:
    experiments/data/queries_with_scenarios.csv
    experiments/data/corpus.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402

TAIL_WORDS = 4
MERGE_THRESHOLD = 0.70


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def scenario_tail(anchor: str, n_words: int = TAIL_WORDS) -> str:
    words = normalize(anchor).split()
    return " ".join(words[-n_words:])


def cluster_scenarios(anchors: list[str]) -> list[int]:
    """Group anchors by exact scenario tail, then merge near-duplicate tails."""
    tails = [scenario_tail(anchor) for anchor in anchors]
    unique_tails = sorted(set(tails))
    if len(unique_tails) == 1:
        return [0] * len(anchors)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), lowercase=True)
    matrix = vectorizer.fit_transform(unique_tails)
    similarity = (matrix @ matrix.T).toarray()
    adjacency = similarity >= MERGE_THRESHOLD
    _, tail_labels = connected_components(adjacency, directed=False)

    tail_to_label = {tail: int(label) for tail, label in zip(unique_tails, tail_labels)}
    return [tail_to_label[tail] for tail in tails]


def recover_templates(anchors: list[str], scenario_labels: list[int]) -> list[int]:
    """Recover the surface-template id of each anchor.

    The generation grammar is `<template prefix> + <scenario clause>` with the
    clause kept verbatim. Within a scenario cluster, the clause is therefore
    the longest common word-suffix of the cluster's anchors; stripping it
    leaves the template prefix, which is shared *across* scenarios of the same
    tool. Tools without the grammar (one-off anchors) get one template per row.
    """
    normalized = [normalize(anchor).split() for anchor in anchors]

    clause_lengths: dict[int, int] = {}
    by_label: dict[int, list[list[str]]] = {}
    for words, label in zip(normalized, scenario_labels):
        by_label.setdefault(label, []).append(words)
    for label, members in by_label.items():
        if len(members) == 1:
            clause_lengths[label] = len(members[0])
            continue
        shortest = min(len(words) for words in members)
        common = 0
        while common < shortest and len({tuple(words[len(words) - common - 1 :]) for words in members}) == 1:
            common += 1
        clause_lengths[label] = common

    template_keys: list[str] = []
    for index, (words, label) in enumerate(zip(normalized, scenario_labels)):
        clause_length = clause_lengths[label]
        prefix = " ".join(words[: len(words) - clause_length]).strip()
        # One-off rows (clause == whole anchor) are their own template.
        template_keys.append(prefix if prefix else f"__row_{index}")

    key_to_id = {key: i for i, key in enumerate(sorted(set(template_keys)))}
    return [key_to_id[key] for key in template_keys]


def load_raw() -> pd.DataFrame:
    frames = []
    for dataset_name, csv_path in (("v1", paths.RAW_V1_CSV), ("v2", paths.RAW_V2_CSV)):
        df = pd.read_csv(csv_path)
        df["dataset"] = dataset_name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["tool"] = df["positive_schema"].map(lambda s: json.loads(s)["name"])
    df["server"] = "github"
    df["origin"] = "author_template"
    return df


def main() -> None:
    paths.ensure_dirs()
    df = load_raw()

    scenario_ids: list[str] = [""] * len(df)
    template_ids: list[str] = [""] * len(df)
    summary_rows: list[dict] = []
    for (dataset_name, tool), group in df.groupby(["dataset", "tool"], sort=True):
        anchors = group["anchor"].tolist()
        labels = cluster_scenarios(anchors)
        templates = recover_templates(anchors, labels)
        sizes: dict[int, int] = {}
        for label in labels:
            sizes[label] = sizes.get(label, 0) + 1
        summary_rows.append(
            {
                "dataset": dataset_name,
                "tool": tool,
                "n_rows": len(group),
                "n_scenarios": len(sizes),
                "n_templates": len(set(templates)),
                "max_cluster": max(sizes.values()),
            }
        )
        for df_index, label, template in zip(group.index, labels, templates):
            scenario_ids[df_index] = f"{dataset_name}:{tool}:s{label:02d}"
            template_ids[df_index] = f"{dataset_name}:{tool}:t{template:02d}"

    df["scenario_id"] = scenario_ids
    df["template_id"] = template_ids
    df["query_id"] = [f"{d}-{i:04d}" for i, d in enumerate(df["dataset"])]

    out = df[
        ["query_id", "dataset", "server", "tool", "scenario_id", "template_id", "origin", "anchor", "positive_schema"]
    ]
    out.to_csv(paths.QUERIES_CSV, index=False)

    corpus: dict[str, dict] = {}
    for _, row in df.drop_duplicates("tool").iterrows():
        corpus[row["tool"]] = {
            "tool": row["tool"],
            "server": row["server"],
            "dataset": row["dataset"],
            "schema": json.loads(row["positive_schema"]),
        }
    paths.CORPUS_JSON.write_text(json.dumps(corpus, indent=1, sort_keys=True), encoding="utf-8")

    summary = pd.DataFrame(summary_rows)
    print(f"rows annotated: {len(df)} | tools: {df['tool'].nunique()} | scenarios: {df['scenario_id'].nunique()}")
    print(summary.to_string(index=False))
    print(f"\nwrote {paths.QUERIES_CSV}")
    print(f"wrote {paths.CORPUS_JSON} ({len(corpus)} tools)")


if __name__ == "__main__":
    main()
