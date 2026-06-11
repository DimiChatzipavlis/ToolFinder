"""Guards against train/test leakage in the experiment splits.

These tests exist because the original evaluation used a random row split over
templated paraphrases, which a 1-NN lookup over training anchors could solve at
96% Recall@1. If they fail, the benchmark numbers are not valid — fix the data,
never the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES_CSV = REPO_ROOT / "experiments" / "data" / "queries_with_scenarios.csv"
SPLITS_DIR = REPO_ROOT / "experiments" / "data" / "splits"

pytestmark = pytest.mark.skipif(
    not QUERIES_CSV.exists(),
    reason="experiment data not generated (run experiments/dataset/annotate_scenarios.py)",
)


def load_queries():
    import pandas as pd

    return pd.read_csv(QUERIES_CSV)


def load_split(name: str) -> dict:
    return json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def test_every_row_is_annotated() -> None:
    df = load_queries()
    assert len(df) == 1500
    assert df["scenario_id"].notna().all()
    assert df["query_id"].is_unique


def test_regime1_buckets_partition_v1() -> None:
    df = load_queries()
    split = load_split("regime1_unseen_queries")
    buckets = [set(split["train"]), set(split["val"]), set(split["test"])]

    assert not (buckets[0] & buckets[1])
    assert not (buckets[0] & buckets[2])
    assert not (buckets[1] & buckets[2])

    v1_ids = set(df[df["dataset"] == "v1"]["query_id"])
    assert buckets[0] | buckets[1] | buckets[2] == v1_ids


def test_regime1_no_scenario_crosses_buckets() -> None:
    df = load_queries().set_index("query_id")
    split = load_split("regime1_unseen_queries")

    scenario_buckets: dict[str, set[str]] = {}
    for bucket in ("train", "val", "test"):
        for query_id in split[bucket]:
            scenario = df.loc[query_id, "scenario_id"]
            scenario_buckets.setdefault(scenario, set()).add(bucket)

    leaking = {s: b for s, b in scenario_buckets.items() if len(b) > 1}
    assert not leaking, f"scenarios in multiple buckets: {leaking}"


def test_regime2_tools_are_unseen_in_training() -> None:
    df = load_queries().set_index("query_id")
    split = load_split("regime2_unseen_tools")

    train_tools = {df.loc[qid, "tool"] for qid in split["train"]}
    val_tools = {df.loc[qid, "tool"] for qid in split["val"]}
    test_tools = {df.loc[qid, "tool"] for qid in split["test"]}

    assert not (train_tools & test_tools), "unseen-tool regime trains on test tools"
    assert not (val_tools & test_tools), "unseen-tool regime validates on test tools"


def test_corpus_contains_all_30_tools_as_distractors() -> None:
    df = load_queries()
    for regime in ("regime1_unseen_queries", "regime2_unseen_tools"):
        split = load_split(regime)
        assert sorted(split["corpus_tools"]) == sorted(df["tool"].unique()), (
            f"{regime} must rank against the full merged corpus, "
            "otherwise unseen-tool evaluation is artificially easy"
        )
