"""Hand-computed checks for the experiment metric implementations."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("sklearn")

from experiments.baselines import Bm25Ranker  # noqa: E402
from experiments.evaluation import metrics  # noqa: E402

RANKINGS = [
    ["a", "b", "c"],  # truth a -> rank 1
    ["b", "a", "c"],  # truth a -> rank 2
    ["b", "c", "a"],  # truth a -> rank 3
    ["b", "c", "d"],  # truth a -> missing
]
TRUTHS = ["a", "a", "a", "a"]


def test_ranks_from_rankings() -> None:
    ranks = metrics.ranks_from_rankings(RANKINGS, TRUTHS)
    assert ranks[0] == 1 and ranks[1] == 2 and ranks[2] == 3
    assert np.isinf(ranks[3])


def test_recall_and_mrr_hand_values() -> None:
    ranks = metrics.ranks_from_rankings(RANKINGS, TRUTHS)
    assert metrics.recall_at_k(ranks, 1).mean() == 0.25
    assert metrics.recall_at_k(ranks, 3).mean() == 0.75
    expected_mrr = (1.0 + 0.5 + 1.0 / 3.0 + 0.0) / 4.0
    assert metrics.reciprocal_rank(ranks).mean() == pytest.approx(expected_mrr)


def test_ndcg_hand_values() -> None:
    ranks = metrics.ranks_from_rankings(RANKINGS, TRUTHS)
    expected = (1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4) + 0.0) / 4.0
    assert metrics.ndcg_at_k(ranks, 10).mean() == pytest.approx(expected)


def test_bootstrap_ci_brackets_mean() -> None:
    values = np.array([0.0, 1.0] * 50)
    mean, low, high = metrics.bootstrap_ci(values, seed=1)
    assert mean == 0.5
    assert low <= mean <= high
    assert 0.3 < low and high < 0.7


def test_bm25_prefers_matching_document() -> None:
    ranker = Bm25Ranker(
        corpus_tools=["create_branch", "list_issues"],
        corpus_texts=[
            '{"name":"create_branch","description":"Create a new branch in a repository"}',
            '{"name":"list_issues","description":"List issues in a repository"}',
        ],
    )
    assert ranker.rank("make a new branch called dev")[0] == "create_branch"
    assert ranker.rank("show me the open issues")[0] == "list_issues"
