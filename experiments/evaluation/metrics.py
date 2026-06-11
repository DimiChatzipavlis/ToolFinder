"""Retrieval metrics with bootstrap confidence intervals.

All metric functions take, per query, the rank position (1-based) of the true
tool in the system's ranking, so every system is evaluated through the same
code path regardless of how it ranks.
"""

from __future__ import annotations

import numpy as np


def ranks_from_rankings(rankings: list[list[str]], truths: list[str]) -> np.ndarray:
    """1-based rank of the true tool per query; np.inf if absent from the ranking."""
    ranks = np.full(len(truths), np.inf)
    for i, (ranking, truth) in enumerate(zip(rankings, truths)):
        for position, tool in enumerate(ranking, start=1):
            if tool == truth:
                ranks[i] = position
                break
    return ranks


def recall_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """Per-query 0/1 hit indicator at cutoff k."""
    return (ranks <= k).astype(float)


def reciprocal_rank(ranks: np.ndarray) -> np.ndarray:
    """Per-query reciprocal rank (0 when the true tool was never ranked)."""
    with np.errstate(divide="ignore"):
        rr = 1.0 / ranks
    return np.where(np.isfinite(rr), rr, 0.0)


def ndcg_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """Per-query NDCG@k under binary relevance with a single relevant item.

    With one relevant document, ideal DCG is 1, so NDCG@k reduces to
    1/log2(rank+1) when rank <= k, else 0.
    """
    gains = 1.0 / np.log2(ranks + 1.0)
    return np.where(ranks <= k, gains, 0.0)


def bootstrap_ci(
    per_query_values: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean with percentile-bootstrap (1-alpha) CI over queries."""
    rng = np.random.default_rng(seed)
    values = np.asarray(per_query_values, dtype=float)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lower), float(upper)


def summarize(rankings: list[list[str]], truths: list[str], seed: int = 0) -> dict:
    """Standard metric block: R@{1,3,5}, MRR, NDCG@10, each with 95% CI."""
    ranks = ranks_from_rankings(rankings, truths)
    block: dict = {"n_queries": len(truths)}
    for name, values in (
        ("recall@1", recall_at_k(ranks, 1)),
        ("recall@3", recall_at_k(ranks, 3)),
        ("recall@5", recall_at_k(ranks, 5)),
        ("mrr", reciprocal_rank(ranks)),
        ("ndcg@10", ndcg_at_k(ranks, 10)),
    ):
        mean, low, high = bootstrap_ci(values, seed=seed)
        block[name] = {"mean": round(mean, 4), "ci95": [round(low, 4), round(high, 4)]}
    return block


def confusion_pairs(rankings: list[list[str]], truths: list[str]) -> dict[str, dict[str, int]]:
    """Top-1 confusion counts: truth -> predicted -> count."""
    table: dict[str, dict[str, int]] = {}
    for ranking, truth in zip(rankings, truths):
        predicted = ranking[0] if ranking else "<none>"
        table.setdefault(truth, {})
        table[truth][predicted] = table[truth].get(predicted, 0) + 1
    return table
