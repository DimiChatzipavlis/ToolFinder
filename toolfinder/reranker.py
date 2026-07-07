"""Optional cross-encoder re-ranking of the bi-encoder shortlist.

The bi-encoder embeds the query and each tool *independently* — fast and
pre-indexable, but it never sees the pair together, so it blurs near-duplicate
tools (`search_issues` vs `search_pull_requests`; `align` vs `map` vs `convert`).
A cross-encoder scores each `(query, tool)` pair **jointly** with full attention:
much better at disambiguating confusable tools, at the cost of one forward pass
per candidate — so it runs only over the bi-encoder's top-k shortlist, never the
whole catalog (classic retrieve-then-rerank).

This is the practical form of "specialize selection to the current catalog"
*without* retraining the index on every change: a stock checkpoint works
**zero-shot** — no labels, no per-catalog training. It is **opt-in**
(`RouterHyperparameters.rerank`) and only worth its latency when the bi-encoder's
top-1 is unreliable (confusable / out-of-domain catalogs, weak agent models); on
distinct catalogs where recall@1 is already ~1.0 it adds cost for no gain.

Measured (GitHub-MCP, 144 unseen queries; `research/experiments/eval_rerank.py`):
enabling rerank lifts router recall@1 0.56 → 0.85 and MRR 0.71 → 0.91.

Caveat: a **fine-tuned** bi-encoder scores higher on its own (recall@1 ~0.99),
and reranking *it* with this stock cross-encoder *hurts* (the cross-encoder
overrides the better base). Treat reranking and fine-tuning as **alternatives** —
this is the no-training fallback, not an add-on to a strong base.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder that re-orders a candidate shortlist."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, device: str | None = None, batch_size: int = 64) -> None:
        self.model_name = model_name
        self._device = device
        self.batch_size = batch_size
        self._model = None  # loaded on first use to keep construction cheap

    def _ensure_model(self):
        if self._model is None:
            import torch
            from sentence_transformers.cross_encoder import CrossEncoder

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("loading cross-encoder reranker %r on %s", self.model_name, device)
            self._model = CrossEncoder(self.model_name, device=device)
        return self._model

    def rank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Score each (query, document) pair jointly and return
        `(original_index, score)` pairs sorted best-first. Empty in → empty out."""
        if not documents:
            return []
        model = self._ensure_model()
        scores = np.asarray(
            model.predict([(query, doc) for doc in documents], batch_size=self.batch_size, show_progress_bar=False),
            dtype=float,
        )
        order = np.argsort(-scores, kind="stable")
        return [(int(i), float(scores[int(i)])) for i in order]
