"""Baseline ranking systems sharing one protocol: rank(query) -> ordered tool names.

Implemented without extra dependencies:
  - RandomRanker: the floor every other number is read against.
  - Bm25Ranker: native BM25 Okapi (k1=1.5, b=0.75); the standard lexical retriever.
  - TfidfRanker: word and character n-gram variants; character n-grams are
    robust to identifier conventions and quantify how lexical the task is.
  - EncoderRanker: any sentence-transformers model (frozen checkpoints serve as
    zero-shot baselines; fine-tuned artifact paths serve as the trained systems).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.representation import lexicalize

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(lexicalize(text).lower())


class RandomRanker:
    name_prefix = "random"

    def __init__(self, corpus_tools: list[str], seed: int = 0) -> None:
        self.corpus_tools = list(corpus_tools)
        self.rng = np.random.default_rng(seed)
        self.name = f"{self.name_prefix}(seed={seed})"

    def rank(self, query: str) -> list[str]:
        del query
        order = self.rng.permutation(len(self.corpus_tools))
        return [self.corpus_tools[i] for i in order]


class Bm25Ranker:
    """BM25 Okapi over tokenized schema documents."""

    def __init__(self, corpus_tools: list[str], corpus_texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.name = "bm25"
        self.corpus_tools = list(corpus_tools)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(text) for text in corpus_texts]
        self.doc_lengths = np.array([len(tokens) for tokens in self.doc_tokens], dtype=float)
        self.avg_doc_length = float(self.doc_lengths.mean())
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))
        n_docs = len(self.doc_tokens)
        self.idf = {
            term: math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        query_tokens = tokenize(query)
        scores = np.zeros(len(self.doc_tokens))
        for doc_index, tf in enumerate(self.term_frequencies):
            length_norm = 1.0 - self.b + self.b * self.doc_lengths[doc_index] / self.avg_doc_length
            score = 0.0
            for term in query_tokens:
                if term not in tf:
                    continue
                frequency = tf[term]
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / (
                    frequency + self.k1 * length_norm
                )
            scores[doc_index] = score
        return scores

    def rank(self, query: str) -> list[str]:
        order = np.argsort(-self.scores(query), kind="stable")
        return [self.corpus_tools[i] for i in order]


class TfidfRanker:
    def __init__(
        self,
        corpus_tools: list[str],
        corpus_texts: list[str],
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (3, 5),
    ) -> None:
        self.name = f"tfidf_{'word' if analyzer == 'word' else 'char'}"
        self.corpus_tools = list(corpus_tools)
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            lowercase=True,
            preprocessor=lexicalize,
        )
        self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)

    def scores(self, query: str) -> np.ndarray:
        query_vector = self.vectorizer.transform([query])
        return np.asarray((query_vector @ self.corpus_matrix.T).todense()).ravel()

    def rank(self, query: str) -> list[str]:
        order = np.argsort(-self.scores(query), kind="stable")
        return [self.corpus_tools[i] for i in order]


class EncoderRanker:
    """Dense ranker over normalized sentence-transformer embeddings.

    Works for zero-shot checkpoints and fine-tuned artifact directories alike;
    cosine similarity is computed exactly (no ANN index) since evaluation
    corpora here are small.
    """

    def __init__(self, name: str, model_path: str, corpus_tools: list[str], corpus_texts: list[str], device: str | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.corpus_tools = list(corpus_tools)
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_path, device=resolved_device)
        self.corpus_embeddings = self._encode(corpus_texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def scores_batch(self, queries: list[str]) -> np.ndarray:
        query_embeddings = self._encode(queries)
        return query_embeddings @ self.corpus_embeddings.T

    def rank_batch(self, queries: list[str]) -> list[list[str]]:
        all_scores = self.scores_batch(queries)
        order = np.argsort(-all_scores, axis=1, kind="stable")
        return [[self.corpus_tools[i] for i in row] for row in order]

    def rank(self, query: str) -> list[str]:
        return self.rank_batch([query])[0]
