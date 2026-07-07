"""Does the cross-encoder reranker actually improve selection? — local, $0 API.

The reranker re-orders the *router's* shortlist, which is model-independent, so we
measure it directly against labeled data — no LLM, no OpenAI tokens (only a
one-time cross-encoder download). We use the GitHub-MCP corpus (which contains the
genuinely confusable tools: `search_issues` vs `search_pull_requests`, `list_*`,
`get_*`) and the **unseen-queries** test split (144 queries) for an honest read.

For each query we rank all 30 tools with rerank OFF (bi-encoder) and ON
(bi-encoder shortlist re-scored by the cross-encoder), and compare Recall@1/@3/@5
and MRR. A win means the reranker pulls the gold tool up; a tie is reported as a
tie (null result), not spun.

Run:  python research/experiments/eval_rerank.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolfinder import UniversalMCPRouter  # noqa: E402
from toolfinder.dynamic_faiss_router import RouterHyperparameters  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
BIENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MISS = 999


def load_catalog() -> list[dict]:
    corpus = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
    catalog = []
    for entry in corpus.values():
        s = entry["schema"]
        catalog.append({"tool_name": s["name"], "description": s.get("description", ""),
                        "inputSchema": s.get("inputSchema", {})})
    return catalog


def load_test_pairs() -> list[tuple[str, str]]:
    test_ids = set(json.loads((DATA / "splits" / "regime1_unseen_queries.json").read_text(encoding="utf-8"))["test"])
    with (DATA / "queries_with_scenarios.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [(r["anchor"], r["tool"]) for r in rows if r["query_id"] in test_ids]


def gold_ranks(rerank: bool, catalog: list[dict], pairs: list[tuple[str, str]], model_name: str = BIENCODER) -> list[int]:
    config = RouterHyperparameters(min_cosine_similarity=-1.0, rerank=rerank, rerank_pool=30)
    router = UniversalMCPRouter(model_name=model_name, config=config)
    router.ingest_server("github", catalog)
    ranks: list[int] = []
    try:
        for anchor, gold in pairs:
            names = [r.tool_name for r in router.route_top_k(anchor, k=30)]
            ranks.append(names.index(gold) if gold in names else MISS)
    finally:
        router.teardown()
    return ranks


def metrics(ranks: list[int]) -> dict[str, float]:
    n = len(ranks)
    return {
        "recall@1": round(sum(r < 1 for r in ranks) / n, 4),
        "recall@3": round(sum(r < 3 for r in ranks) / n, 4),
        "recall@5": round(sum(r < 5 for r in ranks) / n, 4),
        "mrr": round(sum(1.0 / (r + 1) for r in ranks) / n, 4),
    }


def main() -> None:
    catalog = load_catalog()
    pairs = load_test_pairs()
    print(f"[setup] {len(catalog)} tools, {len(pairs)} unseen-query test cases (regime1)\n")

    print("[arm] rerank OFF (bi-encoder only)...")
    off = gold_ranks(False, catalog, pairs)
    print("[arm] rerank ON (cross-encoder re-scores the shortlist)...")
    on = gold_ranks(True, catalog, pairs)

    m_off, m_on = metrics(off), metrics(on)
    improved = sum(n < o for o, n in zip(off, on))
    worsened = sum(n > o for o, n in zip(off, on))

    print("\n" + "=" * 60)
    print("RERANK OFF vs ON  (GitHub-MCP, 144 unseen queries)")
    print("=" * 60)
    print(f"{'metric':>10} | {'OFF':>8} | {'ON':>8} | {'delta':>8}")
    for k in ("recall@1", "recall@3", "recall@5", "mrr"):
        d = round(m_on[k] - m_off[k], 4)
        print(f"{k:>10} | {m_off[k]:>8.4f} | {m_on[k]:>8.4f} | {d:>+8.4f}")
    print(f"\nper-query gold rank: improved={improved}, worsened={worsened}, unchanged={len(pairs)-improved-worsened}")

    out = DATA.parent / "results" / "rerank_eval.json"
    out.write_text(json.dumps({"n": len(pairs), "off": m_off, "on": m_on,
                               "improved": improved, "worsened": worsened}, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    verdict = "helps" if m_on["recall@1"] > m_off["recall@1"] else ("ties" if m_on["recall@1"] == m_off["recall@1"] else "hurts")
    print(f"verdict (recall@1): reranking {verdict} on this catalog.")


if __name__ == "__main__":
    main()
