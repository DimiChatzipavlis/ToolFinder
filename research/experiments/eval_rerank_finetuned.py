"""Completes the professor's suggestion #2 ("specialize selection to the catalog"):
the two practical forms — a **fine-tuned bi-encoder** and the **cross-encoder
reranker** — measured on the same $0 retrieval eval, plus their combination.

2x2 grid on the GitHub-MCP confusable catalog, regime1 **unseen-query** test split
(the fine-tuned MiniLM was trained on the *train* split only, so this is honest):

    base ∈ {stock zero-shot MiniLM, fine-tuned MiniLM} × rerank ∈ {off, on}

No LLM, no OpenAI tokens — selection quality is model-independent.

Run:  python research/experiments/eval_rerank_finetuned.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, for `toolfinder`
sys.path.insert(0, str(HERE))  # sibling import of eval_rerank

from eval_rerank import BIENCODER, gold_ranks, load_catalog, load_test_pairs, metrics  # noqa: E402

FINETUNED = str(HERE / "artifacts" / "biencoder" / "minilm" / "seed42" / "final")


def main() -> None:
    catalog = load_catalog()
    pairs = load_test_pairs()
    print(f"[setup] {len(catalog)} tools, {len(pairs)} unseen-query test cases")
    if not Path(FINETUNED).exists():
        raise SystemExit(f"fine-tuned artifact not found: {FINETUNED} (regenerate on the research branch)")

    bases = [("stock", BIENCODER), ("fine-tuned", FINETUNED)]
    grid: dict[str, dict[str, float]] = {}
    for base_label, model_name in bases:
        for rerank in (False, True):
            label = f"{base_label} | rerank {'ON ' if rerank else 'OFF'}"
            print(f"[arm] {label} ...")
            grid[label] = metrics(gold_ranks(rerank, catalog, pairs, model_name=model_name))

    print("\n" + "=" * 68)
    print("SPECIALIZE SELECTION — fine-tune vs rerank vs both  (GitHub-MCP, 144 q)")
    print("=" * 68)
    print(f"{'arm':<22} | {'recall@1':>8} | {'recall@3':>8} | {'recall@5':>8} | {'mrr':>6}")
    for label, m in grid.items():
        print(f"{label:<22} | {m['recall@1']:>8.4f} | {m['recall@3']:>8.4f} | {m['recall@5']:>8.4f} | {m['mrr']:>6.4f}")

    out = HERE / "results" / "rerank_finetuned_eval.json"
    out.write_text(json.dumps({"n": len(pairs), "grid": grid}, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    print("\nReads the #2 story: how much fine-tuning the bi-encoder helps, how much the")
    print("reranker helps, and whether stacking them still adds on top of a strong base.")


if __name__ == "__main__":
    main()
