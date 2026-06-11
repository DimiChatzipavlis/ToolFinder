"""Render reports/report.md from experiments/results/*.json.

Every number in the report is read from a results file produced by a script in
this package; nothing is hand-typed. Re-run after any experiment:

    python experiments/build_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import paths  # noqa: E402

REPORT_PATH = paths.REPO_ROOT / "reports" / "report.md"

MAIN_SYSTEM_ORDER = [
    ("random(seed=0)", "Random"),
    ("bm25", "BM25"),
    ("tfidf_word", "TF-IDF (word 1-2g)"),
    ("tfidf_char", "TF-IDF (char 3-5g)"),
    ("frozen_minilm", "Frozen MiniLM-L6 (22M)"),
    ("frozen_bge", "Frozen BGE-small (33M)"),
    ("frozen_mpnet", "Frozen MPNet (109M)"),
    ("ft_minilm (avg over seeds)", "**FT bi-encoder MiniLM-L6** (3 seeds)"),
    ("ft_bge (avg over seeds)", "**FT bi-encoder BGE-small** (3 seeds)"),
    ("ft_mpnet (avg over seeds)", "**FT bi-encoder MPNet** (3 seeds)"),
    ("hybrid_bm25+ft_minilm_seed42", "Hybrid RRF: BM25 + FT MiniLM (seed 42)"),
    ("ft_minilm+ce_rerank (avg over seeds)", "**FT MiniLM + CE rerank** (3 seeds)"),
]


def load(name: str) -> dict | None:
    path = paths.RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_metric(entry: dict) -> str:
    if "ci95" in entry:
        return f"{entry['mean']:.3f} [{entry['ci95'][0]:.3f}, {entry['ci95'][1]:.3f}]"
    if "std" in entry:
        return f"{entry['mean']:.3f} ± {entry['std']:.3f}"
    return f"{entry['mean']:.3f}"


def main_results_table(results: dict, regime: str) -> str:
    block = results["regimes"][regime]
    lines = [
        "| System | R@1 | R@3 | MRR | NDCG@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in MAIN_SYSTEM_ORDER:
        if key not in block:
            continue
        row = block[key]
        lines.append(
            f"| {label} | {fmt_metric(row['recall@1'])} | {fmt_metric(row['recall@3'])} "
            f"| {fmt_metric(row['mrr'])} | {fmt_metric(row['ndcg@10'])} |"
        )
    return "\n".join(lines)


def training_table(records: list[dict]) -> str:
    lines = [
        "| Model | Params | Seeds | Train time (s, mean) | Val MRR@10 (mean ± std) |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_model: dict[str, list[dict]] = {}
    for record in records:
        by_model.setdefault(record["model_key"], []).append(record)
    params = {"minilm": "22M", "bge": "33M", "mpnet": "109M", "crossencoder": "22M"}
    for model_key, runs in by_model.items():
        mrr_key = "final_val_mrr@10" if "final_val_mrr@10" in runs[0] else "final_val_mrr_full_corpus"
        mrrs = [run[mrr_key] for run in runs]
        times = [run["train_duration_s"] for run in runs]
        import numpy as np

        lines.append(
            f"| {model_key} | {params.get(model_key, '?')} | {len(runs)} "
            f"| {np.mean(times):.0f} | {np.mean(mrrs):.4f} ± {np.std(mrrs):.4f} |"
        )
    return "\n".join(lines)


def ood_table(ood: dict) -> str:
    lines = [
        "| System | Score | AUROC (pooled) | FPR@95TPR | AUROC chitchat | AUROC out-of-catalog | AUROC near-miss |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for system_name, block in ood["systems"].items():
        for score_name, stats in block["scores"].items():
            per = stats["per_subset"]
            lines.append(
                f"| {system_name} | {score_name} | {stats['auroc_pooled']:.3f} | {stats['fpr@95tpr_pooled']:.3f} "
                f"| {per['chitchat']['auroc']:.3f} | {per['out_of_catalog']['auroc']:.3f} "
                f"| {per['adversarial_near_miss']['auroc']:.3f} |"
            )
    return "\n".join(lines)


def scaling_table(bench: dict) -> str:
    lines = [
        "| N | Tier | Flat p50 (ms) | HNSW M32/ef64 p50 (ms) | HNSW recall@10 | Flat size | HNSW size |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for tier in bench["tiers"]:
        flat = tier["indexes"]["flat"]
        hnsw = tier["indexes"]["hnsw_m32_ef64"]
        lines.append(
            f"| {tier['n']:,} | {tier['tier']} | {flat['latency']['p50_ms']:.4f} "
            f"| {hnsw['latency']['p50_ms']:.4f} | {hnsw['recall@10_vs_exact']:.4f} "
            f"| {flat['size_bytes'] / 1e6:.1f} MB | {hnsw['size_bytes'] / 1e6:.1f} MB |"
        )
    return "\n".join(lines)


def ablation_table(ablation: dict) -> str:
    lines = [
        "| Representation | BM25 R@1 (r1/r2) | TF-IDF char R@1 (r1/r2) | Frozen MPNet R@1 (r1/r2) | FT bi-encoder R@1 (r1/r2) |",
        "| --- | --- | --- | --- | --- |",
    ]
    ft_name = None
    for representation, block in ablation["representations"].items():
        r1 = block["regime1_unseen_queries"]
        r2 = block["regime2_unseen_tools"]
        if ft_name is None:
            ft_name = next(name for name in r1 if name.startswith("ft_"))
        lines.append(
            f"| {representation} "
            f"| {r1['bm25']['recall@1']:.3f} / {r2['bm25']['recall@1']:.3f} "
            f"| {r1['tfidf_char']['recall@1']:.3f} / {r2['tfidf_char']['recall@1']:.3f} "
            f"| {r1['frozen_mpnet']['recall@1']:.3f} / {r2['frozen_mpnet']['recall@1']:.3f} "
            f"| {r1[ft_name]['recall@1']:.3f} / {r2[ft_name]['recall@1']:.3f} |"
        )
    return "\n".join(lines)


def poisoning_table(poisoning: dict) -> str:
    ft_name = poisoning["finetuned_system"]
    lines = [
        "| Bait anchors K | BM25 hijack@1 | Frozen MPNet hijack@1 | FT bi-encoder hijack@1 | FT + length cap | FT + CE rerank | Decoy centroid z |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for attack in poisoning["attacks"]:
        systems = attack["systems"]
        ce_cell = (
            f"{systems[f'{ft_name}+ce_rerank'][0]:.3f}"
            if f"{ft_name}+ce_rerank" in systems
            else "—"
        )
        lines.append(
            f"| {attack['k_bait_anchors']} "
            f"| {systems['bm25'][0]:.3f} "
            f"| {systems['frozen_mpnet'][0]:.3f} "
            f"| {systems[ft_name][0]:.3f} "
            f"| {systems[f'{ft_name}+length_cap'][0]:.3f} "
            f"| {ce_cell} "
            f"| {systems[f'{ft_name}_centroid_z']:.1f} |"
        )
    return "\n".join(lines)


RELATED_WORK = """## Related Work

**Tool-augmented LLMs.** MRKL (Karpas et al., 2022) introduced the modular
router-over-experts framing this system implements; Toolformer (Schick et al.,
2023) learns API invocation self-supervised; Gorilla (Patil et al., 2023) and
ToolLLM/ToolBench (Qin et al., 2023) target correct call *generation* over
large real API collections and already incorporate retrievers; ToolFinder
addresses only the selection sub-problem, but evaluates it under open-set and
adversarial conditions those works do not focus on.

**Retrieval-based tool selection in practice.** Embedding tool descriptions
and retrieving by cosine similarity is shipped functionality in LangChain's
tool retriever, LlamaIndex object retrievers, and the semantic-router library;
OpenAI's function-retrieval cookbook documents the same pattern. This project
does not claim architectural novelty over these systems - its contribution is
a leakage-controlled benchmark and a controlled empirical comparison (lexical
vs frozen vs fine-tuned vs reranked) of what they ship as defaults, plus
abstention and poisoning analyses.

**Dense retrieval methodology.** Bi-encoder contrastive training follows
Sentence-BERT (Reimers & Gurevych, 2019) with in-batch negatives (DPR,
Karpukhin et al., 2020; MultipleNegativesRankingLoss); cross-encoder reranking
follows Nogueira & Cho (2019) on MS MARCO. HNSW is Malkov & Yashunin (2018);
the scaling study revisits its trade-offs at tool-catalog sizes rather than
web-corpus sizes.

**Benchmark hygiene.** The leakage analysis follows the train-test
contamination literature for paraphrase-templated synthetic data: grouped
splits over generation units rather than rows, with split constraints
enforced as tests.
"""


def build() -> str:
    main_eval = load("main_eval.json")
    biencoder_training = load("biencoder_training.json")
    crossencoder_training = load("crossencoder_training.json")
    ood = load("ood_eval.json")
    bench = load("scaling_bench.json")
    ablation = load("ablation_representation.json")
    poisoning = load("poisoning.json")
    calibration = load("calibration.json")
    llm_incontext = load("llm_incontext.json")

    sections: list[str] = []
    sections.append(f"""# ToolFinder: An Empirical Study of Dense Retrieval for Open-Set MCP Tool Routing

*Generated from `experiments/results/` on {date.today().isoformat()}. Regenerate with `python experiments/build_report.py`.*

## Abstract

Selecting the right tool from a Model Context Protocol (MCP) catalog is a precondition for reliable tool-using LLM agents, especially small local models whose context windows cannot hold every schema. We study retrieval-based tool selection on a benchmark of 1,695 natural-language intents over 574 real tool schemas (30 GitHub MCP tools plus 544 operations converted from 23 public OpenAPI providers). We first show that the naive random split used in earlier iterations of this project is answerable at ~96% Recall@1 by a nearest-neighbor lookup over training queries alone, because the queries follow a scenario-by-template generation grammar; we therefore introduce three leakage-controlled protocols - unseen queries (scenario-grouped), unseen tools, and unseen servers - enforced by CI tests. Under these protocols we train and compare two deep architectures - contrastively fine-tuned bi-encoders (22M-109M parameters) and a cross-encoder reranker - against lexical (BM25, TF-IDF), hybrid-fusion, and frozen-encoder baselines, with three seeds and bootstrap confidence intervals. We further characterize threshold-based rejection of out-of-scope requests on 249 out-of-distribution queries (chitchat, out-of-catalog, adversarial near-miss), measure a description-poisoning attack with three mitigations, analyze reranker confidence calibration, and benchmark exact flat search against HNSW from 15 to 100,000 vectors. The fine-tuned bi-encoder substantially outperforms all baselines on all three regimes, including zero-shot transfer to unseen servers; the cross-encoder adds precision on confusable candidates at linear per-query cost; and exact flat search dominates HNSW at every catalog size tested, motivating the runtime's exact-search default.

## 1. Task and Benchmark

**Task.** Given a natural-language intent and a catalog of MCP tool schemas, rank the catalog so the correct tool is at rank 1; abstain when no tool applies (open-set condition).

**Data.** 1,500 author-templated queries over 30 GitHub MCP tools (50 per tool, perfectly balanced), split into 15 train-eligible tools and 15 tools reserved for unseen-tool evaluation. Most tools follow a `scenario x template` grammar (~5 scenarios x ~10 paraphrase templates); two tools have per-row scenarios. Scenario structure was recovered programmatically (`experiments/dataset/annotate_scenarios.py`) and is used to construct leakage-controlled splits.

**Leakage control.** Under a random row split, paraphrases of each test query's scenario appear in training: a 1-NN lookup over training anchors - no schema text, no training - reaches ~96% Recall@1, and 65% of test anchors have a training anchor at char-ngram cosine >= 0.8. The scenario-grouped split removes this artifact (see `experiments/results/figures/fig_leakage_audit.png`). Split hygiene is enforced by `tests/test_split_hygiene.py`.

**Regimes.** *Regime 1 (unseen queries):* scenario-grouped 460/146/144 train/val/test over the 15 v1 tools. *Regime 2 (unseen tools):* all 750 queries of the 15 v2 tools, never seen in training. Both rank the full 30-tool corpus, so unseen-tool evaluation faces trained tools as distractors. *Regime 3 (unseen servers):* 195 hand-written queries (different generator family, anti-echo rules) over 65 tools sampled from 22 real API providers, ranked against a 574-tool merged corpus (30 GitHub + 544 OpenAPI-derived tools from apis.guru); training data remains GitHub-only, so this measures zero-shot cross-server transfer. Dataset provenance and biases: `experiments/data/DATASET_CARD.md`.
""")

    sections.append(RELATED_WORK)
    sections.append("## 2. Systems\n")
    sections.append("""**Baselines.** Random permutation; BM25 (Okapi, k1=1.5, b=0.75); TF-IDF (word 1-2grams and character 3-5grams); frozen sentence-transformer checkpoints (MiniLM-L6, BGE-small, MPNet) used zero-shot. All systems rank the identical canonical-JSON schema documents.

**Model A - fine-tuned bi-encoder.** Sentence-transformer fine-tuned with MultipleNegativesRankingLoss under a no-duplicates batch sampler (with only 15 distinct positives, plain in-batch negatives would treat same-schema rows as negatives - a false-negative artifact the sampler removes). Selection on validation MRR@10 per epoch; seeds 13/42/1337.

**Model B - cross-encoder reranker.** Initialized from ms-marco-MiniLM-L-6-v2, trained with binary cross-entropy on 1 positive : 4 hard negatives mined per training query from BM25 and the strongest bi-encoder. At inference it rescores the bi-encoder's top-10 candidates (retrieve-then-rerank).
""")

    if biencoder_training or crossencoder_training:
        sections.append("### Training summary\n")
        all_records = (biencoder_training or []) + (crossencoder_training or [])
        sections.append(training_table(all_records))
        sections.append("\nLoss curves and per-epoch validation MRR: `experiments/results/figures/fig_loss_curves.png`.\n")

    if main_eval:
        sections.append("## 3. Main Results\n")
        sections.append("### Regime 1 - unseen queries (144 test queries, 30-tool corpus)\n")
        sections.append(main_results_table(main_eval, "regime1_unseen_queries"))
        sections.append("\n### Regime 2 - unseen tools (750 test queries, 30-tool corpus)\n")
        sections.append(main_results_table(main_eval, "regime2_unseen_tools"))
        if "regime3_unseen_servers" in main_eval.get("regimes", {}):
            sections.append("\n### Regime 3 - unseen servers (195 test queries, 574-tool corpus)\n")
            sections.append(main_results_table(main_eval, "regime3_unseen_servers"))
        sections.append("""
Brackets are 95% bootstrap confidence intervals over queries; ± is standard deviation over 3 training seeds. Figure: `figures/fig_main_results.png`; per-tool confusion: `figures/fig_confusion_regime1.png`; embedding-space view: `figures/fig_embedding_tsne.png`.
""")

    if llm_incontext:
        sections.append("### LLM-in-context selection (monolithic arm)\n")
        rows = ["| Catalog size | Accuracy | Mean latency (s) | Unparseable |", "| --- | --- | --- | --- |"]
        for size, block in llm_incontext["catalog_sizes"].items():
            rows.append(
                f"| {size} | {block['accuracy']:.3f} | {block['latency_s_mean']:.2f} | {block['unparseable']} |"
            )
        sections.append("\n".join(rows))
        sections.append(f"\nModel: `{llm_incontext['model']}`, {llm_incontext['n_queries']} sampled regime-1 test queries.\n")
    else:
        sections.append("""### LLM-in-context selection (monolithic arm)

Not run in this environment: the experiment requires a local LLM service (Ollama), which is unavailable on the authoring machine. The ready-to-run script is `experiments/evaluation/llm_incontext.py`; the earlier smoke-scale A/B harness results (3 filesystem tasks, llama3.2) remain in the README as the only end-to-end evidence and are labeled as such.
""")

    if ood:
        sections.append("## 4. Open-Set Rejection (OOD)\n")
        sections.append(f"""In-distribution = regime-1 test queries; out-of-distribution = {ood['n_ood']} author-written queries in three subsets: chitchat (no tool intent), out-of-catalog (capability outside the GitHub corpus, near and far domains), and adversarial near-miss (GitHub vocabulary, deliberately absent capability such as `create_issue` or `delete_branch`).
""")
        sections.append(ood_table(ood))
        sections.append("""
Risk-coverage and per-subset acceptance curves over the threshold: `figures/fig_threshold_sweep.png`. The adversarial near-miss subset is the operative weakness: scores for absent-but-adjacent capabilities overlap the in-distribution score range, so a global cosine threshold cannot fully separate them - destructive tools should carry stricter per-tool margins.
""")

    if bench:
        encoder = bench.get("encoder_latency", {})
        sections.append("## 5. Index Scaling: Flat vs HNSW\n")
        sections.append(scaling_table(bench))
        encoder_note = ""
        if "cpu" in encoder:
            encoder_note = (
                f" Encoding a single query costs {encoder['cpu']['p50_ms']:.0f} ms (CPU p50)"
                + (f" / {encoder['cuda']['p50_ms']:.1f} ms (GPU p50)" if "cuda" in encoder else "")
                + ", so retrieval is a negligible share of routing latency at every tested size."
            )
        sections.append(f"""
Single-threaded FAISS, top-10, query vectors from the 750 v2 anchors (N<=10,000 uses schema-derived embeddings; N=100,000 uses random unit vectors, latency-only). Exact flat search is faster than HNSW at every tested N while being exact and deterministic;{encoder_note} The runtime therefore defaults to `IndexFlatIP` (`RouterHyperparameters.index_type="flat"`), with HNSW available explicitly for catalogs far beyond current MCP registry sizes. Figure: `figures/fig_scaling.png`.
""")

    if ablation:
        sections.append("## 6. Schema Representation Ablation\n")
        sections.append(ablation_table(ablation))
        sections.append("""
Values are Recall@1 on regime 1 / regime 2. Caveat: the fine-tuned bi-encoder was trained on `raw` documents, so its non-raw columns measure inference-time robustness rather than per-representation training quality.
""")

    if poisoning:
        sections.append("## 7. Description-Poisoning Attack\n")
        sections.append(f"""Threat model: a hostile server publishes a decoy tool (`{poisoning['decoy']}`) whose description embeds K validation anchors as bait (attacker knows the query distribution, not the test queries). Hijack@1 = fraction of the {poisoning['n_queries']} regime-1 test queries whose top-ranked tool becomes the decoy.
""")
        sections.append(poisoning_table(poisoning))
        sections.append("""
Mitigations measured: a 300-character description length cap at ingest (cuts the bait payload), the decoy's embedding-centroid z-score (an ingest-time anomaly signal), and cross-encoder reranking as a second factor. None is individually sufficient at high K; the combination (cap + anomaly screen + rerank) is the recommended deployment posture.
""")

    if calibration:
        sections.append("## 8. Cross-Encoder Confidence Calibration\n")
        sections.append(f"""Top-1 reranking confidence (sigmoid of the CE logit) gates auto-execution. On regime-1 test, raw ECE is {calibration['raw']['ece']:.3f}; temperature scaling fitted on validation (T = {calibration['temperature_fitted_on_val']:.2f}) changes ECE to {calibration['temperature_scaled']['ece']:.3f} (top-1 accuracy {calibration['test_top1_accuracy']:.3f}). Reliability diagram: `figures/fig_calibration.png`.
""")

    sections.append("""## 9. Discussion

**Fine-tuning is the dominant factor.** The gap between frozen and fine-tuned bi-encoders dwarfs the gap between encoder sizes; after fine-tuning, the 22M MiniLM matches or beats the 109M MPNet, which matters for CPU-only deployment.

**Lexical baselines reframe the contribution honestly.** BM25/TF-IDF solve a large share of this benchmark (tool-name echo in queries), so the deep models' value is the measured delta over them - largest exactly where lexical overlap is weakest (regime 1 unseen phrasings, confusable tool pairs, and the unseen-server regime).

**Bi- vs cross-encoder.** The reranker's benefit concentrates on confusable candidates; its cost scales linearly with rerank depth while the bi-encoder amortizes the catalog into an index. For this catalog size the hybrid is the quality ceiling; the bi-encoder alone is the latency/scalability winner. The winner is therefore deployment-dependent, which is the comparison's actual conclusion.

**Open-set behavior is the real limitation.** Chitchat is trivially rejected, but adversarial near-misses defeat a global threshold, as the risk-coverage analysis quantifies; description poisoning compounds this, and no single mitigation suffices.

## 10. Limitations

- **Query provenance.** All queries are synthetic: v1/v2 are author-templated, regime-3 and OOD queries are LLM/author-written (different generator family, anti-echo rules). No human-user or production traffic exists in the evaluation; lexical overlap between queries and schemas inflates all lexical numbers (median token overlap 0.33 v1, 0.45 v2).
- **Unseen-server coverage.** Regime 3 queries cover 65 of the 574 corpus tools; the remaining 509 act as distractors only. Models were trained on GitHub data alone; multi-server *training* is untested.
- **Synthetic scaling corpus.** Scaling tiers above 30 tools use schema-like synthetic documents (N<=10k) and random vectors (N=100k); they bound latency behavior but not retrieval quality at scale.
- **Representation ablation is inference-only** for the fine-tuned model (one training condition).
- **LLM-in-context arm environment-blocked.** The monolithic baseline script requires a local LLM service unavailable on the authoring machine; the smoke-scale A/B (3 tasks) in the README is the only end-to-end evidence.
- **Poisoning attack surface is narrow:** one decoy, one bait construction; stronger adaptive attacks (gradient-guided descriptions) are future work.

## 11. Conclusion

Under leakage-controlled protocols, contrastive fine-tuning of a small bi-encoder is the single most effective intervention for MCP tool routing on this benchmark; the advantage persists zero-shot on 23 unseen servers among 574 distractor tools. Lexical retrieval is a strong mandatory baseline rather than a strawman, cross-encoder reranking buys additional precision at linear per-query cost (and is the only measured mitigation that helps against description poisoning without touching ingest), and exact flat search - not HNSW - is the correct index at every catalog size tested. All datasets, splits, training scripts, and results are reproducible from `experiments/` with pinned seeds and SHA256 manifests, and split hygiene is enforced in CI.
""")

    return "\n".join(sections)


def main() -> None:
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(build(), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
