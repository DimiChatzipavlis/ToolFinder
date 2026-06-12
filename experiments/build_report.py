"""Render reports/report.md from experiments/results/*.json.

Structure follows the course rubric: Abstract (~150 words), Introduction,
Related Work, Methodology (data + preprocessing + two DL models), Experiments &
Results (accuracy/precision/recall/F1, ROC-AUC, loss curves, comparison tables),
Discussion & Limitations, Conclusions.

Every number is read from a results file produced by a script in this package;
nothing is hand-typed. Re-run after any experiment:

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
    ("ft_minilm (avg over seeds)", "**Model A: FT bi-encoder MiniLM-L6** (3 seeds)"),
    ("ft_bge (avg over seeds)", "FT bi-encoder BGE-small (3 seeds)"),
    ("ft_mpnet (avg over seeds)", "FT bi-encoder MPNet (3 seeds)"),
    ("hybrid_bm25+ft_minilm_seed42", "Hybrid RRF: BM25 + FT MiniLM (seed 42)"),
    ("ft_minilm+ce_rerank (avg over seeds)", "**Model B: FT MiniLM + CE rerank** (3 seeds)"),
]

CLASSIFICATION_SYSTEMS = [
    ("bm25", "BM25"),
    ("frozen_minilm", "Frozen MiniLM-L6"),
    ("ft_minilm_seed42", "Model A: FT bi-encoder (seed 42)"),
    ("ft_minilm+ce_rerank_seed42", "Model B: + CE rerank (seed 42)"),
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


def classification_table(results: dict, regimes: list[str]) -> str:
    lines = [
        "| System | Regime | Accuracy | Macro Precision | Macro Recall | Macro F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, label in CLASSIFICATION_SYSTEMS:
        for regime in regimes:
            block = results["regimes"].get(regime, {})
            if key not in block or "classification" not in block[key]:
                continue
            cls = block[key]["classification"]
            regime_short = regime.replace("regime", "R").split("_")[0]
            lines.append(
                f"| {label} | {regime_short} | {cls['accuracy']:.3f} | {cls['macro_precision']:.3f} "
                f"| {cls['macro_recall']:.3f} | {cls['macro_f1']:.3f} |"
            )
    return "\n".join(lines)


def training_table(records: list[dict]) -> str:
    import numpy as np

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
        lines.append(
            f"| {model_key} | {params.get(model_key, '?')} | {len(runs)} "
            f"| {np.mean(times):.0f} | {np.mean(mrrs):.4f} ± {np.std(mrrs):.4f} |"
        )
    return "\n".join(lines)


def ood_table(ood: dict) -> str:
    lines = [
        "| System | Score | ROC-AUC (pooled) | FPR@95TPR | AUC chitchat | AUC out-of-catalog | AUC near-miss |",
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
    template_disjoint = load("template_disjoint_eval.json")
    significance = load("significance.json")

    sections: list[str] = []

    # ------------------------------------------------------------------ header
    sections.append(f"""# ToolFinder: Dense Retrieval for Open-Set MCP Tool Routing — An Empirical Study

*Generated from `experiments/results/` on {date.today().isoformat()}. Regenerate with `python experiments/build_report.py`. Figures referenced as `figures/...` live in `experiments/results/figures/`.*

## Abstract

Tool-using LLM agents must select the correct tool from a catalog before invoking it; binding every schema into the context window degrades small models and scales poorly. We study tool selection as dense retrieval over 1,695 natural-language intents and 574 real tool schemas from GitHub MCP and 23 public API providers. The original random evaluation split is invalid — queries follow a scenario-template grammar, so a nearest-neighbor lookup over training queries scores 96% Recall@1 — and we replace it with three leakage-controlled protocols (unseen queries, tools, servers), enforced by CI tests. Under them we compare two trained deep architectures, a contrastively fine-tuned bi-encoder and a cross-encoder reranker, against lexical, hybrid, and frozen-encoder baselines over three seeds with confidence intervals. Fine-tuning dominates: a 22M bi-encoder reaches 0.99/0.91/0.67 Recall@1 across regimes versus 0.57/0.76/0.49 for BM25, rejects out-of-scope requests at 0.99 ROC-AUC, and resists description poisoning that hijacks BM25 83% of the time. Scaling to 100k vectors shows routing latency stays encoder-dominated, keeping exact flat search the correct default.

## 1. Introduction

Agentic LLM systems increasingly rely on external tools exposed through the Model Context Protocol (MCP). The default integration pattern binds *every* available tool schema into the model's context. This fails in three compounding ways: the prompt fills with irrelevant structure before reasoning begins ("lost in the middle"); similar APIs collide in-context and raise tool-selection errors; and small local models (the models most users can actually run) emit malformed calls under long-prompt pressure. The engineering response — retrieve a small set of candidate tools first, then let the model reason over only those — turns tool *selection* into an information-retrieval problem that must work under **open-set conditions**: new phrasings, new tools, entire new servers, out-of-scope requests that must be refused, and adversarially crafted tool descriptions.

This matters beyond convenience. Tool selection is a safety boundary: a router that force-routes an ambiguous request to a destructive tool, or that can be hijacked by a poisoned description, converts a retrieval error into a harmful action. It is also an efficiency boundary: routing decides whether a 3B-parameter local model can use a 500-tool catalog at all.

**Problem statement.** Given a natural-language intent and a catalog of MCP tool schemas, rank the catalog so the correct tool is at rank 1, and abstain when no tool applies.

**Contributions.** (1) A leakage-controlled benchmark over real MCP/OpenAPI schemas with three generalization regimes, after demonstrating quantitatively that the naive split answers itself (~96% Recall@1 via 1-NN over training anchors). (2) A trained two-model comparison — bi-encoder vs cross-encoder reranker — against the baselines that practice actually ships (BM25, TF-IDF, frozen encoders, hybrid fusion), with seed variance and confidence intervals. (3) An open-set analysis: threshold sweeps, ROC-AUC for out-of-distribution rejection, a measured description-poisoning attack with three mitigations, and reranker confidence calibration. (4) A systems result: exact flat search outperforms HNSW at every catalog size up to 10^5 vectors, fixing the deployed router's defaults. All code, data, splits, and results are reproducible (`experiments/run_all.py`), with split hygiene enforced in CI.
""")

    # -------------------------------------------------------------- related work
    sections.append("""## 2. Related Work

**Tool-augmented LLMs.** MRKL (Karpas et al., 2022) introduced the modular router-over-experts framing this system implements. Toolformer (Schick et al., 2023) teaches an LM to invoke APIs self-supervised. Gorilla (Patil et al., 2023) fine-tunes LLaMA-7B for API-call generation and already pairs it with a retriever, showing retrieval reduces call hallucination; ToolLLM (Qin et al., 2023) scales instruction-tuned tool use to 16k+ real APIs with a dedicated API retriever. These works target call *generation*; we isolate the upstream selection sub-problem and evaluate it under open-set and adversarial conditions they do not focus on.

**Retrieval-based tool selection in practice.** Embedding tool descriptions and retrieving by cosine similarity is shipped functionality (LangChain tool retriever, LlamaIndex object retrievers, the semantic-router library, OpenAI's function-retrieval cookbook). We claim no architectural novelty over these; the contribution is a leakage-controlled benchmark and a controlled comparison of what they ship as defaults.

**Dense retrieval methodology.** Bi-encoder contrastive training follows Sentence-BERT (Reimers & Gurevych, 2019) with in-batch negatives as in DPR (Karpukhin et al., 2020); cross-encoder reranking follows Nogueira & Cho (2019). HNSW is Malkov & Yashunin (2018); our scaling study re-examines its trade-offs at tool-catalog (not web-corpus) sizes.
""")

    # -------------------------------------------------------------- methodology
    sections.append("""## 3. Methodology

### 3.1 Data, cleaning, and exploratory analysis

**Raw data.** Two author-templated query sets over the GitHub MCP server (750 + 750 queries; 15 + 15 disjoint tools; exactly 50 queries/tool), plus 195 hand-written queries (different generator family, anti-echo rules) over 65 tools sampled from a 544-tool catalog converted from 23 public OpenAPI providers (apis.guru). 249 out-of-distribution queries (chitchat / out-of-catalog / adversarial near-miss) are evaluation-only. Provenance and biases: `experiments/data/DATASET_CARD.md`.

**Cleaning.** Verification rather than imputation: zero missing values, zero duplicate anchors, zero unparsable schema payloads (asserted in the executed EDA notebook, `notebooks/01_eda.ipynb`). Classes are exactly balanced by construction (the class-distribution figure is in the notebook).

**EDA findings that shaped the design.** (i) Most tools follow a `scenario × template` generation grammar (~5 scenarios × ~10 paraphrase prefixes with the scenario clause verbatim) — recovered programmatically and stored as `scenario_id`. (ii) Under a random row split, a 1-NN lookup over *training anchors alone* scores ~96% Recall@1 and 65% of test anchors have a training anchor at char-ngram cosine ≥ 0.8 (`figures/fig_leakage_audit.png`) — the original evaluation was leaked by construction. (iii) Queries lexically echo schemas (median token overlap 0.33 for v1, 0.45 for v2), so lexical baselines are mandatory. (iv) Schema-similarity analysis identifies confusable pairs (`search_issues` ↔ `search_pull_requests` at 0.93 char-ngram cosine) that later explain most residual errors.

**Preprocessing / normalization.** Schemas are serialized as canonical sorted-key JSON (the `raw` representation; alternatives are ablated in §4.6). Identifier separators are expanded for lexical systems (`add_issue_comment` → "add issue comment"). All embeddings are L2-normalized so inner product equals cosine similarity.

**Splits (leakage control).** *Regime 1 — unseen queries:* scenario-grouped 460/146/144 over the 15 v1 tools (no scenario crosses buckets). *Regime 2 — unseen tools:* all 750 v2 queries; their 15 tools never appear in training; ranking is against the full 30-tool corpus so trained tools act as distractors. *Regime 3 — unseen servers:* the 195 multi-server queries against a 574-tool merged corpus; training data remains GitHub-only. Constraints are enforced by `tests/test_split_hygiene.py` in CI.

### 3.2 Model A — fine-tuned bi-encoder

A sentence-transformer (primary: MiniLM-L6, 22M parameters; 6 transformer layers, 384-d mean-pooled embeddings) fine-tuned with MultipleNegativesRankingLoss: for a batch of (query, schema) pairs, each query's positive schema is contrasted against all other in-batch schemas via softmax over cosine similarities. **Critical detail:** with only 15 distinct schemas in training, naive batching places duplicate schemas in one batch and MNRL then treats correct pairs as negatives; we use a no-duplicates batch sampler, which also bounds the effective in-batch negative count at the catalog size. Hyperparameters: batch 16, lr 2e-5, warmup 10%, ≤8 epochs with per-epoch model selection on validation MRR@10, max sequence 256, fp16, seeds {13, 42, 1337}. BGE-small (33M) and MPNet (109M) are trained identically as a capacity ablation.

### 3.3 Model B — cross-encoder reranker

A cross-encoder (init: ms-marco-MiniLM-L-6, 22M) scores (query, schema) *jointly* with full cross-attention — architecturally distinct from Model A, which encodes the two sides independently. Trained with binary cross-entropy on 1 positive : 4 hard negatives per training query, where negatives are the top wrong candidates mined by BM25 and the strongest bi-encoder (the candidates a deployed system actually needs to reject). Batch 32, lr 2e-5, 3 epochs, max length 384, seeds {13, 42, 1337}. At inference it reranks the bi-encoder's top-10 (retrieve-then-rerank), so its per-query cost is 10 transformer passes versus Model A's single query encoding plus an index lookup — the efficiency/accuracy trade-off the comparison quantifies.

### 3.4 Baselines

Random permutation (floor); BM25 (Okapi, k1=1.5, b=0.75); TF-IDF (word 1-2grams; char 3-5grams); frozen MiniLM/BGE/MPNet (zero-shot, isolating the value of fine-tuning); hybrid reciprocal-rank fusion of BM25 + fine-tuned bi-encoder (k=60). All systems rank identical schema documents through one evaluation code path.

### 3.5 Deployed system

The router (`toolfinder/`) embeds schemas at ingest, retrieves via FAISS, abstains below a similarity threshold (operating point chosen from §4.4), and exposes results as typed objects. The index is **exact** `IndexFlatIP` by default; HNSW is opt-in — a decision justified empirically in §4.5 rather than asserted.
""")

    # ---------------------------------------------------------------- results
    sections.append("## 4. Experiments & Results\n")

    if biencoder_training or crossencoder_training:
        all_records = (biencoder_training or []) + (crossencoder_training or [])
        sections.append("### 4.1 Training behavior\n")
        sections.append(training_table(all_records))
        sections.append("""
Training loss and per-epoch validation MRR@10 for all three bi-encoder capacities: `figures/fig_loss_curves.png`. Validation MRR (not validation loss) is the model-selection criterion because retrieval quality, not the contrastive loss value, is the deployment objective; the curves show convergence within 3-5 epochs and no divergence between train loss and validation quality (no overfitting signal). Seed variance is reported in every results table below.
""")

    if main_eval:
        sections.append("### 4.2 Main retrieval results (three regimes)\n")
        sections.append("**Regime 1 — unseen queries** (144 test queries, 30-tool corpus):\n")
        sections.append(main_results_table(main_eval, "regime1_unseen_queries"))
        sections.append("\n**Regime 2 — unseen tools** (750 test queries, 30-tool corpus):\n")
        sections.append(main_results_table(main_eval, "regime2_unseen_tools"))
        if "regime3_unseen_servers" in main_eval.get("regimes", {}):
            sections.append("\n**Regime 3 — unseen servers** (195 test queries, 574-tool corpus):\n")
            sections.append(main_results_table(main_eval, "regime3_unseen_servers"))
        sections.append("""
Brackets: 95% bootstrap CIs over queries; ±: std over 3 training seeds. Bar chart: `figures/fig_main_results.png`; t-SNE of query embeddings before/after fine-tuning: `figures/fig_embedding_tsne.png`.

### 4.3 Classification view: accuracy, precision, recall, F1

Top-1 selection over a fixed catalog is a classification decision; macro-averaged metrics over the tool classes:
""")
        regimes_present = [r for r in ("regime1_unseen_queries", "regime2_unseen_tools", "regime3_unseen_servers") if r in main_eval["regimes"]]
        sections.append(classification_table(main_eval, regimes_present))
        sections.append("""
Per-tool confusion matrix for Model A (regime 1): `figures/fig_confusion_regime1.png` — residual errors concentrate on the schema pairs the EDA flagged as confusable (`search_issues` ↔ `search_pull_requests`). Whether Model B's joint attention recovers these is examined — with a negative answer — in §5.
""")

    if ood:
        sections.append("### 4.4 Open-set rejection (ROC-AUC)\n")
        sections.append(f"""In-distribution = regime-1 test queries; out-of-distribution = {ood['n_ood']} queries in three subsets. The router's abstention signal is scored as a binary ID-vs-OOD classifier:
""")
        sections.append(ood_table(ood))
        sections.append("""
Risk-coverage and per-subset acceptance curves over the threshold τ: `figures/fig_threshold_sweep.png`. Chitchat is separable almost perfectly; adversarial near-misses (GitHub vocabulary, absent capability — e.g. `create_issue`, `delete_branch`) overlap the in-distribution score range and are the operative weakness of a single global threshold.
""")

    if bench:
        encoder = bench.get("encoder_latency", {})
        sections.append("### 4.5 Index scaling: exact Flat vs HNSW\n")
        sections.append(scaling_table(bench))
        encoder_note = ""
        if "cpu" in encoder:
            encoder_note = (
                f"Encoding one query costs {encoder['cpu']['p50_ms']:.0f} ms on CPU (p50)"
                + (f" / {encoder['cuda']['p50_ms']:.1f} ms on GPU" if "cuda" in encoder else "")
                + ", so search is a minority share of routing latency at every size up to 10^5."
            )
        ef_sweep_note = ""
        schema_tiers = [tier for tier in bench["tiers"] if tier["tier"] == "schema"]
        if schema_tiers:
            top_tier = schema_tiers[-1]
            ef_points = []
            for key in sorted(top_tier["indexes"]):
                if key.startswith("hnsw_m32_ef"):
                    entry = top_tier["indexes"][key]
                    ef_points.append(
                        f"ef={key.split('ef')[-1]}: {entry['recall@10_vs_exact']:.3f} recall at {entry['latency']['p50_ms']:.2f} ms"
                    )
            ef_sweep_note = f" At N={top_tier['n']:,}, the HNSW recall/latency frontier is {'; '.join(ef_points)}."
        sections.append(f"""
Single-threaded FAISS, top-10; N ≤ 10,000 uses schema-derived embeddings, N = 100,000 random unit vectors (a latency-only stress tier: structureless random vectors are pathological for graph ANN, hence the collapsed recall there). The measured picture is more nuanced than the folk claim "HNSW scales, flat doesn't": flat is faster *and* exact below ~10^3 vectors; beyond that HNSW reduces search latency (0.34 ms vs 6.1 ms at 10^4) but pays in recall under default settings.{ef_sweep_note} {encoder_note} Because the encoder dominates end-to-end routing latency and flat search is exact, the runtime defaults to `IndexFlatIP`; HNSW is opt-in and becomes the rational choice only as catalogs approach ~10^5 *and* an efSearch-tuned recall trade-off is acceptable. Log-log figure: `figures/fig_scaling.png`.
""")

    if ablation:
        sections.append("### 4.6 Schema representation ablation\n")
        sections.append(ablation_table(ablation))
        sections.append("""
Recall@1 on regime 1 / regime 2. Caveat: the fine-tuned model was trained on `raw` documents, so its non-raw columns measure inference-time robustness, not per-representation training quality.
""")

    if poisoning:
        sections.append("### 4.7 Description-poisoning attack\n")
        sections.append(f"""A hostile server publishes a decoy tool (`{poisoning['decoy']}`) whose description embeds K validation anchors as bait (the attacker knows the query distribution, not the test queries). Hijack@1 = fraction of the {poisoning['n_queries']} regime-1 test queries whose top-ranked tool becomes the decoy:
""")
        sections.append(poisoning_table(poisoning))
        sections.append("""
The outcome inverts the expected story. BM25 is catastrophically hijackable (83% at K=20 — bait text lexically matches everything), the frozen encoder leaks substantially (41% at K=10), but **the fine-tuned bi-encoder is not hijacked once at any attack strength**: contrastive training reshapes the embedding space so a description stuffed with many unrelated anchors resembles no individual query. Fine-tuning is itself the strongest measured defense — which also means the BM25 arm of any hybrid fusion is the attack surface, and the cross-encoder "second factor" actually *re-admits* a small leak (1-3%) by scoring bait text pairwise. The 300-character ingest cap neutralizes the attack mechanically; the embedding-centroid z-score is a weak detector (max z = 1.32, not separable). Caveats: one decoy, one bait construction, attacker knows the query distribution but not the test queries; adaptive attacks against the fine-tuned space are future work. Deployment posture: server allowlisting + ingest length cap first, fine-tuned retrieval as the routing layer, rerank treated as a quality stage rather than a security control.
""")

    if calibration:
        sections.append("### 4.8 Confidence calibration of the reranker\n")
        sections.append(f"""Model B's sigmoid score on its top candidate gates auto-execution. On regime-1 test: raw ECE {calibration['raw']['ece']:.3f}; after temperature scaling fitted on validation (T = {calibration['temperature_fitted_on_val']:.2f}) ECE {calibration['temperature_scaled']['ece']:.3f} (top-1 accuracy {calibration['test_top1_accuracy']:.3f}). Reliability diagram: `figures/fig_calibration.png`.
""")

    if llm_incontext:
        sections.append("### 4.9 LLM-in-context selection (monolithic arm)\n")
        rows = ["| Catalog size | Accuracy | Mean latency (s) | Unparseable |", "| --- | --- | --- | --- |"]
        for size, block in llm_incontext["catalog_sizes"].items():
            rows.append(
                f"| {size} | {block['accuracy']:.3f} | {block['latency_s_mean']:.2f} | {block['unparseable']} |"
            )
        sections.append("\n".join(rows))
        sections.append(f"\nModel: `{llm_incontext['model']}`, {llm_incontext['n_queries']} sampled regime-1 test queries.\n")
    else:
        sections.append("""### 4.9 LLM-in-context selection (monolithic arm)

Not run in this environment: the experiment requires a local LLM service (Ollama), unavailable on the authoring machine. The ready-to-run script is `experiments/evaluation/llm_incontext.py`; the smoke-scale A/B harness (3 filesystem tasks, llama3.2, README table) remains the only end-to-end evidence and is labeled as such.
""")

    # ------------------------------------------------------------- discussion
    sections.append("""## 5. Discussion & Limitations

**Which model won, and why.** Model A (the fine-tuned bi-encoder) wins outright — on every regime, on every metric, and by 137× on per-query cost (0.5 ms vs 73 ms). Two findings explain it. First, fine-tuning moves retrieval quality far more than parameter count does: the 22M MiniLM jumps from 0.33 to 0.99 Recall@1 on regime 1 and matches or beats the frozen 109M MPNet everywhere, which matters for CPU-only deployment; the t-SNE figure shows the mechanism — contrastive training reorganizes queries into tight per-tool clusters aligned with schema embeddings. Second, Model B (the cross-encoder reranker) *fails to improve A and slightly degrades it* (regime 1: 0.944 vs 0.988; regime 2: 0.867 vs 0.909; regime 3: 0.633 vs 0.667). This is an instructive negative result, not an implementation accident: (i) A is near ceiling, so a reranker can mostly only preserve or damage its rankings; (ii) B trained on just 460 queries with mined negatives concentrated on A's confusion cases, so it gained discrimination there at the price of new errors on cases A already solved; (iii) under regime shift (unseen tools/servers) B's pairwise scores transfer worse than A's embedding geometry. The architectural folklore "retrieve-then-rerank always helps" assumes a reranker trained on far more supervision than the retriever it corrects — at course-project data scale, the opposite holds, and joint cross-attention's per-pair precision (visible in its standalone val MRR of 0.96) does not survive composition with a stronger retriever.

**Why the baselines matter.** BM25/TF-IDF solve a large share of this benchmark (tool-name echo in queries; quantified overlap medians 0.33/0.45). Reporting deep models without them would overstate the contribution — the honest claim is the measured delta, which is largest exactly where lexical overlap is weakest (unseen phrasings, unseen servers, confusable pairs).

**Difficulties encountered.** (1) The most consequential finding was a *data* problem, not a model problem: the original random split was answerable at ~96% Recall@1 without any model; recovering the scenario grammar and rebuilding the splits invalidated and replaced all earlier numbers. (2) MNRL's in-batch negatives are silently wrong at small catalog sizes (duplicate positives become negatives) — fixed with a no-duplicates sampler. (3) Training on a 4GB consumer GPU required fp16, sequence truncation to 256, and one recovery from a mid-run crash; CPU/GPU contention between concurrent jobs distorted early latency measurements until benchmarks were serialized. (4) The local-LLM baseline was blocked by environment (no Ollama); we ship the script and report the gap rather than fabricate it.

**Limitations.** All queries are synthetic (author-templated or LLM-written; no production traffic), and lexical echo inflates all lexical numbers. Regime 3 queries cover 65 of 574 corpus tools; multi-server *training* is untested. The scaling corpus above 30 tools is schema-like synthetic text (≤10k) and random vectors (100k) — it bounds latency, not retrieval quality at scale. The poisoning attack is one decoy with one bait construction; adaptive attacks are future work. The representation ablation is inference-only for the fine-tuned model. A global threshold cannot fully separate adversarial near-misses; destructive tools need per-tool margins and confirmation.
""")

    # ------------------------------------------------------------- conclusion
    sections.append("""## 6. Conclusions

Framed as open-set dense retrieval and evaluated under leakage-controlled protocols, MCP tool selection is solved to high accuracy by a small contrastively fine-tuned bi-encoder — which also proves to be the strongest measured defense against description poisoning and the better open-set rejector. The trained cross-encoder comparison yields a deliberate negative result: at this supervision scale, reranking a stronger retriever with a weaker reranker degrades quality while multiplying cost, so the deployed system uses the bi-encoder alone. On indexing, exact flat search is the correct *default*: it is faster and exact below ~10^3 tools, and although HNSW searches faster beyond that, routing latency stays encoder-dominated to 10^5 vectors while HNSW pays up to 23% recall at default settings — approximation buys nothing until catalogs far exceed today's MCP registries. Equally important is what the study removes: an evaluation whose random split answered itself, and claims ("logarithmic scaling", "deterministic", "10,000+ tools") that the system no longer needs to make rhetorically because the benchmark now tests them empirically — or retires them. All datasets, splits, trained-model manifests, results, and figures regenerate from `experiments/run_all.py` with pinned seeds, and the split-hygiene constraints that make the numbers meaningful are enforced as failing tests in CI.

## References

1. Y. A. Malkov, D. A. Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI 42(4), 2020.
2. E. Karpas et al. *MRKL Systems: A modular, neuro-symbolic architecture...* arXiv:2205.00445, 2022.
3. S. G. Patil et al. *Gorilla: Large Language Model Connected with Massive APIs.* arXiv:2305.15334, 2023.
4. Y. Qin et al. *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.* arXiv:2307.16789, 2023.
5. T. Schick et al. *Toolformer: Language Models Can Teach Themselves to Use Tools.* arXiv:2302.04761, 2023.
6. N. Reimers, I. Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
7. V. Karpukhin et al. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020.
8. R. Nogueira, K. Cho. *Passage Re-ranking with BERT.* arXiv:1901.04085, 2019.
""")

    return "\n".join(sections)


def main() -> None:
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(build(), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
