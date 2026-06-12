# ToolFinder: Dense Retrieval for Open-Set MCP Tool Routing — An Empirical Study

*Generated from `experiments/results/` on 2026-06-12. Regenerate with `python experiments/build_report.py`. Figures referenced as `figures/...` live in `experiments/results/figures/`.*

## Abstract

Tool-using LLM agents must select the correct tool from a catalog before invoking it; binding every schema into the context window degrades small models and scales poorly. We study tool selection as dense retrieval over 1,695 natural-language intents and 574 real tool schemas from GitHub MCP and 23 public API providers. The original random evaluation split is invalid — queries follow a scenario-template grammar, so a nearest-neighbor lookup over training queries scores 96% Recall@1 — and we replace it with three leakage-controlled protocols (unseen queries, tools, servers), enforced by CI tests. Under them we compare two trained deep architectures, a contrastively fine-tuned bi-encoder and a cross-encoder reranker, against lexical, hybrid, and frozen-encoder baselines over three seeds with confidence intervals. Fine-tuning dominates: a 22M bi-encoder reaches 0.99/0.91/0.67 Recall@1 across regimes versus 0.57/0.76/0.49 for BM25, rejects out-of-scope requests at 0.99 ROC-AUC, and resists description poisoning that hijacks BM25 83% of the time. Scaling to 100k vectors shows routing latency stays encoder-dominated, keeping exact flat search the correct default.

## 1. Introduction

Agentic LLM systems increasingly rely on external tools exposed through the Model Context Protocol (MCP). The default integration pattern binds *every* available tool schema into the model's context. This fails in three compounding ways: the prompt fills with irrelevant structure before reasoning begins ("lost in the middle"); similar APIs collide in-context and raise tool-selection errors; and small local models (the models most users can actually run) emit malformed calls under long-prompt pressure. The engineering response — retrieve a small set of candidate tools first, then let the model reason over only those — turns tool *selection* into an information-retrieval problem that must work under **open-set conditions**: new phrasings, new tools, entire new servers, out-of-scope requests that must be refused, and adversarially crafted tool descriptions.

This matters beyond convenience. Tool selection is a safety boundary: a router that force-routes an ambiguous request to a destructive tool, or that can be hijacked by a poisoned description, converts a retrieval error into a harmful action. It is also an efficiency boundary: routing decides whether a 3B-parameter local model can use a 500-tool catalog at all.

**Problem statement.** Given a natural-language intent and a catalog of MCP tool schemas, rank the catalog so the correct tool is at rank 1, and abstain when no tool applies.

**Contributions.** (1) A leakage-controlled benchmark over real MCP/OpenAPI schemas with three generalization regimes, after demonstrating quantitatively that the naive split answers itself (~96% Recall@1 via 1-NN over training anchors). (2) A trained two-model comparison — bi-encoder vs cross-encoder reranker — against the baselines that practice actually ships (BM25, TF-IDF, frozen encoders, hybrid fusion), with seed variance and confidence intervals. (3) An open-set analysis: threshold sweeps, ROC-AUC for out-of-distribution rejection, a measured description-poisoning attack with three mitigations, and reranker confidence calibration. (4) A systems result: exact flat search outperforms HNSW at every catalog size up to 10^5 vectors, fixing the deployed router's defaults. All code, data, splits, and results are reproducible (`experiments/run_all.py`), with split hygiene enforced in CI.

## 2. Related Work

**Tool-augmented LLMs.** MRKL (Karpas et al., 2022) introduced the modular router-over-experts framing this system implements. Toolformer (Schick et al., 2023) teaches an LM to invoke APIs self-supervised. Gorilla (Patil et al., 2023) fine-tunes LLaMA-7B for API-call generation and already pairs it with a retriever, showing retrieval reduces call hallucination; ToolLLM (Qin et al., 2023) scales instruction-tuned tool use to 16k+ real APIs with a dedicated API retriever. These works target call *generation*; we isolate the upstream selection sub-problem and evaluate it under open-set and adversarial conditions they do not focus on.

**Retrieval-based tool selection in practice.** Embedding tool descriptions and retrieving by cosine similarity is shipped functionality (LangChain tool retriever, LlamaIndex object retrievers, the semantic-router library, OpenAI's function-retrieval cookbook). We claim no architectural novelty over these; the contribution is a leakage-controlled benchmark and a controlled comparison of what they ship as defaults.

**Dense retrieval methodology.** Bi-encoder contrastive training follows Sentence-BERT (Reimers & Gurevych, 2019) with in-batch negatives as in DPR (Karpukhin et al., 2020); cross-encoder reranking follows Nogueira & Cho (2019). HNSW is Malkov & Yashunin (2018); our scaling study re-examines its trade-offs at tool-catalog (not web-corpus) sizes.

## 3. Methodology

### 3.1 Data, cleaning, and exploratory analysis

**Raw data.** Two author-templated query sets over the GitHub MCP server (750 + 750 queries; 15 + 15 disjoint tools; exactly 50 queries/tool), plus 195 hand-written queries (different generator family, anti-echo rules) over 65 tools sampled from a 544-tool catalog converted from 23 public OpenAPI providers (apis.guru). 249 out-of-distribution queries (chitchat / out-of-catalog / adversarial near-miss) are evaluation-only. Provenance and biases: `experiments/data/DATASET_CARD.md`.

**Cleaning.** Verification rather than imputation: zero missing values, zero duplicate anchors, zero unparsable schema payloads (asserted in the executed EDA notebook, `notebooks/01_eda.ipynb`). Classes are exactly balanced by construction (the class-distribution figure is in the notebook).

**EDA findings that shaped the design.** (i) Most tools follow a `scenario × template` generation grammar (~5 scenarios × ~10 paraphrase prefixes with the scenario clause verbatim) — recovered programmatically and stored as `scenario_id`. (ii) Under a random row split, a 1-NN lookup over *training anchors alone* scores ~96% Recall@1 and 65% of test anchors have a training anchor at char-ngram cosine ≥ 0.8 (`figures/fig_leakage_audit.png`) — the original evaluation was leaked by construction. (iii) Queries lexically echo schemas (median token overlap 0.33 for v1, 0.45 for v2), so lexical baselines are mandatory. (iv) Schema-similarity analysis identifies confusable pairs (`search_issues` ↔ `search_pull_requests` at 0.93 char-ngram cosine) that later explain most residual errors.

**Preprocessing / normalization.** Schemas are serialized as canonical sorted-key JSON (the `raw` representation; alternatives are ablated in §4.6). Identifier separators are expanded for lexical systems (`add_issue_comment` → "add issue comment"). All embeddings are L2-normalized so inner product equals cosine similarity.

**Splits (leakage control).** *Regime 1 — unseen queries:* scenario-grouped 460/146/144 over the 15 v1 tools (no scenario crosses buckets). *Regime 1b — template-disjoint control:* because the grammar is `template × scenario`, regime 1 still shares the ten surface templates per tool across train and test, so a model could in principle score perfectly by memorizing template→tool mappings while ignoring the scenario clause; regime 1b holds out templates *and* scenarios jointly (304/42/40 rows after discarding mixed blocks) and is the controlled measurement of in-grammar generalization. Regime-1 numbers are therefore reported as in-grammar **upper bounds**. *Regime 2 — unseen tools:* all 750 v2 queries; their 15 tools never appear in training; ranking is against the full 30-tool corpus so trained tools act as distractors. *Regime 3 — unseen servers:* the 195 multi-server queries against a 574-tool merged corpus; training data remains GitHub-only. All constraints, including double disjointness for 1b, are enforced by `tests/test_split_hygiene.py` in CI.

### 3.2 Model A — fine-tuned bi-encoder

A sentence-transformer (primary: MiniLM-L6, 22M parameters; 6 transformer layers, 384-d mean-pooled embeddings) fine-tuned with MultipleNegativesRankingLoss: for a batch of (query, schema) pairs, each query's positive schema is contrasted against all other in-batch schemas via softmax over cosine similarities. **Critical detail:** with only 15 distinct schemas in training, naive batching places duplicate schemas in one batch and MNRL then treats correct pairs as negatives; we use a no-duplicates batch sampler, which also bounds the effective in-batch negative count at the catalog size. Hyperparameters: batch 16, lr 2e-5, warmup 10%, ≤8 epochs with per-epoch model selection on validation MRR@10, max sequence 256, fp16, seeds {13, 42, 1337}. BGE-small (33M) and MPNet (109M) are trained identically as a capacity ablation.

### 3.3 Model B — cross-encoder reranker

A cross-encoder (init: ms-marco-MiniLM-L-6, 22M) scores (query, schema) *jointly* with full cross-attention — architecturally distinct from Model A, which encodes the two sides independently. Trained with binary cross-entropy on 1 positive : 4 hard negatives per training query, where negatives are the top wrong candidates mined by BM25 and the strongest bi-encoder (the candidates a deployed system actually needs to reject). Batch 32, lr 2e-5, 3 epochs, max length 384, seeds {13, 42, 1337}. At inference it reranks the bi-encoder's top-10 (retrieve-then-rerank), so its per-query cost is 10 transformer passes versus Model A's single query encoding plus an index lookup — the efficiency/accuracy trade-off the comparison quantifies.

### 3.4 Baselines

Random permutation (floor); BM25 (Okapi, k1=1.5, b=0.75); TF-IDF (word 1-2grams; char 3-5grams); frozen MiniLM/BGE/MPNet (zero-shot, isolating the value of fine-tuning); hybrid reciprocal-rank fusion of BM25 + fine-tuned bi-encoder (k=60). All systems rank identical schema documents through one evaluation code path.

### 3.5 Deployed system

The router (`toolfinder/`) embeds schemas at ingest, retrieves via FAISS, abstains below a similarity threshold (operating point chosen from §4.4), and exposes results as typed objects. The index is **exact** `IndexFlatIP` by default; HNSW is opt-in — a decision justified empirically in §4.5 rather than asserted.

## 4. Experiments & Results

### 4.1 Training behavior

| Model | Params | Seeds | Train time (s, mean) | Val MRR@10 (mean ± std) |
| --- | --- | --- | --- | --- |
| minilm | 22M | 3 | 221 | 0.9863 ± 0.0000 |
| bge | 33M | 3 | 430 | 1.0000 ± 0.0000 |
| mpnet | 109M | 3 | 1957 | 1.0000 ± 0.0000 |
| crossencoder | 22M | 3 | 535 | 0.9591 ± 0.0029 |

Training loss and per-epoch validation MRR@10 for all three bi-encoder capacities: `figures/fig_loss_curves.png`. Validation MRR (not validation loss) is the model-selection criterion because retrieval quality, not the contrastive loss value, is the deployment objective; the curves show convergence within 3-5 epochs and no divergence between train loss and validation quality (no overfitting signal). Seed variance is reported in every results table below.

### 4.2 Main retrieval results (three regimes)

**Regime 1 — unseen queries** (144 test queries, 30-tool corpus):

| System | R@1 | R@3 | MRR | NDCG@10 |
| --- | --- | --- | --- | --- |
| Random | 0.035 [0.007, 0.069] | 0.083 [0.042, 0.132] | 0.126 [0.098, 0.156] | 0.136 [0.099, 0.173] |
| BM25 | 0.569 [0.493, 0.646] | 0.771 [0.701, 0.833] | 0.688 [0.629, 0.745] | 0.748 [0.695, 0.796] |
| TF-IDF (word 1-2g) | 0.576 [0.493, 0.660] | 0.771 [0.701, 0.840] | 0.692 [0.633, 0.752] | 0.746 [0.694, 0.799] |
| TF-IDF (char 3-5g) | 0.486 [0.403, 0.569] | 0.792 [0.729, 0.854] | 0.663 [0.606, 0.720] | 0.738 [0.691, 0.784] |
| Frozen MiniLM-L6 (22M) | 0.326 [0.257, 0.403] | 0.625 [0.549, 0.694] | 0.510 [0.456, 0.564] | 0.604 [0.557, 0.650] |
| Frozen BGE-small (33M) | 0.604 [0.528, 0.688] | 0.826 [0.764, 0.889] | 0.731 [0.676, 0.784] | 0.789 [0.745, 0.833] |
| Frozen MPNet (109M) | 0.611 [0.528, 0.694] | 0.931 [0.889, 0.965] | 0.772 [0.723, 0.820] | 0.827 [0.790, 0.864] |
| **Model A: FT bi-encoder MiniLM-L6** (3 seeds) | 0.988 ± 0.003 | 1.000 ± 0.000 | 0.994 ± 0.002 | 0.996 ± 0.001 |
| FT bi-encoder BGE-small (3 seeds) | 0.991 ± 0.003 | 1.000 ± 0.000 | 0.995 ± 0.002 | 0.997 ± 0.001 |
| FT bi-encoder MPNet (3 seeds) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Hybrid RRF: BM25 + FT MiniLM (seed 42) | 0.826 [0.764, 0.889] | 0.958 [0.924, 0.993] | 0.891 [0.851, 0.928] | 0.914 [0.881, 0.945] |
| **Model B: FT MiniLM + CE rerank** (3 seeds) | 0.944 ± 0.011 | 1.000 ± 0.000 | 0.972 ± 0.006 | 0.979 ± 0.004 |

**Regime 2 — unseen tools** (750 test queries, 30-tool corpus):

| System | R@1 | R@3 | MRR | NDCG@10 |
| --- | --- | --- | --- | --- |
| Random | 0.043 [0.028, 0.057] | 0.115 [0.092, 0.139] | 0.145 [0.129, 0.160] | 0.168 [0.149, 0.186] |
| BM25 | 0.759 [0.729, 0.788] | 0.945 [0.928, 0.961] | 0.856 [0.837, 0.874] | 0.889 [0.874, 0.903] |
| TF-IDF (word 1-2g) | 0.773 [0.744, 0.803] | 0.965 [0.951, 0.977] | 0.869 [0.851, 0.885] | 0.900 [0.887, 0.913] |
| TF-IDF (char 3-5g) | 0.767 [0.736, 0.795] | 0.969 [0.956, 0.980] | 0.863 [0.844, 0.880] | 0.896 [0.882, 0.909] |
| Frozen MiniLM-L6 (22M) | 0.701 [0.671, 0.736] | 0.956 [0.940, 0.971] | 0.824 [0.806, 0.845] | 0.866 [0.851, 0.882] |
| Frozen BGE-small (33M) | 0.864 [0.839, 0.888] | 1.000 [1.000, 1.000] | 0.929 [0.916, 0.942] | 0.948 [0.938, 0.957] |
| Frozen MPNet (109M) | 0.755 [0.723, 0.783] | 0.967 [0.953, 0.979] | 0.862 [0.843, 0.879] | 0.894 [0.880, 0.908] |
| **Model A: FT bi-encoder MiniLM-L6** (3 seeds) | 0.909 ± 0.003 | 1.000 ± 0.001 | 0.953 ± 0.002 | 0.965 ± 0.001 |
| FT bi-encoder BGE-small (3 seeds) | 0.865 ± 0.002 | 0.994 ± 0.003 | 0.924 ± 0.002 | 0.944 ± 0.002 |
| FT bi-encoder MPNet (3 seeds) | 0.915 ± 0.001 | 1.000 ± 0.000 | 0.958 ± 0.001 | 0.969 ± 0.000 |
| Hybrid RRF: BM25 + FT MiniLM (seed 42) | 0.835 [0.809, 0.859] | 0.989 [0.981, 0.996] | 0.910 [0.896, 0.924] | 0.933 [0.923, 0.944] |
| **Model B: FT MiniLM + CE rerank** (3 seeds) | 0.867 ± 0.004 | 0.978 ± 0.002 | 0.922 ± 0.002 | 0.942 ± 0.002 |

**Regime 3 — unseen servers** (195 test queries, 574-tool corpus):

| System | R@1 | R@3 | MRR | NDCG@10 |
| --- | --- | --- | --- | --- |
| Random | 0.005 [0.000, 0.015] | 0.005 [0.000, 0.015] | 0.014 [0.007, 0.026] | 0.007 [0.000, 0.021] |
| BM25 | 0.487 [0.420, 0.554] | 0.672 [0.605, 0.733] | 0.597 [0.538, 0.652] | 0.636 [0.575, 0.690] |
| TF-IDF (word 1-2g) | 0.359 [0.292, 0.426] | 0.610 [0.544, 0.682] | 0.502 [0.448, 0.557] | 0.554 [0.503, 0.610] |
| TF-IDF (char 3-5g) | 0.426 [0.359, 0.497] | 0.672 [0.605, 0.733] | 0.568 [0.512, 0.625] | 0.625 [0.573, 0.676] |
| Frozen MiniLM-L6 (22M) | 0.451 [0.385, 0.523] | 0.667 [0.600, 0.728] | 0.577 [0.523, 0.635] | 0.623 [0.569, 0.678] |
| Frozen BGE-small (33M) | 0.554 [0.482, 0.621] | 0.774 [0.718, 0.831] | 0.681 [0.629, 0.730] | 0.736 [0.691, 0.779] |
| Frozen MPNet (109M) | 0.436 [0.364, 0.508] | 0.703 [0.641, 0.764] | 0.595 [0.541, 0.651] | 0.649 [0.598, 0.703] |
| **Model A: FT bi-encoder MiniLM-L6** (3 seeds) | 0.667 ± 0.008 | 0.874 ± 0.011 | 0.775 ± 0.004 | 0.817 ± 0.003 |
| FT bi-encoder BGE-small (3 seeds) | 0.644 ± 0.005 | 0.863 ± 0.011 | 0.761 ± 0.005 | 0.802 ± 0.007 |
| FT bi-encoder MPNet (3 seeds) | 0.727 ± 0.005 | 0.942 ± 0.002 | 0.831 ± 0.004 | 0.866 ± 0.004 |
| Hybrid RRF: BM25 + FT MiniLM (seed 42) | 0.595 [0.523, 0.661] | 0.769 [0.713, 0.831] | 0.707 [0.656, 0.757] | 0.752 [0.704, 0.798] |
| **Model B: FT MiniLM + CE rerank** (3 seeds) | 0.632 ± 0.009 | 0.863 ± 0.009 | 0.753 ± 0.005 | 0.800 ± 0.003 |

Brackets: 95% bootstrap CIs over queries; ±: std over 3 training seeds. Bar chart: `figures/fig_main_results.png`; t-SNE of query embeddings before/after fine-tuning: `figures/fig_embedding_tsne.png`.

**Template-disjoint control (regime 1b).** Regime 1's near-perfect scores are in-grammar upper bounds: its test rows reuse training-set surface templates. Retraining on the doubly-disjoint split (templates and scenarios both unseen; 40 test rows) gives the controlled number, with the 1-NN-over-training-queries probe quantifying residual surface leakage on each split:

| System | R@1 | R@3 | MRR |
| --- | --- | --- | --- |
| Random | 0.025 [0.000, 0.075] | 0.025 [0.000, 0.075] | 0.104 [0.070, 0.161] |
| 1-NN over training queries (leakage probe) | 0.650 [0.500, 0.800] | 0.875 [0.750, 0.975] | 0.774 [0.674, 0.877] |
| BM25 | 0.475 [0.325, 0.625] | 0.650 [0.500, 0.800] | 0.588 [0.465, 0.711] |
| TF-IDF (char 3-5g) | 0.425 [0.275, 0.575] | 0.725 [0.575, 0.850] | 0.603 [0.493, 0.711] |
| Frozen MiniLM-L6 | 0.325 [0.200, 0.475] | 0.600 [0.450, 0.750] | 0.498 [0.385, 0.615] |
| **FT MiniLM, retrained on 1b** (3 seeds) | 0.958 ± 0.012 | 1.000 ± 0.000 | 0.978 ± 0.007 |
| *(same probe on regime 1, for contrast)* | 0.868 [0.806, 0.917] | 0.979 [0.951, 1.000] | 0.922 [0.886, 0.952] |

**Statistical significance.** Paired bootstrap over test queries (10,000 resamples) of the fine-tuned MiniLM against BM25, per seed and regime — the fine-tuning advantage is significant everywhere, with confidence intervals excluding zero: regime1 unseen queries: ΔR@1 = +0.417..+0.424 (95% CI [+0.340, +0.507] across seeds), p ≤ 0.0001; regime2 unseen tools: ΔR@1 = +0.147..+0.155 (95% CI [+0.113, +0.189] across seeds), p ≤ 0.0001; regime3 unseen servers: ΔR@1 = +0.169..+0.190 (95% CI [+0.092, +0.267] across seeds), p ≤ 0.0001. Full table: `results/significance.json`.

### 4.3 Classification view: accuracy, precision, recall, F1

Top-1 selection over a fixed catalog is a classification decision; macro-averaged metrics over the tool classes:

| System | Regime | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| BM25 | R1 | 0.569 | 0.509 | 0.395 | 0.426 |
| BM25 | R2 | 0.759 | 0.574 | 0.517 | 0.530 |
| BM25 | R3 | 0.487 | 0.345 | 0.244 | 0.272 |
| Frozen MiniLM-L6 | R1 | 0.326 | 0.242 | 0.186 | 0.180 |
| Frozen MiniLM-L6 | R2 | 0.701 | 0.580 | 0.526 | 0.526 |
| Frozen MiniLM-L6 | R3 | 0.451 | 0.300 | 0.214 | 0.235 |
| Model A: FT bi-encoder (seed 42) | R1 | 0.993 | 0.938 | 0.931 | 0.934 |
| Model A: FT bi-encoder (seed 42) | R2 | 0.905 | 0.772 | 0.754 | 0.756 |
| Model A: FT bi-encoder (seed 42) | R3 | 0.667 | 0.507 | 0.380 | 0.416 |
| Model B: + CE rerank (seed 42) | R1 | 0.944 | 0.896 | 0.882 | 0.885 |
| Model B: + CE rerank (seed 42) | R2 | 0.867 | 0.585 | 0.542 | 0.558 |
| Model B: + CE rerank (seed 42) | R3 | 0.641 | 0.499 | 0.375 | 0.410 |

Per-tool confusion matrix for Model A (regime 1): `figures/fig_confusion_regime1.png` — residual errors concentrate on the schema pairs the EDA flagged as confusable (`search_issues` ↔ `search_pull_requests`). Whether Model B's joint attention recovers these is examined — with a negative answer — in §5.

### 4.4 Open-set rejection (ROC-AUC)

In-distribution = regime-1 test queries; out-of-distribution = 249 queries in three subsets. The router's abstention signal is scored as a binary ID-vs-OOD classifier:

| System | Score | ROC-AUC (pooled) | FPR@95TPR | AUC chitchat | AUC out-of-catalog | AUC near-miss |
| --- | --- | --- | --- | --- | --- | --- |
| ft_bge_seed13 | max_sim | 0.988 | 0.076 | 1.000 | 0.988 | 0.964 |
| ft_bge_seed13 | margin | 0.968 | 0.185 | 0.982 | 0.972 | 0.931 |
| frozen_mpnet | max_sim | 0.910 | 0.369 | 0.991 | 0.906 | 0.757 |
| frozen_mpnet | margin | 0.681 | 0.912 | 0.741 | 0.653 | 0.618 |

Risk-coverage and per-subset acceptance curves over the threshold τ: `figures/fig_threshold_sweep.png`. Chitchat is separable almost perfectly; adversarial near-misses (GitHub vocabulary, absent capability — e.g. `create_issue`, `delete_branch`) overlap the in-distribution score range and are the operative weakness of a single global threshold.

### 4.5 Index scaling: exact Flat vs HNSW

| N | Tier | Flat p50 (ms) | HNSW M32/ef64 p50 (ms) | HNSW recall@10 | Flat size | HNSW size |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | schema | 0.0078 | 0.0108 | 1.0000 | 0.0 MB | 0.1 MB |
| 100 | schema | 0.0192 | 0.0317 | 1.0000 | 0.3 MB | 0.3 MB |
| 1,000 | schema | 0.3183 | 0.1561 | 0.9975 | 3.1 MB | 3.3 MB |
| 10,000 | schema | 6.1370 | 0.3381 | 0.9136 | 30.7 MB | 33.4 MB |
| 100,000 | random | 42.6321 | 1.0510 | 0.0948 | 307.2 MB | 334.4 MB |

Single-threaded FAISS, top-10; N ≤ 10,000 uses schema-derived embeddings, N = 100,000 random unit vectors (a latency-only stress tier: structureless random vectors are pathological for graph ANN, hence the collapsed recall there). The measured picture is more nuanced than the folk claim "HNSW scales, flat doesn't": flat is faster *and* exact below ~10^3 vectors; beyond that HNSW reduces search latency (0.34 ms vs 6.1 ms at 10^4) but pays in recall under default settings. At N=10,000, the HNSW recall/latency frontier is ef=128: 0.962 recall at 0.63 ms; ef=16: 0.769 recall at 0.10 ms; ef=256: 0.981 recall at 1.34 ms; ef=64: 0.914 recall at 0.34 ms. Encoding one query costs 86 ms on CPU (p50) / 33.2 ms on GPU, so search is a minority share of routing latency at every size up to 10^5. Because the encoder dominates end-to-end routing latency and flat search is exact, the runtime defaults to `IndexFlatIP`; HNSW is opt-in and becomes the rational choice only as catalogs approach ~10^5 *and* an efSearch-tuned recall trade-off is acceptable. Log-log figure: `figures/fig_scaling.png`.

### 4.6 Schema representation ablation

| Representation | BM25 R@1 (r1/r2) | TF-IDF char R@1 (r1/r2) | Frozen MPNet R@1 (r1/r2) | FT bi-encoder R@1 (r1/r2) |
| --- | --- | --- | --- | --- |
| raw | 0.569 / 0.759 | 0.486 / 0.767 | 0.611 / 0.755 | 0.986 / 0.865 |
| minified | 0.514 / 0.753 | 0.514 / 0.749 | 0.625 / 0.796 | 0.986 / 0.860 |
| name_desc | 0.646 / 0.749 | 0.688 / 0.809 | 0.750 / 0.811 | 0.986 / 0.907 |
| desc_only | 0.590 / 0.715 | 0.653 / 0.664 | 0.715 / 0.768 | 0.986 / 0.876 |

Recall@1 on regime 1 / regime 2. Caveat: the fine-tuned model was trained on `raw` documents, so its non-raw columns measure inference-time robustness, not per-representation training quality.

### 4.7 Description-poisoning attack

A hostile server publishes a decoy tool (`workspace_notes_sync`) whose description embeds K validation anchors as bait (the attacker knows the query distribution, not the test queries). Hijack@1 = fraction of the 144 regime-1 test queries whose top-ranked tool becomes the decoy:

| Bait anchors K | BM25 hijack@1 | Frozen MPNet hijack@1 | FT bi-encoder hijack@1 | FT + length cap | FT + CE rerank | Decoy centroid z |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.3 |
| 1 | 0.035 | 0.000 | 0.000 | 0.000 | 0.000 | 0.6 |
| 5 | 0.514 | 0.153 | 0.000 | 0.000 | 0.028 | 1.0 |
| 10 | 0.729 | 0.410 | 0.000 | 0.000 | 0.014 | 1.3 |
| 20 | 0.833 | 0.188 | 0.000 | 0.000 | 0.035 | 1.2 |

The outcome inverts the expected story. BM25 is catastrophically hijackable (83% at K=20 — bait text lexically matches everything), the frozen encoder leaks substantially (41% at K=10), but **the fine-tuned bi-encoder is not hijacked once at any attack strength**: contrastive training reshapes the embedding space so a description stuffed with many unrelated anchors resembles no individual query. Fine-tuning is itself the strongest measured defense — which also means the BM25 arm of any hybrid fusion is the attack surface, and the cross-encoder "second factor" actually *re-admits* a small leak (1-3%) by scoring bait text pairwise. The 300-character ingest cap neutralizes the attack mechanically; the embedding-centroid z-score is a weak detector (max z = 1.32, not separable). Caveats: one decoy, one bait construction, attacker knows the query distribution but not the test queries; adaptive attacks against the fine-tuned space are future work. Deployment posture: server allowlisting + ingest length cap first, fine-tuned retrieval as the routing layer, rerank treated as a quality stage rather than a security control.

### 4.8 Confidence calibration of the reranker

Model B's sigmoid score on its top candidate gates auto-execution. On regime-1 test: raw ECE 0.131; after temperature scaling fitted on validation (T = 0.95) ECE 0.128 (top-1 accuracy 0.944). Reliability diagram: `figures/fig_calibration.png`.

### 4.9 LLM-in-context selection (monolithic arm)

Not run in this environment: the experiment requires a local LLM service (Ollama), unavailable on the authoring machine. The ready-to-run script is `experiments/evaluation/llm_incontext.py`; the smoke-scale A/B harness (3 filesystem tasks, llama3.2, README table) remains the only end-to-end evidence and is labeled as such.

## 5. Discussion & Limitations

**Which model won, and why.** Model A (the fine-tuned bi-encoder) wins outright — on every regime, on every metric, and by 137× on per-query cost (0.5 ms vs 73 ms). Two findings explain it. First, fine-tuning moves retrieval quality far more than parameter count does: the 22M MiniLM jumps from 0.33 to 0.99 Recall@1 on regime 1 and matches or beats the frozen 109M MPNet everywhere, which matters for CPU-only deployment; the t-SNE figure shows the mechanism — contrastive training reorganizes queries into tight per-tool clusters aligned with schema embeddings. Second, Model B (the cross-encoder reranker) *fails to improve A and slightly degrades it* (regime 1: 0.944 vs 0.988; regime 2: 0.867 vs 0.909; regime 3: 0.633 vs 0.667). This is an instructive negative result, not an implementation accident: (i) A is near ceiling, so a reranker can mostly only preserve or damage its rankings; (ii) B trained on just 460 queries with mined negatives concentrated on A's confusion cases, so it gained discrimination there at the price of new errors on cases A already solved; (iii) under regime shift (unseen tools/servers) B's pairwise scores transfer worse than A's embedding geometry. The architectural folklore "retrieve-then-rerank always helps" assumes a reranker trained on far more supervision than the retriever it corrects — at course-project data scale, the opposite holds, and joint cross-attention's per-pair precision (visible in its standalone val MRR of 0.96) does not survive composition with a stronger retriever.

**Why the baselines matter.** BM25/TF-IDF solve a large share of this benchmark (tool-name echo in queries; quantified overlap medians 0.33/0.45). Reporting deep models without them would overstate the contribution — the honest claim is the measured delta, which is largest exactly where lexical overlap is weakest (unseen phrasings, unseen servers, confusable pairs).

**Difficulties encountered.** (1) The most consequential finding was a *data* problem, not a model problem: the original random split was answerable at ~96% Recall@1 without any model; recovering the scenario grammar and rebuilding the splits invalidated and replaced all earlier numbers. (2) MNRL's in-batch negatives are silently wrong at small catalog sizes (duplicate positives become negatives) — fixed with a no-duplicates sampler. (3) Training on a 4GB consumer GPU required fp16, sequence truncation to 256, and one recovery from a mid-run crash; CPU/GPU contention between concurrent jobs distorted early latency measurements until benchmarks were serialized. (4) The local-LLM baseline was blocked by environment (no Ollama); we ship the script and report the gap rather than fabricate it.

**Limitations.** All queries are synthetic with a *known generation grammar* (author-templated or LLM-written; no production traffic, no human-written test set): regime-1 numbers are in-grammar upper bounds because surface templates cross the split — quantified and controlled by regime 1b, but the grammar itself remains the data's ceiling, and lexical echo inflates all lexical numbers. The library's **shipped default is the zero-shot encoder, not the evaluated best**: the fine-tuned weights that produce the headline numbers are regenerable from pinned seeds but not committed, and zero-shot dense retrieval loses to BM25 here (disclosed in README, router docstring, and STATUS.md). The LLM-in-context arm is environment-blocked (script ships, unrun). Regime 3 queries cover 65 of 574 corpus tools; multi-server *training* is untested. The scaling corpus above 30 tools is schema-like synthetic text (≤10k) and random vectors (100k) — it bounds latency, not retrieval quality at scale. The poisoning attack is one decoy with one bait construction; adaptive attacks are future work. The representation ablation is inference-only for the fine-tuned model. A global threshold cannot fully separate adversarial near-misses; destructive tools need per-tool margins and confirmation. Latency numbers are single-hardware, English-only queries throughout.

## 6. Conclusions

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
