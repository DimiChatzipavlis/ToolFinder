# ToolFinder — Evaluation

This documents **what we evaluated, how, with which scripts, and how rigorous it
is** — honestly, including where it is *not* yet production-grade. Every claim in
the README maps to a reproducible script here. The evaluation code lives on the
[`research`-branch / `legacy/`](../legacy/README.md) side of the repo
(`legacy/experiments/`); the source under test ships in [`toolfinder/`](../toolfinder/README.md).

## The question we set out to answer

Is ToolFinder a valuable MCP-server **wrapper** versus the simple "bind every
downstream tool to the LLM" approach — and **on which axis**: cost, accuracy, or
plain feasibility? We deliberately tried to *disprove* it, and report the axes
where it does **not** win.

## How the evaluation stays objective and stable

- **Objective metrics — no LLM-as-judge.** We use gold-labeled retrieval
  (Recall@k / MRR), end-task success **verified on disk**, **API-reported** token
  counts (`usage.*`, including `cached_tokens`), and forced-pick selection checked
  against a gold label. Nothing is scored by another model.
- **Deterministic where possible.** The bi-encoder and cross-encoder are
  deterministic, so the **retrieval evaluations reproduce exactly** on re-run —
  `n=1` is sufficient and stable for those.
- **Sampling controlled where the LLM is in the loop.** LLM arms are stochastic;
  we mitigate with **repeats** (e.g. 3 trials/cell) or larger query sets, and we
  flag small samples as noise.
- **Leakage control.** Retrieval is measured on held-out splits: `regime1`
  (unseen queries) and `regime4` (unseen *servers*) — not on training data.

## Evaluation scripts at a glance

| Script (`legacy/experiments/`) | What it measures | Uses | API $ | Objective / stable |
| --- | --- | --- | --- | --- |
| `bridge_ab.py` | A/B agent-loop harness (foundation) | tiktoken, OpenAI SDK, real filesystem MCP server, on-disk `verify()` | varies | yes / sampled |
| `bridge_scaling.py` | tokens + success vs catalog size, 3 arms; router recall@1/@3 (free axis) | real fs tools + 574-tool distractor pool, FAISS router, GPT-5.4 | low | yes / n=1 |
| `bridge_cache_aware.py` | **modeled** prompt-cache cost re-scoring | per-turn token structure from `bridge_scaling_gpt.json` | **$0** | yes / deterministic |
| `bridge_cache_measured.py` | **measured** `cached_tokens` + top-5 success | live GPT-5.4 `usage.prompt_tokens_details.cached_tokens` | small | yes / n=1 |
| `bridge_selection_accuracy.py` | forced single-pick accuracy vs N | 8 probes, gpt-4.1-mini vs gpt-5.4 vs router, `tool_choice="required"` | small | yes / small-n |
| `eval_rerank.py` | reranker lift, Recall@1/@3/@5 + MRR | GitHub corpus, `regime1` unseen-query test (144), cross-encoder | **$0** | yes / deterministic |
| `eval_rerank_finetuned.py` | fine-tune vs rerank vs both (2×2) | stock + fine-tuned MiniLM artifact, same 144 | **$0** | yes / deterministic |
| `eval_gateway_vs_baseline.py` | gateway vs bind-all: success + tokens, 3 trials | api_arm, real fs task, N=15/60/120, gpt-4.1-mini | ~$0.13 | yes / repeated |
| `eval_selection_at_scale.py` | selection accuracy + **bind feasibility** at scale | 574-tool pool, `regime4` held-out (24 q), N=60/250/574, gpt-4.1-mini | ~$0.10 | yes / small-n |
| `gateway_openapi_demo.py` | live OpenAPI gateway (ingest, route, dispatch) | Swagger Petstore spec over HTTP, OpenAPIClient | tiny | demonstration |
| `gateway_heterogeneous_demo.py` | one gateway over **MCP + OpenAPI at once** | real filesystem MCP server + Petstore, on-disk `verify()` | tiny | demonstration |

## What each script does inside (and what it proves)

**`bridge_ab.py`** — the shared agent loop (`run_arm`): binds a tool set, lets the
model call tools until done, counts tokens (`tiktoken` + API `usage`), and checks
task success on disk (`verify`). Backend-agnostic via the OpenAI SDK. Everything
else builds on it. *Exercises:* `mcp_adapter.DynamicMCPClient`, `UniversalMCPRouter`.

**`bridge_scaling.py`** — grows the catalog from 14 → N by padding the real
filesystem tools with real distractors from the 574-tool pool, and runs three
arms (baseline = all tools / `find_tools`+`call_tool` / `route_and_call`). The
*free axis* (no API) reports per-turn schema-token weight and router recall@1/@3;
the *API axis* reports end-to-end tokens + success. **Result:** uncached cost
≈ linear for baseline, flat for the gateway (~15× at N=120 with GPT-5.4).

**`bridge_cache_aware.py`** — re-scores those runs under prompt caching, **modeled
from the per-turn structure** (the static prefix repeats each turn). **Result:**
the "15×" becomes ~6–10× cached. Pure computation, deterministic, **$0**.

**`bridge_cache_measured.py`** — the live check: logs the API's actual
`cached_tokens` per turn and a top-5 arm. **Result:** confirms the model within
~1–2% at N≥60 (cached share 18%→70%→73% for N=14→120); top-5 selection completes
the task 100%. Validates `bridge_cache_aware.py`.

**`bridge_selection_accuracy.py`** — isolates selection: one forced tool call per
probe (`tool_choice="required"`), checked against gold, across N, for a weak model
(gpt-4.1-mini) vs a strong one (gpt-5.4) vs the router. **Result:** router 100%,
weak model 62–75%, strong 88–100% — routing helps weak models; for strong models
the value is cost.

**`eval_rerank.py`** — pure retrieval: rank all 30 GitHub tools per query (144
unseen), rerank OFF vs ON, report Recall@1/@3/@5 + MRR. **Result:** recall@1
**0.56 → 0.85**, MRR 0.71 → 0.91. Deterministic, **$0**. *Exercises:*
`dynamic_faiss_router`, `reranker`.

**`eval_rerank_finetuned.py`** — the 2×2 {stock, fine-tuned MiniLM} × {rerank
off, on} on the same 144. **Result:** fine-tune **0.99** > rerank 0.85 > stock
0.56; they are alternatives (stacking a stock cross-encoder on a fine-tuned base
*hurts*). Deterministic, **$0**.

**`eval_gateway_vs_baseline.py`** — strict end-task A/B: same model, same
filesystem task, baseline vs gateway, **3 trials/cell**, N=15/60/120. **Result:**
3/3 success both ways; gateway ~1.3× / 5× / 10× cheaper (uncached). Cost win that
grows with N; no accuracy gap on an easy task. *Exercises:* the full bridge path.

**`eval_selection_at_scale.py`** — the scaled, confusable stress test: 574-tool
pool, `regime4` held-out-server queries, baseline (bind all N, forced pick) vs
gateway (router→top-5, forced pick), N=60/250/574. **Result:** binding all tools
is **rejected by the API past ~128** (`tools array too long`), so baseline is
N/A at 250/574 while the gateway runs; where binding works (N=60) baseline 0.96 >
gateway 0.75 (gateway is router-bound). **Value at scale = feasibility + cost, not
accuracy.**

**`gateway_openapi_demo.py` / `gateway_heterogeneous_demo.py`** — live, end-to-end
demonstrations (not statistical): ToolFinder ingests a real OpenAPI spec
(Petstore) and a real MCP server (filesystem) **at once**, an agent binds only
ToolFinder's two tools, routes cross-source, and a filesystem op is **verified on
disk** (a real success, independent of any web API's uptime). Shows 20–32×
context reduction and per-server fault isolation.

## Source code under test (reference map)

| Module (`toolfinder/`) | Implements | Exercised by |
| --- | --- | --- |
| [`dynamic_faiss_router.py`](../toolfinder/dynamic_faiss_router.py) | `UniversalMCPRouter`: FAISS retrieval, cosine threshold abstention, server-aware hierarchical routing, rerank integration | every eval |
| [`reranker.py`](../toolfinder/reranker.py) | `CrossEncoderReranker` (opt-in) | `eval_rerank*`, `eval_selection_at_scale` |
| [`mcp_adapter.py`](../toolfinder/mcp_adapter.py) | `DynamicMCPClient` (stdio MCP) | `bridge_*`, gateway demos |
| [`openapi_adapter.py`](../toolfinder/openapi_adapter.py) | `OpenAPIClient` (REST via OpenAPI) | gateway demos |
| [`mcp_server.py`](../toolfinder/mcp_server.py) | FastMCP bridge: `find_tools`/`call_tool`/`route_and_call`/`get_stats`/`refresh`, per-server fault isolation | `tests/test_mcp_server.py`; demos replicate its dispatch |

## Headline results

- **Cost:** gateway ~10× fewer tokens at 120 tools (uncached); ~6–10× under
  caching (modeled + measured); marginal below ~30 tools.
- **Feasibility:** bind-all is **impossible past ~128 tools** (API limit) — the
  gateway is the only option at 250/574 tools.
- **Accuracy:** cross-encoder rerank lifts retrieval recall@1 0.56→0.85; a
  fine-tuned encoder 0.56→0.99. End-task accuracy is a wash when binding is
  feasible and the model is capable; the gateway's accuracy ceiling is the
  router's recall.

## Does this meet production-level evaluation rigor? (honest)

**What is already sound (good practice):**
- Objective metrics only (on-disk success, gold-labeled retrieval, API token
  counts) — **no LLM-as-judge**, no circularity.
- **Deterministic, reproducible** retrieval evaluations (re-run → identical).
- **Leakage-controlled** splits (unseen queries, unseen servers).
- Real execution (real MCP servers, real HTTP), and CI unit tests for the bridge
  logic incl. fault isolation.

**What is *not* production-grade (demonstration-level):**
- **Small samples** for the LLM-in-the-loop arms (n=1–3 trials; 8–24 queries) and
  **no confidence intervals** on those arms.
- **One end-task family** (filesystem create→edit→read) and **single hardware**.
- **Synthetic queries** (template- or LLM-generated), not human-written — external
  validity is unproven; the fine-tuned 0.99 is on a shared-template split and is
  likely optimistic.
- One/two models (gpt-4.1-mini, gpt-5.4); no latency-under-load or concurrency.

**What production-grade would require:** hundreds of **human-written** queries;
multiple real task families with verifiable outcomes; **repeats with bootstrap
CIs / paired significance** on every LLM arm; several models; and load/latency
tests. The retrieval evaluations here are the closest to that bar (objective,
deterministic, leakage-controlled); the end-task LLM evaluations are honest
**demonstrations**, not statistically powered benchmarks.

> In short: the **methodology is objective and leakage-aware**, and the
> **conclusions are directional and reproducible** — but the **statistical
> weight is demonstration-level**, not production-certified. Claims in the README
> are scoped accordingly.
