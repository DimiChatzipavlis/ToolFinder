# ToolFinder: Neural Semantic Router for MCP

🧠 Stop stuffing your LLM context windows. ToolFinder is a zero-hallucination routing middleware that dynamically connects Local SLMs to massive Model Context Protocol (MCP) ecosystems without OOM crashes.

It separates tool selection from tool execution, so your model only sees the few schemas it actually needs. The result is lower latency, tighter prompts, and a much safer path from local-model experimentation to production-grade MCP orchestration.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-compatible-black)](https://modelcontextprotocol.io/)
[![FAISS](https://img.shields.io/badge/retrieval-FAISS-orange)](https://github.com/facebookresearch/faiss)
[![LangGraph](https://img.shields.io/badge/integration-LangGraph-green)](https://github.com/langchain-ai/langgraph)

## 🚨 The Problem: Context Bloat & Lost in the Middle

Blindly binding `50` MCP tools to a small local model like `llama3.2` is a systems mistake.

- The prompt fills with irrelevant schemas before reasoning even begins.
- The model spends tokens comparing tools instead of using them.
- Similar APIs start colliding in-context, which increases tool-selection errors and malformed calls.
- Even when the right tool is chosen, smaller models often emit partially invalid JSON under long prompt pressure.

This is the classic "lost in the middle" failure mode applied to MCP orchestration: the right tool may exist in context, but the model has to wade through too much irrelevant structure to use it reliably.

## 🎯 The Solution: Semantic Anchoring

ToolFinder turns MCP tool usage into a retrieval problem first.

- A contrastive bi-encoder built on `sentence-transformers/all-mpnet-base-v2` embeds queries and MCP schemas into the same vector space.
- A FAISS-backed similarity index retrieves only the top-k candidate tools for the user’s query.
- The model then reasons over a tiny, relevant tool surface instead of the entire ecosystem.

The rigorous data science, datasets, and semester-project evaluation pipeline live in [academic_research](academic_research). That folder contains the training corpora, notebooks, and zero-shot evaluation stack behind the retrieval layer.

> Technical note:
> The current runtime implementation uses FAISS `IndexFlatIP` for exact dense retrieval with very low observed latency. The architectural scaling story generalizes cleanly to ANN indexes when larger tool graphs justify sublinear search.

## 📊 Empirical Benchmarks: The Proof

The repository contains two proof surfaces:

- A focused LangGraph A/B benchmark showing the first-turn efficiency win of semantic routing.
- A self-bootstrapping multi-task evaluator that continuously refreshes the benchmark table below.

### Headline Result

- ~95% prompt payload reduction in the focused LangGraph proof (`9110` chars to `485` chars)
- ~84% latency reduction in the same first-turn comparison (`85.52s` to `13.71s`)
- Sub-20ms to low-double-digit-ms routing in the LangGraph routing path

### Auto-Updating Benchmark

The block below is maintained by `python examples/eval_toolfinder.py`. Preserve the markers so the automated suite can continue injecting the latest metrics.

<!-- EVAL_TABLE_START -->
_Last auto-updated: 2026-03-12 19:20:44_

| Metric | Naive Baseline | ToolFinder Enabled |
| --- | --- | --- |
| Tasks Run | 3 | 3 |
| Average Tools In Context | 14 | 2 |
| Average Context Payload (Chars) | 9106 | 1450 |
| Average Total Latency (s) | 49.98 | 22.69 |
| Average Inference Latency (s) | 49.9 | 22.57 |
| Successful Tool Calls | 3/3 | 3/3 |
| Expected Tool Matches | 3/3 | 3/3 |
| State Verified | 3/3 | 3/3 |

Task outcomes:
- T1_READ: naive=`read_text_file` verified=`True`, toolfinder=`read_text_file` verified=`True`
- T2_WRITE: naive=`write_file` verified=`True`, toolfinder=`write_file` verified=`True`
- T3_LIST: naive=`list_directory` verified=`True`, toolfinder=`list_directory` verified=`True`
<!-- EVAL_TABLE_END -->

<details>
<summary>What changed between the headline proof and the live table?</summary>

The headline LangGraph benchmark measures a narrower, first-turn filesystem task and highlights the raw routing advantage. The auto-updating evaluator is broader: it runs multiple tasks, performs correctness checks, bootstraps and tears down the sandbox, and averages end-to-end inference time across the suite. The exact percentages shift, but the systems conclusion remains the same: routing a tiny tool subset is materially cheaper and safer than context stuffing.

</details>

## 🛡️ Features & Protections

ToolFinder hardens both selection and execution.

- Semantic routing narrows the prompt to the top-k MCP tools before inference.
- Strict schema enforcement injects `additionalProperties: false` into object schemas to reject speculative keys.
- AST recovery parsing salvages Python-style dicts and malformed local-model outputs when strict JSON fails.
- ReAct execution loops let the agent observe failures, retry, and continue rather than crash on the first malformed response.
- Idempotency guards and bounded scratchpads prevent repeated actions and runaway context growth.

## ⚡ Quickstart & Integration

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

Minimal integration with LangChain or LangGraph:

```python
from toolfinder.dynamic_faiss_router import UniversalMCPRouter
from langchain_ollama import ChatOllama
router = UniversalMCPRouter(); [router.add_tool(tool) for tool in mcp_server_tools]; router.build_index()
llm = ChatOllama(model="llama3.2")
response = llm.bind_tools(router.route_top_k("Write a summary to output.txt", k=2)).invoke("Write a summary to output.txt")
```

For a complete end-to-end proof, run:

```bash
python examples/eval_toolfinder.py
python -u examples/langgraph_integration/benchmark_agent.py
python -u examples/langgraph_integration/baseline_agent.py
```

## 🗂️ Repository Structure

- [toolfinder](toolfinder): Core package. FAISS routing, MCP ingestion, schema hardening, parsing recovery, and autonomous execution.
- [examples](examples): Integration proofs. LangGraph benchmark, baseline comparison, self-bootstrapping evaluator, and orchestration demos.
- [academic_research](academic_research): Semester project assets. Training data, notebooks, model artifacts, and evaluation code underpinning the semantic routing layer.

## 🔬 Why This Architecture Works

ToolFinder treats tool use as a systems architecture problem, not a prompt formatting trick.

- Retrieval handles scale.
- Middleware validation handles malformed outputs.
- ReAct orchestration handles recovery.

That separation is the reason the same architecture can support small local SLMs, larger hosted models, and expanding MCP ecosystems without turning every new server into prompt debt.

## 📄 For Senior Review

If you want the engineering whitepaper summary rather than the developer landing page, see [ARCHITECTURE_REPORT.md](ARCHITECTURE_REPORT.md).
