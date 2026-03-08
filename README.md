# Neural MCP Router: Zero-Shot Tool Selection for AI Agents

## 🚀 The Vision
Large Language Models are currently bottlenecked by "Context Stuffing." When an AI Agent is connected to 50+ Model Context Protocol (MCP) tools, pasting every massive JSON schema into the system prompt consumes thousands of tokens, increases latency, and causes hallucination.

**The Neural MCP Router** solves this. By decoupling **Tool Selection** from **Tool Execution**, this system uses a lightweight 109M parameter Bi-Encoder to mathematically map human intent directly to the correct JSON schema in **<50 milliseconds**. The LLM only sees the exact tool it needs, guaranteeing zero-hallucination execution.

## 🧠 Core Architecture
This repository has been hardened from an academic prototype into a production-safe routing layer:

* **Approximate Nearest Neighbor (ANN) Indexing:** Replaces flat $O(N)$ PyTorch tensor scans with Facebook's `FAISS` library, allowing the router to scale to thousands of tools logarithmically.
* **Smart Minification:** Dynamically strips nested parameter descriptions from JSON schemas during the embedding phase to prevent 512-token context truncation, while preserving top-level semantic hooks.
* **Deep JSON Validation:** Intercepts the LLM's raw output and enforces strict type, enum, and required-field compliance via Python's `jsonschema` library before execution.
* **Dynamic Cache Isolation:** Persists FAISS indexes to disk to eliminate $O(N)$ embedding overhead on startup.

## 📊 Empirical Performance
Evaluated on a strictly held-out dataset of unseen MCP tools, the router achieves:

* **Zero-Shot Recall@1:** `94.13%`
* **Zero-Shot Recall@3:** `100.00%`
* **Average Latency:** `~41.20 ms` (CPU)

## 🗺️ Roadmap: The Agentic Future
To transform this router into a fully autonomous, community-ready AI Agent framework, the following features are actively on the roadmap:

### 1. Hierarchical Routing (Multi-Server Ontology)
Currently, the router operates in a flat vector space. To support 50+ different MCP servers such as GitHub, Slack, Notion, and Jira, we will introduce a two-stage routing engine:

* **Layer 1:** Server Selection, for example routing the request to Slack.
* **Layer 2:** Tool Selection through FAISS retrieval within the selected server corpus.

### 2. The Action Layer (MCP Client Integration)
The router currently acts as the **Brain**, generating validated JSON. The next step is adding the **Hands**. We will integrate an official MCP Client SDK to securely execute the generated JSON payload against live servers and return the tool output back to the LLM for final synthesis.

### 3. Continuous Ingestion (Self-Healing)
A pipeline to automatically ingest, minify, embed, and cache new tools the moment a user connects a new MCP server, requiring zero manual dataset generation or fine-tuning.

### 4. Package Distribution
Abstracting the core `mcp_router.py` logic into a pip-installable package, `pip install neural-mcp-router`, so developers can drop it into any existing LangChain, LlamaIndex, or custom AI agent loop in three lines of code.

## 💻 Quickstart
1. Clone the repository and install dependencies.

```bash
git clone https://github.com/DimiChatzipavlis/ToolFinder.git
cd ToolFinder
pip install -r requirements.txt
```

2. Extract the model weights into `models/best_mcp_router`.

```bash
unzip best_mcp_router.zip -d models/
```

3. Run the zero-shot benchmark.

```bash
python evaluate_zero_shot.py
```

4. Run the interactive local agent.

```bash
python local_copilot_chat.py
```
