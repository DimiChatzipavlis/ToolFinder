# Neural Routing for Model Context Protocol (MCP) Servers

Abstract: This Semester Project replaces traditional LLM "context-stuffing" with a 109M parameter Bi-Encoder (all-mpnet-base-v2). By executing semantic vector search over JSON schemas, the system routes natural language to exact API tools in <60ms, eliminating hallucination and token waste.

## 1. Project Architecture

Data Synthesis: Explain the automated generation of mcp_routing_dataset.csv (15 core GitHub tools) and mcp_routing_dataset_v2.csv (15 unseen tools) using a 30/40/30 linguistic variance rule.

The Neural Router: Describe the use of MultipleNegativesRankingLoss to train the encoder to map human intent to complex JSON schemas.

The Execution Bridge: Explain how local_copilot_chat.py connects the deterministic neural router to a localized quantized LLM (Llama 3.2 via Ollama) to generate executable tool arguments.

## 2. Empirical Verification

Base Performance: State that the 109M parameter model achieved 100% Recall@1 on the training distribution.

Zero-Shot Generalization: Highlight that on the completely unseen V2 dataset, the model achieved 93.87% Recall@1 and 100% Recall@3 at 43ms latency, mathematically proving semantic generalization rather than dataset memorization.

## 3. Future Work: Multi-Server Scalability

Detail how the architecture scales to N-Servers. Explain the concept of Hierarchical Routing (Layer 1: Server Selection -> Layer 2: Tool Selection) and Continuous Ingestion (auto-generating synthetic pairs for new MCP servers to fine-tune the vector space).

## 4. Usage

```bash
unzip best_mcp_router.zip -d models/
python evaluate_zero_shot.py
python local_copilot_chat.py
```

4. Architectural Limitations & Future Work
While the dense retrieval prototype successfully demonstrates sub-60ms zero-shot routing, scaling this architecture to an enterprise environment requires addressing several theoretical and mechanical limits:

Context Window Truncation: The all-mpnet-base-v2 encoder has a strict 512-token limit. Analysis shows V1/V2 schemas average between 216 and 301 tokens, with maximums hitting 511 tokens. Future iterations must implement schema summarization or semantic chunking.

Latency Scaling (O(Nd)): The current architecture computes cosine similarity against every schema embedding simultaneously. Scaling to 5,000+ multi-server tools requires replacing the flat tensor scan with an Approximate Nearest Neighbor (ANN) index.

Semantic Collisions: The router operates in a flat namespace. If distinct MCP servers expose identically named tools with different structural arguments, the vector space may collapse them. Future work will introduce Hierarchical Routing (Layer 1: Server Selection -> Layer 2: Tool Selection).

Execution Validation: The pipeline relies on the localized LLM to natively respect prompt constraints. To guarantee zero hallucination in production, an intermediate validation layer (e.g., Pydantic or JSON Schema validation) must be wrapped around the LLM's output before API execution.
