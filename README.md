# ToolFinder: Neural Routing for MCP Servers

This is a Semester Project demonstrating how a Bi-Encoder neural network can replace standard LLM context-stuffing by routing natural language to specific Model Context Protocol (MCP) JSON schemas.

## 1. Project Architecture

Data Generation: Synthesized 750 rows of training data (15 distinct GitHub MCP tools) using live server schemas.

Neural Router: Trained using sentence-transformers and MultipleNegativesRankingLoss.

Agent Bridge: A local execution script (agent_bridge.py) that maps user queries to schemas in < 60ms and generates token-optimized prompts for SLMs.

## 2. Benchmarking Results

| Model | Params | Train Time (s) | Recall@1 (%) | Recall@3 (%) | Latency (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| all-MiniLM-L6-v2 | 22M | 13.68 | 98.67 | 100 | 8.4 |
| bge-small-en-v1.5 | 33M | 36.98 | 99.33 | 100 | 14.73 |
| all-mpnet-base-v2 | 109M | 94.15 | 100.00 | 100 | 17.66 |

Note: all-mpnet-base-v2 was selected for the final deployment due to achieving mathematically perfect Recall@1.

## 3. How to Run Locally

Clone the repo and run pip install -r requirements.txt.

Ensure the best_mcp_router model is extracted into models/best_mcp_router/.

Run python agent_bridge.py to process a query and generate the optimized SLM prompt.
