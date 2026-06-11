# examples/ — Runnable Demos and Benchmark Harnesses

Everything here exercises the *deployed* library (`toolfinder/`), as opposed to
the research pipeline in `experiments/`.

| Script | What it does | Requires |
| --- | --- | --- |
| `eval_toolfinder.py` | The naive-vs-routed A/B harness: runs identical filesystem tasks against a local model with (a) every tool bound vs (b) top-k routed tools; verifies task state; `--update-readme` rewrites the table between the `EVAL_TABLE` markers in the root README. Smoke-scale (n=3 tasks) — treat as plumbing evidence, not statistics. | Ollama (`llama3.2`), Node (`npx @modelcontextprotocol/server-filesystem`) |
| `tri_server_demo.py` | Multi-server agent demo: memory + sqlite + fetch MCP servers behind one router, a 4-step goal executed by the ReAct agent, with per-iteration routing latency and context-size metrics. (Renamed from `prove_scalability.py` — it demonstrates orchestration, not scaling; the scaling *benchmark* is `experiments/benchmarks/scaling_bench.py`.) | Ollama, Node |
| `verify_react_agent.py` | Minimal agent sanity check against a single filesystem server. | Ollama, Node |
| `langgraph_integration/benchmark_agent.py` | LangGraph graph with a semantic-router node; logs routed tools, context chars saved, and routing latency per turn. | Ollama, Node, `pip install -e ".[langgraph]"` |
| `langgraph_integration/baseline_agent.py` | The same graph without routing (all tools bound) for comparison. | same |
| `ToolFinder_StepByStep.ipynb` | Guided walkthrough notebook for the library API. | — |

## Environment notes

- All agent demos need a **local Ollama service** (`ollama pull llama3.2`) and
  **Node** for the `npx`-launched MCP servers. Without Ollama the scripts fail
  fast with a connection error — nothing is mocked.
- The sandbox directory used by the filesystem demos is recreated under
  `langgraph_integration/sandbox/` and gitignored.
