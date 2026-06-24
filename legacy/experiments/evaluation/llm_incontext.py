"""LLM-in-context tool selection baseline (the monolithic arm).

Asks a local LLM (via Ollama's HTTP API) to pick the right tool when ALL
candidate schemas are stuffed into the prompt, at several catalog sizes. This
is the comparison the routing architecture is premised on: accuracy and
latency of in-context selection should degrade as the catalog grows, while
retrieval-based routing stays flat.

ENVIRONMENT REQUIREMENT: a running Ollama service (http://localhost:11434)
with the target model pulled (default llama3.2). This experiment was NOT run
in the authoring environment because Ollama was unavailable; the script exists
so the result can be produced on any machine with a local model. The report
marks this arm as environment-blocked rather than reporting fabricated numbers.

Usage:
    ollama pull llama3.2
    python experiments/evaluation/llm_incontext.py --catalog-sizes 15 30 --n-queries 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402
from experiments.representation import represent_name_desc  # noqa: E402

OLLAMA_URL = "http://localhost:11434"

PROMPT_TEMPLATE = """You are a tool-routing assistant. Given a user request and a numbered catalog of tools, answer with ONLY the number of the single best tool.

TOOLS:
{catalog}

USER REQUEST: {query}

Answer with only the number of the best tool:"""


def check_service() -> None:
    try:
        httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Ollama service not reachable at {OLLAMA_URL} ({type(exc).__name__}). "
            "Install and start Ollama, `ollama pull llama3.2`, then re-run."
        ) from exc


def ask(client: httpx.Client, model: str, prompt: str) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=300,
    )
    elapsed_s = time.perf_counter() - started
    response.raise_for_status()
    return response.json().get("response", ""), elapsed_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--catalog-sizes", nargs="+", type=int, default=[15, 30])
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    check_service()
    paths.ensure_dirs()
    rng = random.Random(args.seed)

    queries = pd.read_csv(paths.QUERIES_CSV).set_index("query_id")
    split = json.loads(
        (paths.SPLITS_DIR / "regime1_unseen_queries.json").read_text(encoding="utf-8")
    )
    corpus = json.loads(paths.CORPUS_JSON.read_text(encoding="utf-8"))
    test_ids = rng.sample(split["test"], min(args.n_queries, len(split["test"])))

    output: dict = {"model": args.model, "n_queries": len(test_ids), "catalog_sizes": {}}
    with httpx.Client() as client:
        for catalog_size in args.catalog_sizes:
            correct = 0
            latencies: list[float] = []
            unparseable = 0
            for query_id in test_ids:
                row = queries.loc[query_id]
                truth = row["tool"]
                distractors = [tool for tool in corpus if tool != truth]
                rng.shuffle(distractors)
                tools = distractors[: catalog_size - 1] + [truth]
                rng.shuffle(tools)
                catalog_text = "\n".join(
                    f"{i + 1}. {represent_name_desc(corpus[tool]['schema'])}"
                    for i, tool in enumerate(tools)
                )
                answer, elapsed_s = ask(
                    client, args.model, PROMPT_TEMPLATE.format(catalog=catalog_text, query=row["anchor"])
                )
                latencies.append(elapsed_s)
                digits = "".join(ch for ch in answer.strip()[:4] if ch.isdigit())
                if not digits:
                    unparseable += 1
                    continue
                choice = int(digits)
                if 1 <= choice <= len(tools) and tools[choice - 1] == truth:
                    correct += 1

            block = {
                "accuracy": round(correct / len(test_ids), 4),
                "unparseable": unparseable,
                "latency_s_mean": round(sum(latencies) / len(latencies), 2),
            }
            output["catalog_sizes"][str(catalog_size)] = block
            print(f"catalog={catalog_size}: {block}")

    out_path = paths.RESULTS_DIR / "llm_incontext.json"
    out_path.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
