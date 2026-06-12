"""Build and execute notebooks/02_toolfinder_live.ipynb.

A self-contained, runnable evidence notebook: on Colab it clones the GitHub
repo and installs dependencies; locally it uses the checkout. It then (a) runs
the router LIVE on the real 30-tool corpus, (b) reproduces the lexical-baseline
table rows live and checks them against the committed results, (c) renders the
committed result tables and figures, (d) runs a small LIVE Flat-vs-HNSW timing
at N in {15, 100, 1000}, and (e) if a GPU is present, fine-tunes MiniLM for two
epochs live and shows the before/after Recall@1.

Usage (also executes the notebook so outputs are committed):
    python experiments/build_live_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import paths  # noqa: E402

NOTEBOOK_DIR = paths.REPO_ROOT / "notebooks"
GITHUB_URL = "https://github.com/DimiChatzipavlis/ToolFinder.git"
COLAB_BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/DimiChatzipavlis/ToolFinder/blob/main/notebooks/02_toolfinder_live.ipynb)"
)

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        f"""# ToolFinder — Live Evidence Notebook

{COLAB_BADGE}

This notebook is the runnable companion to `reports/report.md`. It works in
two modes:

- **Google Colab:** clones the repository from GitHub and installs dependencies
  (first run takes a few minutes), then everything below runs live.
- **Local checkout:** detects the repo and runs directly.

What runs **live** here: the deployed router on the real 30-tool GitHub MCP
corpus (routing, scores, abstention), the lexical baselines reproduced
against the committed results, an exact-vs-HNSW timing microbenchmark, and —
if a GPU is available — a 2-epoch bi-encoder fine-tune with before/after
accuracy. The full multi-seed results are loaded from
`experiments/results/` (regenerable via `python experiments/run_all.py`).""",
    ),
    (
        "code",
        f"""import os, sys, subprocess
from pathlib import Path

if Path.cwd().name == "notebooks":
    os.chdir("..")
if not Path("toolfinder").exists():
    print("cloning ToolFinder from GitHub...")
    subprocess.run(["git", "clone", "--depth", "1", "{GITHUB_URL}"], check=True)
    os.chdir("ToolFinder")

try:
    import faiss, sentence_transformers, pandas, sklearn, matplotlib  # noqa: F401
except ImportError:
    print("installing ToolFinder + experiment dependencies (a few minutes on Colab)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[experiments]"], check=True)

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))
print("repo root:", REPO_ROOT)""",
    ),
    (
        "markdown",
        """## 1. The router, live

We ingest the 30 real GitHub MCP tool schemas and route natural-language
intents. MiniLM is used here for CPU speed; the runtime default (MPNet) behaves
identically architecturally. Note the **exact flat index** — see section 4 for
why approximate search buys nothing at these catalog sizes.""",
    ),
    (
        "code",
        """import json
from toolfinder import UniversalMCPRouter, RouteNotFoundError, to_openai_tools

corpus = json.loads((REPO_ROOT / "experiments/data/corpus.json").read_text(encoding="utf-8"))
router = UniversalMCPRouter(model_name="sentence-transformers/all-MiniLM-L6-v2")
for tool, entry in corpus.items():
    schema = entry["schema"]
    router.add_tool(
        {"name": schema["name"], "description": schema["description"], "inputSchema": schema["inputSchema"]},
        server_name=entry["server"],
    )
count = router.build_index()
print(f"indexed {count} tools | index type: {type(router.faiss_index).__name__}")""",
    ),
    (
        "code",
        """for query in [
    "open a pull request to merge the auth cleanup work",
    "who is on the platform team?",
    "I need the contents of src/config.yaml from the repo",
]:
    results = router.route_top_k(query, k=3)
    formatted = ", ".join(f"{r.tool_name} ({r.score:.3f})" for r in results)
    print(f"{query!r}\\n  -> {formatted}\\n")

print("bindable schema for the last top hit:")
print(json.dumps(to_openai_tools(results[:1])[0]["function"]["name"], indent=2))""",
    ),
    (
        "markdown",
        """### Abstention on out-of-scope queries

Below the similarity threshold the router refuses instead of force-routing —
the operating point is chosen from the measured risk-coverage analysis
(`experiments/results/ood_eval.json`), not guessed.""",
    ),
    (
        "code",
        """for query in ["what's the weather in Athens tomorrow?", "play my focus playlist"]:
    try:
        result = router.route(query)
        print(f"{query!r} -> ROUTED to {result.tool_name} (score {result.score:.3f})")
    except RouteNotFoundError:
        top = router.route_top_k(query, k=1)
        print(f"{query!r} -> ABSTAINED (no candidate above threshold)")""",
    ),
    (
        "markdown",
        """## 2. Baselines reproduced live

BM25 and TF-IDF run in seconds on CPU; we evaluate them on the leakage-controlled
regime-1 test split and check the numbers against the committed results file —
the table in the report is reproducible, not copy-pasted.""",
    ),
    (
        "code",
        """import pandas as pd
from experiments.baselines import Bm25Ranker, TfidfRanker
from experiments.evaluation import metrics
from experiments.representation import represent_raw

queries = pd.read_csv(REPO_ROOT / "experiments/data/queries_with_scenarios.csv").set_index("query_id")
split = json.loads((REPO_ROOT / "experiments/data/splits/regime1_unseen_queries.json").read_text(encoding="utf-8"))
corpus_tools = sorted(corpus)
corpus_texts = [represent_raw(corpus[t]["schema"]) for t in corpus_tools]
test_ids = split["test"]
anchors = queries.loc[test_ids, "anchor"].tolist()
truths = queries.loc[test_ids, "tool"].tolist()

live = {}
for system in (Bm25Ranker(corpus_tools, corpus_texts),
               TfidfRanker(corpus_tools, corpus_texts, analyzer="char_wb", ngram_range=(3, 5))):
    rankings = [system.rank(a) for a in anchors]
    ranks = metrics.ranks_from_rankings(rankings, truths)
    live[system.name] = round(float(metrics.recall_at_k(ranks, 1).mean()), 4)

committed_path = REPO_ROOT / "experiments/results/main_eval.json"
committed = json.loads(committed_path.read_text(encoding="utf-8")) if committed_path.exists() else None
print(f"{'system':12s} {'live R@1':>9s} {'committed':>10s}")
for name, value in live.items():
    committed_value = (
        committed["regimes"]["regime1_unseen_queries"][name]["recall@1"]["mean"] if committed else float("nan")
    )
    match = "OK" if committed and abs(value - committed_value) < 1e-6 else ""
    print(f"{name:12s} {value:9.4f} {committed_value:10.4f}  {match}")""",
    ),
    (
        "markdown",
        """## 3. The committed evidence

Full multi-seed results across the three regimes (unseen queries / unseen tools /
unseen servers), produced by `experiments/run_all.py`.""",
    ),
    (
        "code",
        """def regime_frame(regime: str) -> pd.DataFrame:
    rows = []
    for name, block in committed["regimes"][regime].items():
        if "recall@1" not in block or name.startswith("random"):
            continue
        entry = {"system": name, "R@1": block["recall@1"]["mean"], "R@3": block["recall@3"]["mean"], "MRR": block["mrr"]["mean"]}
        rows.append(entry)
    return pd.DataFrame(rows).sort_values("R@1", ascending=False).reset_index(drop=True)

for regime in committed["regimes"]:
    print(f"==== {regime} ====")
    display(regime_frame(regime).head(12))""",
    ),
    (
        "code",
        """from IPython.display import Image, display
for figure_name in [
    "fig_main_results.png",
    "fig_leakage_audit.png",
    "fig_loss_curves.png",
    "fig_threshold_sweep.png",
    "fig_scaling.png",
]:
    figure_path = REPO_ROOT / "experiments/results/figures" / figure_name
    if figure_path.exists():
        print(figure_name)
        display(Image(str(figure_path)))""",
    ),
    (
        "markdown",
        """## 4. Exact vs approximate search, live

The report's scaling study runs to 100k vectors; here is the small live version.
At tool-catalog sizes, exact flat inner-product search is *faster* than HNSW
graph traversal — and exact. Query encoding (tens of ms) dominates either.""",
    ),
    (
        "code",
        """import time
import numpy as np
import faiss
from experiments.benchmarks.scaling_bench import generate_synthetic_schemas
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
documents = [represent_raw(corpus[t]["schema"]) for t in corpus_tools] + generate_synthetic_schemas(1000 - 30)
embeddings = encoder.encode(documents, batch_size=64, convert_to_numpy=True, show_progress_bar=False).astype("float32")
faiss.normalize_L2(embeddings)
query_vectors = encoder.encode(anchors[:50], convert_to_numpy=True, show_progress_bar=False).astype("float32")
faiss.normalize_L2(query_vectors)
faiss.omp_set_num_threads(1)

print(f"{'N':>6s} {'flat p50 (ms)':>14s} {'hnsw p50 (ms)':>14s} {'hnsw recall@10':>15s}")
for n in (15, 100, 1000):
    sub = embeddings[:n]
    flat = faiss.IndexFlatIP(sub.shape[1]); flat.add(sub)
    hnsw = faiss.IndexHNSWFlat(sub.shape[1], 32, faiss.METRIC_INNER_PRODUCT); hnsw.hnsw.efSearch = 64; hnsw.add(sub)
    _, truth_ids = flat.search(query_vectors, 10)
    timings = {}
    for label, index in (("flat", flat), ("hnsw", hnsw)):
        per_query = []
        for repeat in range(10):
            for i in range(len(query_vectors)):
                start = time.perf_counter()
                index.search(query_vectors[i:i+1], 10)
                per_query.append((time.perf_counter() - start) * 1000)
        timings[label] = float(np.percentile(per_query, 50))
    _, approx_ids = hnsw.search(query_vectors, 10)
    recall = float(np.mean([len(set(a) & set(t)) / 10 for a, t in zip(approx_ids, truth_ids)]))
    print(f"{n:6d} {timings['flat']:14.4f} {timings['hnsw']:14.4f} {recall:15.4f}")""",
    ),
    (
        "markdown",
        """## 5. Live fine-tuning (GPU only)

If this runtime has a GPU (Colab: Runtime → Change runtime type → GPU), the cell
below fine-tunes MiniLM for two epochs on the regime-1 training scenarios with
the no-duplicates batch sampler, then re-evaluates on the held-out test split —
the frozen-vs-fine-tuned delta is the report's headline effect, reproduced in
~2 minutes. On CPU the cell skips itself.""",
    ),
    (
        "code",
        """import torch

if not torch.cuda.is_available():
    print("no GPU in this runtime - skipping the live fine-tune (full results are in section 3)")
else:
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers
    from experiments.baselines import EncoderRanker

    text_by_tool = dict(zip(corpus_tools, corpus_texts))
    train_df = queries.loc[split["train"]]
    train_dataset = Dataset.from_dict({
        "anchor": train_df["anchor"].tolist(),
        "positive": [text_by_tool[t] for t in train_df["tool"]],
    })

    def recall_at_1(model_path_or_obj) -> float:
        ranker = EncoderRanker("eval", model_path_or_obj, corpus_tools, corpus_texts) \
            if isinstance(model_path_or_obj, str) else model_path_or_obj
        rankings = ranker.rank_batch(anchors)
        ranks = metrics.ranks_from_rankings(rankings, truths)
        return float(metrics.recall_at_k(ranks, 1).mean())

    frozen_r1 = recall_at_1("sentence-transformers/all-MiniLM-L6-v2")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    model.max_seq_length = 256
    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir="/tmp/live_ft", num_train_epochs=2, per_device_train_batch_size=16,
            learning_rate=2e-5, warmup_ratio=0.1, fp16=True,
            batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no",
            logging_steps=20, report_to="none", seed=42,
        ),
        train_dataset=train_dataset,
        loss=losses.MultipleNegativesRankingLoss(model),
    )
    trainer.train()
    model.save("/tmp/live_ft/final")
    tuned_r1 = recall_at_1("/tmp/live_ft/final")
    print(f"\\nRecall@1 on regime-1 test: frozen {frozen_r1:.4f} -> fine-tuned (2 epochs) {tuned_r1:.4f}")""",
    ),
    (
        "markdown",
        """## Where to go next

- Full report (auto-generated from results): `reports/report.md`
- Pipeline documentation and stage map: `experiments/README.md`
- Dataset provenance and biases: `experiments/data/DATASET_CARD.md`
- Security posture: `SECURITY.md`""",
    ),
]


def build() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    for cell_type, source in CELLS:
        if cell_type == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            notebook.cells.append(nbformat.v4.new_code_cell(source))
    return notebook


def main() -> None:
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    notebook = build()
    client = NotebookClient(
        notebook, timeout=1200, kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    client.execute()
    out_path = NOTEBOOK_DIR / "02_toolfinder_live.ipynb"
    nbformat.write(notebook, out_path)
    executed = sum(1 for cell in notebook.cells if cell.cell_type == "code" and cell.outputs)
    print(f"wrote {out_path} ({executed} code cells with saved outputs)")


if __name__ == "__main__":
    main()
