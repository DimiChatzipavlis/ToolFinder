"""Build and execute notebooks/01_eda.ipynb with committed outputs.

The notebook is generated programmatically so it stays in sync with the
pipeline, then executed top-to-bottom so every figure and table in it is real
saved output (the original course notebook shipped with no outputs at all).

Usage:
    python experiments/build_eda_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import paths  # noqa: E402

NOTEBOOK_DIR = paths.REPO_ROOT / "notebooks"


CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# ToolFinder Dataset: Exploratory Data Analysis

This notebook documents the query/schema corpus used by the ToolFinder
experiments. Source data: `academic_research/mcp_routing_dataset{,_v2}.csv`,
annotated with recovered scenario structure by
`experiments/dataset/annotate_scenarios.py`.

Key facts established below:

1. **Composition** — 1,500 queries over 30 GitHub MCP tools (15 train-eligible,
   15 reserved for unseen-tool evaluation), perfectly balanced at 50 queries/tool.
2. **Generation structure** — most tools follow a `scenario x template` grammar
   (~5 scenarios x ~10 paraphrase templates), which is exactly why random row
   splits leak: paraphrases of one scenario land on both sides of the split.
3. **Leakage audit** — under the original random split, a trivial 1-NN lookup
   over training anchors scores ~96% Recall@1; the scenario-grouped split
   removes this artifact.
4. **Lexical overlap** — queries share substantial vocabulary with their target
   schemas (tool-name echo), which is why BM25/TF-IDF are mandatory baselines.
""",
    ),
    (
        "code",
        """import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

queries = pd.read_csv(REPO_ROOT / "experiments/data/queries_with_scenarios.csv")
corpus = json.loads((REPO_ROOT / "experiments/data/corpus.json").read_text(encoding="utf-8"))
print(f"queries: {len(queries)} | tools: {queries['tool'].nunique()} | scenarios: {queries['scenario_id'].nunique()}")
queries.head(3)""",
    ),
    (
        "markdown",
        "## 1. Dataset card: composition, balance, and data quality",
    ),
    (
        "code",
        """card = queries.groupby("dataset").agg(
    n_queries=("query_id", "count"),
    n_tools=("tool", "nunique"),
    n_scenarios=("scenario_id", "nunique"),
)
card["queries_per_tool"] = card["n_queries"] // card["n_tools"]
card""",
    ),
    (
        "markdown",
        """### Data quality: missing values and duplicates

The cleaning step verifies rather than imputes: the corpus is checked for
missing values, duplicate anchors, and unparsable schema JSON. Anything found
here would be a dataset-construction bug, not something to fill in.""",
    ),
    (
        "code",
        """missing = queries.isna().sum()
print("missing values per column:")
print(missing.to_string())
print(f"\\nduplicate anchors: {queries['anchor'].duplicated().sum()}")
unparsable = 0
for schema in queries["positive_schema"]:
    try:
        json.loads(schema)
    except json.JSONDecodeError:
        unparsable += 1
print(f"unparsable schema payloads: {unparsable}")
assert missing.sum() == 0 and unparsable == 0, "data quality violation"
print("\\nverdict: no missing values, no unparsable schemas; no imputation required")""",
    ),
    (
        "markdown",
        "### Class distribution (queries per tool)",
    ),
    (
        "code",
        """per_tool_counts = queries.groupby(["dataset", "tool"]).size().reset_index(name="n")
fig, ax = plt.subplots(figsize=(10, 3.2))
colors = per_tool_counts["dataset"].map({"v1": "#4878a8", "v2": "#e1812c"})
ax.bar(range(len(per_tool_counts)), per_tool_counts["n"], color=colors)
ax.set_xticks(range(len(per_tool_counts)))
ax.set_xticklabels(per_tool_counts["tool"], rotation=90, fontsize=6)
ax.set_ylabel("queries")
ax.set_title("Class balance: 50 queries per tool by construction (blue=v1 train-eligible, orange=v2 eval-only)")
plt.show()
print(per_tool_counts["n"].describe().loc[["min", "max"]].to_string())""",
    ),
    (
        "code",
        """per_tool = queries.groupby(["dataset", "tool"]).agg(
    rows=("query_id", "count"), scenarios=("scenario_id", "nunique")
).reset_index()
print("Scenario counts per tool (5 = templated 5x10 grammar; ~50 = one scenario per row):")
per_tool.groupby(["dataset", "scenarios"]).size().rename("n_tools").reset_index()""",
    ),
    (
        "markdown",
        "## 2. Query length distribution",
    ),
    (
        "code",
        """lengths = queries["anchor"].str.split().str.len()
fig, ax = plt.subplots(figsize=(7, 3.5))
for dataset, group in queries.groupby("dataset"):
    ax.hist(group["anchor"].str.split().str.len(), bins=range(3, 22), alpha=0.6, label=dataset)
ax.set_xlabel("anchor length (words)"); ax.set_ylabel("queries"); ax.legend()
ax.set_title(f"Anchor lengths: median {int(lengths.median())} words, range {lengths.min()}-{lengths.max()}")
plt.show()""",
    ),
    (
        "markdown",
        """## 3. The scenario x template grammar (why random splits leak)

One concrete scenario group: the same scenario clause rendered through ten
different prefixes. Any split that separates rows instead of scenarios puts
near-duplicates of every test query into training.""",
    ),
    (
        "code",
        """example_scenario = (
    queries[queries["tool"] == "create_or_update_file"].groupby("scenario_id").size().idxmax()
)
for anchor in queries[queries["scenario_id"] == example_scenario]["anchor"]:
    print(" -", anchor)""",
    ),
    (
        "code",
        """from IPython.display import Image, display
figure_path = REPO_ROOT / "experiments/results/figures/fig_leakage_audit.png"
if figure_path.exists():
    display(Image(str(figure_path)))
else:
    print("run experiments/figures.py first")""",
    ),
    (
        "markdown",
        """## 4. Query-schema lexical overlap (why lexical baselines are mandatory)

Fraction of each query's content words that also appear in its target schema
text. High overlap means part of the task is solvable by string matching, so
dense models must be read as deltas over BM25/TF-IDF, not in isolation.""",
    ),
    (
        "code",
        """import re

def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")))

schema_tokens = {tool: tokens(json.dumps(entry["schema"])) for tool, entry in corpus.items()}
overlap = queries.apply(
    lambda row: len(tokens(row["anchor"]) & schema_tokens[row["tool"]]) / max(1, len(tokens(row["anchor"]))),
    axis=1,
)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist([overlap[queries["dataset"] == "v1"], overlap[queries["dataset"] == "v2"]],
        bins=20, label=["v1", "v2"])
ax.set_xlabel("fraction of query tokens present in target schema"); ax.set_ylabel("queries")
ax.set_title(f"Query-schema lexical overlap (median {overlap.median():.2f})"); ax.legend()
plt.show()
print(f"median overlap v1: {overlap[queries['dataset']=='v1'].median():.2f} | "
      f"v2: {overlap[queries['dataset']=='v2'].median():.2f}")""",
    ),
    (
        "markdown",
        "## 5. Inter-tool schema confusability",
    ),
    (
        "code",
        """from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tools_sorted = sorted(corpus)
schema_texts = [json.dumps(corpus[t]["schema"], sort_keys=True) for t in tools_sorted]
matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(schema_texts)
similarity = cosine_similarity(matrix)
np.fill_diagonal(similarity, 0)

fig, ax = plt.subplots(figsize=(8, 7))
image = ax.imshow(similarity, cmap="magma")
ax.set_xticks(range(30)); ax.set_xticklabels(tools_sorted, rotation=90, fontsize=6)
ax.set_yticks(range(30)); ax.set_yticklabels(tools_sorted, fontsize=6)
ax.set_title("Pairwise schema similarity (char-ngram TF-IDF)")
fig.colorbar(image, shrink=0.8)
plt.show()

pairs = [(tools_sorted[i], tools_sorted[j], similarity[i, j])
         for i in range(30) for j in range(i + 1, 30)]
print("Most confusable schema pairs:")
for a, b, s in sorted(pairs, key=lambda p: -p[2])[:5]:
    print(f"  {s:.2f}  {a} <-> {b}")""",
    ),
    (
        "markdown",
        """## 6. Split summary

Splits are scenario-grouped (regime 1) and tool-disjoint (regime 2); the
hygiene constraints are enforced by `tests/test_split_hygiene.py` in CI.""",
    ),
    (
        "code",
        """for regime in ("regime1_unseen_queries", "regime2_unseen_tools"):
    spec = json.loads((REPO_ROOT / f"experiments/data/splits/{regime}.json").read_text(encoding="utf-8"))
    print(f"{regime}: train={len(spec['train'])} val={len(spec['val'])} test={len(spec['test'])} "
          f"corpus={len(spec['corpus_tools'])} tools")""",
    ),
]


def build() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for cell_type, source in CELLS:
        if cell_type == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            notebook.cells.append(nbformat.v4.new_code_cell(source))
    return notebook


def main() -> None:
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    notebook = build()
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK_DIR)}})
    client.execute()
    out_path = NOTEBOOK_DIR / "01_eda.ipynb"
    nbformat.write(notebook, out_path)
    executed_code_cells = sum(1 for cell in notebook.cells if cell.cell_type == "code" and cell.outputs)
    print(f"wrote {out_path} ({executed_code_cells} code cells with saved outputs)")


if __name__ == "__main__":
    main()
