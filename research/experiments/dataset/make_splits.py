"""Build leakage-controlled splits from the scenario-annotated queries.

Two evaluation regimes:

regime1_unseen_queries
    Group-aware split of the v1 queries: all rows of a scenario go to the same
    bucket, targeting 70/15/15 with at least one scenario per bucket per tool.
    Measures generalization to new phrasings/scenarios of known tools.

regime2_unseen_tools
    Train = regime1 train. Test = every v2 query. The 15 v2 tools never appear
    in training. Measures zero-shot generalization to unseen tools.

Both regimes are evaluated against the full 30-tool corpus, so unseen-tool
evaluation faces the trained tools as distractors (the previous protocol
searched only the 15 v2 tools, which is strictly easier).

Output: experiments/data/splits/{regime}.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import paths  # noqa: E402

SEED = 42
TARGET_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
BUCKETS = ("train", "val", "test")


def split_tool_scenarios(
    scenario_sizes: dict[str, int],
    rng: random.Random,
) -> dict[str, str]:
    """Assign each scenario to a bucket, keeping every bucket non-empty per tool.

    Greedy largest-first assignment to the bucket furthest below its target row
    fraction; seeds each bucket with one scenario first so 5-scenario tools end
    up 3/1/1.
    """
    scenarios = list(scenario_sizes)
    rng.shuffle(scenarios)
    scenarios.sort(key=lambda s: -scenario_sizes[s])

    total_rows = sum(scenario_sizes.values())
    assigned: dict[str, str] = {}
    bucket_rows = dict.fromkeys(BUCKETS, 0)

    seed_order = ("train", "test", "val")
    for bucket, scenario in zip(seed_order, scenarios):
        assigned[scenario] = bucket
        bucket_rows[bucket] += scenario_sizes[scenario]

    for scenario in scenarios[len(seed_order):]:
        deficits = {
            bucket: TARGET_FRACTIONS[bucket] - bucket_rows[bucket] / total_rows
            for bucket in BUCKETS
        }
        bucket = max(deficits, key=lambda b: deficits[b])
        assigned[scenario] = bucket
        bucket_rows[bucket] += scenario_sizes[scenario]

    return assigned


def main() -> None:
    paths.ensure_dirs()
    df = pd.read_csv(paths.QUERIES_CSV)
    corpus_tools = sorted(df["tool"].unique())
    rng = random.Random(SEED)

    v1 = df[df["dataset"] == "v1"]
    bucket_ids: dict[str, list[str]] = {bucket: [] for bucket in BUCKETS}
    for _, tool_group in v1.groupby("tool", sort=True):
        sizes = tool_group.groupby("scenario_id").size().to_dict()
        assignment = split_tool_scenarios(sizes, rng)
        for _, row in tool_group.iterrows():
            bucket_ids[assignment[row["scenario_id"]]].append(row["query_id"])

    regime1 = {
        "regime": "regime1_unseen_queries",
        "seed": SEED,
        "corpus_tools": corpus_tools,
        "train": sorted(bucket_ids["train"]),
        "val": sorted(bucket_ids["val"]),
        "test": sorted(bucket_ids["test"]),
    }

    regime2 = {
        "regime": "regime2_unseen_tools",
        "seed": SEED,
        "corpus_tools": corpus_tools,
        "train": regime1["train"],
        "val": regime1["val"],
        "test": sorted(df[df["dataset"] == "v2"]["query_id"].tolist()),
    }

    # Regime 1b: the template-disjoint control. The generation grammar is
    # template x scenario; regime 1 holds out scenarios but every surface
    # template still appears on both sides, so a model can ace it by learning
    # template->tool mappings. Here test rows use templates AND scenarios never
    # seen in training (rows in mixed blocks are discarded), which is the
    # controlled measurement of in-grammar generalization.
    bucket1b: dict[str, list[str]] = {bucket: [] for bucket in BUCKETS}
    discarded = 0
    rng1b = random.Random(SEED + 1)
    for _, tool_group in v1.groupby("tool", sort=True):
        templates = sorted(tool_group["template_id"].unique())
        scenarios = sorted(tool_group["scenario_id"].unique())
        if len(templates) <= 12 and len(scenarios) <= 8:
            rng1b.shuffle(templates)
            rng1b.shuffle(scenarios)
            template_bucket = {t: ("train" if i < 6 else "val" if i < 8 else "test") for i, t in enumerate(templates)}
            scenario_bucket = {s: ("train" if i < 3 else "val" if i < 4 else "test") for i, s in enumerate(scenarios)}
            for _, row in tool_group.iterrows():
                tb = template_bucket[row["template_id"]]
                sb = scenario_bucket[row["scenario_id"]]
                if tb == sb:
                    bucket1b[tb].append(row["query_id"])
                else:
                    discarded += 1
        else:
            # One-off tools: every row is its own template, so a scenario-grouped
            # split is template-disjoint by construction.
            sizes = tool_group.groupby("scenario_id").size().to_dict()
            assignment = split_tool_scenarios(sizes, rng1b)
            for _, row in tool_group.iterrows():
                bucket1b[assignment[row["scenario_id"]]].append(row["query_id"])

    regime1b = {
        "regime": "regime1b_template_disjoint",
        "seed": SEED,
        "corpus_tools": corpus_tools,
        "discarded_mixed_block_rows": discarded,
        "train": sorted(bucket1b["train"]),
        "val": sorted(bucket1b["val"]),
        "test": sorted(bucket1b["test"]),
    }

    for spec in (regime1, regime2, regime1b):
        out_path = paths.SPLITS_DIR / f"{spec['regime']}.json"
        out_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        sizes = {bucket: len(spec[bucket]) for bucket in BUCKETS}
        print(f"{spec['regime']}: {sizes} (corpus={len(spec['corpus_tools'])} tools)")


if __name__ == "__main__":
    main()
