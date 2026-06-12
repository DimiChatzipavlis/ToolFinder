"""Run the full experiment pipeline in dependency order.

    python experiments/run_all.py            # everything (data -> train -> eval -> report)
    python experiments/run_all.py --from eval  # resume from a stage

Stages: data, train, crossencoder, eval, ood, ablation, scaling, figures, report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STAGES: list[tuple[str, list[list[str]]]] = [
    (
        "data",
        [
            ["experiments/dataset/annotate_scenarios.py"],
            ["experiments/dataset/make_splits.py"],
            ["experiments/dataset/make_ood.py"],
            ["experiments/dataset/build_multiserver_catalog.py"],
            ["experiments/dataset/make_multiserver_queries.py"],
        ],
    ),
    (
        "train",
        [
            ["experiments/models/biencoder.py"],
            [
                "experiments/models/biencoder.py",
                "--models", "minilm",
                "--split-name", "regime1b_template_disjoint",
                "--artifact-root", "biencoder_r1b",
                "--results-name", "biencoder_training_r1b.json",
            ],
        ],
    ),
    (
        "crossencoder",
        [
            ["experiments/models/hard_negatives.py"],
            ["experiments/models/crossencoder.py"],
        ],
    ),
    ("eval", [["experiments/evaluation/evaluate.py"]]),
    (
        "controls",
        [
            ["experiments/evaluation/eval_template_disjoint.py"],
            ["experiments/evaluation/significance.py"],
        ],
    ),
    ("ood", [["experiments/evaluation/ood.py"]]),
    ("ablation", [["experiments/ablation_representation.py"]]),
    ("scaling", [["experiments/benchmarks/scaling_bench.py"]]),
    ("attacks", [["experiments/attacks/poisoning.py"]]),
    ("calibration", [["experiments/evaluation/calibration.py"]]),
    ("figures", [
        ["experiments/figures.py"],
        ["experiments/build_eda_notebook.py"],
        ["experiments/build_live_notebook.py"],
    ]),
    ("report", [["experiments/build_report.py"], ["experiments/build_manifest.py"]]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", default="data", choices=[name for name, _ in STAGES])
    args = parser.parse_args()

    started = False
    for stage_name, commands in STAGES:
        if stage_name == args.start:
            started = True
        if not started:
            continue
        print(f"\n========== stage: {stage_name} ==========")
        for command in commands:
            print(f"$ python {' '.join(command)}")
            result = subprocess.run([sys.executable, *command], cwd=REPO_ROOT)
            if result.returncode != 0:
                print(f"stage '{stage_name}' failed at: {command}")
                sys.exit(result.returncode)
    print("\npipeline complete")


if __name__ == "__main__":
    main()
