"""Figures for the bridge crossover study (reads results/bridge_scaling_*.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import paths  # noqa: E402

BLUE, ORANGE, GREEN, MUTED = "#4878a8", "#e1812c", "#3a923a", "#64748b"


def fig_free() -> None:
    path = paths.RESULTS_DIR / "bridge_scaling_free.json"
    if not path.exists():
        print("[skip] bridge_scaling_free.json missing")
        return
    pts = json.loads(path.read_text(encoding="utf-8"))["points"]
    ns = [p["n_tools"] for p in pts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    ax1.plot(ns, [p["baseline_schema_tokens"] for p in pts], "o-", color=BLUE, label="baseline (all tools bound)")
    ax1.plot(ns, [p["findcall_schema_tokens"] for p in pts], "s--", color=ORANGE, label="bridge: find+call (2 tools)")
    ax1.plot(ns, [p["single_schema_tokens"] for p in pts], "^--", color=GREEN, label="bridge: single tool")
    ax1.set_xlabel("catalog size N (tools available to the agent)")
    ax1.set_ylabel("tool-schema tokens carried per turn")
    ax1.set_title("Per-turn context weight vs catalog size")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(ns, [p["router_recall@1"] for p in pts], "o-", color=BLUE, label="router recall@1")
    ax2.plot(ns, [p["router_recall@3"] for p in pts], "s--", color=GREEN, label="router recall@3")
    ax2.set_xlabel("catalog size N (correct tool buried among N-14 distractors)")
    ax2.set_ylabel("selection recall over filesystem intents")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Does the router still find the right tool at scale?")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    out = paths.FIGURES_DIR / "fig_bridge_scaling_free.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_api() -> None:
    path = paths.RESULTS_DIR / "bridge_scaling_gpt.json"
    if not path.exists():
        print("[skip] bridge_scaling_gpt.json missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    sizes = sorted(int(n) for n in data["sizes"])
    if len(sizes) < 2:
        print(f"[skip] only {len(sizes)} catalog size(s) in api results — rerun with more --api-sizes for the crossover plot")
        return

    def series(arm):
        xs, ys = [], []
        for n in sizes:
            cell = data["sizes"][str(n)].get(arm, {})
            if "total_tokens" in cell:
                xs.append(n)
                ys.append(cell["total_tokens"])
        return xs, ys

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for arm, color, marker, label in (
        ("baseline", BLUE, "o-", "baseline (all N tools bound)"),
        ("find_call", ORANGE, "s--", "bridge: find+call"),
        ("single", GREEN, "^--", "bridge: single tool"),
    ):
        xs, ys = series(arm)
        if xs:
            ax.plot(xs, ys, marker, color=color, label=label)
    ax.set_xlabel("catalog size N")
    ax.set_ylabel(f"total tokens for the task ({data['model']})")
    ax.set_title("End-to-end token cost vs catalog size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = paths.FIGURES_DIR / "fig_bridge_scaling_api.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    paths.ensure_dirs()
    fig_free()
    fig_api()
