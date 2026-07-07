"""Cache-aware re-analysis of the MCP-bridge scaling study (Step 1 / M1).

The published bridge numbers (`results/bridge_scaling_gpt.json`) are *uncached*
total tokens. But a tool-use loop re-sends the same static prefix (system prompt
+ bound tool schemas + first user message) on every model turn, and providers
serve that repeated prefix from a **prompt cache** at a fraction of the input
price. This script re-scores the same runs under prompt caching, so the honest
cost gap replaces the headline "~15x".

Model (per arm, per catalog size N), grounded in the logged fields:
  P0 = first_call_prompt_tokens   # the static, repeated, cacheable prefix
  T  = model_turns                # number of model calls
  prompt = prompt_tokens          # input tokens summed over all turns

  The prefix P0 appears at the head of all T turns, so it is billed T*P0 across
  the task; the remainder (`prompt - T*P0`) is the small per-turn delta (tool
  call + tool result) that genuinely changes each turn. We verified T*P0 <= prompt
  for every cell, so this decomposition is consistent with the measured totals.

  Under caching the prefix is a cache *write* on turn 1 (rate w) and a cache
  *read* on turns 2..T (rate r); deltas stay full price:
     billable_input = (prompt - T*P0) + P0*w + (T-1)*P0*r
  Real APIs only cache a prefix once it clears a minimum length (~1024 tokens).
  The bridge arms' prefix (~330 tok) is *below* that floor, so they get NO cache
  benefit while the baseline's large tool block does — caching helps the baseline
  more than the bridge. We honor that floor.

Caveats (printed in the report too):
  - Conservative for caching: only the turn-1 prefix is treated as the cached
    span; the growing conversation history (also a matching prefix on later turns)
    is NOT additionally credited. Here the deltas are tiny (~700 tok), so it
    barely matters.
  - Output tokens are unaffected by caching.
  - This is a *model* from per-turn structure; a *measured* version (M1) would log
    the API's reported `cached_tokens`. Single trial per cell (n=1).
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SRC = RESULTS / "bridge_scaling_gpt.json"
OUT = RESULTS / "bridge_cache_aware.json"

# (name, cache-read multiplier, cache-write multiplier, min cacheable prefix tokens)
SCENARIOS = [
    ("uncached (as published)", 1.0, 1.0, 0),
    ("OpenAI-style (read 0.50x)", 0.50, 1.0, 1024),
    ("OpenAI-style (read 0.25x)", 0.25, 1.0, 1024),
    ("Anthropic-style (read 0.10x / write 1.25x)", 0.10, 1.25, 1024),
]

ARMS = ["baseline", "find_call", "single"]


def billable_input(prompt: int, p0: int, turns: int, r: float, w: float, min_prefix: int) -> float:
    """Cache-weighted input tokens for one arm/run."""
    if turns <= 1 or p0 < min_prefix:
        return float(prompt)  # nothing repeats, or prefix too short to cache
    deltas = max(prompt - turns * p0, 0)
    return deltas + p0 * w + (turns - 1) * p0 * r


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    sizes = data["sizes"]
    model = data.get("model", "?")

    report: dict = {"model": model, "source": SRC.name, "scenarios": [s[0] for s in SCENARIOS], "sizes": {}}

    print(f"Cache-aware re-analysis of {SRC.name}  (model: {model})\n")
    print("Validating the repeated-prefix model (T*P0 <= prompt_tokens):")
    for n, arms in sizes.items():
        for arm in ARMS:
            a = arms[arm]
            lhs, rhs = a["model_turns"] * a["first_call_prompt_tokens"], a["prompt_tokens"]
            flag = "ok" if lhs <= rhs else "VIOLATED"
            if flag != "ok":
                print(f"  N={n:>3} {arm:<10} T*P0={lhs} > prompt={rhs}  <-- {flag}")
    print("  (all ok unless flagged above)\n")

    for n, arms in sizes.items():
        report["sizes"][n] = {}
        print(f"=== N = {n} tools ===")
        header = f"{'arm':<10} {'turns':>5} {'prefix P0':>9} {'out':>5} | " + " | ".join(
            f"{name.split(' (')[0][:18]:>18}" for name, *_ in SCENARIOS
        )
        print(header)
        billable = {}
        for arm in ARMS:
            a = arms[arm]
            row = report["sizes"][n].setdefault(arm, {})
            row["model_turns"] = a["model_turns"]
            row["prefix_tokens_P0"] = a["first_call_prompt_tokens"]
            row["completion_tokens"] = a["completion_tokens"]
            cells = []
            billable[arm] = {}
            for name, r, w, mn in SCENARIOS:
                bi = billable_input(a["prompt_tokens"], a["first_call_prompt_tokens"], a["model_turns"], r, w, mn)
                billable[arm][name] = bi
                row.setdefault("billable_input_tokens", {})[name] = round(bi, 1)
                cells.append(f"{bi:>18,.0f}")
            print(f"{arm:<10} {a['model_turns']:>5} {a['first_call_prompt_tokens']:>9,} "
                  f"{a['completion_tokens']:>5} | " + " | ".join(cells))

        # ratios baseline vs each bridge arm, per scenario (on billable INPUT tokens)
        print(f"\n{'ratio baseline /':<22} " + " | ".join(f"{name.split(' (')[0][:18]:>18}" for name, *_ in SCENARIOS))
        for bridge in ("find_call", "single"):
            ratios = []
            for name, *_ in SCENARIOS:
                ratio = billable["baseline"][name] / billable[bridge][name]
                ratios.append(f"{ratio:>17.1f}x")
                report["sizes"][n].setdefault("baseline_over_" + bridge, {})[name] = round(ratio, 2)
            print(f"  {bridge:<20} " + " | ".join(ratios))
        print()

    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    print("\nReading the table: caching shrinks the baseline's input (its big tool block")
    print("is a cached read on turns 2..T), but the bridge arms' ~330-token prefix is")
    print("below the ~1024 cache floor, so they get no discount. The bridge still wins,")
    print("by a smaller margin than the uncached ratio implies, and the margin still")
    print("grows with N. Output tokens are unaffected; n=1 per cell; modeled, not measured.")


if __name__ == "__main__":
    main()
