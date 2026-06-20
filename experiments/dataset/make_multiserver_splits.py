"""Server-disjoint split for R1: train the bi-encoder on SOME servers, test on
entirely held-out servers.

The 195 multi-server queries span 22 servers. We partition the *servers* (not
the queries) into train/val/test, so the test servers — and their tools — are
never seen in training. Ranking is against the full multi-server corpus, so the
held-out tools compete with every trained tool as a distractor.

This is the controlled measurement R1 asks for: does training on a diversity of
servers improve generalization to brand-new servers, versus the GitHub-only
model? Output: experiments/data/splits/regime4_multiserver.json
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
N_TEST_SERVERS = 7
N_VAL_SERVERS = 3


def main() -> None:
    paths.ensure_dirs()
    queries = pd.read_csv(paths.DATA_DIR / "queries_multiserver.csv")
    corpus = json.loads((paths.DATA_DIR / "corpus_multiserver.json").read_text(encoding="utf-8"))
    corpus_tools = sorted(corpus)

    servers = sorted(queries["server"].unique())
    rng = random.Random(SEED)
    rng.shuffle(servers)
    test_servers = set(servers[:N_TEST_SERVERS])
    val_servers = set(servers[N_TEST_SERVERS:N_TEST_SERVERS + N_VAL_SERVERS])
    train_servers = set(servers[N_TEST_SERVERS + N_VAL_SERVERS:])

    def ids(server_set):
        return sorted(queries[queries["server"].isin(server_set)]["query_id"].tolist())

    spec = {
        "regime": "regime4_multiserver",
        "seed": SEED,
        "corpus_file": "corpus_multiserver.json",
        "queries_files": ["queries_multiserver.csv"],
        "corpus_tools": corpus_tools,
        "train_servers": sorted(train_servers),
        "val_servers": sorted(val_servers),
        "test_servers": sorted(test_servers),
        "train": ids(train_servers),
        "val": ids(val_servers),
        "test": ids(test_servers),
    }

    # Hygiene: a test server's tools must never appear in training queries.
    train_tools = set(queries[queries["server"].isin(train_servers)]["tool"])
    test_tools = set(queries[queries["server"].isin(test_servers)]["tool"])
    assert not (train_tools & test_tools), "server-disjoint split leaked a tool across train/test"
    assert all(t in corpus for t in test_tools), "test tool missing from corpus"

    out = paths.SPLITS_DIR / "regime4_multiserver.json"
    out.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"servers: {len(servers)} -> train {len(train_servers)} / val {len(val_servers)} / test {len(test_servers)}")
    print(f"queries: train {len(spec['train'])} / val {len(spec['val'])} / test {len(spec['test'])}")
    print(f"test servers (unseen): {sorted(test_servers)}")
    print(f"corpus for ranking: {len(corpus_tools)} tools")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
