# Dataset Card: ToolFinder MCP Routing Benchmark

## Composition

| Subset | Queries | Tools | Servers | Role |
| --- | --- | --- | --- | --- |
| v1 (GitHub) | 750 | 15 | 1 | training + unseen-query eval (regime 1) |
| v2 (GitHub) | 750 | 15 | 1 | unseen-tool eval only (regime 2) |
| ms (multi-server) | 195 | 65 queried / 544 in corpus | 23 | unseen-server eval only (regime 3) |
| OOD | 249 | — | — | abstention eval only |

Corpora: `corpus.json` (30 GitHub MCP tools), `corpus_multiserver.json`
(30 GitHub + 544 tools converted from real OpenAPI specs via apis.guru;
provenance per tool in `catalogs/multiserver_catalog.json`).

## Generation provenance

- **v1/v2 anchors** are author-templated: most tools follow a
  `scenario × template` grammar (~5 scenarios × ~10 paraphrase prefixes with
  the scenario clause kept verbatim). Scenario structure was recovered
  programmatically (`dataset/annotate_scenarios.py`, tail-grouping + fuzzy
  merge) and is stored in the `scenario_id` column.
- **ms anchors** were hand-written by a different model family (Claude) against
  sampled catalog tools, with an anti-echo rule (avoid operationId tokens and
  description phrasing). Eval-only.
- **OOD queries** (chitchat / out-of-catalog / adversarial near-miss) are
  author-written, eval-only (`dataset/make_ood.py`).

## Splits and leakage control

Random row splits over v1 are answerable at ~96% Recall@1 by a 1-NN lookup over
training anchors (paraphrase leakage). Committed splits are therefore:

- `regime1_unseen_queries` — scenario-grouped 460/146/144 (no scenario crosses
  buckets; enforced by `tests/test_split_hygiene.py` in CI).
- `regime2_unseen_tools` — train/val from regime 1; test = all v2 queries;
  corpus = all 30 tools (trained tools act as distractors).
- `regime3_unseen_servers` — train/val from regime 1 (GitHub only); test = all
  ms queries; corpus = 574 tools across 24 servers.

## Known biases and limitations

- v2 anchors lexically echo tool names (median query→schema token overlap 0.45
  vs 0.33 for v1), inflating lexical-baseline scores on regime 2.
- All queries are synthetic (author- or LLM-written); no human-user or
  production traffic is included.
- The GitHub corpus is a single domain; regime 3 mitigates but its queries
  cover 65 of 544 corpus tools.
- OOD sets are author-written and small (249 queries).

## License / intended use

Schemas originate from the public GitHub MCP server and public OpenAPI specs
(apis.guru, CC0). Queries are released for research/teaching evaluation of
tool-retrieval systems.
