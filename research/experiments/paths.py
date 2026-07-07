"""Central path constants for the experiments package."""

from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent

DATA_DIR = EXPERIMENTS_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
OOD_DIR = DATA_DIR / "ood"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
ARTIFACTS_DIR = EXPERIMENTS_DIR / "artifacts"

RAW_V1_CSV = REPO_ROOT / "academic_research" / "mcp_routing_dataset.csv"
RAW_V2_CSV = REPO_ROOT / "academic_research" / "mcp_routing_dataset_v2.csv"

QUERIES_CSV = DATA_DIR / "queries_with_scenarios.csv"
CORPUS_JSON = DATA_DIR / "corpus.json"


def ensure_dirs() -> None:
    for path in (DATA_DIR, SPLITS_DIR, OOD_DIR, RESULTS_DIR, FIGURES_DIR, DIAGNOSTICS_DIR, ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
