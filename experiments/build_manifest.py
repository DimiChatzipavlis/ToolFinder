"""SHA256 manifest of datasets, splits, results, and trained model artifacts.

Datasets/splits/results are committed; model weights are not (see
academic_research/MODEL_ARTIFACTS.md) - the manifest pins their hashes so a
retrained or transferred artifact can be verified against the published runs.

Usage:
    python experiments/build_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import paths  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file():
                entries[str(path.relative_to(paths.REPO_ROOT)).replace("\\", "/")] = sha256_file(path)
    return entries


def main() -> None:
    paths.ensure_dirs()
    manifest = {
        "datasets": hash_tree(paths.DATA_DIR, ("*.csv", "*.json")),
        "raw_sources": hash_tree(paths.REPO_ROOT / "academic_research", ("mcp_routing_dataset*.csv",)),
        "results": hash_tree(paths.RESULTS_DIR, ("*.json",)),
        "model_artifacts": hash_tree(paths.ARTIFACTS_DIR, ("*.safetensors", "config.json")),
    }
    counts = {section: len(entries) for section, entries in manifest.items()}
    out_path = paths.RESULTS_DIR / "artifact_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path} ({counts})")


if __name__ == "__main__":
    main()
