# Model Artifact Policy

This repository is source-first. Large training and inference artifacts are not committed.

## What Stays Out Of Git

- Model weights (`.safetensors`)
- Tokenizer export blobs (`tokenizer.json`)
- FAISS indexes (`*.index`)
- Generated cache metadata (`*_faiss_meta.json`)

## Storage And Transfer

Store these artifacts in an enterprise artifact registry (for example: S3 + checksum manifest, GitHub Releases, or an internal package registry).

## Retrieval Contract

1. Keep a versioned artifact bundle per model release.
2. Include SHA256 checksums for each file.
3. Fetch artifacts during setup, not from the git repository.
4. Validate checksums before loading artifacts at runtime.

## Operational Note

If local artifacts are present from experiments, remove them before release packaging to prevent accidental transfer of large binaries.
