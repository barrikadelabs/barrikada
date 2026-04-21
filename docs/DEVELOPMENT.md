# Development Guide

This guide covers developer-facing workflows for Barrikada.

## Local setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Pull runtime artifacts and datasets:

```bash
./scripts/pull_artifacts.sh
```

## Private artifacts repository

Barrikada runtime artifacts and datasets are managed in:

- https://github.com/barrikadelabs/barrikade-artifacts

One-time Git LFS setup per machine:

```bash
brew install git-lfs
git lfs install
```

Update and push artifacts from the artifacts repository:

```bash
git add datasets core/layer_b/signatures core/layer_c/outputs core/layer_c/train/outputs core/layer_d/outputs core/layer_e/outputs
git commit -m "Update Barrikada artifacts and datasets"
git push origin <artifact-branch>
```

## Pull artifacts and datasets into this repo

From the Barrikada codebase root:

```bash
./scripts/pull_artifacts.sh
```

Optional flags:

```bash
./scripts/pull_artifacts.sh --ref <artifact-branch>
./scripts/pull_artifacts.sh --source-prefix artifacts
./scripts/pull_artifacts.sh --delete
```

## If files were committed before LFS

Run once in the artifacts repository:

```bash
git lfs migrate import --include="datasets/**,core/layer_b/signatures/embeddings/**,core/layer_b/signatures/extracted/*.yar,core/layer_c/outputs/**,core/layer_c/train/outputs/**,core/layer_d/outputs/**,core/layer_e/outputs/**"
git push --force-with-lease
```

## Docker workflow

See `docs/DOCKER.md` for image build, compose, health checks, and runtime environment variables.

## Quality checks

```bash
pytest -q
```

## Examples

```bash
python examples/quickstart.py
python examples/basic_detection.py
```
