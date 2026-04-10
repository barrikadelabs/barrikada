# Artifact Strategy

Barrikada keeps runtime model artifacts outside git history.

## Why

- Reduces repository bloat.
- Prevents accidental public exposure of trained model binaries.
- Keeps package distribution slim and faster to install.

## What Is External

- Layer B embedding indexes and encoder checkpoints.
- Layer C classifier binary (`classifier.joblib`).
- Layer D local model directory.

## Fetch Flow

Use the CLI once per machine or environment:

```bash
barrikada fetch-artifacts --base-url <BARRIKADA_ARTIFACT_BASE_URL>
```

Artifacts are downloaded under:

- `~/.barrikada/artifacts` by default.
- `BARRIKADA_ARTIFACTS_DIR` when overridden.

## Build Manifest

When publishing new artifacts:

```bash
python scripts/build_artifact_manifest.py --core-dir core --out artifact-manifest.json
```

Publish `manifest.json` and listed files at your artifact base URL.
