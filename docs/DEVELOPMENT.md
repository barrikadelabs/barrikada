# Development Guide

This guide covers developer-facing workflows for Barrikada.

## Local setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Models and datasets are automatically downloaded from Google Cloud Storage on first use.
   No configuration needed — see `docs/MODEL_HOSTING.md` for details.

## Model distribution

Runtime models and datasets are distributed via Google Cloud Storage public buckets.

**Key points:**
- No credentials needed for download (public read access)
- Models auto-fetch on first import from `core/` pipeline
- Docker containers auto-download models at startup
- Manual download available via scripts in `scripts/gcs_download.py`

For complete documentation, see `docs/MODEL_HOSTING.md`.

## Training Layer B encoders

To train custom dual-encoder models for Layer B:

```bash
python core/layer_b/extraction/train_dual_encoder.py
```

After training, rebuild signatures:

```bash
python core/layer_b/extraction/extract_signature_patterns.py
```

These scripts are also accessible via wrapper scripts in `scripts/` for convenience:

```bash
python scripts/train_dual_encoder.py
python scripts/extract_signature_patterns.py
```
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
