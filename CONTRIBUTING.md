# Contributing

## Setup

1. Create and activate a Python environment.
2. Install dependencies from `requirements.txt`.
3. Run the pipeline or use Docker to automatically fetch runtime artifacts:

**Local (auto-fetches on first import):**
```bash
python examples/quickstart.py
```

**Docker (auto-fetches at startup):**
```bash
docker-compose up
```

For model hosting details and manual download workflows, see `docs/MODEL_HOSTING.md`.
For full local setup and container workflows, see `docs/DEVELOPMENT.md`.

## Local Checks

```bash
pytest -q
```

## Pull Requests

- Keep changes focused and small.
- Add or update tests for behavior changes.
- Avoid committing generated outputs and model binaries.
