# Contributing

## Setup

1. Create and activate a Python environment.
2. Install dependencies from `requirements.txt`.
3. Fetch runtime artifacts if needed:

```bash
barrikada fetch-artifacts --base-url <BARRIKADA_BASE_URL>
```

## Local Checks

```bash
pytest -q
```

## Pull Requests

- Keep changes focused and small.
- Add or update tests for behavior changes.
- Avoid committing generated outputs and model binaries.
