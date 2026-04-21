# Examples

These examples are intentionally small and runnable.

## Prerequisites

1. Install the package dependencies.
2. Fetch runtime artifacts and datasets once:

```bash
./scripts/pull_artifacts.sh
```

## Run

```bash
python examples/quickstart.py
python examples/basic_detection.py
```

If artifacts are missing, the runtime raises a `FileNotFoundError` with the exact path and required environment override.
