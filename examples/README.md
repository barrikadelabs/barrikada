# Examples

These examples are intentionally small and runnable.

## Prerequisites

1. Install the package dependencies.
2. Fetch runtime artifacts once:

```bash
barrikada fetch-artifacts --base-url <YOUR_ARTIFACT_BASE_URL>
```

## Run

```bash
python examples/quickstart.py
python examples/basic_detection.py
```

If artifacts are missing, the runtime raises a `FileNotFoundError` with the exact path and required environment override.
