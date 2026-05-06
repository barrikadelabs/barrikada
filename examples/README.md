# Examples

These examples are intentionally small and runnable.

## Prerequisites

1. Install the package dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Models are auto-downloaded on first run from Google Cloud Storage public buckets.
   No credentials needed — see `docs/MODEL_HOSTING.md` for details.

## Run

```bash
python examples/quickstart.py
python examples/basic_detection.py
```

If models are missing, the runtime will attempt to download them automatically.
For manual download or offline setup, see `docs/MODEL_HOSTING.md`.
