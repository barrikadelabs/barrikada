# Barrikada Open Day Demo

This demo showcases live attack-vs-defense behavior:

- WITH BARRIKADA: layered pipeline verdict and explanation (always runs on Analyze)
- WITHOUT SECURITY: optional direct model output from the local teacher model (runs only when clicked)

## Run

From repository root:

```bash
source .venv/bin/activate
streamlit run demo/streamlit_app.py
```

## Runtime requirement

The demo uses the local teacher checkpoint for baseline and Layer E execution.

If required artifacts are missing, point the demo at `core/layer_e/outputs/teacher/merged_teacher` or set `BARRIKADA_LAYER_E_TEACHER_LOCAL_MODEL_DIR`.

## Open Day flow

1. Click one of the attack presets.
2. Click Analyze to run the prompt through Barrikada.
3. Optionally click Run Base LLM (without security) for comparison.
4. Open layer tabs to explain where and why the verdict was made.
