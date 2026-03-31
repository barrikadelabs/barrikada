# Barrikada Open Day Demo

This demo showcases live attack-vs-defense behavior:

- WITH BARRIKADA: layered pipeline verdict and explanation (always runs on Analyze)
- WITHOUT SECURITY: optional direct model output from Ollama (runs only when clicked)

## Run

From repository root:

```bash
source .venv/bin/activate
streamlit run demo/streamlit_app.py
```

## Runtime requirement

The demo uses local Ollama for baseline and Layer E execution.

If required models are missing, pull them before running, for example:

```bash
ollama pull qwen3.5:2b
```

## Open Day flow

1. Click one of the attack presets.
2. Click Analyze to run the prompt through Barrikada.
3. Optionally click Run Base LLM (without security) for comparison.
4. Open layer tabs to explain where and why the verdict was made.
