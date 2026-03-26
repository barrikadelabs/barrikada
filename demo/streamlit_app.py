from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from core.orchestrator import PIPipeline
from core.settings import Settings
from demo.demo_utils import (
    ATTACK_PRESETS,
    VERDICT_COLORS,
    build_explanations,
    layer_statuses,
    run_unprotected_baseline,
    summarize_pipeline,
)


st.set_page_config(
    page_title="Barrikada Open Day Demo",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --demo-bg: #0a0a0a;
                --demo-bg-elevated: #141414;
                --demo-bg-soft: #1d1d1d;
                --demo-text: #f5f5f5;
                --demo-text-muted: #bdbdbd;
                --demo-surface: #121212;
                --demo-border: #2a2a2a;
            }
            .stApp {
                background: radial-gradient(circle at 14% 18%, #1a1a1a 0%, #0f0f0f 36%, #060606 100%);
                color: var(--demo-text);
            }
            [data-testid="stAppViewContainer"],
            [data-testid="stMainBlockContainer"],
            [data-testid="stMainBlockContainer"] p,
            [data-testid="stMainBlockContainer"] li,
            [data-testid="stMainBlockContainer"] label,
            [data-testid="stMainBlockContainer"] h1,
            [data-testid="stMainBlockContainer"] h2,
            [data-testid="stMainBlockContainer"] h3 {
                color: var(--demo-text) !important;
            }
            .headline {
                font-size: 2.3rem;
                font-weight: 700;
                letter-spacing: 0.4px;
                margin-bottom: 0.2rem;
                color: var(--demo-text);
            }
            .subhead {
                color: var(--demo-text-muted);
                margin-bottom: 1rem;
            }
            .verdict-card {
                padding: 1rem 1.2rem;
                border-radius: 12px;
                border: 1px solid var(--demo-border);
                background: var(--demo-surface);
                box-shadow: 0 8px 22px rgba(0, 0, 0, 0.45);
                color: var(--demo-text);
            }
            .mono {
                font-family: "Source Code Pro", "Fira Mono", monospace;
            }
            .stMarkdown a {
                color: #e3e3e3 !important;
            }
            [data-testid="stMetricLabel"] *,
            [data-testid="stMetricValue"] {
                color: var(--demo-text) !important;
            }
            .stTextArea textarea,
            .stTextInput input,
            [data-testid="stCodeBlock"] pre {
                color: var(--demo-text) !important;
                background-color: var(--demo-bg-elevated) !important;
                border: 1px solid var(--demo-border) !important;
            }
            .stButton button[kind="secondary"] {
                color: var(--demo-text) !important;
                background: var(--demo-bg-soft) !important;
                border: 1px solid #4a4a4a !important;
            }
            .stButton button[kind="primary"] {
                color: #111111 !important;
                background: #e8e8e8 !important;
                border: 1px solid #f3f3f3 !important;
            }
            [data-baseweb="tab-list"] button {
                color: #d6d6d6 !important;
                background: #1a1a1a !important;
                border: 1px solid #2f2f2f !important;
            }
            [data-baseweb="tab-list"] button[aria-selected="true"] {
                color: #ffffff !important;
                background: #3a3a3a !important;
                border: 1px solid #6a6a6a !important;
                border-radius: 8px;
            }
            [data-testid="stAlert"] {
                border-radius: 10px;
                background: #171717 !important;
                border: 1px solid #3a3a3a !important;
            }
            [data-testid="stAlert"] * {
                color: #f1f1f1 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=True)
def get_pipeline() -> PIPipeline:
    return PIPipeline()


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return Settings()


def render_summary(summary):
    verdict = summary["final_verdict"]
    decision_layer = summary["decision_layer"]
    confidence = float(summary["confidence_score"])
    total_ms = float(summary["total_processing_time_ms"])
    color = VERDICT_COLORS.get(verdict, "#444444")

    st.markdown(
        (
            "<div class='verdict-card'>"
            f"<h2 style='margin:0;color:{color};'>Final verdict: {verdict.upper()}</h2>"
            f"<p style='margin:0.4rem 0 0 0;'>Decided at Layer {decision_layer} | "
            f"Confidence {confidence:.2f} | Total {total_ms:.1f} ms</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_layer_block(layer_name, summary, status_map):
    data = summary["layers"].get(layer_name)
    timing = summary["timings"].get(layer_name)
    status = status_map.get(layer_name, "skipped")

    st.markdown(f"### Layer {layer_name} ({status})")
    if timing is not None:
        st.caption(f"Processing time: {float(timing):.2f} ms")

    if data is None:
        st.info("Layer not executed due to early verdict.")
        return

    if layer_name == "A":
        flags = data.get("flags") or []
        col1, col2, col3 = st.columns(3)
        col1.metric("Suspicious", str(bool(data.get("suspicious"))))
        col2.metric("Flag count", len(flags))
        col3.metric("Confidence", f"{float(data.get('confidence_score', 0.0)):.2f}")
        if flags:
            st.write("Flags:", ", ".join(str(flag) for flag in flags))
        st.write("Decoded text preview:")
        st.code(str(data.get("processed_text", ""))[:600])
        return

    if layer_name == "B":
        col1, col2, col3 = st.columns(3)
        col1.metric("Verdict", str(data.get("verdict", "n/a")).upper())
        col2.metric("Attack similarity", f"{float(data.get('attack_similarity', 0.0)):.2f}")
        col3.metric("Benign similarity", f"{float(data.get('benign_similarity', 0.0)):.2f}")
        matches = data.get("matches") or []
        if matches:
            st.write("Signature matches")
            st.dataframe(matches, use_container_width=True)
        else:
            st.info("No signature matches.")
        return

    if layer_name in {"C", "D"}:
        col1, col2, col3 = st.columns(3)
        col1.metric("Verdict", str(data.get("verdict", "n/a")).upper())
        col2.metric("Risk probability", f"{float(data.get('probability_score', 0.0)):.2f}")
        col3.metric("Confidence", f"{float(data.get('confidence_score', 0.0)):.2f}")
        return

    if layer_name == "E":
        col1, col2 = st.columns(2)
        col1.metric("Decision", str(data.get("decision", "n/a")).upper())
        col2.metric("Model", str(data.get("model", "n/a")))
        st.write("Rationale")
        st.info(str(data.get("rationale", "No rationale returned.")))
        with st.expander("Raw Layer E response"):
            st.code(str(data.get("raw_response", ""))[:1200])
        return


def run_demo(prompt):
    pipeline = get_pipeline()
    return summarize_pipeline(pipeline.detect(prompt))


def run_baseline(prompt):
    settings = get_settings()
    baseline = run_unprotected_baseline(prompt, settings)
    return {
        "output": baseline.output,
        "model": baseline.model,
        "latency_ms": baseline.latency_ms,
    }


def main():
    inject_styles()
    st.markdown("<div class='headline'>Hack the AI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subhead'>Live prompt attack vs layered defense with verdict and rationale.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Try an attack")
    preset_cols = st.columns(5)
    for idx, (name, text) in enumerate(ATTACK_PRESETS.items()):
        if preset_cols[idx % 5].button(name, use_container_width=True):
            st.session_state["prompt_input"] = text

    presenter_mode = st.toggle("Presenter mode", value=True, help="Reduces clutter and emphasizes verdict visuals.")

    prompt = st.text_area(
        "Prompt input",
        key="prompt_input",
        height=180 if presenter_mode else 240,
        placeholder="Enter a prompt to test Barrikada defense...",
    )

    action_cols = st.columns([1, 1, 5])
    run_clicked = action_cols[0].button("Analyze", type="primary", use_container_width=True)
    clear_clicked = action_cols[1].button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state["prompt_input"] = ""
        st.session_state.pop("result", None)
        st.rerun()

    if run_clicked:
        if not prompt.strip():
            st.warning("Please enter a prompt before analysis.")
        else:
            with st.spinner("Running Barrikada pipeline..."):
                try:
                    protected = run_demo(prompt)
                    st.session_state["result"] = {
                        "protected": protected,
                        "baseline": None,
                        "prompt": prompt,
                    }
                except Exception as exc:
                    st.error(f"Demo execution failed: {exc}")

    if "result" not in st.session_state:
        st.info("Run analysis to see attack-vs-defense results.")
        return

    result = st.session_state["result"]
    protected = result["protected"]
    baseline = result["baseline"]
    analyzed_prompt = result["prompt"]

    render_summary(protected)

    st.markdown("### With Barrikada (protected)")
    verdict = protected["final_verdict"].upper()
    decision_layer = protected["decision_layer"]
    confidence = float(protected["confidence_score"])
    st.success(
        f"Verdict: {verdict} | Decided at Layer {decision_layer} | Confidence: {confidence:.2f}"
    )
    st.write("How it was detected")
    for bullet in build_explanations(protected):
        st.write(f"- {bullet}")

    st.markdown("### Optional comparison: without security")
    baseline_cols = st.columns([2, 3])
    with baseline_cols[0]:
        run_baseline_clicked = st.button("Run Base LLM (without security)", use_container_width=True)
    with baseline_cols[1]:
        st.caption("Runs only when clicked and uses the same prompt that was analyzed by Barrikada.")

    if run_baseline_clicked:
        with st.spinner("Running base LLM without Barrikada..."):
            try:
                st.session_state["result"]["baseline"] = run_baseline(analyzed_prompt)
                baseline = st.session_state["result"]["baseline"]
            except Exception as exc:
                st.error(f"Baseline execution failed: {exc}")

    if baseline is not None:
        st.caption(f"Model: {baseline['model']} | Latency: {float(baseline['latency_ms']):.1f} ms")
        st.code(baseline["output"] or "[No response returned]")
    else:
        st.info("Baseline has not been run yet.")

    status_map = layer_statuses(protected)
    layer_tabs = st.tabs([
        "Layer A",
        "Layer B",
        "Layer C",
        "Layer D",
        "Layer E",
        "Timing",
        "Raw JSON",
    ])

    with layer_tabs[0]:
        render_layer_block("A", protected, status_map)
    with layer_tabs[1]:
        render_layer_block("B", protected, status_map)
    with layer_tabs[2]:
        render_layer_block("C", protected, status_map)
    with layer_tabs[3]:
        render_layer_block("D", protected, status_map)
    with layer_tabs[4]:
        render_layer_block("E", protected, status_map)
    with layer_tabs[5]:
        timings = protected["timings"]
        for layer_name in ["A", "B", "C", "D", "E"]:
            value = timings.get(layer_name)
            if value is None:
                st.write(f"Layer {layer_name}: skipped")
            else:
                st.write(f"Layer {layer_name}: {float(value):.2f} ms")
    with layer_tabs[6]:
        st.json(
            {
                "prompt": result["prompt"],
                "protected": protected,
                "baseline": baseline,
            }
        )


if __name__ == "__main__":
    main()
