import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM

from core.orchestrator import PIPipeline
from models.verdicts import FinalVerdict

DEFAULT_MODEL_NAME = "qwen3.5:2b"


class BarrikadaAgent:
    """LLM agent with Barrikada screening on every inbound message."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_history: int = 20):
        self.pipeline = PIPipeline()

        llm = OllamaLLM(model=model_name)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant. "
                    "Answer the user's questions clearly and concisely.",
                ),
                MessagesPlaceholder(variable_name="history", optional=True),
                ("user", "{question}"),
            ]
        )
        self.chain = prompt | llm | StrOutputParser()
        self.history: list[HumanMessage | AIMessage] = []
        self.max_history = max_history

    def invoke(self, question: str, max_retries: int = 3) -> dict:
        """Screen question through Barrikada, then (if allowed) call the LLM."""
        scan = self.pipeline.detect(question)

        result = {
            "barrikada_verdict": scan.final_verdict.value,
            "decision_layer": scan.decision_layer.value,
            "confidence": scan.confidence_score,
            "pipeline_time_ms": scan.total_processing_time_ms,
            "agent_response": None,
        }

        if scan.final_verdict == FinalVerdict.BLOCK:
            result["agent_response"] = "[BLOCKED by Barrikada]"
            return result

        trimmed = self.history[-self.max_history :]
        for attempt in range(max_retries):
            try:
                response = self.chain.invoke({"question": question, "history": trimmed})
                self.history.append(HumanMessage(content=question))
                self.history.append(AIMessage(content=response))
                result["agent_response"] = response
                return result
            except Exception as exc:
                if attempt == max_retries - 1:
                    result["agent_response"] = f"[LLM ERROR: {str(exc)[:120]}]"
                    return result

        return result

    def clear_history(self) -> None:
        self.history.clear()


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def evaluate(
    csv_path: str = "datasets/barrikada_test.csv",
    model_name: str = DEFAULT_MODEL_NAME,
    max_samples: int | None = None,
):
    """Run every prompt in csv_path through the Barrikada-wrapped agent."""
    project_root = _resolve_project_root()
    candidate = Path(csv_path)
    csv_file = candidate if candidate.is_absolute() else project_root / candidate

    df = pd.read_csv(csv_file)
    if max_samples:
        df = df.head(max_samples)

    agent = BarrikadaAgent(model_name=model_name)

    print(f"Dataset : {csv_file.name}  ({len(df)} prompts)")
    print(f"Model   : {model_name}")
    print("=" * 60)

    rows = []
    correct = 0
    blocked = 0

    for i, (_, row) in enumerate(df.iterrows(), 1):
        text, label = row["text"], int(row["label"])
        out = agent.invoke(text)

        predicted_block = out["barrikada_verdict"] == "block"
        is_malicious = label == 1
        is_correct = predicted_block == is_malicious

        if predicted_block:
            blocked += 1
        if is_correct:
            correct += 1

        rows.append(
            {
                "text": text,
                "true_label": label,
                "barrikada_verdict": out["barrikada_verdict"],
                "decision_layer": out["decision_layer"],
                "confidence": out["confidence"],
                "pipeline_time_ms": out["pipeline_time_ms"],
                "agent_response": out["agent_response"],
                "is_correct": is_correct,
            }
        )

        tag = "PASS" if is_correct else "FAIL"
        print(
            f"[{i}/{len(df)}] {tag:4s}  "
            f"verdict={out['barrikada_verdict']:5s}  "
            f"layer={out['decision_layer']}  "
            f"conf={out['confidence']:.2f}  "
            f"time={out['pipeline_time_ms']:.1f}ms  "
            f"true={'mal' if is_malicious else 'ben'}"
        )

    accuracy = correct / len(df) * 100
    block_rate = blocked / len(df) * 100

    print("\n" + "=" * 60)
    print(f"Accuracy   : {accuracy:.2f}%  ({correct}/{len(df)})")
    print(f"Block rate : {block_rate:.1f}%  ({blocked}/{len(df)})")
    print("=" * 60)

    results_dir = project_root / "test_results" / "agent"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_df = pd.DataFrame(rows)
    results_csv = results_dir / f"eval_{timestamp}.csv"
    results_df.to_csv(results_csv, index=False)

    summary = {
        "timestamp": timestamp,
        "dataset": csv_file.name,
        "total": len(df),
        "correct": correct,
        "accuracy_pct": round(accuracy, 2),
        "blocked": blocked,
        "block_rate_pct": round(block_rate, 2),
        "model": model_name,
    }
    summary_file = results_dir / f"summary_{timestamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nResults -> {results_csv}")
    print(f"Summary -> {summary_file}")
    return results_df


def interactive(model_name: str = DEFAULT_MODEL_NAME) -> None:
    """Chat with the Barrikada-wrapped agent in the terminal."""
    agent = BarrikadaAgent(model_name=model_name)
    print("Barrikada Agent  (type 'quit' to exit, 'clear' to reset history)\n")

    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if question.lower() in ("clear", "reset"):
                agent.clear_history()
                print("[history cleared]\n")
                continue
            if not question:
                continue

            out = agent.invoke(question)
            print(
                f"[{out['barrikada_verdict'].upper()} | "
                f"layer {out['decision_layer']} | "
                f"{out['pipeline_time_ms']:.1f}ms]"
            )
            print(f"Agent: {out['agent_response']}\n")

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")
