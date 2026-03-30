import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Allow running as a script: `python scripts/security_cost_eval.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import PIPipeline


@dataclass
class CostModel:
    compute_cost_per_ms: float = 0.0
    layer_e_input_cost_per_1k_tokens: float = 0.0
    layer_e_output_cost_per_1k_tokens: float = 0.0
    sla_ms: float = 50.0
    latency_penalty_per_ms_over_sla: float = 0.0
    false_positive_penalty: float = 0.0
    false_negative_penalty: float = 0.0


def _latency_summary(values):
    arr = values.to_numpy(dtype=float)
    if arr.size == 0:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _prediction_from_verdict(verdict):
    # Binary projection for system-level metrics: allow -> benign (0), block/flag -> malicious (1)
    return 0 if verdict == "allow" else 1


def _compute_cost_components(row, cost):
    total_ms = float(row["total_time_ms"] or 0.0)
    prompt_tokens = int(row["layer_e_prompt_tokens"] or 0)
    completion_tokens = int(row["layer_e_completion_tokens"] or 0)

    compute_cost = total_ms * cost.compute_cost_per_ms
    token_cost = (
        (prompt_tokens / 1000.0) * cost.layer_e_input_cost_per_1k_tokens
        + (completion_tokens / 1000.0) * cost.layer_e_output_cost_per_1k_tokens
    )
    latency_over_sla = max(0.0, total_ms - cost.sla_ms)
    latency_penalty = latency_over_sla * cost.latency_penalty_per_ms_over_sla

    y_true = int(row["true_label"] or 0)
    y_pred = int(row["predicted_label"] or 0)
    fp_penalty = cost.false_positive_penalty if y_true == 0 and y_pred == 1 else 0.0
    fn_penalty = cost.false_negative_penalty if y_true == 1 and y_pred == 0 else 0.0

    total_cost = compute_cost + token_cost + latency_penalty + fp_penalty + fn_penalty

    return {
        "compute_cost": float(compute_cost),
        "token_cost": float(token_cost),
        "latency_penalty": float(latency_penalty),
        "fp_penalty": float(fp_penalty),
        "fn_penalty": float(fn_penalty),
        "total_cost": float(total_cost),
    }


def evaluate_dataset(csv_path, text_col, label_col, max_samples, cost_model, ):
    pipeline = PIPipeline()

    df = pd.read_csv(csv_path)
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)

    rows = []

    for _, row in df.iterrows():
        text = str(row[text_col])
        y_true = int(row[label_col])

        result = pipeline.detect(text)

        layer_e_result = result.layer_e_result or {}
        verdict = result.final_verdict.value
        y_pred = _prediction_from_verdict(verdict)

        record = {
            "text": text,
            "true_label": y_true,
            "predicted_verdict": verdict,
            "predicted_label": y_pred,
            "decision_layer": result.decision_layer.value,
            "confidence_score": float(result.confidence_score),
            "is_correct": bool(y_true == y_pred),
            "total_time_ms": float(result.total_processing_time_ms),
            "layer_a_time_ms": float(result.layer_a_time_ms or 0.0),
            "layer_b_time_ms": float(result.layer_b_time_ms or 0.0),
            "layer_c_time_ms": float(result.layer_c_time_ms or 0.0),
            "layer_d_time_ms": float(result.layer_d_time_ms or 0.0),
            "layer_e_time_ms": float(result.layer_e_time_ms or 0.0),
            "layer_e_called": bool(result.decision_layer.value == "E"),
            "layer_e_model": str(layer_e_result.get("model", "")),
            "layer_e_prompt_tokens": int(layer_e_result.get("prompt_tokens") or 0),
            "layer_e_completion_tokens": int(layer_e_result.get("completion_tokens") or 0),
            "layer_e_total_tokens": int(layer_e_result.get("total_tokens") or 0),
        }

        record.update(_compute_cost_components(pd.Series(record), cost_model))
        rows.append(record)

    out_df = pd.DataFrame(rows)

    y_true_arr = out_df["true_label"].astype(int).to_numpy()
    y_pred_arr = out_df["predicted_label"].astype(int).to_numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="binary",
        zero_division=0,
    )

    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))

    total_cost = float(out_df["total_cost"].sum()) if not out_df.empty else 0.0
    n_rows = max(1, int(len(out_df)))

    malicious_total = int(np.sum(y_true_arr == 1))
    blocked_malicious = tp
    cost_per_1k = float((total_cost / n_rows) * 1000.0)
    cost_per_malicious_blocked = float(total_cost / blocked_malicious) if blocked_malicious > 0 else None

    baseline_allow_all_fn_cost = float(malicious_total * cost_model.false_negative_penalty)
    actual_fn_cost = float(fn * cost_model.false_negative_penalty)
    expected_loss_avoided = float(baseline_allow_all_fn_cost - actual_fn_cost)

    summary = {
        "rows": int(len(out_df)),
        "cost_model": asdict(cost_model),
        "quality": {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)) if len(out_df) else 0.0,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            },
            "false_positive_rate": float(fp / max(1, (fp + tn))),
            "false_negative_rate": float(fn / max(1, (fn + tp))),
        },
        "latency_ms": {
            "total": _latency_summary(out_df["total_time_ms"]),
            "layer_a": _latency_summary(out_df["layer_a_time_ms"]),
            "layer_b": _latency_summary(out_df["layer_b_time_ms"]),
            "layer_c": _latency_summary(out_df["layer_c_time_ms"]),
            "layer_d": _latency_summary(out_df["layer_d_time_ms"]),
            "layer_e": _latency_summary(out_df["layer_e_time_ms"]),
        },
        "routing": {
            "decision_layer_distribution": out_df["decision_layer"].value_counts().to_dict(),
            "verdict_distribution": out_df["predicted_verdict"].value_counts().to_dict(),
            "layer_e_invocation_rate": float(np.mean(out_df["layer_e_called"].astype(int))) if len(out_df) else 0.0,
        },
        "cost": {
            "total_cost": total_cost,
            "avg_cost_per_request": float(total_cost / n_rows),
            "avg_cost_per_1000_requests": cost_per_1k,
            "cost_per_malicious_blocked": cost_per_malicious_blocked,
            "expected_loss_avoided_vs_allow_all": expected_loss_avoided,
            "breakdown_totals": {
                "compute_cost": float(out_df["compute_cost"].sum()) if len(out_df) else 0.0,
                "token_cost": float(out_df["token_cost"].sum()) if len(out_df) else 0.0,
                "latency_penalty": float(out_df["latency_penalty"].sum()) if len(out_df) else 0.0,
                "fp_penalty": float(out_df["fp_penalty"].sum()) if len(out_df) else 0.0,
                "fn_penalty": float(out_df["fn_penalty"].sum()) if len(out_df) else 0.0,
            },
            "breakdown_per_request": {
                "compute_cost": float(out_df["compute_cost"].mean()) if len(out_df) else 0.0,
                "token_cost": float(out_df["token_cost"].mean()) if len(out_df) else 0.0,
                "latency_penalty": float(out_df["latency_penalty"].mean()) if len(out_df) else 0.0,
                "fp_penalty": float(out_df["fp_penalty"].mean()) if len(out_df) else 0.0,
                "fn_penalty": float(out_df["fn_penalty"].mean()) if len(out_df) else 0.0,
            },
        },
    }

    return out_df, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Offline system evaluation with security cost accounting")
    parser.add_argument("--csv", default="datasets/barrikada_test.csv", help="Path to evaluation CSV")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--label-col", default="label", help="Binary label column name (0 benign, 1 malicious)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap")
    parser.add_argument(
        "--out-dir",
        default="test_results/system_eval",
        help="Output directory for detailed CSV and summary JSON",
    )

    parser.add_argument("--compute-cost-per-ms", type=float, default=0.0)
    parser.add_argument("--layer-e-input-cost-per-1k-tokens", type=float, default=0.0)
    parser.add_argument("--layer-e-output-cost-per-1k-tokens", type=float, default=0.0)
    parser.add_argument("--sla-ms", type=float, default=50.0)
    parser.add_argument("--latency-penalty-per-ms-over-sla", type=float, default=0.0)
    parser.add_argument("--false-positive-penalty", type=float, default=0.0)
    parser.add_argument("--false-negative-penalty", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()

    cost_model = CostModel(
        compute_cost_per_ms=float(args.compute_cost_per_ms),
        layer_e_input_cost_per_1k_tokens=float(args.layer_e_input_cost_per_1k_tokens),
        layer_e_output_cost_per_1k_tokens=float(args.layer_e_output_cost_per_1k_tokens),
        sla_ms=float(args.sla_ms),
        latency_penalty_per_ms_over_sla=float(args.latency_penalty_per_ms_over_sla),
        false_positive_penalty=float(args.false_positive_penalty),
        false_negative_penalty=float(args.false_negative_penalty),
    )

    detailed_df, summary = evaluate_dataset(
        csv_path=args.csv,
        text_col=args.text_col,
        label_col=args.label_col,
        max_samples=args.max_samples,
        cost_model=cost_model,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_path = out_dir / f"system_eval_{ts}.csv"
    summary_path = out_dir / f"system_eval_summary_{ts}.json"

    detailed_df.to_csv(detailed_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Rows used: {summary['rows']}")
    print(f"Accuracy: {summary['quality']['accuracy']:.4f}")
    print(f"F1: {summary['quality']['f1']:.4f}")
    print(f"Avg latency (ms): {summary['latency_ms']['total']['avg']:.2f}")
    print(f"Avg cost / 1000 requests: {summary['cost']['avg_cost_per_1000_requests']:.6f}")
    print(f"Detailed CSV: {detailed_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
