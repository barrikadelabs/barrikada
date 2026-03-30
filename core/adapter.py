import logging

from core.orchestrator import PIPipeline
from models.verdicts import FinalVerdict


class AdapterPolicy:
    def __init__(self, block_on_flag=False, fail_closed=True):
        self.block_on_flag = block_on_flag
        self.fail_closed = fail_closed


class BarrikadaAdapter:
    def __init__(
        self,
        pipeline = None,
        *,
        policy = None,
    ):
        self.pipeline = pipeline or PIPipeline()
        self.policy = policy or AdapterPolicy()
        self.logger = logging.getLogger(__name__)

    def detect(self, text):
        safe_text = text or ""

        try:
            result = self.pipeline.detect(safe_text)
            verdict = self._normalize_verdict(result.final_verdict)
            flagged = verdict == FinalVerdict.FLAG.value
            blocked = verdict == FinalVerdict.BLOCK.value

            if flagged and self.policy.block_on_flag:
                blocked = True

            return {
                "blocked": blocked,
                "flagged": flagged,
                "reason": self._build_reason(result),
                "confidence": float(getattr(result, "confidence_score", 0.0) or 0.0),
                "metadata": {
                    "verdict": verdict,
                    "decision_layer": self._enum_value(getattr(result, "decision_layer", None), default="unknown"),
                    "input_hash": getattr(result, "input_hash", ""),
                    "latency_ms": float(getattr(result, "total_processing_time_ms", 0.0) or 0.0),
                },
            }
        except Exception as exc:
            self.logger.exception("Barrikada detect failed")
            blocked = self.policy.fail_closed
            return {
                "blocked": blocked,
                "flagged": False,
                "reason": "detection_error",
                "confidence": 0.0,
                "metadata": {
                    "verdict": "error",
                    "decision_layer": "error",
                    "input_hash": "",
                    "latency_ms": 0.0,
                    "error": str(exc),
                },
            }

    @staticmethod
    def _normalize_verdict(verdict):
        value = BarrikadaAdapter._enum_value(verdict, default=str(verdict).lower())
        if value in {FinalVerdict.ALLOW.value, FinalVerdict.FLAG.value, FinalVerdict.BLOCK.value}:
            return value
        return FinalVerdict.BLOCK.value

    @staticmethod
    def _enum_value(value, default):
        if value is None:
            return default
        return str(getattr(value, "value", value)).lower()

    @staticmethod
    def _build_reason(result):
        layer_e = getattr(result, "layer_e_result", None) or {}
        rationale = layer_e.get("rationale")
        if rationale:
            return str(rationale)

        layer_b = getattr(result, "layer_b_result", None) or {}
        matches = layer_b.get("matches") or []
        if matches:
            return f"layer_b_match_count={len(matches)}"

        layer_c = getattr(result, "layer_c_result", None) or {}
        prob = layer_c.get("probability_score")
        if prob is not None:
            try:
                return f"layer_c_probability={float(prob):.3f}"
            except (TypeError, ValueError):
                return "layer_c_probability=unknown"

        decision_layer = BarrikadaAdapter._enum_value(getattr(result, "decision_layer", None), default="unknown")
        verdict = BarrikadaAdapter._normalize_verdict(getattr(result, "final_verdict", None))
        return f"layer_{decision_layer}_{verdict}"