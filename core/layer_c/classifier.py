import os
import platform
import time
from dataclasses import dataclass

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from models.LayerCResult import LayerCResult


@dataclass(frozen=True)
class Thresholds:
    low: float = 0.35
    high: float = 0.85

    def validate(self):
        if not (0.0 <= self.low <= 1.0 and 0.0 <= self.high <= 1.0):
            raise ValueError("Thresholds must be within [0,1]")
        if self.low >= self.high:
            raise ValueError("Expected low < high")


class Classifier:
    def __init__(self, model_path, embedding_model="all-mpnet-base-v2", low=0.35, high=0.85, ):
        self._configure_runtime()
        device = self._resolve_device()
        self.encoder = SentenceTransformer(embedding_model, device=device)
        artifact = joblib.load(model_path)

        self.model = artifact.get("model")
        self.calibrator = artifact.get("calibrator")

        if self.model is None or not hasattr(self.model, "predict_proba"):
            raise ValueError("Layer C model artifact does not contain a valid predict_proba model")

        self.thresholds = Thresholds(low=low, high=high)
        self.thresholds.validate()

##### SAFE MDOE - remove once ONNX is in place. 
    @staticmethod
    def _resolve_device() -> str:
        forced = os.getenv("BARRIKADA_LAYER_C_DEVICE", "").strip().lower()
        if forced in {"cpu", "cuda", "mps"}:
            return forced

        if platform.system() == "Darwin":
            return "cpu"

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _configure_runtime() -> None:
        safe_mode = os.getenv("BARRIKADA_LAYER_C_SAFE_MODE")
        if safe_mode is None:
            safe_mode = "1" if platform.system() == "Darwin" else "0"

        if safe_mode != "0":
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            for env_var in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                os.environ.setdefault(env_var, "1")

            import torch

            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except (ValueError, RuntimeError):
                pass

#######
    def predict(self, input_text):
        start_time = time.time()

        emb = self.encoder.encode([input_text], normalize_embeddings=True)
        probability_score = float(self.model.predict_proba(emb)[:, 1][0])
        if self.calibrator is not None:
            probability_score = float(self.calibrator.predict(np.array([probability_score]))[0])

        if probability_score < self.thresholds.low:
            verdict = "allow"
        elif probability_score < self.thresholds.high:
            verdict = "flag"
        else:
            verdict = "block"

        # Confidence: distance from the decision boundary
        if verdict == "allow":
            confidence_score = 1.0 - probability_score
        elif verdict == "block":
            confidence_score = probability_score
        else:
            # Middle band = uncertain
            confidence_score = 0.5

        processing_time_ms = (time.time() - start_time) * 1000.0

        return LayerCResult(
            verdict=verdict,
            probability_score=probability_score,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
        )

    def predict_dict(self, input_text):
        """Something to get a simple dict output for API responses."""
        res = self.predict(input_text)
        return {"score": res.probability_score, "decision": res.verdict}

    def predict_batch(self, texts):
        """Return raw probability scores for a batch of texts."""
        embs = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        probs = self.model.predict_proba(embs)[:, 1]
        if self.calibrator is not None:
            probs = self.calibrator.predict(probs)
        return probs