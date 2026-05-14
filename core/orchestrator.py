import hashlib
import logging
import time

from core.artifacts import ensure_runtime_artifacts
from core.settings import Settings
from models.PipelineResult import PipelineResult

from models.Verdicts import DecisionLayer, FinalVerdict


log = logging.getLogger(__name__)


class PIPipeline:
    def __init__(self):
        log.info("Initializing Barrikade pipeline")
        ensure_runtime_artifacts()

        from core.layer_a.pipeline import analyze_text
        from core.layer_b.signature_engine import SignatureEngine
        from core.layer_c.classifier import Classifier
        from core.layer_d.classifier import LayerDClassifier
        from core.layer_e.llm_judge import LLMJudge

        settings = Settings()

        self.layer_a_analyze = analyze_text
        self.layer_b_engine = SignatureEngine()
        self.layer_c_classifier = Classifier(
            model_path=settings.model_path,
            embedding_model=settings.layer_c_embedding_model,
            low=settings.layer_c_low_threshold,
            high=settings.layer_c_high_threshold,
        )
        self.layer_d_classifier = LayerDClassifier(
            model_dir=settings.layer_d_output_dir,
            low=settings.layer_d_low_threshold,
            high=settings.layer_d_high_threshold,
            max_length=settings.layer_d_max_length,
        )
        self.layer_e_judge = LLMJudge(
            model_dir=settings.layer_e_model_dir,
            model_name=settings.layer_e_model_dir,
            temperature=settings.layer_e_temperature,
            timeout_s=settings.layer_e_timeout_s,
            max_retries=settings.layer_e_max_retries,
            max_new_tokens=settings.layer_e_max_new_tokens,
            no_think_default=settings.layer_e_no_think_default,
        )
        log.info("Barrikade pipeline ready")

    def _create_result(self, input_hash, start_time, layer_a_result, final_verdict, decision_layer, confidence_score, layer_b_result=None, layer_c_result=None, layer_d_result=None, layer_e_result_dict=None, layer_e_time_ms=None, ):
        total_time = (time.time() - start_time) * 1000
        return PipelineResult(
            input_hash=input_hash,
            total_processing_time_ms=total_time,
            layer_a_result=layer_a_result.to_dict(),
            layer_a_time_ms=layer_a_result.processing_time_ms,
            layer_b_result=layer_b_result.to_dict() if layer_b_result else None,
            layer_b_time_ms=layer_b_result.processing_time_ms if layer_b_result else None,
            layer_c_result=layer_c_result.to_dict() if layer_c_result else None,
            layer_c_time_ms=layer_c_result.processing_time_ms if layer_c_result else None,
            layer_d_result=layer_d_result.to_dict() if layer_d_result else None,
            layer_d_time_ms=layer_d_result.processing_time_ms if layer_d_result else None,
            layer_e_result=layer_e_result_dict,
            layer_e_time_ms=layer_e_time_ms,
            final_verdict=final_verdict,
            decision_layer=decision_layer,
            confidence_score=confidence_score,
        )

    def detect(self, input_text):
        start_time = time.time()
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]

        #Layer A
        layer_a_result = self.layer_a_analyze(input_text)
        analysis_text = layer_a_result.processed_text

        # Hard-block from Layer A (high-confidence flags)
        if layer_a_result.get_verdict() == "block":
            return self._create_result(
                input_hash, start_time, layer_a_result,
                final_verdict=FinalVerdict.BLOCK,
                decision_layer=DecisionLayer.LAYER_A,
                confidence_score=layer_a_result.confidence_score,
            )

        #Layer B
        layer_b_result = self.layer_b_engine.detect(analysis_text) #type: ignore

        # MALICIOUS signatures => block immediately
        if layer_b_result.verdict == "block" or layer_b_result.verdict == "allow":
            return self._create_result(
                input_hash, start_time, layer_a_result,
                layer_b_result=layer_b_result,
                final_verdict=FinalVerdict(layer_b_result.verdict),
                decision_layer=DecisionLayer.LAYER_B,
                confidence_score=layer_b_result.confidence_score,
            )

        #Layer C
        # Anything not blocked by Layer B is screened by the ML classifier.
        layer_c_result = self.layer_c_classifier.predict(analysis_text)

        if layer_c_result.verdict == "block" or layer_c_result.verdict == "allow":
            return self._create_result(
                input_hash, start_time, layer_a_result,
                layer_b_result=layer_b_result,
                layer_c_result=layer_c_result,
                final_verdict=FinalVerdict(layer_c_result.verdict),
                decision_layer=DecisionLayer.LAYER_C,
                confidence_score=layer_c_result.confidence_score,
            )
    
        #Layer D
        layer_d_result = self.layer_d_classifier.predict(analysis_text)

        if layer_d_result.verdict == "block" or layer_d_result.verdict == "allow":
            return self._create_result(
                input_hash, start_time, layer_a_result,
                layer_b_result=layer_b_result,
                layer_c_result=layer_c_result,
                layer_d_result=layer_d_result,
                final_verdict=FinalVerdict(layer_d_result.verdict),
                decision_layer=DecisionLayer.LAYER_D,
                confidence_score=layer_d_result.confidence_score,
            )

        #Layer E
        layer_e_start = time.time()
        layer_e_result = self.layer_e_judge.call_judge(analysis_text)
        layer_e_time_ms = (time.time() - layer_e_start) * 1000

        # Fallback if LLM judge returns None unexpectedly
        layer_e_result_dict = layer_e_result.model_dump()
        layer_e_verdict = FinalVerdict.BLOCK if layer_e_result.decision == "block" else FinalVerdict.ALLOW

        return self._create_result(
            input_hash, start_time, layer_a_result,
            layer_b_result=layer_b_result,
            layer_c_result=layer_c_result,
            layer_d_result=layer_d_result,
            layer_e_result_dict=layer_e_result_dict,
            layer_e_time_ms=layer_e_time_ms,
            final_verdict=layer_e_verdict,
            decision_layer=DecisionLayer.LAYER_E,
            confidence_score=1.0,  # LLM judge gives binary decisions
        )

    
