import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.orchestrator import PIPipeline
from core.settings import Settings

log = logging.getLogger(__name__)

@dataclass
class AppState:
    pipeline: PIPipeline | None = None
    startup_error: str | None = None


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    include_diagnostics: bool = False


class DetectResponse(BaseModel):
    final_verdict: str
    decision_layer: str
    confidence_score: float
    total_processing_time_ms: float
    result: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    pipeline_initialized: bool
    ollama_reachable: bool
    details: str | None = None


state = AppState()


def _check_ollama(base_url: str, timeout_s: float = 2.0):
    endpoint = f"{base_url.rstrip('/')}/api/tags"
    req = request.Request(endpoint, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if 200 <= resp.status < 300:
                return True, None, body
            return False, f"Unexpected status {resp.status} from {endpoint}", None
    except (error.URLError, error.HTTPError, TimeoutError) as exc:
        return False, str(exc), None


def _has_ollama_model(tags_response: dict[str, Any] | None, model_name: str) -> bool:
    if not tags_response:
        return False
    models = tags_response.get("models", [])
    for model in models:
        name = str(model.get("name", "")).strip()
        if name == model_name:
            return True
    return False


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state.pipeline = PIPipeline()
        state.startup_error = None
        log.info("Barrikada pipeline initialized")
    except Exception as exc:  # pragma: no cover
        state.pipeline = None
        state.startup_error = str(exc)
        log.exception("Failed to initialize Barrikada pipeline")
    yield


app = FastAPI(
    title="Barrikada Detection API",
    version="0.0.1",
    description="Production API for the Barrikada detection pipeline.",
    lifespan=lifespan,
)


@app.get("/health/live", response_model=HealthResponse)
def live():
    return HealthResponse(status="alive")


@app.get("/health/ready", response_model=ReadinessResponse)
def ready():
    settings = Settings()
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(state.startup_error or "Pipeline not initialized"),
        )

    ollama_base_url = os.getenv("LAYER_E_OLLAMA_BASE_URL", settings.layer_e_ollama_base_url)
    ollama_ok, ollama_err, tags_response = _check_ollama(ollama_base_url)
    if not ollama_ok:
        raise HTTPException(
            status_code=503,
            detail=f"Layer E backend unreachable: {ollama_err}",
        )

    judge_mode = settings.layer_e_judge_mode.strip().lower()
    required_model = (
        settings.layer_e_runtime_finetuned_model
        if judge_mode == "finetuned"
        else settings.layer_e_runtime_base_model
    )
    required_model = os.getenv("LAYER_E_RUNTIME_MODEL", required_model)
    if not _has_ollama_model(tags_response, required_model):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Layer E model '{required_model}' not found in Ollama. "
                "Pull it with: ollama pull "
                f"{required_model}"
            ),
        )

    return ReadinessResponse(
        status="ready",
        pipeline_initialized=True,
        ollama_reachable=True,
        details=None,
    )


@app.post("/v1/detect", response_model=DetectResponse)
def detect(payload: DetectRequest):
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=state.startup_error or "Pipeline unavailable",
        )

    try:
        result = state.pipeline.detect(payload.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        log.exception("Detection request failed")
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

    details = result.to_dict() if payload.include_diagnostics else None
    return DetectResponse(
        final_verdict=result.final_verdict.value,
        decision_layer=result.decision_layer.value,
        confidence_score=result.confidence_score,
        total_processing_time_ms=result.total_processing_time_ms,
        result=details,
    )
