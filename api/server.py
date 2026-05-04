import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.orchestrator import PIPipeline

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
    details: str | None = None


state = AppState()


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
    title="Barrikade Detection API",
    version="0.0.1",
    description="Production API for the Barrikade detection pipeline.",
    lifespan=lifespan,
)


@app.get("/health/live", response_model=HealthResponse)
def live():
    return HealthResponse(status="alive")


@app.get("/health/ready", response_model=ReadinessResponse)
def ready():
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(state.startup_error or "Pipeline not initialized"),
        )

    return ReadinessResponse(
        status="ready",
        pipeline_initialized=True,
        details="Layer E judge active.",
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
