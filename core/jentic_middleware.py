import json
import logging

from core.adapter import BarrikadaAdapter

LOGGER = logging.getLogger(__name__)
DEFAULT_GUARDED_ROUTES = ("/search", "/execute", "/tasks")

_ADAPTER_SINGLETON = None


def get_guard_adapter():
    """Return the process-wide guard adapter singleton."""
    global _ADAPTER_SINGLETON
    if _ADAPTER_SINGLETON is None:
        _ADAPTER_SINGLETON = BarrikadaAdapter()
    return _ADAPTER_SINGLETON


def set_guard_adapter(adapter):
    """Override the singleton adapter (useful for tests and custom bootstrap)."""
    global _ADAPTER_SINGLETON
    _ADAPTER_SINGLETON = adapter


def warmup_guard_adapter(
    adapter=None,
    *,
    warmup_text="healthcheck",
):
    """Force model initialization on startup with a safe dummy prompt."""
    guard = adapter or get_guard_adapter()
    return guard.detect(warmup_text)


def get_guard_readiness(adapter=None):
    """Return readiness status for health probes and startup checks."""
    guard = adapter or get_guard_adapter()
    try:
        outcome = warmup_guard_adapter(guard)
        healthy = outcome.get("metadata", {}).get("verdict") != "error"
        return {
            "ready": bool(healthy),
            "error": None if healthy else outcome.get("metadata", {}).get("error", "unknown"),
            "last_warmup": outcome,
        }
    except Exception as exc:
        LOGGER.exception("Guard readiness check failed")
        return {
            "ready": False,
            "error": str(exc),
            "last_warmup": None,
        }


def _is_guarded_route(method, path, guarded_routes):
    if method.upper() != "POST":
        return False
    return any(path.startswith(prefix) for prefix in guarded_routes)


def _extract_text_from_body(body):
    if not body:
        return ""
    try:
        payload = json.loads(body.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            return body.decode("utf-8", errors="ignore")

        return (
            str(payload.get("query") or "")
            or str(payload.get("input") or "")
            or str(payload.get("prompt") or "")
        )
    except Exception:
        return body.decode("utf-8", errors="ignore")


def _default_block_response(decision):
    return {
        "status_code": 403,
        "content": {
            "error": "prompt_injection_detected",
            "detail": decision,
        },
    }


async def dispatch_guard(
    request,
    call_next,
    *,
    adapter=None,
    guarded_routes=DEFAULT_GUARDED_ROUTES,
    block_response_factory=None,
):
    """
    Request guard dispatch intended for FastAPI/Starlette middleware usage.

    The request object is expected to expose:
      - method: str
      - url.path: str
      - async body() -> bytes
    """
    method = getattr(request, "method", "")
    path = getattr(getattr(request, "url", None), "path", "")

    if not _is_guarded_route(method, path, guarded_routes):
        return await call_next(request)

    body = await request.body()
    text = _extract_text_from_body(body)

    guard = adapter or get_guard_adapter()
    decision = guard.detect(text)
    if decision.get("blocked"):
        factory = block_response_factory or _default_block_response
        return factory(decision)

    response = await call_next(request)
    if decision.get("flagged") and hasattr(response, "headers"):
        response.headers["X-Barrikada-Flagged"] = "true"
        response.headers["X-Barrikada-Layer"] = decision.get("metadata", {}).get("decision_layer", "unknown")
    return response


def create_fastapi_guard(
    *,
    adapter=None,
    guarded_routes=DEFAULT_GUARDED_ROUTES,
):
    """Create a FastAPI-compatible middleware function returning JSONResponse on block."""

    def _json_response_factory(decision):
        from starlette.responses import JSONResponse

        return JSONResponse(
            status_code=403,
            content={
                "error": "prompt_injection_detected",
                "detail": decision,
            },
        )

    async def _middleware(request, call_next):
        return await dispatch_guard(
            request,
            call_next,
            adapter=adapter,
            guarded_routes=guarded_routes,
            block_response_factory=_json_response_factory,
        )

    return _middleware