from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import check_database
from app.core.types import utcnow
from app.face_engine.factory import get_face_provider, get_liveness_provider

router = APIRouter(tags=["Health"])


@router.get("/health")
def health():
    return {
        "success": True,
        "message": "Service is healthy",
        "data": {
            "status": "ok",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV,
            "timestamp": utcnow(),
        },
        "meta": None,
    }


@router.get("/ready")
def ready():
    checks: dict[str, dict] = {}
    ready_state = True
    try:
        check_database()
        checks["database"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        ready_state = False
        checks["database"] = {"ok": False, "error": type(exc).__name__}

    try:
        face_provider = get_face_provider()
        available = face_provider.is_available()
        checks["face_provider"] = {
            "ok": available,
            "name": face_provider.name,
            "model_version": face_provider.model_version,
        }
        ready_state = ready_state and available
    except Exception as exc:  # noqa: BLE001
        ready_state = False
        checks["face_provider"] = {"ok": False, "error": type(exc).__name__}

    try:
        liveness_provider = get_liveness_provider()
        available = liveness_provider.is_available()
        liveness_disabled = liveness_provider.name == "disabled"
        liveness_mode = settings.FACE_LIVENESS_MODE
        liveness_enabled = liveness_provider.name not in ("disabled", "mock")
        check: dict = {
            "ok": True,
            "name": liveness_provider.name,
            "disabled": liveness_disabled,
            "mode": liveness_mode,
            "enabled": liveness_enabled,
        }
        if liveness_provider.name == "minifasnet":
            check["model_version"] = liveness_provider.model_version
            check["available"] = available
        elif liveness_provider.name == "mock":
            check["available"] = available
        checks["liveness_provider"] = check
        # Disabled liveness is an intentional configuration, not a failure.
        # The service can still operate without anti-spoofing protection.
    except Exception as exc:  # noqa: BLE001
        ready_state = False
        checks["liveness_provider"] = {"ok": False, "error": type(exc).__name__}

    payload = {
        "success": ready_state,
        "message": "Service is ready" if ready_state else "Service is not ready",
        "data": {"ready": ready_state, "checks": checks, "timestamp": utcnow()},
        "meta": None,
    }
    return JSONResponse(status_code=200 if ready_state else 503, content=_jsonable(payload))


def _jsonable(payload: dict) -> dict:
    from fastapi.encoders import jsonable_encoder

    return jsonable_encoder(payload)
