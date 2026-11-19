from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.core.exceptions import ProviderUnavailableError
from app.face_engine.interfaces import FaceRecognitionProvider, LivenessProvider
from app.face_engine.providers.disabled_liveness import DisabledLivenessProvider
from app.face_engine.providers.http_liveness import HttpLivenessProvider
from app.face_engine.providers.insightface_provider import InsightFaceProvider
from app.face_engine.providers.minifasnet_liveness import MiniFASNetLivenessProvider
from app.face_engine.providers.mock import MockFaceRecognitionProvider, MockLivenessProvider
from app.face_engine.providers.opencv_sface import OpenCvSfaceProvider


@lru_cache
def get_face_provider() -> FaceRecognitionProvider:
    if settings.FACE_RECOGNITION_PROVIDER == "mock":
        if settings.is_production:
            raise ProviderUnavailableError("Mock face provider cannot be used in production")
        return MockFaceRecognitionProvider()
    if settings.FACE_RECOGNITION_PROVIDER == "insightface":
        return InsightFaceProvider()
    if settings.FACE_RECOGNITION_PROVIDER == "opencv_sface":
        return OpenCvSfaceProvider()
    raise ProviderUnavailableError(
        "No face recognition provider is configured. Set FACE_RECOGNITION_PROVIDER."
    )


@lru_cache
def get_liveness_provider() -> LivenessProvider:
    if settings.FACE_LIVENESS_PROVIDER == "mock":
        if settings.is_production:
            raise ProviderUnavailableError("Mock liveness provider cannot be used in production")
        return MockLivenessProvider()
    if settings.FACE_LIVENESS_PROVIDER == "http":
        return HttpLivenessProvider()
    if settings.FACE_LIVENESS_PROVIDER == "disabled":
        return DisabledLivenessProvider()
    if settings.FACE_LIVENESS_PROVIDER == "minifasnet":
        return MiniFASNetLivenessProvider()
    raise ProviderUnavailableError(
        "No liveness provider is configured. Set FACE_LIVENESS_PROVIDER."
    )
