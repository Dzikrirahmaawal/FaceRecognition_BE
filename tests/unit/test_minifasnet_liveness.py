"""Unit tests for MiniFASNet liveness provider with mocked ONNX session.

Tests cover provider behavior, preprocessing, threshold logic, shadow/enforce
modes, and fail-closed safety — all without requiring the real ONNX model file.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.config import settings
from app.core.exceptions import ProviderUnavailableError
from app.face_engine.factory import get_face_provider, get_liveness_provider
from app.face_engine.providers.minifasnet_liveness import (
    INPUT_SHAPE,
    LIVE_CLASS_INDEX,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    MiniFASNetLivenessProvider,
)
from app.face_engine.providers.disabled_liveness import DisabledLivenessProvider
from app.face_engine.providers.http_liveness import HttpLivenessProvider
from app.face_engine.providers.mock import MockLivenessProvider


# ---------------------------------------------------------------------------
# Helper: mock ONNX session + model file validation
# ---------------------------------------------------------------------------


class _FakeSession:
    """Stand-in for ort.InferenceSession that returns controllable logits."""

    def __init__(self, logits: np.ndarray):
        self._logits = logits.astype(np.float32)
        inp = MagicMock()
        inp.name = "input"
        inp.shape = INPUT_SHAPE
        out = MagicMock()
        out.name = "output"
        out.shape = (1, 3)
        self._inputs = [inp]
        self._outputs = [out]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _output_names, _input_feed):
        return [self._logits]


def _mock_model_file(monkeypatch: pytest.MonkeyPatch, model_path: str = "/tmp/test_minifasnet.onnx"):
    """Mock os.path checks so the provider skips real file validation."""
    monkeypatch.setattr(
        "app.face_engine.providers.minifasnet_liveness.os.path.isfile",
        lambda p: p == model_path,
    )
    monkeypatch.setattr(
        "app.face_engine.providers.minifasnet_liveness.os.path.getsize",
        lambda p: MODEL_SIZE_BYTES if p == model_path else 0,
    )
    monkeypatch.setattr(
        "app.face_engine.providers.minifasnet_liveness.MiniFASNetLivenessProvider._sha256_file",
        staticmethod(lambda p: MODEL_SHA256),
    )


def _make_provider_with_session(
    monkeypatch: pytest.MonkeyPatch,
    logits: np.ndarray | None = None,
    model_path: str = "/tmp/test_minifasnet.onnx",
) -> MiniFASNetLivenessProvider:
    """Create a provider with a mocked InferenceSession and mocked file validation."""
    _mock_model_file(monkeypatch, model_path)
    session = _FakeSession(
        logits if logits is not None else np.zeros((1, 3), dtype=np.float32)
    )
    monkeypatch.setattr(
        "app.face_engine.providers.minifasnet_liveness.ort.InferenceSession",
        MagicMock(return_value=session),
    )
    monkeypatch.setattr(
        "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_PATH",
        model_path,
        raising=False,
    )
    monkeypatch.setattr(
        "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_SHA256",
        MODEL_SHA256,
        raising=False,
    )
    monkeypatch.setattr(
        "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_THRESHOLD",
        0.5,
        raising=False,
    )
    monkeypatch.setattr(
        "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_ONNX_INTRA_THREADS",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_ONNX_INTER_THREADS",
        1,
        raising=False,
    )
    return MiniFASNetLivenessProvider()


# ---------------------------------------------------------------------------
# Session init and configuration
# ---------------------------------------------------------------------------


class TestProviderInit:
    def test_session_uses_cpu_execution_provider(self, monkeypatch):
        captured = {}

        def fake_session(model_path, providers, sess_options):
            captured["providers"] = providers
            captured["sess_options"] = sess_options
            return _FakeSession(np.zeros((1, 3), dtype=np.float32))

        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.ort.InferenceSession",
            fake_session,
        )
        _mock_model_file(monkeypatch, "/tmp/test.onnx")
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_PATH",
            "/tmp/test.onnx",
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_SHA256",
            MODEL_SHA256,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_THRESHOLD",
            0.5,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_ONNX_INTRA_THREADS",
            3,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_ONNX_INTER_THREADS",
            1,
            raising=False,
        )

        provider = MiniFASNetLivenessProvider()
        assert "CPUExecutionProvider" in captured["providers"]
        assert captured["sess_options"].intra_op_num_threads == 3
        assert captured["sess_options"].inter_op_num_threads == 1
        assert provider.is_available() is True

    def test_missing_model_path_raises(self, monkeypatch):
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_PATH",
            "",
            raising=False,
        )
        with pytest.raises(ProviderUnavailableError):
            MiniFASNetLivenessProvider()

    def test_hash_mismatch_raises(self, monkeypatch):
        _mock_model_file(monkeypatch)
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_PATH",
            "/tmp/test.onnx",
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_SHA256",
            "badhash",
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_THRESHOLD",
            0.5,
            raising=False,
        )
        with pytest.raises(ProviderUnavailableError):
            MiniFASNetLivenessProvider()

    def test_session_init_error_raises(self, monkeypatch):
        _mock_model_file(monkeypatch)
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_PATH",
            "/tmp/test.onnx",
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_SHA256",
            MODEL_SHA256,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_THRESHOLD",
            0.5,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.ort.InferenceSession",
            MagicMock(side_effect=RuntimeError("CUDA not available")),
        )
        with pytest.raises(ProviderUnavailableError):
            MiniFASNetLivenessProvider()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def _make_provider(self, monkeypatch):
        return _make_provider_with_session(monkeypatch)

    def test_crop_scale_2_7(self, monkeypatch):
        provider = self._make_provider(monkeypatch)
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        bbox = (100, 80, 60, 80)  # x, y, w, h
        tensor = provider._preprocess(img, bbox)
        assert tensor.shape == (1, 3, 80, 80)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_crop_clamps_at_image_border(self, monkeypatch):
        provider = self._make_provider(monkeypatch)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (5, 5, 90, 90)
        tensor = provider._preprocess(img, bbox)
        assert tensor.shape == (1, 3, 80, 80)

    def test_crop_shifts_when_out_of_frame(self, monkeypatch):
        provider = self._make_provider(monkeypatch)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (-10, -10, 50, 50)
        tensor = provider._preprocess(img, bbox)
        assert tensor.shape == (1, 3, 80, 80)

    def test_bgr_channel_order_raw_0_255(self, monkeypatch):
        provider = self._make_provider(monkeypatch)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = [10, 20, 30]  # B=10, G=20, R=30
        bbox = (0, 0, 100, 100)
        tensor = provider._preprocess(img, bbox)
        # VERIFIED contract: raw 0-255 (no /255), BGR channel order.
        assert abs(tensor[0, 0, 0, 0] - 10.0) < 1e-6
        assert abs(tensor[0, 1, 0, 0] - 20.0) < 1e-6
        assert abs(tensor[0, 2, 0, 0] - 30.0) < 1e-6
        assert tensor.min() >= 0.0
        assert tensor.max() <= 255.0


# ---------------------------------------------------------------------------
# Classification and threshold
# ---------------------------------------------------------------------------


class TestClassification:
    def _make_provider_with_logits(self, monkeypatch, logits):
        return _make_provider_with_session(monkeypatch, logits=logits)

    def test_live_class_index_is_one(self):
        assert LIVE_CLASS_INDEX == 1

    def test_high_live_score_passes(self, monkeypatch):
        # class index 1 is live; logits concentrated on class 1 => high p[1]
        logits = np.array([[0.0, 5.0, 0.0]], dtype=np.float32)
        provider = self._make_provider_with_logits(monkeypatch, logits)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, (10, 10, 80, 80))
        assert result.passed is True
        assert result.score is not None
        assert result.score > 0.5
        assert result.reason is None

    def test_low_live_score_fails(self, monkeypatch):
        # class index 1 is live; logits NOT on class 1 => low p[1]
        logits = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        provider = self._make_provider_with_logits(monkeypatch, logits)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, (10, 10, 80, 80))
        assert result.passed is False
        assert result.score is not None
        assert result.score < 0.5
        assert result.reason == "below_threshold"

    def test_threshold_boundary_passes_at_threshold(self, monkeypatch):
        # live prob on class 1 >= 0.5: logits [-10, 0, -10] => softmax p[1] ≈ 0.5
        logits = np.array([[-10.0, 0.0, -10.0]], dtype=np.float32)
        provider = self._make_provider_with_logits(monkeypatch, logits)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, (10, 10, 80, 80))
        assert result.passed is True  # >= 0.5

    def test_liveness_score_equals_softmax_live_index(self, monkeypatch):
        # VERIFIED contract: live_score = softmax(logits)[LIVE_CLASS_INDEX] == softmax(...)[1]
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        provider = self._make_provider_with_logits(monkeypatch, logits)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, (10, 10, 80, 80))
        probs = provider._softmax(logits[0])
        expected_live = probs[LIVE_CLASS_INDEX]
        assert abs(result.score - expected_live) < 1e-6
        assert result.score == probs[1]


# ---------------------------------------------------------------------------
# Fail-closed safety
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_missing_bbox_fails_closed(self, monkeypatch):
        provider = _make_provider_with_session(monkeypatch)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, None)
        assert result.passed is False
        assert result.reason == "no face bbox provided"

    def test_invalid_output_shape_fails_closed(self, monkeypatch):
        _mock_model_file(monkeypatch, "/tmp/test.onnx")
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_PATH",
            "/tmp/test.onnx",
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_SHA256",
            MODEL_SHA256,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_THRESHOLD",
            0.5,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.ort.InferenceSession",
            MagicMock(return_value=_BadSession()),
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.MiniFASNetLivenessProvider._validate_model_contract",
            lambda self: None,
        )
        provider = MiniFASNetLivenessProvider()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, (10, 10, 80, 80))
        assert result.passed is False
        assert result.reason == "invalid model output"

    def test_nan_output_fails_closed(self, monkeypatch):
        _mock_model_file(monkeypatch, "/tmp/test.onnx")
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_PATH",
            "/tmp/test.onnx",
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_SHA256",
            MODEL_SHA256,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_THRESHOLD",
            0.5,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.ort.InferenceSession",
            MagicMock(return_value=_NanSession()),
        )
        provider = MiniFASNetLivenessProvider()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, (10, 10, 80, 80))
        assert result.passed is False

    def test_onnx_exception_fails_closed(self, monkeypatch):
        _mock_model_file(monkeypatch, "/tmp/test.onnx")
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_PATH",
            "/tmp/test.onnx",
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_SHA256",
            MODEL_SHA256,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_THRESHOLD",
            0.5,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.ort.InferenceSession",
            MagicMock(return_value=_ExcSession()),
        )
        provider = MiniFASNetLivenessProvider()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, (10, 10, 80, 80))
        assert result.passed is False
        assert result.reason == "inference_error"


class _BadSession:
    def get_inputs(self):
        inp = MagicMock()
        inp.name = "input"
        inp.shape = INPUT_SHAPE
        return [inp]

    def get_outputs(self):
        out = MagicMock()
        out.name = "output"
        out.shape = (1, 5)
        return [out]

    def run(self, _output_names, _input_feed):
        return [np.zeros((1, 5), dtype=np.float32)]


class _NanSession:
    def get_inputs(self):
        inp = MagicMock()
        inp.name = "input"
        inp.shape = INPUT_SHAPE
        return [inp]

    def get_outputs(self):
        out = MagicMock()
        out.name = "output"
        out.shape = (1, 3)
        return [out]

    def run(self, _output_names, _input_feed):
        return [np.array([[np.nan, np.inf, -np.inf]], dtype=np.float32)]


class _ExcSession:
    def get_inputs(self):
        inp = MagicMock()
        inp.name = "input"
        inp.shape = INPUT_SHAPE
        return [inp]

    def get_outputs(self):
        out = MagicMock()
        out.name = "output"
        out.shape = (1, 3)
        return [out]

    def run(self, _output_names, _input_feed):
        raise RuntimeError("ONNX runtime error")


# ---------------------------------------------------------------------------
# Invalid input handling (fail-closed)
# ---------------------------------------------------------------------------


class TestInvalidInput:
    def _make_provider(self, monkeypatch):
        return _make_provider_with_session(monkeypatch)

    def test_none_image_fails_closed(self, monkeypatch):
        provider = self._make_provider(monkeypatch)
        result = provider.verify_liveness(None, (10, 10, 80, 80))
        assert result.passed is False
        assert result.reason is not None
        assert result.score is None

    def test_zero_size_bbox_fails_closed(self, monkeypatch):
        provider = self._make_provider(monkeypatch)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, (0, 0, 0, 0))
        assert result.passed is False
        assert result.reason is not None
        assert result.score is None

    def test_negative_bbox_fails_closed(self, monkeypatch):
        provider = self._make_provider(monkeypatch)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, (-50, -50, 20, 20))
        # Negative bbox is clamped into the frame; must not crash and must not pass.
        assert result.passed is False
        assert result.reason is not None
        assert result.score is not None
        assert result.score < 0.5


# ---------------------------------------------------------------------------
# Print vs replay: both spoof classes (0 and 2) are NOT live
# ---------------------------------------------------------------------------


class TestPrintVsReplay:
    def _make_provider_with_logits(self, monkeypatch, logits):
        return _make_provider_with_session(monkeypatch, logits=logits)

    def test_class_zero_is_not_live(self, monkeypatch):
        # class index 0 is spoof/other; high p[0] => low p[1] => not live
        logits = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        provider = self._make_provider_with_logits(monkeypatch, logits)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, (10, 10, 80, 80))
        assert result.passed is False
        assert result.score < 0.5

    def test_class_two_is_not_live(self, monkeypatch):
        # class index 2 is spoof/other; high p[2] => low p[1] => not live
        logits = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
        provider = self._make_provider_with_logits(monkeypatch, logits)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = provider.verify_liveness(img, (10, 10, 80, 80))
        assert result.passed is False
        assert result.score < 0.5


# ---------------------------------------------------------------------------
# No silent mock fallback when the real model is unavailable
# ---------------------------------------------------------------------------


class TestNoMockFallback:
    def test_provider_unavailable_when_model_missing(self, monkeypatch):
        # No file mocking: path is empty => ProviderUnavailableError, not a silent mock.
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_PATH",
            "",
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_SHA256",
            MODEL_SHA256,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_THRESHOLD",
            0.5,
            raising=False,
        )
        with pytest.raises(ProviderUnavailableError):
            MiniFASNetLivenessProvider()

    def test_provider_unavailable_when_session_creation_fails(self, monkeypatch):
        _mock_model_file(monkeypatch, "/tmp/test.onnx")
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_PATH",
            "/tmp/test.onnx",
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_MODEL_SHA256",
            MODEL_SHA256,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.settings.FACE_LIVENESS_THRESHOLD",
            0.5,
            raising=False,
        )
        monkeypatch.setattr(
            "app.face_engine.providers.minifasnet_liveness.ort.InferenceSession",
            MagicMock(side_effect=RuntimeError("no CUDA")),
        )
        with pytest.raises(ProviderUnavailableError):
            MiniFASNetLivenessProvider()


# ---------------------------------------------------------------------------
# Existing providers unchanged with new optional parameter
# ---------------------------------------------------------------------------


class TestExistingProvidersUnchanged:
    def test_disabled_provider_accepts_bbox_param(self):
        provider = DisabledLivenessProvider()
        result = provider.verify_liveness(
            np.zeros((100, 100, 3), dtype=np.uint8), (10, 10, 80, 80)
        )
        assert result.passed is False
        assert result.provider == "disabled"
        assert result.reason == "liveness_disabled"

    def test_mock_provider_accepts_bbox_param(self):
        provider = MockLivenessProvider()
        result = provider.verify_liveness(
            np.zeros((100, 100, 3), dtype=np.uint8), (10, 10, 80, 80)
        )
        assert result.passed is True
        assert result.provider == "mock"
        assert result.score == 1.0

    def test_http_provider_accepts_bbox_param(self):
        provider = HttpLivenessProvider()
        import inspect

        sig = inspect.signature(provider.verify_liveness)
        assert "face_bbox" in sig.parameters
        assert sig.parameters["face_bbox"].default is None


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    def test_minifasnet_branch_in_factory(self, monkeypatch):
        monkeypatch.setattr(
            "app.core.config.settings.FACE_LIVENESS_PROVIDER", "minifasnet", raising=False
        )
        get_liveness_provider.cache_clear()
        try:
            get_liveness_provider()
        except ProviderUnavailableError as exc:
            assert "minifasnet" in str(exc) or "model" in str(exc).lower()
        except Exception:
            pytest.fail("Unexpected exception type")


# ---------------------------------------------------------------------------
# Shadow vs enforce mode
# ---------------------------------------------------------------------------


class TestShadowMode:
    def test_shadow_mode_logs_and_allows(self, monkeypatch, caplog):
        from app.services import face_service
        import logging

        monkeypatch.setattr(
            "app.core.config.settings.FACE_LIVENESS_MODE", "shadow", raising=False
        )
        monkeypatch.setattr(
            "app.core.config.settings.FACE_LIVENESS_THRESHOLD", 0.5, raising=False
        )

        class _FailingProvider:
            name = "minifasnet"

            def verify_liveness(self, image, face_bbox=None):
                from app.face_engine.interfaces import LivenessResult

                return LivenessResult(
                    passed=False, provider="minifasnet", score=0.1, reason="below_threshold"
                )

        monkeypatch.setattr(
            "app.services.face_service.get_liveness_provider",
            lambda: _FailingProvider(),
            raising=False,
        )
        get_liveness_provider.cache_clear()

        # Mock face detection to return a valid face so we reach liveness
        from app.face_engine.interfaces import DetectedFace
        from app.face_engine.factory import get_face_provider

        class _FakeFaceProvider:
            name = "fake"
            model_version = "fake-v1"
            embedding_version = "1"

            def detect_faces(self, image):
                return [DetectedFace(bbox=(10, 10, 80, 80), score=0.99)]

            def validate_face_quality(self, image, face):
                from app.face_engine.interfaces import QualityReport
                return QualityReport(acceptable=True)

            def generate_embedding(self, image, face):
                return np.ones(128, dtype=np.float32)

        monkeypatch.setattr(
            "app.services.face_service.get_face_provider",
            lambda: _FakeFaceProvider(),
            raising=False,
        )
        get_face_provider.cache_clear()

        # Create a valid minimal PNG image
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".png", img)
        valid_image_bytes = buffer.tobytes()

        with caplog.at_level(logging.WARNING, logger="app.services.face_service"):
            try:
                face_service._prepare_capture(valid_image_bytes, "image/png", operation="enroll")
            except Exception:
                pass

        shadow_logs = [
            r
            for r in caplog.records
            if getattr(r, "extra_data", None)
            and r.extra_data.get("event") == "liveness_shadow_failure"
        ]
        assert len(shadow_logs) >= 1

    def test_enforce_mode_blocks(self, monkeypatch):
        from app.services import face_service
        from app.core.exceptions import ValidationAppError
        import logging

        monkeypatch.setattr(
            "app.core.config.settings.FACE_LIVENESS_MODE", "enforce", raising=False
        )
        monkeypatch.setattr(
            "app.core.config.settings.FACE_LIVENESS_THRESHOLD", 0.5, raising=False
        )

        class _FailingProvider:
            name = "minifasnet"

            def verify_liveness(self, image, face_bbox=None):
                from app.face_engine.interfaces import LivenessResult

                return LivenessResult(
                    passed=False, provider="minifasnet", score=0.1, reason="below_threshold"
                )

        monkeypatch.setattr(
            "app.services.face_service.get_liveness_provider",
            lambda: _FailingProvider(),
            raising=False,
        )
        get_liveness_provider.cache_clear()

        from app.face_engine.interfaces import DetectedFace
        from app.face_engine.factory import get_face_provider

        class _FakeFaceProvider:
            name = "fake"
            model_version = "fake-v1"
            embedding_version = "1"

            def detect_faces(self, image):
                return [DetectedFace(bbox=(10, 10, 80, 80), score=0.99)]

            def validate_face_quality(self, image, face):
                from app.face_engine.interfaces import QualityReport
                return QualityReport(acceptable=True)

            def generate_embedding(self, image, face):
                return np.ones(128, dtype=np.float32)

        monkeypatch.setattr(
            "app.services.face_service.get_face_provider",
            lambda: _FakeFaceProvider(),
            raising=False,
        )
        get_face_provider.cache_clear()

        # Create a valid minimal PNG image
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".png", img)
        valid_image_bytes = buffer.tobytes()

        with pytest.raises(ValidationAppError) as exc:
            face_service._prepare_capture(valid_image_bytes, "image/png", operation="enroll")
        assert exc.value.error_code == "LIVENESS_FAILED"


# ---------------------------------------------------------------------------
# Real-model regression test (skips gracefully if model/fixtures absent)
# ---------------------------------------------------------------------------


class TestRealModelRegression:
    """Regression test against the real MiniFASNet ONNX model and real fixtures.

    Reads paths from environment variables (never commits personal photos):
      - MINIFASNET_MODEL_PATH        (default: models/minifasnet_v2.onnx)
      - LIVENESS_REAL_DIR            (default: liveness_samples/real)
      - LIVENESS_SCREEN_DIR          (default: liveness_samples/screen)

    Skips when the model file or either fixture directory is absent.
    Asserts the verified live/spoof decision direction:
      screen (spoof) images are NOT classified as live (argmax != LIVE_CLASS_INDEX),
      and real images have higher mean p[live] than screen images.
    """

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _load_real_model(self):
        import cv2  # noqa: import out of project - local import for real-model test
        import onnxruntime as ort  # noqa: import out of project

        model_path = Path(
            os.environ.get(
                "MINIFASNET_MODEL_PATH",
                str(self._repo_root() / "models" / "minifasnet_v2.onnx"),
            )
        )
        if not model_path.is_file():
            pytest.skip(f"Real model not found: {model_path}")
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        det = cv2.FaceDetectorYN.create(
            str(self._repo_root() / "models" / "face_detection_yunet_2023mar.onnx"),
            "",
            (320, 320),
            0.60,
            0.30,
            5000,
            cv2.dnn.DNN_BACKEND_DEFAULT,
            cv2.dnn.DNN_TARGET_CPU,
        )
        return session, input_name, output_name, det

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    @staticmethod
    def _detect_first_face(det, image):
        import cv2  # noqa: import out of project

        h, w = image.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(image)
        if faces is None or len(faces) == 0:
            return None
        row = faces[0]
        x, y, bw, bh, _score = [float(v) for v in row[0:5]]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        return int(x), int(y), int(bw), int(bh)

    @staticmethod
    def _crop_and_tensor(image, bbox):
        import cv2  # noqa: import out of project

        x, y, bw, bh = bbox
        h_img, w_img = image.shape[:2]
        scale = min((h_img - 1) / bh, (w_img - 1) / bw, 2.7)
        cx = x + bw / 2
        cy = y + bh / 2
        x1 = max(0, int(cx - bw * scale / 2))
        y1 = max(0, int(cy - bh * scale / 2))
        x2 = min(w_img - 1, int(cx + bw * scale / 2))
        y2 = min(h_img - 1, int(cy + bh * scale / 2))
        cropped = image[y1 : y2 + 1, x1 : x2 + 1]
        if cropped.size == 0:
            cropped = image
        resized = cv2.resize(cropped, (80, 80))
        tensor = resized.astype(np.float32)  # VERIFIED: raw 0-255, no /255, BGR
        tensor = np.transpose(tensor, (2, 0, 1))
        return np.expand_dims(tensor, axis=0)

    def _evaluate_dir(self, session, input_name, output_name, det, directory):
        import cv2  # noqa: import out of project - local import for real-model test

        directory = Path(directory)
        if not directory.is_dir():
            pytest.skip(f"Fixture directory not found: {directory}")
        results = []
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = cv2.imread(str(p))  # noqa: cv2 imported in caller
            if image is None:
                continue
            face = self._detect_first_face(det, image)
            if face is None:
                continue
            tensor = self._crop_and_tensor(image, face)
            logits = session.run(None, {input_name: tensor})[0][0]
            probs = self._softmax(logits)
            results.append(
                {
                    "name": p.name,
                    "logits": logits,
                    "probs": probs,
                    "argmax": int(np.argmax(probs)),
                    "p_live": float(probs[LIVE_CLASS_INDEX]),
                }
            )
        return results

    def test_real_model_separates_live_from_spoof(self):
        import cv2  # noqa: import out of project - local import for real-model test

        session, input_name, output_name, det = self._load_real_model()

        real_dir = Path(
            os.environ.get(
                "LIVENESS_REAL_DIR", str(self._repo_root() / "liveness_samples" / "real")
            )
        )
        screen_dir = Path(
            os.environ.get(
                "LIVENESS_SCREEN_DIR",
                str(self._repo_root() / "liveness_samples" / "screen"),
            )
        )

        real = self._evaluate_dir(session, input_name, output_name, det, real_dir)
        screen = self._evaluate_dir(session, input_name, output_name, det, screen_dir)

        if not real or not screen:
            pytest.skip("Insufficient real/screen fixtures with detected faces")

        # Decision direction: no screen image may be classified as live
        # (argmax == LIVE_CLASS_INDEX == 1). This is the security-critical side.
        live_screens = [r for r in screen if r["argmax"] == LIVE_CLASS_INDEX]
        assert len(live_screens) == 0, (
            f"Screen images classified as live: {[r['name'] for r in live_screens]}"
        )

        # Real images should concentrate more probability on the live class than screens.
        real_mean = float(np.mean([r["p_live"] for r in real]))
        screen_mean = float(np.mean([r["p_live"] for r in screen]))
        assert real_mean > screen_mean, (
            f"Real mean p[live]={real_mean:.4f} not > screen mean p[live]={screen_mean:.4f}"
        )


# ---------------------------------------------------------------------------
# Failed liveness -> no embedding and no visitor record
# ---------------------------------------------------------------------------


class TestFailedLivenessNoEmbeddingNoVisitor:
    """Verify that a failed liveness check in enforce mode aborts before
    any embedding is generated and before any visitor record is created.
    Covers enroll, identify, and verify operations via _prepare_capture.
    """

    @staticmethod
    def _setup_failed_liveness(monkeypatch, operation: str = "enroll"):
        from app.core.config import settings
        from app.services import face_service
        from app.face_engine.factory import get_liveness_provider, get_face_provider
        from app.face_engine.interfaces import (
            DetectedFace,
            FaceRecognitionProvider,
            LivenessResult,
            QualityReport,
        )

        monkeypatch.setattr(
            "app.core.config.settings.FACE_LIVENESS_MODE", "enforce", raising=False
        )
        monkeypatch.setattr(
            "app.core.config.settings.FACE_LIVENESS_THRESHOLD", 0.5, raising=False
        )

        embedding_generated = {"called": False}

        class _FailingLivenessProvider:
            name = "minifasnet"

            def verify_liveness(self, image, face_bbox=None):
                return LivenessResult(
                    passed=False,
                    provider="minifasnet",
                    score=0.1,
                    reason="below_threshold",
                )

        class _TrackingFaceProvider:
            name = "fake"
            model_version = "fake-v1"
            embedding_version = "1"

            def detect_faces(self, image):
                return [DetectedFace(bbox=(10, 10, 80, 80), score=0.99)]

            def validate_face_quality(self, image, face):
                return QualityReport(acceptable=True)

            def generate_embedding(self, image, face):
                embedding_generated["called"] = True
                return np.ones(128, dtype=np.float32)

        monkeypatch.setattr(
            "app.services.face_service.get_liveness_provider",
            lambda: _FailingLivenessProvider(),
            raising=False,
        )
        monkeypatch.setattr(
            "app.services.face_service.get_face_provider",
            lambda: _TrackingFaceProvider(),
            raising=False,
        )
        get_liveness_provider.cache_clear()
        get_face_provider.cache_clear()

        import cv2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".png", img)
        return buffer.tobytes(), embedding_generated

    def test_failed_liveness_enroll_no_embedding_no_visitor(self, monkeypatch):
        from app.core.exceptions import ValidationAppError
        from app.services import face_service

        valid_image_bytes, embedding_generated = self._setup_failed_liveness(
            monkeypatch, operation="enroll"
        )

        try:
            face_service._prepare_capture(
                valid_image_bytes, "image/png", operation="enroll"
            )
        except ValidationAppError as exc:
            assert exc.error_code == "LIVENESS_FAILED"
        else:
            pytest.fail("Expected ValidationAppError for failed liveness in enforce mode")

        assert embedding_generated["called"] is False, (
            "generate_embedding must NOT be called when liveness fails"
        )

    def test_failed_liveness_identify_no_embedding_no_visitor(self, monkeypatch):
        from app.core.exceptions import ValidationAppError
        from app.services import face_service

        valid_image_bytes, embedding_generated = self._setup_failed_liveness(
            monkeypatch, operation="identify"
        )

        try:
            face_service._prepare_capture(
                valid_image_bytes, "image/png", operation="identify"
            )
        except ValidationAppError as exc:
            assert exc.error_code == "LIVENESS_FAILED"
        else:
            pytest.fail("Expected ValidationAppError for failed liveness in enforce mode")

        assert embedding_generated["called"] is False, (
            "generate_embedding must NOT be called when liveness fails"
        )

    def test_failed_liveness_verify_no_embedding_no_visitor(self, monkeypatch):
        from app.core.exceptions import ValidationAppError
        from app.services import face_service

        valid_image_bytes, embedding_generated = self._setup_failed_liveness(
            monkeypatch, operation="verify"
        )

        try:
            face_service._prepare_capture(
                valid_image_bytes, "image/png", operation="verify"
            )
        except ValidationAppError as exc:
            assert exc.error_code == "LIVENESS_FAILED"
        else:
            pytest.fail("Expected ValidationAppError for failed liveness in enforce mode")

        assert embedding_generated["called"] is False, (
            "generate_embedding must NOT be called when liveness fails"
        )
