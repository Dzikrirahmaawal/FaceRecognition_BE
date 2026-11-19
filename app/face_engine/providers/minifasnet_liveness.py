"""MiniFASNet V2 CPU-only passive liveness / anti-spoofing provider.

Uses the MiniFASNet V2 (2.7_80x80) ONNX model from minivision-ai's
Silent-Face-Anti-Spoofing project, distributed via
garciafido/minifasnet-v2-anti-spoofing-onnx (Apache 2.0).

Model: minifasnet_v2.onnx
- Input:  [batch, 3, 80, 80] float32, BGR, raw range [0, 255] (NO /255 division)
- Output: [batch, 3] float32 raw logits (softmax NOT baked in; applied in provider)
- Classes (verified): index 0 = spoof/other, index 1 = Real Face (live), index 2 = spoof/other
  → live index = 1 (agrees with upstream test.py:45-51; contradicts the redistributing
  model card which claims index 0)
- Liveness score = p[live] = softmax(logits)[LIVE_CLASS_INDEX]

VERIFIED CONTRACT (Checkpoint B, real webcam images):
  - pixel range: raw 0-255 float (dividing by 255 destroys real/screen separation)
  - channel order: BGR (cv2.imread native; RGB erases separation)
  - live index: 1 (real faces concentrate probability on class 1; screens do not)
  - softmax: applied by the provider after inference
  - crop: 2.7x scale around bbox center, resize to 80x80 square

CPU only: onnxruntime CPUExecutionProvider, no PyTorch, no CUDA.
"""

from __future__ import annotations

import hashlib
import os
import threading
from time import perf_counter

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - onnxruntime is a required dependency
    raise ImportError(
        "onnxruntime is required for the minifasnet liveness provider. "
        "Install it with: pip install onnxruntime"
    ) from exc

from app.core.config import settings
from app.core.exceptions import ProviderUnavailableError
from app.face_engine.interfaces import LivenessProvider, LivenessResult

# Pinned model metadata (matches the downloaded ONNX file)
MODEL_SIZE_BYTES = 1_744_116
MODEL_SHA256 = "d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b"

INPUT_NAME = "input"
OUTPUT_NAME = "output"
INPUT_SHAPE = (1, 3, 80, 80)
CROP_SCALE = 2.7
# VERIFIED: live class index = 1 (Real Face), per upstream test.py:45-51 and
# Checkpoint B real-image experiments (real faces -> high p[1], screens -> low p[1]).
# The redistributing model card incorrectly claims index 0; empirical evidence wins.
LIVE_CLASS_INDEX = 1


class MiniFASNetLivenessProvider(LivenessProvider):
    """CPU-only passive liveness provider using MiniFASNet V2 ONNX."""

    name = "minifasnet"
    model_version = "minifasnet-v2-2.7-80x80"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._load_error: str | None = None

        model_path = settings.FACE_LIVENESS_MODEL_PATH
        expected_sha = settings.FACE_LIVENESS_MODEL_SHA256

        try:
            self._validate_model_file(model_path, expected_sha)
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
                sess_options=self._build_session_options(),
            )
            # Inspect actual input/output names and shapes
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._validate_model_contract()
        except Exception as exc:
            self._session = None
            self._load_error = str(exc)
            raise ProviderUnavailableError(
                f"MiniFASNet liveness provider unavailable: {exc}"
            ) from exc

    def _build_session_options(self) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.FACE_LIVENESS_ONNX_INTRA_THREADS
        options.inter_op_num_threads = settings.FACE_LIVENESS_ONNX_INTER_THREADS
        return options

    def _validate_model_file(self, model_path: str, expected_sha: str) -> None:
        if not model_path:
            raise ProviderUnavailableError("FACE_LIVENESS_MODEL_PATH is not configured")
        if not os.path.isfile(model_path):
            raise ProviderUnavailableError(
                f"Liveness model file not found: {model_path}"
            )
        actual_size = os.path.getsize(model_path)
        if actual_size != MODEL_SIZE_BYTES:
            raise ProviderUnavailableError(
                f"Liveness model size mismatch: {actual_size} != {MODEL_SIZE_BYTES}"
            )
        actual_sha = self._sha256_file(model_path)
        if actual_sha != expected_sha:
            raise ProviderUnavailableError(
                f"Liveness model SHA-256 mismatch: {actual_sha} != {expected_sha}"
            )

    @staticmethod
    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _validate_model_contract(self) -> None:
        if self._input_name != INPUT_NAME:
            raise ProviderUnavailableError(
                f"Unexpected input name: {self._input_name} (expected {INPUT_NAME})"
            )
        if self._output_name != OUTPUT_NAME:
            raise ProviderUnavailableError(
                f"Unexpected output name: {self._output_name} (expected {OUTPUT_NAME})"
            )
        # Validate input shape: batch dim is dynamic, rest must be 3x80x80
        inp = self._session.get_inputs()[0]
        if list(inp.shape[1:]) != list(INPUT_SHAPE[1:]):
            raise ProviderUnavailableError(
                f"Unexpected input shape: {inp.shape} (expected {INPUT_SHAPE})"
            )
        # Validate output: batch x 3 classes
        out = self._session.get_outputs()[0]
        if list(out.shape[1:]) != [3]:
            raise ProviderUnavailableError(
                f"Unexpected output shape: {out.shape} (expected [batch, 3])"
            )

    def _crop_face(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Crop face region with 2.7× scale around center, clamp/shift at image borders.

        Mirrors the upstream MiniFASNet preprocessing (filename prefix 2.7_80x80):
        expand the bbox around its center, clamp into the image frame, then always
        resize to the fixed square 80x80 (out_w x out_h) required by the model.
        """
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]

        # Clamp scale so the crop never exceeds image bounds
        scale = min((h_img - 1) / h, (w_img - 1) / w, CROP_SCALE)
        new_w = w * scale
        new_h = h * scale

        center_x = x + w / 2
        center_y = y + h / 2

        x1 = max(0, int(center_x - new_w / 2))
        y1 = max(0, int(center_y - new_h / 2))
        x2 = min(w_img - 1, int(center_x + new_w / 2))
        y2 = min(h_img - 1, int(center_y + new_h / 2))

        cropped = image[y1 : y2 + 1, x1 : x2 + 1]
        if cropped.size == 0:
            # Degenerate crop: fall back to the original image
            cropped = image

        resized = cv2.resize(cropped, (80, 80))
        return resized

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def _preprocess(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Build the NCHW float32 input tensor expected by the ONNX model.

        VERIFIED contract: raw 0-255 float pixels (NO division by 255), BGR channel
        order (cv2.imread native). Dividing by 255 was empirically shown to destroy
        real/screen separation on Checkpoint B images.
        """
        cropped = self._crop_face(image, bbox)
        # BGR channel order (OpenCV native), raw pixel range [0, 255] (no /255)
        tensor = cropped.astype(np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def verify_liveness(
        self, image: np.ndarray, face_bbox: tuple[int, int, int, int] | None = None
    ) -> LivenessResult:
        if self._session is None:
            return LivenessResult(
                passed=False,
                provider=self.name,
                score=None,
                reason=self._load_error or "model not loaded",
                model_version=self.model_version,
            )

        if face_bbox is None:
            return LivenessResult(
                passed=False,
                provider=self.name,
                score=None,
                reason="no face bbox provided",
                model_version=self.model_version,
            )

        try:
            input_tensor = self._preprocess(image, face_bbox)
            with self._lock:
                start = perf_counter()
                outputs = self._session.run(
                    None, {self._input_name: input_tensor}
                )
                latency_ms = (perf_counter() - start) * 1000.0

            logits = outputs[0]
            if logits.shape != (1, 3) or not np.all(np.isfinite(logits)):
                return LivenessResult(
                    passed=False,
                    provider=self.name,
                    score=None,
                    reason="invalid model output",
                    model_version=self.model_version,
                )

            probs = self._softmax(logits[0])
            # VERIFIED: live score = softmax(logits)[LIVE_CLASS_INDEX] where
            # LIVE_CLASS_INDEX = 1 (Real Face). Do NOT use 1 - (p[print] + p[replay]);
            # that formula assumed the old (wrong) class ordering.
            live_score = float(probs[LIVE_CLASS_INDEX])
            if not (0.0 <= live_score <= 1.0):
                return LivenessResult(
                    passed=False,
                    provider=self.name,
                    score=None,
                    reason="invalid live probability",
                    model_version=self.model_version,
                )

            passed = live_score >= settings.FACE_LIVENESS_THRESHOLD
            return LivenessResult(
                passed=passed,
                provider=self.name,
                score=live_score,
                reason=None if passed else "below_threshold",
                latency_ms=latency_ms,
                model_version=self.model_version,
            )

        except Exception as exc:  # noqa: BLE001 - fail closed on any inference error
            return LivenessResult(
                passed=False,
                provider=self.name,
                score=None,
                reason="inference_error",
                model_version=self.model_version,
            )

    def is_available(self) -> bool:
        return self._session is not None