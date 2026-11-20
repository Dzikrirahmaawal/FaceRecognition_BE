"""OpenCV YuNet + SFace face recognition provider.

Uses OpenCV DNN with official OpenCV Zoo ONNX models:
- YuNet face detection (MIT License)
- SFace face recognition (Apache License 2.0)

Model files are not bundled in this repository. They must be present at the
paths configured by FACE_DETECTION_MODEL_PATH and FACE_RECOGNITION_MODEL_PATH
before the provider can be used.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from app.core.config import settings
from app.core.exceptions import ProviderUnavailableError
from app.face_engine.detection import assess_quality
from app.face_engine.interfaces import (
    DetectedFace,
    FaceRecognitionProvider,
    QualityReport,
)

_DETECTOR_OUTPUT_DIM = 15
_LANDMARK_COUNT = 5
_LANDMARK_VALUES = _DETECTOR_OUTPUT_DIM - 1  # 14: 4 bbox + 1 score + 9 remaining (5 landmarks have 10 coords, we store 9 nearest)


class OpenCvSfaceProvider(FaceRecognitionProvider):
    name = "opencv_sface"
    model_version = settings.FACE_MODEL_VERSION
    embedding_version = settings.FACE_EMBEDDING_VERSION

    def __init__(self) -> None:
        self._lock = threading.Lock()
        detection_path = settings.FACE_DETECTION_MODEL_PATH
        recognition_path = settings.FACE_RECOGNITION_MODEL_PATH
        if not detection_path:
            raise ProviderUnavailableError("FACE_DETECTION_MODEL_PATH is not configured")
        if not recognition_path:
            raise ProviderUnavailableError("FACE_RECOGNITION_MODEL_PATH is not configured")
        try:
            self._detector = cv2.FaceDetectorYN.create(
                detection_path,
                "",
                (320, 320),
                settings.FACE_DETECTION_SCORE_THRESHOLD,
                settings.FACE_DETECTION_NMS_THRESHOLD,
                settings.FACE_DETECTION_TOP_K,
                cv2.dnn.DNN_BACKEND_DEFAULT,
                cv2.dnn.DNN_TARGET_CPU,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(
                recognition_path,
                "",
                cv2.dnn.DNN_BACKEND_DEFAULT,
                cv2.dnn.DNN_TARGET_CPU,
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Failed to load OpenCV YuNet/SFace models: {exc}"
            ) from exc

    def _face_box_row(self, face: DetectedFace) -> np.ndarray:
        row = np.zeros(_DETECTOR_OUTPUT_DIM, dtype=np.float32)
        x, y, w, h = face.bbox
        row[0] = float(x)
        row[1] = float(y)
        row[2] = float(w)
        row[3] = float(h)
        # Landmarks from DetectedFace (indices 5-13 in the original detector row);
        # if fewer than 5 landmarks are available, remaining slots stay zero.
        lm = face.landmarks
        for i in range(min(_LANDMARK_COUNT, len(lm) // 2)):
            row[1 + i] = float(lm[2 * i])     # landmark x
            row[1 + _LANDMARK_COUNT + i] = float(lm[2 * i + 1])  # landmark y
        # Note: the original code set row[14] = score; we keep that convention
        # but the actual score normalisation happens in generate_embedding.
        row[14] = 1.0  # placeholder; actual score used differently
        return row

    def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        with self._lock:
            self._detector.setInputSize((width, height))
            _, faces = self._detector.detect(image)
        if faces is None or len(faces) == 0:
            return []
        results: list[DetectedFace] = []
        for row in faces:
            # The YuNet detector outputs 15 values per face:
            # indices 0-3: bounding box (x, y, w, h) in 320x320 normalized coords
            # index 4: often unused/conflict depending on model version
            # indices 5-13: 5 facial landmarks (x,y pairs would be 10 values, stored in 9 slots here)
            # index 14: confidence score (raw, not necessarily 0-1)
            # We extract:
            #   bbox from indices 0-3
            #   score normalized from index 14
            #   landmarks from indices 5-13 (9 values = up to 4.5 landmark pairs; we take min(5, available))
            if len(row) >= _DETECTOR_OUTPUT_DIM:
                # bbox: indices 0-3
                x_raw = float(row[0])
                y_raw = float(row[1])
                w_raw = float(row[2])
                h_raw = float(row[3])
                # scale bbox to original image size
                x = max(0, min(x_raw * width, width - 1))
                y = max(0, min(y_raw * height, height - 1))
                w = max(1, min(w_raw * width, width - x))
                h = max(1, min(h_raw * height, height - y))

                # score from index 14, normalized to 0-1 range (raw scores can be >1)
                raw_score = float(row[14])
                score = min(raw_score / 1000.0, 1.0)  # rough normalization; adjust if needed

                # landmarks from indices 5-13 (9 values = up to 4.5 pairs; we take up to 5)
                lm: list[float] = []
                for i in range(5):
                    idx_x = 1 + i  # first approx: indices 1-5 may contain landmarks
                    idx_y = 1 + _LANDMARK_COUNT + i  # second approx: indices 7-11
                    # Be safe: only add if within bounds
                    if idx_x < _DETECTOR_OUTPUT_DIM - 1:
                        lm.append(float(row[idx_x]))
                    if idx_y < _DETECTOR_OUTPUT_DIM - 1 and len(lm) < _LANDMARK_COUNT * 2:
                        lm.append(float(row[idx_y]))
                # Pad to even number and _LANDMARK_COUNT pairs
                while len(lm) < _LANDMARK_COUNT * 2:
                    lm.append(1.0)
                lm = tuple(lm[: _LANDMARK_COUNT * 2])

                results.append(
                    DetectedFace(
                        bbox=(int(x), int(y), int(w), int(h)),
                        score=score,
                        landmarks=lm,
                    )
                )
            else:
                # Fallback: old behavior with just bbox and score
                x, y, w, h, score = [float(value) for value in row[0:5]]
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))
                results.append(
                    DetectedFace(bbox=(int(x), int(y), int(w), int(h)), score=score)
                )
        return results

    def validate_face_quality(self, image: np.ndarray, face: DetectedFace) -> QualityReport:
        return assess_quality(image, face)

    def generate_embedding(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        with self._lock:
            aligned = self._recognizer.alignCrop(image, self._face_box_row(face))
            feature = self._recognizer.feature(aligned)
        embedding = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        return embedding / norm if norm else embedding

    def compare_embeddings(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        if embedding_a.shape != embedding_b.shape:
            return 0.0
        a = embedding_a.astype(np.float64).reshape(-1)
        b = embedding_b.astype(np.float64).reshape(-1)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.clip(float(np.dot(a, b) / denom), -1.0, 1.0))

    def is_available(self) -> bool:
        return self._detector is not None and self._recognizer is not None