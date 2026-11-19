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
        row[14] = float(face.score)
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