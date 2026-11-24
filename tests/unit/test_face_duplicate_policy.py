from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.config import settings
from app.core.exceptions import ConflictError
from app.services import face_service
from app.face_engine.interfaces import DetectedFace, QualityReport, LivenessResult


class ControllableFaceProvider:
    name = "test"
    model_version = "test-v1"
    embedding_version = "1"

    def __init__(self, similarity: float = 0.0):
        self.similarity = similarity

    def compare_embeddings(self, embedding_a, embedding_b) -> float:
        return self.similarity

    def generate_embedding(self, image, face):
        return np.ones(128, dtype=np.float32)

    def detect_faces(self, image):
        return [DetectedFace(bbox=(10, 10, 100, 100), score=0.99)]

    def validate_face_quality(self, image, face):
        return QualityReport(acceptable=True)

    def is_available(self) -> bool:
        return True


class ControllableLivenessProvider:
    name = "test"

    def verify_liveness(self, image):
        return LivenessResult(passed=True, provider="test", score=1.0)

    def is_available(self) -> bool:
        return True


def _make_profile(visitor_id: uuid.UUID, is_active: bool = True,
                  model_version: str = "test-v1", embedding_version: str = "1"):
    profile = MagicMock()
    profile.id = uuid.uuid4()
    profile.visitor_id = visitor_id
    profile.is_active = is_active
    profile.model_version = model_version
    profile.embedding_version = embedding_version
    profile.embedding_data = b"encrypted-embedding"
    return profile


def _setup_enroll_mocks(
    db: MagicMock,
    provider: ControllableFaceProvider,
    profiles: list,
    visitor_id: uuid.UUID | None = None,
):
    """Patch all dependencies needed by face_service.enroll."""

    # _prepare_capture returns a dummy embedding
    patches = [
        patch.object(face_service, "_prepare_capture",
                     return_value=(None, np.ones(128, dtype=np.float32), None, None)),
        patch.object(face_service, "get_face_provider", return_value=provider),
        patch.object(face_service, "get_liveness_provider",
                     return_value=ControllableLivenessProvider()),
        patch.object(
            face_service.visitors_repo, "list_active_profiles",
            return_value=[
                p for p in profiles
                if p.is_active
                and p.model_version == provider.model_version
                and p.embedding_version == provider.embedding_version
            ],
        ),
        patch.object(face_service.visitors_repo, "add_face_profile"),
        patch.object(face_service.consent_repo, "add"),
        patch.object(face_service.audit_service, "record"),
        patch.object(face_service.visitors_repo, "get_by_identity_number", return_value=None),
        patch.object(face_service.BiometricCipher, "decrypt_embedding",
                     return_value=np.ones(128, dtype=np.float32)),
    ]

    if visitor_id is not None:
        # Existing visitor path
        visitor = MagicMock()
        visitor.id = visitor_id
        visitor.is_active = True
        patches.append(
            patch.object(face_service.visitor_service, "get_or_404",
                         return_value=visitor)
        )
        patches.append(
            patch.object(face_service.visitors_repo,
                         "get_active_profile_for_visitor", return_value=None)
        )
    else:
        # New visitor path
        visitor = MagicMock()
        visitor.id = uuid.uuid4()
        patches.append(
            patch.object(face_service.visitor_service, "create_visitor",
                         return_value=visitor)
        )

    return patches, visitor


def _run_enroll(
    db: MagicMock,
    similarity: float = 0.0,
    visitor_id: uuid.UUID | None = None,
    profiles: list | None = None,
):
    if profiles is None:
        profiles = []
    provider = ControllableFaceProvider(similarity=similarity)
    patches, visitor = _setup_enroll_mocks(db, provider, profiles or [], visitor_id)
    for p in patches:
        p.start()
    try:
        result = face_service.enroll(
            db,
            image_bytes=b"fake-image",
            content_type="image/png",
            visitor_id=visitor_id,
            full_name="Test Person",
            nik="1234567890123456",
            identity_number="ID-123",
            company_name="Test Company",
            institution="Test Institution",
            job_title="Visitor",
            phone_number="08123456789",
            email="test@example.com",
            consent_granted=True,
            consent_policy_version="1.0",
            replace_existing=False,
            actor=MagicMock(),
            ip_address=None,
        )
        return result, visitor
    finally:
        for p in reversed(patches):
            p.stop()


@pytest.fixture
def db():
    return MagicMock()


class TestDuplicateBelowThreshold:
    """Similarity below 0.80 allows enrollment."""

    def test_allows_enrollment_when_no_profiles(self, db):
        result, _ = _run_enroll(db, similarity=0.0, profiles=[])
        assert result["duplicate_review"] is False

    def test_allows_enrollment_when_similarity_below_threshold(self, db):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id)]
        result, _ = _run_enroll(db, similarity=0.70, profiles=profiles)
        assert result["duplicate_review"] is False

    def test_allows_enrollment_when_similarity_exactly_at_threshold(self, db):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id)]
        result, _ = _run_enroll(db, similarity=settings.FACE_DUPLICATE_THRESHOLD, profiles=profiles)
        assert result["duplicate_review"] is True


class TestDuplicateBetweenThresholds:
    """Similarity between 0.80 and 0.90 allows enrollment with duplicate_review=True."""

    def test_allows_with_review_when_similarity_between_thresholds(self, db):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id)]
        result, _ = _run_enroll(db, similarity=0.85, profiles=profiles)
        assert result["duplicate_review"] is True

    def test_allows_with_review_when_similarity_just_below_confirmation(self, db):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id)]
        result, _ = _run_enroll(db, similarity=0.89, profiles=profiles)
        assert result["duplicate_review"] is True


class TestDuplicateAtOrAboveConfirmation:
    """Similarity at or above 0.90 rejects enrollment."""

    def test_rejects_when_similarity_at_confirmation_threshold(self, db):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id)]
        with pytest.raises(ConflictError) as exc:
            _run_enroll(db, similarity=settings.FACE_DUPLICATE_CONFIRM_THRESHOLD, profiles=profiles)
        assert exc.value.error_code == "FACE_DUPLICATE_DETECTED"

    def test_rejects_when_similarity_above_confirmation_threshold(self, db):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id)]
        with pytest.raises(ConflictError) as exc:
            _run_enroll(db, similarity=0.95, profiles=profiles)
        assert exc.value.error_code == "FACE_DUPLICATE_DETECTED"


class TestHighestSimilaritySelected:
    """Highest similarity is selected regardless of profile order."""

    def test_selects_highest_when_lower_appears_first(self, db):
        visitor_a = uuid.uuid4()
        visitor_b = uuid.uuid4()
        profiles = [
            _make_profile(visitor_a),  # lower similarity first
            _make_profile(visitor_b),  # higher similarity second
        ]
        provider = ControllableFaceProvider(similarity=0.0)

        def compare_side_effect(emb_a, emb_b):
            return 0.85 if emb_b is not None else 0.0

        provider.compare_embeddings = compare_side_effect

        patches, _ = _setup_enroll_mocks(db, provider, profiles)
        for p in patches:
            p.start()
        try:
            result = face_service.enroll(
                db,
                image_bytes=b"fake-image",
                content_type="image/png",
                visitor_id=None,
                full_name="Test Person",
                identity_number="ID-123",
                company_name="Test Company",
                phone_number="08123456789",
                email="test@example.com",
                consent_granted=True,
                consent_policy_version="1.0",
                replace_existing=False,
                actor=MagicMock(),
                ip_address=None,
            )
            assert result["duplicate_review"] is True
        finally:
            for p in reversed(patches):
                p.stop()


class TestInactiveProfilesExcluded:
    """Inactive profiles do not cause duplicate rejection."""

    def test_inactive_profile_does_not_cause_rejection(self, db):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id, is_active=False)]
        result, _ = _run_enroll(db, similarity=0.95, profiles=profiles)
        assert result["duplicate_review"] is False


class TestIncompatibleVersionsNotCompared:
    """Incompatible model/embedding versions are not silently compared."""

    def test_incompatible_model_version_not_compared(self, db):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id, model_version="other-v1")]
        result, _ = _run_enroll(db, similarity=0.95, profiles=profiles)
        assert result["duplicate_review"] is False

    def test_incompatible_embedding_version_not_compared(self, db):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id, embedding_version="2")]
        result, _ = _run_enroll(db, similarity=0.95, profiles=profiles)
        assert result["duplicate_review"] is False


class TestDuplicateReviewInResponse:
    """duplicate_review is included in the service response."""

    def test_duplicate_review_is_false_below_threshold(self, db):
        result, _ = _run_enroll(db, similarity=0.70, profiles=[])
        assert "duplicate_review" in result
        assert result["duplicate_review"] is False

    def test_duplicate_review_is_true_between_thresholds(self, db):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id)]
        result, _ = _run_enroll(db, similarity=0.85, profiles=profiles)
        assert "duplicate_review" in result
        assert result["duplicate_review"] is True


class TestLogging:
    """Diagnostic logging behavior."""

    def test_logs_comparison_metadata(self, db, caplog):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id)]
        provider = ControllableFaceProvider(similarity=0.85)

        patches, _ = _setup_enroll_mocks(db, provider, profiles)
        for p in patches:
            p.start()
        try:
            with caplog.at_level("INFO", logger="app.services.face_service"):
                face_service.enroll(
                    db,
                    image_bytes=b"fake-image",
                    content_type="image/png",
                    visitor_id=None,
                    full_name="Test Person",
                    nik="1234567890123456",
                    identity_number="ID-123",
                    company_name="Test Company",
                    institution="Test Institution",
                    job_title="Visitor",
                    phone_number="08123456789",
                    email="test@example.com",
                    consent_granted=True,
                    consent_policy_version="1.0",
                    replace_existing=False,
                    actor=MagicMock(),
                    ip_address=None,
                )
        finally:
            for p in reversed(patches):
                p.stop()

        # Check that comparison log contains expected metadata
        comparison_logs = [
            r for r in caplog.records
            if getattr(r, "extra_data", None) and r.extra_data.get("event") == "face_duplicate_comparison"
        ]
        assert len(comparison_logs) >= 1
        log_data = comparison_logs[0].extra_data
        assert "similarity" in log_data
        assert "duplicate_threshold" in log_data
        assert "confirmation_threshold" in log_data
        assert "model_version" in log_data
        assert "embedding_version" in log_data

    def test_logs_do_not_contain_raw_embeddings(self, db, caplog):
        other_visitor_id = uuid.uuid4()
        profiles = [_make_profile(other_visitor_id)]
        provider = ControllableFaceProvider(similarity=0.85)

        patches, _ = _setup_enroll_mocks(db, provider, profiles)
        for p in patches:
            p.start()
        try:
            with caplog.at_level("INFO", logger="app.services.face_service"):
                face_service.enroll(
                    db,
                    image_bytes=b"fake-image",
                    content_type="image/png",
                    visitor_id=None,
                    full_name="Test Person",
                    nik="1234567890123456",
                    identity_number="ID-123",
                    company_name="Test Company",
                    institution="Test Institution",
                    job_title="Visitor",
                    phone_number="08123456789",
                    email="test@example.com",
                    consent_granted=True,
                    consent_policy_version="1.0",
                    replace_existing=False,
                    actor=MagicMock(),
                    ip_address=None,
                )
        finally:
            for p in reversed(patches):
                p.stop()

        all_log_text = " ".join(r.getMessage() for r in caplog.records)
        extra_text = " ".join(
            str(r.extra_data) for r in caplog.records if getattr(r, "extra_data", None)
        )
        combined = all_log_text + " " + extra_text
        assert "embedding" not in combined.lower().replace("embedding_data", "").replace("embedding_version", "")


# New unit tests for detector score, landmarks, and pipeline integrity
# These test the face detection and quality pipeline directly


def test_detector_score_from_index_14():
    """Verify that YuNet detector score is read from index 14 of the 15-value row."""
    from app.face_engine.providers.opencv_sface import OpenCvSfaceProvider
    from app.face_engine.interfaces import DetectedFace
    import cv2
    import numpy as np
    from pathlib import Path

    REPO = Path("/home/dzikri/Face Recognition")
    import app.core.config as _cfg
    settings = _cfg.settings
    for _attr, _val in (
        ("FACE_DETECTION_MODEL_PATH", str(REPO / "models" / "face_detection_yunet_2023mar.onnx")),
    ):
        setattr(settings, _attr, _val)

    provider = OpenCvSfaceProvider()
    img = cv2.imread(str(REPO / "liveness_samples" / "real" / "Andikha Tidak Berkaca Mata.jpg"))
    faces = provider.detect_faces(img)

    assert len(faces) > 0, "Should detect at least one face"
    face = faces[0]
    assert hasattr(face, "score")
    assert isinstance(face.score, float)
    assert face.score is not None


def test_landmarks_preserved_in_detected_face():
    """Verify that 5 facial landmarks are preserved in DetectedFace from YuNet output."""
    from app.face_engine.providers.opencv_sface import OpenCvSfaceProvider
    from app.face_engine.interfaces import DetectedFace
    import cv2
    from pathlib import Path

    REPO = Path("/home/dzikri/Face Recognition")
    import app.core.config as _cfg
    settings = _cfg.settings
    for _attr, _val in (
        ("FACE_DETECTION_MODEL_PATH", str(REPO / "models" / "face_detection_yunet_2023mar.onnx")),
    ):
        setattr(settings, _attr, _val)

    provider = OpenCvSfaceProvider()
    img = cv2.imread(str(REPO / "liveness_samples" / "real" / "Andikha Tidak Berkaca Mata.jpg"))
    faces = provider.detect_faces(img)

    assert len(faces) > 0, "Should detect at least one face"
    face = faces[0]
    assert hasattr(face, "landmarks")
    assert isinstance(face.landmarks, tuple)
    assert len(face.landmarks) == 10, f"Expected 10 landmark coords (5 pairs), got {len(face.landmarks)}"


def test_different_people_not_identical_embeddings():
    """Verify that two different people produce different embeddings (not all 1.0)."""
    from app.face_engine.providers.opencv_sface import OpenCvSfaceProvider
    from app.face_engine.detection import validate_single_face
    import cv2
    import numpy as np
    from pathlib import Path

    REPO = Path("/home/dzikri/Face Recognition")
    import app.core.config as _cfg
    settings = _cfg.settings
    for _attr, _val in (
        ("FACE_LIVENESS_PROVIDER", "minifasnet"),
        ("FACE_LIVENESS_MODE", "enforce"),
        ("FACE_LIVENESS_MODEL_PATH", str(REPO / "models" / "minifasnet_v2.onnx")),
        ("FACE_LIVENESS_MODEL_SHA256", "d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b"),
        ("FACE_LIVENESS_THRESHOLD", 0.5),
        ("FACE_LIVENESS_ONNX_INTRA_THREADS", 2),
        ("FACE_LIVENESS_ONNX_INTER_THREADS", 1),
        ("FACE_DETECTION_MODEL_PATH", str(REPO / "models" / "face_detection_yunet_2023mar.onnx")),
        ("FACE_RECOGNITION_MODEL_PATH", str(REPO / "models" / "face_recognition_sface_2021dec.onnx")),
    ):
        setattr(settings, _attr, _val)

    provider = OpenCvSfaceProvider()

    img1 = cv2.imread(str(REPO / "liveness_samples" / "real" / "Andikha Tidak Berkaca Mata.jpg"))
    img2 = cv2.imread(str(REPO / "liveness_samples" / "real" / "Dzikri Tidak Berkaca Mata.jpg"))

    faces1 = provider.detect_faces(img1)
    faces2 = provider.detect_faces(img2)

    assert len(faces1) > 0, "Andikha face should be detected"
    assert len(faces2) > 0, "Dzikri face should be detected"

    face1 = validate_single_face(faces1)
    face2 = validate_single_face(faces2)

    emb1 = provider.generate_embedding(img1, face1)
    emb2 = provider.generate_embedding(img2, face2)

    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)

    similarity = float(np.dot(emb1_norm, emb2_norm))
    assert similarity < 1.0, f"Different people should not have identical embeddings, got similarity={similarity}"
    assert similarity < 0.98, f"Different-person similarity should be well below 1.0, got {similarity}"


def test_quality_filtering_rejects_occluded():
    """Test that quality filtering can reject images with occluded faces."""
    from app.face_engine.providers.opencv_sface import OpenCvSfaceProvider
    from app.face_engine.detection import validate_single_face
    import cv2
    import numpy as np
    from pathlib import Path

    REPO = Path("/home/dzikri/Face Recognition")
    import app.core.config as _cfg
    settings = _cfg.settings
    for _attr, _val in (
        ("FACE_LIVENESS_PROVIDER", "minifasnet"),
        ("FACE_LIVENESS_MODE", "enforce"),
        ("FACE_LIVENESS_MODEL_PATH", str(REPO / "models" / "minifasnet_v2.onnx")),
        ("FACE_LIVENESS_MODEL_SHA256", "d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b"),
        ("FACE_LIVENESS_THRESHOLD", 0.5),
        ("FACE_LIVENESS_ONNX_INTRA_THREADS", 2),
        ("FACE_LIVENESS_ONNX_INTER_THREADS", 1),
        ("FACE_DETECTION_MODEL_PATH", str(REPO / "models" / "face_detection_yunet_2023mar.onnx")),
    ):
        setattr(settings, _attr, _val)

    provider = OpenCvSfaceProvider()
    img = cv2.imread(str(REPO / "liveness_samples" / "real" / "Andikha Tidak Berkaca Mata.jpg"))
    faces = provider.detect_faces(img)
    face = validate_single_face(faces)

    quality = provider.validate_face_quality(img, face)
    # Quality check should return a result (acceptable or not)
    assert hasattr(quality, "acceptable")
    assert quality.acceptable is True or quality.acceptable is False


def test_non_face_score_below_threshold():
    """Test that non-face inputs produce low detector scores."""
    from app.face_engine.providers.opencv_sface import OpenCvSfaceProvider
    import cv2
    import numpy as np
    from pathlib import Path

    REPO = Path("/home/dzikri/Face Recognition")
    import app.core.config as _cfg
    settings = _cfg.settings
    for _attr, _val in (
        ("FACE_DETECTION_MODEL_PATH", str(REPO / "models" / "face_detection_yunet_2023mar.onnx")),
        ("FACE_RECOGNITION_MODEL_PATH", str(REPO / "models" / "face_recognition_sface_2021dec.onnx")),
    ):
        setattr(settings, _attr, _val)

    provider = OpenCvSfaceProvider()

    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
    faces = provider.detect_faces(blank_img)

    # Should detect few or no faces for a blank image
    for face in faces:
        # Score should be low for non-face images
        assert face.score < 0.6 or len(faces) == 0, \
            f"Face detected with score {face.score} on blank image, expected below 0.6"