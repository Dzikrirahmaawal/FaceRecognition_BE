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
        finally:
            for p in reversed(patches):
                p.stop()

        all_log_text = " ".join(r.getMessage() for r in caplog.records)
        extra_text = " ".join(
            str(r.extra_data) for r in caplog.records if getattr(r, "extra_data", None)
        )
        combined = all_log_text + " " + extra_text
        assert "embedding" not in combined.lower().replace("embedding_data", "").replace("embedding_version", "")