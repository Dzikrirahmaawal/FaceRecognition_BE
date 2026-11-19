"""Disabled liveness provider.

Used when FACE_LIVENESS_PROVIDER=disabled. Anti-spoofing protection is
intentionally turned off. This provider never returns a successful
anti-spoofing result; it always reports passed=False with reason
"liveness_disabled" so that enrollment, identification, and verification
operations fail safely when liveness is required but not configured.
"""

from __future__ import annotations

import numpy as np

from app.face_engine.interfaces import LivenessProvider, LivenessResult


class DisabledLivenessProvider(LivenessProvider):
    name = "disabled"

    def verify_liveness(
        self, image: np.ndarray, face_bbox: tuple[int, int, int, int] | None = None
    ) -> LivenessResult:  # noqa: ARG002 - intentionally unused
        return LivenessResult(
            passed=False,
            provider=self.name,
            score=None,
            reason="liveness_disabled",
        )

    def is_available(self) -> bool:
        return True