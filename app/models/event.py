from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, Unicode
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.types import GUID, utcnow
from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.attendance import AttendanceLog
    from app.models.user import User


class Event(TimestampMixin, Base):
    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint("ends_at > starts_at", name="ck_events_ends_after_starts"),
        Index("ix_events_starts_at_is_active", "starts_at", "is_active"),
        Index("ix_events_is_active", "is_active"),
        Index("ix_events_name", "name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Unicode(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Unicode(1000), nullable=True)
    location: Mapped[str | None] = mapped_column(Unicode(200), nullable=True)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    creator: Mapped["User | None"] = relationship(
        foreign_keys=[created_by], lazy="noload"
    )
    attendance_logs: Mapped[list["AttendanceLog"]] = relationship(
        back_populates="event", lazy="noload"
    )
