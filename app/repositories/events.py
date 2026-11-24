from __future__ import annotations

import uuid
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.types import ensure_utc
from app.models.event import Event
from app.repositories.base import paginate


def _tz() -> ZoneInfo:
    return ZoneInfo(settings.TIMEZONE)


def day_bounds(day: date) -> tuple[datetime, datetime]:
    tz = _tz()
    start = datetime.combine(day, time.min, tzinfo=tz).astimezone(timezone.utc)
    end = (
        datetime.combine(day, time.min, tzinfo=tz) + timedelta(days=1)
    ).astimezone(timezone.utc)
    return start, end


def get(db: Session, event_id: uuid.UUID) -> Event | None:
    return db.get(Event, event_id)


def list_events(
    db: Session,
    page: int,
    page_size: int,
    search: str | None = None,
    is_active: bool | None = None,
    include_deleted: bool = False,
    date_from: date | None = None,
    date_to: date | None = None,
):
    stmt = select(Event).order_by(Event.starts_at.asc(), Event.name.asc())
    if not include_deleted:
        stmt = stmt.where(Event.deleted_at.is_(None))
    if is_active is not None:
        stmt = stmt.where(Event.is_active == is_active)
    if search:
        like = f"%{search}%"
        stmt = stmt.where(
            Event.name.ilike(like)
            | Event.location.ilike(like)
            | Event.description.ilike(like)
        )
    if date_from is not None:
        stmt = stmt.where(Event.ends_at >= ensure_utc(datetime.combine(date_from, time.min)))
    if date_to is not None:
        stmt = stmt.where(Event.starts_at <= ensure_utc(datetime.combine(date_to, time.max)))
    return paginate(db, stmt, page, page_size)


def get_available_today(db: Session, day: date) -> list[Event]:
    start, end = day_bounds(day)
    stmt = (
        select(Event)
        .where(
            Event.deleted_at.is_(None),
            Event.is_active == True,  # noqa: E712
            Event.starts_at < end,
            Event.ends_at > start,
        )
        .order_by(Event.starts_at.asc(), Event.name.asc())
    )
    return list(db.scalars(stmt).all())


def add(db: Session, event: Event) -> Event:
    db.add(event)
    return event
