from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictError, NotFoundError, ValidationAppError
from app.core.types import utcnow
from app.models.event import Event
from app.models.user import User
from app.repositories import events as events_repo
from app.schemas.event import EventCreate, EventUpdate
from app.services import audit_service


def get_or_404(db: Session, event_id: uuid.UUID) -> Event:
    event = events_repo.get(db, event_id)
    if event is None:
        raise NotFoundError("Event not found", error_code="EVENT_NOT_FOUND")
    return event


def create_event(
    db: Session,
    *,
    data: EventCreate,
    actor: User,
    ip_address: str | None,
) -> Event:
    if data.ends_at <= data.starts_at:
        raise ValidationAppError(
            "ends_at must be after starts_at", error_code="ENDS_AT_MUST_BE_AFTER_STARTS_AT"
        )
    event = Event(
        name=data.name.strip(),
        description=data.description,
        location=data.location,
        starts_at=data.starts_at,
        ends_at=data.ends_at,
        is_active=True,
        created_by=actor.id,
    )
    events_repo.add(db, event)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError("Event could not be created", error_code="EVENT_CONFLICT") from exc
    audit_service.record(
        db,
        action="EVENT_CREATED",
        entity_type="event",
        entity_id=event.id,
        description=f'Event "{event.name}" created',
        user_id=actor.id,
        ip_address=ip_address,
    )
    db.commit()
    db.refresh(event)
    return event


def list_events(
    db: Session,
    *,
    page: int,
    page_size: int,
    search: str | None,
    is_active: bool | None,
    include_deleted: bool,
    date_from: date | None,
    date_to: date | None,
):
    return events_repo.list_events(
        db,
        page=page,
        page_size=page_size,
        search=search,
        is_active=is_active,
        include_deleted=include_deleted,
        date_from=date_from,
        date_to=date_to,
    )


def update_event(
    db: Session,
    *,
    event_id: uuid.UUID,
    data: EventUpdate,
    actor: User,
    ip_address: str | None,
) -> Event:
    event = get_or_404(db, event_id)
    changes = data.model_dump(exclude_unset=True)
    starts_at = changes.get("starts_at", event.starts_at)
    ends_at = changes.get("ends_at", event.ends_at)
    if ends_at <= starts_at:
        raise ValidationAppError(
            "ends_at must be after starts_at", error_code="ENDS_AT_MUST_BE_AFTER_STARTS_AT"
        )
    for field, value in changes.items():
        setattr(event, field, value)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError("Event update violates a constraint", error_code="EVENT_CONFLICT") from exc
    audit_service.record(
        db,
        action="EVENT_UPDATED",
        entity_type="event",
        entity_id=event.id,
        description=f"Fields updated: {', '.join(sorted(changes.keys()))}",
        user_id=actor.id,
        ip_address=ip_address,
    )
    db.refresh(event)
    return event


def soft_delete_event(
    db: Session,
    *,
    event_id: uuid.UUID,
    actor: User,
    ip_address: str | None,
) -> Event:
    event = get_or_404(db, event_id)
    if event.deleted_at is not None:
        raise ConflictError("Event is already deleted", error_code="EVENT_ALREADY_DELETED")
    now = utcnow()
    event.deleted_at = now
    event.is_active = False
    event.updated_at = now
    audit_service.record(
        db,
        action="EVENT_DELETED",
        entity_type="event",
        entity_id=event.id,
        description=f'Event "{event.name}" soft-deleted; attendance history is retained',
        user_id=actor.id,
        ip_address=ip_address,
    )
    db.commit()
    db.refresh(event)
    return event


def get_available_events(db: Session, day: date) -> list[Event]:
    return events_repo.get_available_today(db, day)
