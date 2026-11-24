from __future__ import annotations

import uuid
from datetime import date, datetime

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.api.dependencies import Actor, require_actor, require_permissions
from app.core.database import get_db
from app.core.permissions import Permission
from app.models.user import User
from app.repositories.base import paginate
from app.schemas.common import pagination_meta, success_payload
from app.schemas.event import EventAvailableOut, EventCreate, EventOut, EventUpdate
from app.services import event_service, report_service

router = APIRouter(prefix="/events", tags=["Events"])


def _ip(request: Request) -> str | None:
    return request.client.host if request.client else None


@router.post("", status_code=201)
def create_event(
    payload: EventCreate,
    request: Request,
    user: User = Depends(require_permissions(Permission.EVENT_MANAGE)),
    db: Session = Depends(get_db),
):
    event = event_service.create_event(
        db, data=payload, actor=user, ip_address=_ip(request)
    )
    return success_payload(
        data=EventOut.model_validate(event).model_dump(mode="json"),
        message="Event created",
    )


@router.get("")
def list_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    search: str | None = Query(None, max_length=200),
    is_active: bool | None = Query(None),
    include_deleted: bool = Query(False),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    _: User = Depends(require_permissions(Permission.EVENT_READ)),
    db: Session = Depends(get_db),
):
    items, total = event_service.list_events(
        db,
        page=page,
        page_size=page_size,
        search=search,
        is_active=is_active,
        include_deleted=include_deleted,
        date_from=date_from,
        date_to=date_to,
    )
    return success_payload(
        data=[EventOut.model_validate(item).model_dump(mode="json") for item in items],
        message="Events retrieved",
        meta=pagination_meta(page, page_size, total),
    )


@router.get("/{event_id}")
def get_event(
    event_id: uuid.UUID,
    _: User = Depends(require_permissions(Permission.EVENT_READ)),
    db: Session = Depends(get_db),
):
    event = event_service.get_or_404(db, event_id)
    return success_payload(
        data=EventOut.model_validate(event).model_dump(mode="json"),
        message="Event retrieved",
    )


@router.patch("/{event_id}")
def update_event(
    event_id: uuid.UUID,
    payload: EventUpdate,
    request: Request,
    user: User = Depends(require_permissions(Permission.EVENT_MANAGE)),
    db: Session = Depends(get_db),
):
    event = event_service.update_event(
        db, event_id=event_id, data=payload, actor=user, ip_address=_ip(request)
    )
    return success_payload(
        data=EventOut.model_validate(event).model_dump(mode="json"),
        message="Event updated",
    )


@router.put("/{event_id}")
def replace_event(
    event_id: uuid.UUID,
    payload: EventUpdate,
    request: Request,
    user: User = Depends(require_permissions(Permission.EVENT_MANAGE)),
    db: Session = Depends(get_db),
):
    if payload.name is None:
        from app.core.exceptions import ValidationAppError

        raise ValidationAppError(
            "name is required for full update", error_code="NAME_REQUIRED"
        )
    event = event_service.update_event(
        db, event_id=event_id, data=payload, actor=user, ip_address=_ip(request)
    )
    return success_payload(
        data=EventOut.model_validate(event).model_dump(mode="json"),
        message="Event updated",
    )


@router.delete("/{event_id}")
def delete_event(
    event_id: uuid.UUID,
    request: Request,
    user: User = Depends(require_permissions(Permission.EVENT_MANAGE)),
    db: Session = Depends(get_db),
):
    event = event_service.soft_delete_event(
        db, event_id=event_id, actor=user, ip_address=_ip(request)
    )
    return success_payload(
        data=EventOut.model_validate(event).model_dump(mode="json"),
        message="Event soft-deleted. Attendance history linked to the event is retained.",
    )


@router.get("/available")
def available_events(
    day: date | None = Query(None, alias="date"),
    actor: Actor = Depends(require_actor(Permission.EVENT_READ, allow_kiosk=True)),
    db: Session = Depends(get_db),
):
    target_day = day or report_service.local_today()
    events = event_service.get_available_events(db, target_day)
    return success_payload(
        data=[EventAvailableOut.model_validate(item).model_dump(mode="json") for item in events],
        message="Available events retrieved",
    )
