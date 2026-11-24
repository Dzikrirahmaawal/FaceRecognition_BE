from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.dependencies import require_permissions
from app.core.database import get_db
from app.core.permissions import Permission
from app.models.user import User
from app.schemas.common import success_payload
from app.schemas.report import (
    DailyReportOut,
    DashboardOut,
    EventGuestsOut,
    EventsSummaryOut,
    MonthlyReportOut,
    VisitorReportOut,
)
from app.services import report_service

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/dashboard")
def dashboard(
    _: User = Depends(require_permissions(Permission.REPORT_READ)),
    db: Session = Depends(get_db),
):
    data = report_service.dashboard(db)
    return success_payload(
        data=DashboardOut.model_validate(data).model_dump(mode="json"),
        message="Dashboard metrics retrieved",
    )


@router.get("/attendance/daily")
def daily(
    day: date | None = Query(None, alias="date"),
    _: User = Depends(require_permissions(Permission.REPORT_READ)),
    db: Session = Depends(get_db),
):
    report = report_service.daily_report(db, day or report_service.local_today())
    return success_payload(
        data=DailyReportOut.model_validate(report).model_dump(mode="json"),
        message="Daily attendance report retrieved",
    )


@router.get("/attendance/monthly")
def monthly(
    year: int | None = Query(None, ge=2000, le=2100),
    month: int | None = Query(None, ge=1, le=12),
    _: User = Depends(require_permissions(Permission.REPORT_READ)),
    db: Session = Depends(get_db),
):
    today = report_service.local_today()
    report = report_service.monthly_report(db, year or today.year, month or today.month)
    return success_payload(
        data=MonthlyReportOut.model_validate(report).model_dump(mode="json"),
        message="Monthly attendance report retrieved",
    )


@router.get("/visitors")
def visitors(
    search: str | None = Query(None, max_length=150),
    limit: int = Query(500, ge=1, le=2000),
    _: User = Depends(require_permissions(Permission.REPORT_READ)),
    db: Session = Depends(get_db),
):
    report = report_service.visitor_report(db, search=search, limit=limit)
    return success_payload(
        data=VisitorReportOut.model_validate(report).model_dump(mode="json"),
        message="Visitor report retrieved",
    )


@router.get("/events/summary")
def events_summary(
    day: date | None = Query(None, alias="date"),
    _: User = Depends(require_permissions(Permission.REPORT_READ)),
    db: Session = Depends(get_db),
):
    report = report_service.events_summary(db, day or report_service.local_today())
    return success_payload(
        data=EventsSummaryOut.model_validate(report).model_dump(mode="json"),
        message="Per-event attendance summary retrieved",
    )


@router.get("/events/{event_id}/guests")
def event_guests(
    event_id: uuid.UUID,
    day: date | None = Query(None, alias="date"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permissions(Permission.REPORT_READ)),
    db: Session = Depends(get_db),
):
    report = report_service.event_guests(db, event_id, day or report_service.local_today(), page, page_size)
    return success_payload(
        data=EventGuestsOut.model_validate(report).model_dump(mode="json"),
        message="Event guests retrieved",
    )


@router.get("/events/export")
def events_export(
    day: date | None = Query(None, alias="date"),
    event_id: uuid.UUID | None = Query(None),
    _: User = Depends(require_permissions(Permission.REPORT_READ)),
    db: Session = Depends(get_db),
):
    content, media_type, filename = report_service.export_events_excel(
        db, day or report_service.local_today(), event_id
    )
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/attendance/export")
def export(
    report_type: str = Query("daily", pattern="^(daily|monthly|visitors)$"),
    file_format: str = Query("csv", alias="format", pattern="^(csv|xlsx)$"),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    year: int | None = Query(None, ge=2000, le=2100),
    month: int | None = Query(None, ge=1, le=12),
    _: User = Depends(require_permissions(Permission.REPORT_READ)),
    db: Session = Depends(get_db),
):
    content, media_type, filename = report_service.export_report(
        db,
        report_type=report_type,
        file_format=file_format,
        date_from=date_from,
        date_to=date_to,
        year=year,
        month=month,
    )
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
