from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import false, select
from sqlalchemy.orm import Session, joinedload

from app.models.attendance import AttendanceLog
from app.models.enums import AttendanceEventType


def get(db: Session, attendance_id: uuid.UUID) -> AttendanceLog | None:
    stmt = (
        select(AttendanceLog)
        .where(AttendanceLog.id == attendance_id)
        .options(
            joinedload(AttendanceLog.visit),
            joinedload(AttendanceLog.visitor),
            joinedload(AttendanceLog.kiosk),
            joinedload(AttendanceLog.event),
        )
    )
    return db.scalars(stmt).first()


def get_original_for_visit(
    db: Session, visit_id: uuid.UUID, event_type: AttendanceEventType, for_update: bool = False
) -> AttendanceLog | None:
    stmt = select(AttendanceLog).where(
        AttendanceLog.visit_id == visit_id,
        AttendanceLog.event_type == event_type,
        AttendanceLog.is_correction == false(),
    )
    if for_update:
        stmt = stmt.with_for_update()
    return db.scalars(stmt).first()


def list_attendance(
    db: Session,
    page: int,
    page_size: int,
    visitor_id: uuid.UUID | None = None,
    event_type: AttendanceEventType | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    kiosk_id: uuid.UUID | None = None,
    event_id: uuid.UUID | None = None,
):
    stmt = (
        select(AttendanceLog)
        .options(
            joinedload(AttendanceLog.visit),
            joinedload(AttendanceLog.visitor),
            joinedload(AttendanceLog.kiosk),
            joinedload(AttendanceLog.event),
        )
        .order_by(AttendanceLog.event_timestamp.desc())
    )
    if visitor_id:
        stmt = stmt.where(AttendanceLog.visitor_id == visitor_id)
    if event_type:
        stmt = stmt.where(AttendanceLog.event_type == event_type)
    if event_id is not None:
        stmt = stmt.where(AttendanceLog.event_id == event_id)
    if kiosk_id:
        stmt = stmt.where(AttendanceLog.kiosk_id == kiosk_id)
    if date_from:
        stmt = stmt.where(AttendanceLog.event_timestamp >= date_from)
    if date_to:
        stmt = stmt.where(AttendanceLog.event_timestamp <= date_to)
    return stmt


def get_event_check_in_for_visitor_day(
    db: Session,
    visitor_id: uuid.UUID,
    event_id: uuid.UUID,
    start: datetime,
    end: datetime,
) -> AttendanceLog | None:
    stmt = select(AttendanceLog).where(
        AttendanceLog.visitor_id == visitor_id,
        AttendanceLog.event_id == event_id,
        AttendanceLog.event_type == AttendanceEventType.CHECK_IN,
        AttendanceLog.is_correction == false(),
        AttendanceLog.event_timestamp >= start,
        AttendanceLog.event_timestamp < end,
    )
    return db.scalars(stmt.order_by(AttendanceLog.event_timestamp.asc())).first()


def get_open_check_ins(
    db: Session,
    visitor_id: uuid.UUID,
    start: datetime,
    end: datetime,
    event_id: uuid.UUID | None = None,
) -> list[AttendanceLog]:
    stmt = select(AttendanceLog).where(
        AttendanceLog.visitor_id == visitor_id,
        AttendanceLog.event_type == AttendanceEventType.CHECK_IN,
        AttendanceLog.is_correction == false(),
        AttendanceLog.event_timestamp >= start,
        AttendanceLog.event_timestamp < end,
    )
    if event_id is not None:
        stmt = stmt.where(AttendanceLog.event_id == event_id)
    check_ins = list(db.scalars(stmt.order_by(AttendanceLog.event_timestamp.asc())).all())
    open_check_ins: list[AttendanceLog] = []
    for check_in in check_ins:
        has_check_out = db.scalars(
            select(AttendanceLog).where(
                AttendanceLog.visitor_id == visitor_id,
                AttendanceLog.event_type == AttendanceEventType.CHECK_OUT,
                AttendanceLog.is_correction == false(),
                AttendanceLog.event_timestamp >= start,
                AttendanceLog.event_timestamp < end,
                (AttendanceLog.event_id == check_in.event_id)
                | (AttendanceLog.event_id.is_(None) & check_in.event_id.is_(None)),
            )
        ).first()
        if has_check_out is None:
            open_check_ins.append(check_in)
    return open_check_ins


def add(db: Session, log: AttendanceLog) -> AttendanceLog:
    db.add(log)
    return log
