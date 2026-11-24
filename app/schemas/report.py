from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field


class DashboardOut(BaseModel):
    total_visitors: int
    active_visitors: int
    visitors_currently_inside: int
    today_check_ins: int
    today_check_outs: int
    pending_visits: int
    active_kiosks: int
    generated_at: datetime
    timezone: str


class DailyAttendanceRow(BaseModel):
    hour: int
    check_ins: int
    check_outs: int


class DailyReportOut(BaseModel):
    date: date
    timezone: str
    total_check_ins: int
    total_check_outs: int
    rows: list[DailyAttendanceRow]


class MonthlyAttendanceRow(BaseModel):
    date: date
    check_ins: int
    check_outs: int


class MonthlyReportOut(BaseModel):
    year: int
    month: int
    timezone: str
    total_check_ins: int
    total_check_outs: int
    rows: list[MonthlyAttendanceRow]


class VisitorReportRow(BaseModel):
    visitor_id: uuid.UUID
    visitor_code: str
    full_name: str
    company_name: str | None = None
    visit_count: int
    last_visit_at: datetime | None = None


class VisitorReportOut(BaseModel):
    items: list[VisitorReportRow]
    total: int


class ExportQuery(BaseModel):
    report_type: str = Field(pattern="^(daily|monthly|visitors)$")
    format: str = Field(default="csv", pattern="^(csv|xlsx)$")
    date_from: date | None = None
    date_to: date | None = None
    year: int | None = None
    month: int | None = None


class EventSummaryRow(BaseModel):
    event_id: uuid.UUID
    name: str
    starts_at: datetime
    ends_at: datetime
    location: str | None = None
    total_guests: int
    checked_out: int
    still_present: int


class EventsSummaryOut(BaseModel):
    date: date
    timezone: str
    events: list[EventSummaryRow]
    unique_guests_total: int


class EventGuestRow(BaseModel):
    visitor_id: uuid.UUID
    nik: str | None = None
    full_name: str
    job_title: str | None = None
    institution: str | None = None
    phone_number: str | None = None
    check_in_at: datetime | None = None
    check_out_at: datetime | None = None


class EventGuestsOut(BaseModel):
    event_id: uuid.UUID
    date: date
    items: list[EventGuestRow]
    total: int
    page: int
    page_size: int
    total_pages: int
