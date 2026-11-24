from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class EventBase(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    location: str | None = Field(default=None, max_length=200)
    starts_at: datetime
    ends_at: datetime


class EventCreate(EventBase):
    pass


class EventUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    location: str | None = Field(default=None, max_length=200)
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    is_active: bool | None = None


class EventOut(ORMModel):
    id: uuid.UUID
    name: str
    description: str | None = None
    location: str | None = None
    starts_at: datetime
    ends_at: datetime
    is_active: bool
    created_by: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class EventAvailableOut(EventOut):
    """Events a kiosk may offer for check-in today (same shape as EventOut)."""
