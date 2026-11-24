from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from app.schemas.common import ORMModel


class VisitorCreate(BaseModel):
    full_name: str = Field(min_length=1, max_length=150)
    identity_number: str | None = Field(default=None, max_length=64)
    company_name: str | None = Field(default=None, max_length=150)
    phone_number: str | None = Field(default=None, max_length=30)
    email: EmailStr | None = None
    job_title: str | None = Field(default=None, max_length=150)


class VisitorUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=150)
    identity_number: str | None = Field(default=None, max_length=64)
    company_name: str | None = Field(default=None, max_length=150)
    phone_number: str | None = Field(default=None, max_length=30)
    email: EmailStr | None = None
    job_title: str | None = Field(default=None, max_length=150)
    is_active: bool | None = None


class VisitorSummary(ORMModel):
    id: uuid.UUID
    visitor_code: str
    full_name: str
    company_name: str | None = None


class VisitorOut(ORMModel):
    id: uuid.UUID
    visitor_code: str
    full_name: str
    identity_number: str | None = None
    company_name: str | None = None
    phone_number: str | None = None
    email: EmailStr | None = None
    job_title: str | None = None
    is_active: bool
    has_active_face_profile: bool = False
    created_at: datetime
    updated_at: datetime


class FaceProfileOut(ORMModel):
    id: uuid.UUID
    visitor_id: uuid.UUID
    model_version: str
    embedding_version: str
    is_active: bool
    enrolled_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class ConsentWithdrawRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=255)
    policy_version: str | None = Field(default=None, max_length=50)


class BiometricDeleteRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=255)
    delete_face_profiles: bool = True
