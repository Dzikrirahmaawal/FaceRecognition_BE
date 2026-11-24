"""Add events, event link on attendance, job title on visitors.

Revision ID: 0002_events_and_event_attendance
Revises: 0001_initial_schema
Create Date: 2026-11-24
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql
from sqlalchemy import String, DateTime, Boolean

revision: str = "0003_events_and_event_attendance"
down_revision: Union[str, None] = "0002_user_biometric_profiles"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EVENT_ID_COLUMN = "event_id"
EVENT_FK_NAME = "fk_attendance_logs_event_id_events"
EVENT_INDEX_NAME = "ix_attendance_logs_event_id"
UNIQUE_VISITOR_EVENT_DAY_NAME = "uq_attendance_logs_visitor_event_day"
ATTENDANCE_DATE_COLUMN = "attendance_date"


def _is_mssql() -> bool:
    return op.get_bind().dialect.name == "mssql"


def _dt_type():
    """DateTime column type: DATETIMEOFFSET for MSSQL, timezone-aware DateTime for SQLite."""
    return mssql.DATETIMEOFFSET() if _is_mssql() else DateTime(timezone=True)


def _uuid_type():
    """UUID column type: UNIQUEIDENTIFIER for MSSQL, String(36) for SQLite."""
    return mssql.UNIQUEIDENTIFIER() if _is_mssql() else String(36)


def _bit_type():
    """BIT column type: for MSSQL, Boolean for SQLite."""
    return mssql.BIT() if _is_mssql() else Boolean()


def _nvarchar(length: int):
    """NVARCHAR column type."""
    return mssql.NVARCHAR(length=length) if _is_mssql() else String(length)


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("id", _uuid_type(), nullable=False),
        sa.Column("name", _nvarchar(200), nullable=False),
        sa.Column("description", _nvarchar(1000), nullable=True),
        sa.Column("location", _nvarchar(200), nullable=True),
        sa.Column("starts_at", _dt_type(), nullable=False),
        sa.Column("ends_at", _dt_type(), nullable=False),
        sa.Column("is_active", _bit_type(), nullable=False, server_default=sa.text("1")),
        sa.Column(
            "created_by", _uuid_type(), nullable=True
        ),
        sa.Column(
            "created_at",
            _dt_type(),
            nullable=False,
            server_default=sa.text("SYSUTCDATETIME()") if _is_mssql() else sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            _dt_type(),
            nullable=False,
            server_default=sa.text("SYSUTCDATETIME()") if _is_mssql() else sa.func.now(),
        ),
        sa.Column("deleted_at", _dt_type(), nullable=True),
        sa.CheckConstraint("ends_at > starts_at", name="ck_events_ends_after_starts"),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], name="fk_events_created_by_users", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_events"),
    )
    op.create_index("ix_events_starts_at_is_active", "events", ["starts_at", "is_active"])
    op.create_index("ix_events_is_active", "events", ["is_active"])
    op.create_index("ix_events_name", "events", ["name"])

    with op.batch_alter_table("attendance_logs") as batch_op:
        batch_op.add_column(sa.Column(EVENT_ID_COLUMN, _uuid_type(), nullable=True))
        batch_op.create_foreign_key(
            EVENT_FK_NAME,
            "events",
            [EVENT_ID_COLUMN],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(EVENT_INDEX_NAME, [EVENT_ID_COLUMN])

    op.add_column(
        "visitors",
        sa.Column("job_title", _nvarchar(150), nullable=True),
    )

    # One guest may not check in twice to the same event on the same local day.
    # The day part is computed in the business timezone (Asia/Jakarta = +07:00).
    if _is_mssql():
        op.execute(
            f'ALTER TABLE attendance_logs ADD {ATTENDANCE_DATE_COLUMN} '
            f"AS (CAST(SWITCHOFFSET(event_timestamp, '+07:00') AS DATE)) PERSISTED"
        )
        op.create_index(
            UNIQUE_VISITOR_EVENT_DAY_NAME,
            "attendance_logs",
            ["visitor_id", EVENT_ID_COLUMN, ATTENDANCE_DATE_COLUMN],
            unique=True,
            mssql_where=sa.text(f"{EVENT_ID_COLUMN} IS NOT NULL"),
        )
    else:
        # SQLite (tests/throwaway verification): expression partial index.
        op.execute(
            f'CREATE UNIQUE INDEX {UNIQUE_VISITOR_EVENT_DAY_NAME} '
            f"ON attendance_logs "
            f"(visitor_id, {EVENT_ID_COLUMN}, strftime('%Y-%m-%d', event_timestamp, '+7 hours')) "
            f"WHERE {EVENT_ID_COLUMN} IS NOT NULL"
        )


def downgrade() -> None:
    op.drop_index(UNIQUE_VISITOR_EVENT_DAY_NAME, table_name="attendance_logs")
    if _is_mssql():
        op.execute(f"ALTER TABLE attendance_logs DROP COLUMN {ATTENDANCE_DATE_COLUMN}")
    op.drop_constraint(EVENT_FK_NAME, table_name="attendance_logs", type_="foreignkey")
    op.drop_index(EVENT_INDEX_NAME, table_name="attendance_logs")
    op.drop_column("attendance_logs", EVENT_ID_COLUMN)

    op.drop_index("ix_events_name", table_name="events")
    op.drop_index("ix_events_is_active", table_name="events")
    op.drop_index("ix_events_starts_at_is_active", table_name="events")
    op.drop_table("events")

    op.drop_column("visitors", "job_title")
