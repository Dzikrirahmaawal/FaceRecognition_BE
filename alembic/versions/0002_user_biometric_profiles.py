"""Add user biometric profiles table.

Revision ID: 0002_user_biometric_profiles
Revises: 0001_initial_schema
Create Date: 2026-11-12
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

revision: str = "0002_user_biometric_profiles"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_biometric_profiles",
        sa.Column("id", mssql.UNIQUEIDENTIFIER(), nullable=False),
        sa.Column("user_id", mssql.UNIQUEIDENTIFIER(), nullable=False),
        sa.Column("embedding_data", mssql.VARBINARY(), nullable=True),
        sa.Column("model_version", mssql.NVARCHAR(length=50), nullable=False),
        sa.Column("embedding_version", mssql.NVARCHAR(length=20), nullable=False),
        sa.Column("is_active", mssql.BIT(), nullable=False, server_default=sa.text("1")),
        sa.Column("enrolled_at", mssql.DATETIMEOFFSET(), nullable=False),
        sa.Column("updated_at", mssql.DATETIMEOFFSET(), nullable=False, server_default=sa.text("SYSUTCDATETIME()")),
        sa.Column("deleted_at", mssql.DATETIMEOFFSET(), nullable=True),
        sa.Column("deletion_reason", mssql.NVARCHAR(length=255), nullable=True),
        sa.Column("created_at", mssql.DATETIMEOFFSET(), nullable=False, server_default=sa.text("SYSUTCDATETIME()")),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_user_biometric_profiles_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_biometric_profiles"),
    )
    op.create_index("ix_user_biometric_profiles_user_id", "user_biometric_profiles", ["user_id"])
    op.create_index("ix_user_biometric_profiles_is_active", "user_biometric_profiles", ["is_active"])
    op.create_index(
        "uq_user_biometric_profiles_active_user",
        "user_biometric_profiles",
        ["user_id"],
        unique=True,
        mssql_where=sa.text("is_active = 1"),
    )


def downgrade() -> None:
    op.drop_index("uq_user_biometric_profiles_active_user", table_name="user_biometric_profiles")
    op.drop_index("ix_user_biometric_profiles_is_active", table_name="user_biometric_profiles")
    op.drop_index("ix_user_biometric_profiles_user_id", table_name="user_biometric_profiles")
    op.drop_table("user_biometric_profiles")
