# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # Embedded SQL is formatted as executable migration text.

"""Move the benchmarked Atlas voice from ``dax`` to ``capella``.

Revision ID: 20260903_0026
Revises:     20260901_0025
Create Date: 2026-09-03

The code registry pins the voice each provider is measured with, and the
database is authoritative for the roster, so the seeded row from
20260831_0023 has to follow the literal or the two disagree about what an
Atlas number was produced with.

Both are stock voices from ``/v1/models``, so this changes which voice the
next collection runs against and nothing about how it is scored. Existing
results stay as they are: they were produced with ``dax`` and relabelling
them would attribute one voice's numbers to another.
"""

from __future__ import annotations

from alembic import op

revision = "20260903_0026"
down_revision = "20260901_0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Point the Atlas TTS row at ``capella``."""
    op.get_bind().exec_driver_sql(
        """
        UPDATE benchmarks_v2.models
        SET voice = 'capella',
            updated_by_user_id = 'migration:20260903_0026', updated_by_email = NULL,
            updated_at = now()
        WHERE modality = 'TTS' AND provider = 'atlas' AND model = 'atlas-tts'
          AND voice = 'dax';
        """
    )


def downgrade() -> None:
    """Return the row to ``dax``, skipping it if the admin API has edited it since."""
    op.get_bind().exec_driver_sql(
        """
        UPDATE benchmarks_v2.models
        SET voice = 'dax',
            updated_by_user_id = 'migration:20260831_0023', updated_by_email = NULL,
            updated_at = now()
        WHERE updated_by_user_id = 'migration:20260903_0026'
          AND modality = 'TTS' AND provider = 'atlas' AND model = 'atlas-tts';
        """
    )
