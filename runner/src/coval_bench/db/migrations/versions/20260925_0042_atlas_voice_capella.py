# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0
"""Repoint the atlas TTS voice from ``dax`` to ``capella``.

``models`` is unique on (modality, provider, model), so the benchmarked voice is
an attribute of the single atlas row rather than a row of its own: this is an
update, not an insert.

The atlas endpoint now serves exactly one voice. ``dax`` was pinned by
``20260831_0023`` when the gateway still fronted the full stock set; it is no
longer routable and the provider rejects it at the ``start`` frame with
``Unknown voice "dax". Supported: capella``, which would fail every item of a
registry-driven run rather than degrade.
"""

from __future__ import annotations

from alembic import op

revision = "20260925_0042"
down_revision = "20260921_0041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """UPDATE benchmarks_v2.models
              SET voice = 'capella',
                  updated_by_user_id = 'migration:20260925_0042'
            WHERE modality = 'TTS' AND provider = 'atlas' AND model = 'atlas-tts'"""
    )


def downgrade() -> None:
    op.execute(
        """UPDATE benchmarks_v2.models
              SET voice = 'dax',
                  updated_by_user_id = 'migration:20260831_0023'
            WHERE modality = 'TTS' AND provider = 'atlas' AND model = 'atlas-tts'"""
    )
