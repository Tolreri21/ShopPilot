"""add hnsw index

Revision ID: 8b396509da33
Revises: 6651b2de3410
Create Date: 2026-11-12 18:57:16.327666

"""

from collections.abc import Sequence

from alembic import op

revision: str = "8b396509da33"
down_revision: str | Sequence[str] | None = "6651b2de3410"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX products_embedding_idx ON products USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX products_embedding_idx")
