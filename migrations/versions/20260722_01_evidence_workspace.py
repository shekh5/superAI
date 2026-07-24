"""Create Evidence Workspace tables and search indexes."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import VECTOR

revision = "20260722_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("must_change_password", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_table(
        "workspace_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "owner_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_workspace_sessions_owner_id", "workspace_sessions", ["owner_id"])
    op.create_table(
        "workspace_documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(36),
            sa.ForeignKey("workspace_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("document_type", sa.String(30), nullable=False),
        sa.Column("content_type", sa.String(150), nullable=False),
        sa.Column("s3_key", sa.String(800), nullable=False, unique=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("failure_reason", sa.String(500), nullable=True),
        sa.Column("unit_label", sa.String(30), nullable=False),
        sa.Column("unit_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("extracted_chars", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_workspace_documents_session_id", "workspace_documents", ["session_id"]
    )
    op.create_index("ix_workspace_documents_status", "workspace_documents", ["status"])
    op.create_index(
        "ix_workspace_documents_expires_at", "workspace_documents", ["expires_at"]
    )
    op.create_table(
        "workspace_chunks",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("workspace_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("location", sa.String(300), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("embedding", VECTOR(384), nullable=False),
    )
    op.create_index("ix_workspace_chunks_document_id", "workspace_chunks", ["document_id"])
    op.create_index(
        "ix_workspace_chunks_document_ordinal",
        "workspace_chunks",
        ["document_id", "ordinal"],
    )
    op.execute(
        "CREATE INDEX ix_workspace_chunks_search ON workspace_chunks "
        "USING gin (to_tsvector('simple', search_text))"
    )
    op.execute(
        "CREATE INDEX ix_workspace_chunks_embedding ON workspace_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_table("workspace_chunks")
    op.drop_table("workspace_documents")
    op.drop_table("workspace_sessions")
    op.drop_table("users")
