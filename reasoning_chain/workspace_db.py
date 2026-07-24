"""Durable ownership, document metadata, and pgvector storage."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .workspace_config import workspace_settings


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(20), default="member")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class WorkspaceSession(Base):
    __tablename__ = "workspace_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="New Chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class WorkspaceDocument(Base):
    __tablename__ = "workspace_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("workspace_sessions.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    document_type: Mapped[str] = mapped_column(String(30))
    content_type: Mapped[str] = mapped_column(String(150), default="application/octet-stream")
    s3_key: Mapped[str] = mapped_column(String(800), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    failure_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    unit_label: Mapped[str] = mapped_column(String(30), default="page")
    unit_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    extracted_chars: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    chunks: Mapped[list["WorkspaceChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class WorkspaceChunk(Base):
    __tablename__ = "workspace_chunks"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("workspace_documents.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    location: Mapped[str] = mapped_column(String(300))
    text: Mapped[str] = mapped_column(Text)
    search_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(
        VECTOR(workspace_settings.embedding_dimensions)
    )
    document: Mapped[WorkspaceDocument] = relationship(back_populates="chunks")


Index("ix_workspace_chunks_document_ordinal", WorkspaceChunk.document_id, WorkspaceChunk.ordinal)

_engine = None
_session_factory = None


def engine():
    global _engine, _session_factory
    if _engine is None:
        if not workspace_settings.database_url:
            raise RuntimeError("DATABASE_URL is required")
        _engine = create_engine(
            workspace_settings.database_url,
            pool_pre_ping=True,
            pool_recycle=300,
        )
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


@contextmanager
def db_session():
    global _session_factory
    if _session_factory is None:
        engine()
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
