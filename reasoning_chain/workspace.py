"""S3 originals, queued extraction, hybrid retrieval, and safe evidence previews."""

from __future__ import annotations

import logging
import mimetypes
import os
import uuid
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import boto3
import redis
from rq import Queue, Retry
from sqlalchemy import delete, func, select

from .documents import (
    SUPPORTED_DOCUMENT_TYPES,
    DocumentError,
    DocumentMetadata,
    DocumentRetrieval,
    DocumentSettings,
    extract_document,
)
from .workspace_config import workspace_settings
from .workspace_db import (
    WorkspaceChunk,
    WorkspaceDocument,
    WorkspaceSession,
    db_session,
)

logger = logging.getLogger("reasoning_chain.workspace")
_s3 = None
_embedder = None


def _s3_client():
    global _s3
    if _s3 is None:
        kwargs = {"region_name": workspace_settings.aws_region}
        if workspace_settings.s3_endpoint_url:
            kwargs["endpoint_url"] = workspace_settings.s3_endpoint_url
        _s3 = boto3.client("s3", **kwargs)
    return _s3


def _embedding_model():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(workspace_settings.embedding_model)
    return _embedder


def embed_passages(values: list[str]) -> list[list[float]]:
    if not values:
        return []
    prefixed = [f"passage: {value}" for value in values]
    vectors = _embedding_model().encode(prefixed, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def embed_query(value: str) -> list[float]:
    vector = _embedding_model().encode([f"query: {value}"], normalize_embeddings=True)[0]
    return vector.tolist()


def _validate_signature(filename: str, data: bytes) -> None:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_DOCUMENT_TYPES))
        raise DocumentError(f"unsupported document type; allowed extensions: {allowed}")
    if extension == ".pdf" and not data.startswith(b"%PDF-"):
        raise DocumentError("file content is not a valid PDF")
    if extension in {".docx", ".xlsx", ".pptx"} and not data.startswith(b"PK"):
        raise DocumentError("file content does not match its Office document extension")
    if b"\x00" in data[:4096] and extension in {".txt", ".md", ".csv", ".tsv"}:
        raise DocumentError("text document contains binary content")


def _metadata(document: WorkspaceDocument) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document.id,
        session_id=document.session_id,
        filename=document.filename,
        document_type=document.document_type,
        unit_label=document.unit_label,
        unit_count=document.unit_count,
        status=document.status,
        page_count=document.unit_count,
        chunk_count=document.chunk_count,
        extracted_chars=document.extracted_chars,
        created_at=document.created_at.isoformat(),
        failure_reason=document.failure_reason,
    )


class EvidenceWorkspaceStore:
    def __init__(self, settings: DocumentSettings | None = None):
        self.settings = settings or DocumentSettings()

    def ensure_session(
        self, owner_id: str, session_id: str, title: str, created_at: datetime
    ) -> None:
        with db_session() as session:
            existing = session.get(WorkspaceSession, session_id)
            if existing and existing.owner_id != owner_id:
                raise DocumentError("session not found")
            if existing:
                existing.title = title[:200]
                existing.updated_at = datetime.now(timezone.utc)
            else:
                session.add(
                    WorkspaceSession(
                        id=session_id,
                        owner_id=owner_id,
                        title=title[:200],
                        created_at=created_at,
                    )
                )

    def _purge_expired(self, owner_id: str) -> None:
        now = datetime.now(timezone.utc)
        with db_session() as session:
            expired = session.scalars(
                select(WorkspaceDocument)
                .join(WorkspaceSession)
                .where(
                    WorkspaceSession.owner_id == owner_id,
                    WorkspaceDocument.expires_at <= now,
                )
            ).all()
            keys = [document.s3_key for document in expired]
            for document in expired:
                session.delete(document)
        for key in keys:
            try:
                _s3_client().delete_object(Bucket=workspace_settings.s3_bucket, Key=key)
            except Exception:
                logger.warning("failed to delete expired S3 object %s", key)

    def owns_session(self, owner_id: str, session_id: str) -> bool:
        with db_session() as session:
            return bool(
                session.scalar(
                    select(WorkspaceSession.id).where(
                        WorkspaceSession.id == session_id,
                        WorkspaceSession.owner_id == owner_id,
                    )
                )
            )

    def list_sessions(self, owner_id: str, limit: int = 100) -> list[dict]:
        with db_session() as session:
            rows = session.scalars(
                select(WorkspaceSession)
                .where(WorkspaceSession.owner_id == owner_id)
                .order_by(WorkspaceSession.updated_at.desc())
                .limit(limit)
            ).all()
            return [
                {"id": row.id, "title": row.title, "created_at": row.created_at.isoformat()}
                for row in rows
            ]

    def queue_upload(
        self,
        owner_id: str,
        session_id: str,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> DocumentMetadata:
        self._purge_expired(owner_id)
        if not self.owns_session(owner_id, session_id):
            raise DocumentError("session not found")
        safe_name = Path(filename or "document.txt").name[:120]
        if len(data) > self.settings.max_file_bytes:
            raise DocumentError(
                f"document exceeds the {self.settings.max_file_bytes // (1024 * 1024)} MB limit"
            )
        _validate_signature(safe_name, data)
        extension = Path(safe_name).suffix.lower()
        document_type, unit_label = SUPPORTED_DOCUMENT_TYPES[extension]
        with db_session() as session:
            count = session.scalar(
                select(func.count(WorkspaceDocument.id)).where(
                    WorkspaceDocument.session_id == session_id
                )
            )
            if count >= self.settings.max_documents_per_session:
                raise DocumentError(
                    f"session already has {self.settings.max_documents_per_session} documents"
                )
            document_id = str(uuid.uuid4())
            s3_key = f"users/{owner_id}/sessions/{session_id}/{document_id}/{safe_name}"
            _s3_client().put_object(
                Bucket=workspace_settings.s3_bucket,
                Key=s3_key,
                Body=data,
                ContentType=content_type or mimetypes.guess_type(safe_name)[0]
                or "application/octet-stream",
                ServerSideEncryption="AES256",
            )
            document = WorkspaceDocument(
                id=document_id,
                session_id=session_id,
                filename=safe_name,
                document_type=document_type,
                content_type=content_type or "application/octet-stream",
                s3_key=s3_key,
                status="queued",
                unit_label=unit_label,
                expires_at=datetime.now(timezone.utc)
                + timedelta(seconds=self.settings.ttl_seconds),
            )
            session.add(document)
            session.flush()
            result = _metadata(document)
        try:
            connection = redis.Redis.from_url(
                os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            )
            Queue("document-ingestion", connection=connection).enqueue(
                ingest_document_job,
                document_id,
                job_timeout=900,
                retry=Retry(max=3, interval=[10, 30, 60]),
            )
        except Exception:
            with db_session() as session:
                session.execute(
                    delete(WorkspaceDocument).where(WorkspaceDocument.id == document_id)
                )
            _s3_client().delete_object(Bucket=workspace_settings.s3_bucket, Key=s3_key)
            raise DocumentError("document queue is unavailable")
        return result

    def list(self, owner_id: str, session_id: str) -> list[DocumentMetadata]:
        self._purge_expired(owner_id)
        if not self.owns_session(owner_id, session_id):
            raise DocumentError("session not found")
        with db_session() as session:
            documents = session.scalars(
                select(WorkspaceDocument)
                .where(WorkspaceDocument.session_id == session_id)
                .order_by(WorkspaceDocument.created_at)
            ).all()
            return [_metadata(document) for document in documents]

    def status(self, owner_id: str, document_id: str) -> DocumentMetadata | None:
        self._purge_expired(owner_id)
        with db_session() as session:
            document = session.scalar(
                select(WorkspaceDocument)
                .join(WorkspaceSession)
                .where(
                    WorkspaceDocument.id == document_id,
                    WorkspaceSession.owner_id == owner_id,
                )
            )
            return _metadata(document) if document else None

    def delete(self, owner_id: str, session_id: str, document_id: str) -> bool:
        self._purge_expired(owner_id)
        with db_session() as session:
            document = session.scalar(
                select(WorkspaceDocument)
                .join(WorkspaceSession)
                .where(
                    WorkspaceDocument.id == document_id,
                    WorkspaceDocument.session_id == session_id,
                    WorkspaceSession.owner_id == owner_id,
                )
            )
            if not document:
                return False
            s3_key = document.s3_key
            session.delete(document)
        _s3_client().delete_object(Bucket=workspace_settings.s3_bucket, Key=s3_key)
        return True

    def view_url(self, owner_id: str, document_id: str) -> dict | None:
        self._purge_expired(owner_id)
        with db_session() as session:
            document = session.scalar(
                select(WorkspaceDocument)
                .join(WorkspaceSession)
                .where(
                    WorkspaceDocument.id == document_id,
                    WorkspaceSession.owner_id == owner_id,
                    WorkspaceDocument.status == "ready",
                )
            )
            if not document:
                return None
            url = _s3_client().generate_presigned_url(
                "get_object",
                Params={"Bucket": workspace_settings.s3_bucket, "Key": document.s3_key},
                ExpiresIn=workspace_settings.presigned_url_seconds,
            )
            return {
                "url": url,
                "document_type": document.document_type,
                "expires_in": workspace_settings.presigned_url_seconds,
            }

    def preview(self, owner_id: str, document_id: str, chunk_id: str) -> dict | None:
        self._purge_expired(owner_id)
        with db_session() as session:
            row = session.execute(
                select(WorkspaceChunk, WorkspaceDocument)
                .join(WorkspaceDocument)
                .join(WorkspaceSession)
                .where(
                    WorkspaceChunk.id == chunk_id,
                    WorkspaceDocument.id == document_id,
                    WorkspaceSession.owner_id == owner_id,
                )
            ).first()
            if not row:
                return None
            chunk, document = row
            return {
                "chunk_id": chunk.id,
                "document_id": document.id,
                "filename": document.filename,
                "document_type": document.document_type,
                "page": chunk.page,
                "location": chunk.location,
                "text": chunk.text,
            }

    def retrieve(self, owner_id: str, session_id: str, query: str) -> DocumentRetrieval:
        self._purge_expired(owner_id)
        if not query.strip() or not self.owns_session(owner_id, session_id):
            return DocumentRetrieval()
        query_vector = embed_query(query)
        candidate_limit = self.settings.retrieval_chunks * 4
        with db_session() as session:
            scope = (
                WorkspaceDocument.session_id == session_id,
                WorkspaceDocument.status == "ready",
                WorkspaceSession.owner_id == owner_id,
            )
            semantic = session.execute(
                select(WorkspaceChunk, WorkspaceDocument)
                .join(WorkspaceDocument)
                .join(WorkspaceSession)
                .where(*scope)
                .order_by(WorkspaceChunk.embedding.cosine_distance(query_vector))
                .limit(candidate_limit)
            ).all()
            lexical_score = func.ts_rank_cd(
                func.to_tsvector("simple", WorkspaceChunk.search_text),
                func.plainto_tsquery("simple", query),
            )
            lexical = session.execute(
                select(WorkspaceChunk, WorkspaceDocument, lexical_score.label("rank"))
                .join(WorkspaceDocument)
                .join(WorkspaceSession)
                .where(*scope, lexical_score > 0)
                .order_by(lexical_score.desc())
                .limit(candidate_limit)
            ).all()

        scores: dict[str, float] = {}
        candidates = {}
        for rank, (chunk, document) in enumerate(semantic, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0) + (
                workspace_settings.semantic_weight / (60 + rank)
            )
            candidates[chunk.id] = (chunk, document)
        for rank, (chunk, document, _) in enumerate(lexical, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0) + (
                workspace_settings.lexical_weight / (60 + rank)
            )
            candidates[chunk.id] = (chunk, document)
        ordered = sorted(scores, key=scores.get, reverse=True)
        selected = []
        per_document: dict[str, int] = {}
        for chunk_id in ordered:
            chunk, document = candidates[chunk_id]
            if per_document.get(document.id, 0) >= 3:
                continue
            selected.append((chunk, document, scores[chunk_id]))
            per_document[document.id] = per_document.get(document.id, 0) + 1
            if len(selected) == workspace_settings.retrieval_chunks:
                break
        if not selected:
            return DocumentRetrieval()

        passages = []
        rendered = []
        for chunk, document, score in selected:
            citation = f"[{document.filename}, {chunk.location}]"
            passage = {
                "chunk_id": chunk.id,
                "document_id": document.id,
                "filename": document.filename,
                "document_type": document.document_type,
                "page": chunk.page,
                "location": chunk.location,
                "citation": citation,
                "text": chunk.text,
                "retrieval_score": round(score, 6),
                "embedding": list(chunk.embedding),
            }
            passages.append(passage)
            rendered.append(
                f'<passage chunk_id="{escape(chunk.id, quote=True)}" '
                f'document_id="{escape(document.id, quote=True)}" '
                f'citation="{escape(citation, quote=True)}">'
                f"{escape(chunk.text)}</passage>"
            )
        return DocumentRetrieval(
            context='<document_context trust="untrusted">\n'
            + "\n".join(rendered)
            + "\n</document_context>",
            citations=[passage["citation"] for passage in passages],
            document_ids=list(dict.fromkeys(passage["document_id"] for passage in passages)),
            chunk_count=len(passages),
            passages=passages,
        )


def ingest_document_job(document_id: str) -> None:
    """RQ entry point. It is idempotent for a document that has already completed."""
    settings = DocumentSettings()
    s3_key = None
    try:
        with db_session() as session:
            document = session.get(WorkspaceDocument, document_id)
            if not document or document.status == "ready":
                return
            document.status = "processing"
            s3_key = document.s3_key
            filename = document.filename
        response = _s3_client().get_object(
            Bucket=workspace_settings.s3_bucket, Key=s3_key
        )
        data = response["Body"].read(settings.max_file_bytes + 1)
        extracted, chunks = extract_document(data, filename, settings)
        vectors = embed_passages([chunk.text for chunk in chunks])
        with db_session() as session:
            document = session.get(WorkspaceDocument, document_id)
            session.execute(
                delete(WorkspaceChunk).where(WorkspaceChunk.document_id == document_id)
            )
            for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
                session.add(
                    WorkspaceChunk(
                        id=f"{document_id}:c{ordinal}",
                        document_id=document_id,
                        ordinal=ordinal,
                        page=chunk.page,
                        location=chunk.location,
                        text=chunk.text,
                        search_text=f"{filename} {chunk.location} {chunk.text}",
                        embedding=vector,
                    )
                )
            document.status = "ready"
            document.unit_label = extracted.unit_label
            document.unit_count = extracted.unit_count
            document.chunk_count = len(chunks)
            document.extracted_chars = extracted.extracted_chars
            document.failure_reason = None
    except Exception as exc:
        logger.exception("document ingestion failed for %s", document_id)
        safe_reason = str(exc)[:500] or "document processing failed"
        with db_session() as session:
            document = session.get(WorkspaceDocument, document_id)
            if document:
                document.status = "failed"
                document.failure_reason = safe_reason
        raise
