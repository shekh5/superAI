"""
FastAPI endpoints for the reasoning chain.

Wire into your existing app with:

    from reasoning_chain.router import router as chain_router
    app.include_router(chain_router, prefix="/chain")

Production note on the Redis logging below:
Every run gets a request_id and its full ChainTrace is stored under that
key. This is the minimum viable version of what LangSmith / Helicone give
you in Phase 5 -- a way to pull up "what did the agent actually do for
this specific request" after the fact, instead of only having whatever
made it into stdout logs.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from .chain import decompose_goal, run_chain, summarize_context
from .context import ContextBundle, ContextMessage, ContextSettings, RedisContextStore
from .documents import (
    DocumentError,
    DocumentMetadata,
    DocumentSettings,
    RedisDocumentStore,
)
from .schemas import ChainTrace, Plan, SessionMessage, SessionMetadata
from .workspace_config import workspace_settings

logger = logging.getLogger("reasoning_chain")
router = APIRouter()
context_settings = ContextSettings()
document_settings = DocumentSettings()

try:
    import redis

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    _redis = redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
    _redis.ping()
except Exception as e:
    _redis = None
    logger.warning(f"Redis unavailable ({e}) -- chain traces will not be persisted")


def _log_trace(trace: ChainTrace) -> None:
    if _redis is None:
        return
    try:
        # Save trace content
        _redis.set(f"chain_trace:{trace.request_id}", trace.model_dump_json(), ex=60 * 60 * 24)
        # Add request_id to traces list for history visualization
        _redis.lpush("chain_traces_list", trace.request_id)
        # Keep only the last 100 trace IDs
        _redis.ltrim("chain_traces_list", 0, 99)
    except Exception as e:
        logger.warning(f"failed to persist trace {trace.request_id}: {e}")


def _context_store() -> Optional[RedisContextStore]:
    return RedisContextStore(_redis, context_settings) if _redis is not None else None


def _document_store() -> Optional[RedisDocumentStore]:
    return RedisDocumentStore(_redis, document_settings) if _redis is not None else None


def _identity(request: Request):
    if not workspace_settings.enabled:
        return None
    from .auth import require_identity

    return require_identity(request)


def _workspace_store():
    from .workspace import EvidenceWorkspaceStore

    return EvidenceWorkspaceStore(document_settings)


def _context_session_id(session_id: str, owner_id: Optional[str] = None) -> str:
    return f"{owner_id}:{session_id}" if owner_id else session_id


def _load_context(session_id: str) -> ContextBundle:
    store = _context_store()
    if store is None:
        return ContextBundle()
    try:
        context = store.load(session_id)
        if len(context.recent) > context_settings.compaction_threshold:
            store.compact(session_id, summarize_context)
            context = store.load(session_id)
        return context
    except Exception as e:
        logger.warning(f"failed to load session context: {e}")
        return ContextBundle()


def _attach_document_context(
    session_id: str,
    goal: str,
    conversation: ContextBundle,
    owner_id: Optional[str] = None,
) -> ContextBundle:
    store = _workspace_store() if workspace_settings.enabled else _document_store()
    if store is None:
        return conversation
    try:
        retrieval = (
            store.retrieve(owner_id, session_id, goal)
            if workspace_settings.enabled
            else store.retrieve(session_id, goal)
        )
    except Exception as exc:
        logger.warning(f"failed to retrieve session documents: {exc}")
        return conversation
    return ContextBundle(
        summary=conversation.summary,
        recent=conversation.recent,
        document_context=retrieval.context,
        document_citations=retrieval.citations,
        document_ids=retrieval.document_ids,
        document_chunks=retrieval.chunk_count,
        document_passages=retrieval.passages,
    )


def _compact_context(store: RedisContextStore, session_id: str) -> None:
    try:
        store.compact(session_id, summarize_context)
    except Exception as e:
        # A failed summary must never fail the user's request or delete source messages.
        logger.warning(f"failed to compact session context: {e}")


@router.post("/plan", response_model=Plan)
def plan_only(goal: str):
    """Returns the decomposition only -- no tools executed. Use this to
    sanity-check the model's reasoning before wiring up execution."""
    try:
        return decompose_goal(goal)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"planning failed: {e}") from e


@router.post("/run", response_model=ChainTrace)
def run(
    request: Request,
    goal: str,
    session_id: Optional[str] = None,
    temperature: Optional[float] = Query(default=None, ge=0.0, le=1.0),
):
    """Run the bounded ReAct loop with optional conversation memory."""
    identity = _identity(request)
    owner_id = identity.user_id if identity else None
    if session_id and identity and not _workspace_store().owns_session(owner_id, session_id):
        raise HTTPException(status_code=404, detail="session not found")
    context_session_id = _context_session_id(session_id, owner_id) if session_id else None
    conversation = _load_context(context_session_id) if context_session_id else ContextBundle()
    if session_id:
        conversation = _attach_document_context(session_id, goal, conversation, owner_id)

    try:
        trace = run_chain(goal, conversation=conversation, temperature=temperature)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"chain failed: {e}") from e

    if session_id:
        trace.session_id = session_id

    _log_trace(trace)

    if session_id and _redis:
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            user_msg = {
                "sender": "user",
                "text": goal,
                "timestamp": timestamp,
            }
            agent_msg = {
                "sender": "agent",
                "text": trace.verify.final_summary,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trace": trace.model_dump(),
            }
            store = RedisContextStore(_redis, context_settings)
            store.append_display(context_session_id, user_msg, agent_msg)
            store.append(
                context_session_id,
                ContextMessage(role="user", text=goal, timestamp=timestamp),
            )
            store.append(
                context_session_id,
                ContextMessage(
                    role="model",
                    text=trace.verify.final_summary,
                    timestamp=agent_msg["timestamp"],
                ),
            )
            _compact_context(store, context_session_id)
        except Exception as e:
            logger.warning(f"failed to append session messages: {e}")

    return trace


@router.post(
    "/session/{session_id}/documents",
    response_model=DocumentMetadata,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    session_id: str,
    request: Request,
    response: Response,
    file: UploadFile = File(...),
):
    """Extract a supported document and attach its searchable chunks to a session."""
    identity = _identity(request)
    store = _workspace_store() if workspace_settings.enabled else _document_store()
    if store is None:
        raise HTTPException(status_code=503, detail="document storage is not configured")
    try:
        data = await file.read(document_settings.max_file_bytes + 1)
        if len(data) > document_settings.max_file_bytes:
            limit_mb = document_settings.max_file_bytes // (1024 * 1024)
            raise DocumentError(f"document exceeds the {limit_mb} MB limit")
        if workspace_settings.enabled:
            response.status_code = status.HTTP_202_ACCEPTED
            return await run_in_threadpool(
                store.queue_upload,
                identity.user_id,
                session_id,
                file.filename,
                file.content_type or "application/octet-stream",
                data,
            )
        return await run_in_threadpool(store.ingest, session_id, file.filename, data)
    except DocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@router.get(
    "/session/{session_id}/documents",
    response_model=list[DocumentMetadata],
)
def list_session_documents(session_id: str, request: Request):
    """List documents currently available to one chat session."""
    identity = _identity(request)
    store = _workspace_store() if workspace_settings.enabled else _document_store()
    if store is None:
        return []
    try:
        return (
            store.list(identity.user_id, session_id)
            if workspace_settings.enabled
            else store.list(session_id)
        )
    except DocumentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning(f"failed to list session documents: {exc}")
        return []


@router.delete("/session/{session_id}/documents/{document_id}")
def delete_session_document(session_id: str, document_id: str, request: Request):
    """Delete extracted document text and detach it from the session."""
    identity = _identity(request)
    store = _workspace_store() if workspace_settings.enabled else _document_store()
    if store is None:
        raise HTTPException(status_code=503, detail="document storage is not configured")
    try:
        deleted = (
            store.delete(identity.user_id, session_id, document_id)
            if workspace_settings.enabled
            else store.delete(session_id, document_id)
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="document not found")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="failed to delete document") from exc
    return {"status": "deleted", "document_id": document_id}


@router.get("/documents/{document_id}/status", response_model=DocumentMetadata)
def document_status(document_id: str, request: Request):
    if not workspace_settings.enabled:
        raise HTTPException(status_code=404, detail="Evidence Workspace is disabled")
    identity = _identity(request)
    metadata = _workspace_store().status(identity.user_id, document_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="document not found")
    return metadata


@router.get("/documents/{document_id}/view-url")
def document_view_url(document_id: str, request: Request):
    if not workspace_settings.enabled:
        raise HTTPException(status_code=404, detail="Evidence Workspace is disabled")
    identity = _identity(request)
    result = _workspace_store().view_url(identity.user_id, document_id)
    if not result:
        raise HTTPException(status_code=404, detail="document not ready")
    return result


@router.get("/documents/{document_id}/chunks/{chunk_id}/preview")
def document_chunk_preview(document_id: str, chunk_id: str, request: Request):
    if not workspace_settings.enabled:
        raise HTTPException(status_code=404, detail="Evidence Workspace is disabled")
    identity = _identity(request)
    result = _workspace_store().preview(identity.user_id, document_id, chunk_id)
    if not result:
        raise HTTPException(status_code=404, detail="evidence not found")
    return result


@router.get("/sessions")
def list_sessions(request: Request, limit: int = 100):
    """Retrieves list of all saved chat session metadata summaries."""
    identity = _identity(request)
    if workspace_settings.enabled:
        return _workspace_store().list_sessions(identity.user_id, min(limit, 100))
    if _redis is None:
        return []
    try:
        session_ids = _redis.lrange("chain_sessions_list", 0, limit - 1)
        sessions = []
        for s_id in session_ids:
            raw = _redis.get(f"session:{s_id}:metadata")
            if raw:
                sessions.append(json.loads(raw))
        return sessions
    except Exception as e:
        logger.warning(f"failed to list sessions: {e}")
        return []


@router.post("/session/{session_id}/metadata")
def update_session_metadata(session_id: str, meta: SessionMetadata, request: Request):
    """Updates or inserts the metadata summary of a session (e.g. title)."""
    identity = _identity(request)
    if workspace_settings.enabled:
        try:
            created_at = datetime.fromisoformat(meta.created_at.replace("Z", "+00:00"))
            _workspace_store().ensure_session(
                identity.user_id, session_id, meta.title, created_at
            )
            return {"status": "ok"}
        except (ValueError, DocumentError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if _redis is None:
        return {"status": "error", "message": "Redis storage is not configured"}
    try:
        _redis.set(
            f"session:{session_id}:metadata",
            meta.model_dump_json(),
            ex=context_settings.session_ttl_seconds,
        )
        session_ids = _redis.lrange("chain_sessions_list", 0, -1)
        if session_id not in session_ids:
            _redis.lpush("chain_sessions_list", session_id)
            _redis.ltrim("chain_sessions_list", 0, 99)
        return {"status": "ok"}
    except Exception as e:
        logger.warning(f"failed to save session metadata: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/session/{session_id}")
def get_session_messages(session_id: str, request: Request):
    """Retrieves all message turns logged for a given session ID."""
    identity = _identity(request)
    owner_id = identity.user_id if identity else None
    if identity and not _workspace_store().owns_session(owner_id, session_id):
        raise HTTPException(status_code=404, detail="session not found")
    context_session_id = _context_session_id(session_id, owner_id)
    if _redis is None:
        return []
    try:
        raw_msgs = _redis.lrange(f"session:{context_session_id}:messages", 0, -1)
        return [json.loads(raw) for raw in raw_msgs]
    except Exception as e:
        logger.warning(f"failed to fetch session messages: {e}")
        return []


@router.post("/session/{session_id}/message")
def append_session_message(session_id: str, message: SessionMessage, request: Request):
    """Logs an arbitrary chat message turn into a session's history log."""
    identity = _identity(request)
    owner_id = identity.user_id if identity else None
    if identity and not _workspace_store().owns_session(owner_id, session_id):
        raise HTTPException(status_code=404, detail="session not found")
    context_session_id = _context_session_id(session_id, owner_id)
    if _redis is None:
        return {"status": "error", "message": "Redis storage is not configured"}
    try:
        store = RedisContextStore(_redis, context_settings)
        store.append_display(context_session_id, message.model_dump())
        clean = ContextMessage.from_raw(message.model_dump())
        if clean:
            store.append(context_session_id, clean)
            _compact_context(store, context_session_id)
        return {"status": "ok"}
    except Exception as e:
        logger.warning(f"failed to append session message: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/trace/{request_id}")
def get_trace(request_id: str, request: Request):
    identity = _identity(request)
    if _redis is None:
        raise HTTPException(status_code=503, detail="trace storage not configured")
    raw = _redis.get(f"chain_trace:{request_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="trace not found")
    trace = json.loads(raw)
    if identity:
        session_id = trace.get("session_id")
        if not session_id or not _workspace_store().owns_session(identity.user_id, session_id):
            raise HTTPException(status_code=404, detail="trace not found")
    return trace


@router.get("/traces")
def list_traces(request: Request, limit: int = 20):
    """Retrieves metadata of the most recent traces recorded in Redis."""
    identity = _identity(request)
    if _redis is None:
        return []
    try:
        # Retrieve recent request IDs
        ids = _redis.lrange("chain_traces_list", 0, limit - 1)
        traces = []
        for request_id in ids:
            raw = _redis.get(f"chain_trace:{request_id}")
            if raw:
                trace_data = json.loads(raw)
                if identity:
                    session_id = trace_data.get("session_id")
                    if not session_id or not _workspace_store().owns_session(
                        identity.user_id, session_id
                    ):
                        continue
                # Keep payload light: return metadata summaries
                traces.append({
                    "request_id": trace_data.get("request_id"),
                    "goal": trace_data.get("goal"),
                    "start_time": trace_data.get("start_time"),
                    "total_latency_ms": trace_data.get("total_latency_ms"),
                    "satisfied": trace_data.get("verify", {}).get("satisfied", False),
                    "repair_rounds": trace_data.get("repair_rounds"),
                    "step_count": len(trace_data.get("results", [])),
                    "total_tokens": trace_data.get("total_tokens", 0),
                })
        return traces
    except Exception as e:
        logger.warning(f"failed to list traces: {e}")
        return []
