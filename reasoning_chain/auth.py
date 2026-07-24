"""Invite-only application accounts and Redis-backed browser sessions."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256

import redis
from argon2 import PasswordHasher
from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import select

from .workspace_config import workspace_settings
from .workspace_db import User, db_session

router = APIRouter(prefix="/auth", tags=["authentication"])
admin_router = APIRouter(prefix="/admin", tags=["administration"])
_passwords = PasswordHasher()
_sessions = redis.Redis.from_url(
    os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
)


@dataclass(frozen=True)
class Identity:
    user_id: str
    email: str
    role: str
    must_change_password: bool = False


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str = Field(min_length=8, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)


class CreateUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: EmailStr
    temporary_password: str = Field(min_length=12, max_length=256)
    role: str = Field(default="member", pattern="^(member|admin)$")


def _session_key(token: str) -> str:
    return f"workspace:auth:{token}"


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        workspace_settings.session_cookie_name,
        token,
        max_age=workspace_settings.session_ttl_seconds,
        httponly=True,
        secure=os.environ.get("COOKIE_SECURE", "true").lower() != "false",
        samesite="lax",
        path="/",
    )


def _create_session(response: Response, user: User) -> None:
    token = secrets.token_urlsafe(32)
    _sessions.hset(
        _session_key(token),
        mapping={
            "user_id": user.id,
            "email": user.email,
            "role": user.role,
            "must_change_password": "1" if user.must_change_password else "0",
        },
    )
    _sessions.expire(_session_key(token), workspace_settings.session_ttl_seconds)
    _set_cookie(response, token)


def optional_identity(request: Request) -> Identity | None:
    if not workspace_settings.auth_required:
        return None
    token = request.cookies.get(workspace_settings.session_cookie_name)
    if not token:
        raise HTTPException(status_code=401, detail="authentication required")
    values = _sessions.hgetall(_session_key(token))
    if not values:
        raise HTTPException(status_code=401, detail="session expired")
    _sessions.expire(_session_key(token), workspace_settings.session_ttl_seconds)
    return Identity(
        user_id=values["user_id"],
        email=values["email"],
        role=values["role"],
        must_change_password=values.get("must_change_password") == "1",
    )


def require_identity(request: Request) -> Identity:
    identity = optional_identity(request)
    if identity is None:
        raise HTTPException(status_code=401, detail="authentication required")
    if identity.must_change_password and request.url.path != "/auth/change-password":
        raise HTTPException(status_code=403, detail="password change required")
    return identity


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response):
    client = request.client.host if request.client else "unknown"
    email_fingerprint = sha256(payload.email.lower().encode()).hexdigest()[:16]
    attempt_key = f"workspace:login-attempts:{client}:{email_fingerprint}"
    if int(_sessions.get(attempt_key) or 0) >= 5:
        raise HTTPException(status_code=429, detail="too many login attempts; try again later")
    with db_session() as session:
        user = session.scalar(select(User).where(User.email == payload.email.lower()))
        valid = False
        if user and user.is_active:
            try:
                valid = _passwords.verify(user.password_hash, payload.password)
            except Exception:
                valid = False
        if not valid:
            pipe = _sessions.pipeline()
            pipe.incr(attempt_key)
            pipe.expire(attempt_key, 900)
            pipe.execute()
            raise HTTPException(status_code=401, detail="invalid email or password")
        _sessions.delete(attempt_key)
        _create_session(response, user)
        return {
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "must_change_password": user.must_change_password,
        }


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response):
    token = request.cookies.get(workspace_settings.session_cookie_name)
    if token:
        _sessions.delete(_session_key(token))
    response.delete_cookie(workspace_settings.session_cookie_name, path="/")


@router.get("/me")
def me(request: Request):
    identity = require_identity(request)
    return identity.__dict__


@router.post("/change-password")
def change_password(payload: ChangePasswordRequest, request: Request, response: Response):
    identity = optional_identity(request)
    if identity is None:
        raise HTTPException(status_code=401, detail="authentication required")
    with db_session() as session:
        user = session.get(User, identity.user_id)
        try:
            current_valid = bool(user) and _passwords.verify(
                user.password_hash, payload.current_password
            )
        except Exception:
            current_valid = False
        if not current_valid:
            raise HTTPException(status_code=400, detail="current password is incorrect")
        user.password_hash = _passwords.hash(payload.new_password)
        user.must_change_password = False
        token = request.cookies.get(workspace_settings.session_cookie_name)
        if token:
            _sessions.delete(_session_key(token))
        _create_session(response, user)
    return {"status": "ok"}


@admin_router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(payload: CreateUserRequest, request: Request):
    identity = require_identity(request)
    if identity.role != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    with db_session() as session:
        existing = session.scalar(select(User).where(User.email == payload.email.lower()))
        if existing:
            raise HTTPException(status_code=409, detail="account already exists")
        user = User(
            email=payload.email.lower(),
            password_hash=_passwords.hash(payload.temporary_password),
            role=payload.role,
            must_change_password=True,
        )
        session.add(user)
        session.flush()
        return {"id": user.id, "email": user.email, "role": user.role}


def bootstrap_admin() -> None:
    """Idempotently create the first administrator from deployment secrets."""
    email = os.environ.get("BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()
    password = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD", "")
    if not email or not password:
        return
    with db_session() as session:
        if session.scalar(select(User.id).limit(1)):
            return
        session.add(
            User(
                email=email,
                password_hash=_passwords.hash(password),
                role="admin",
                must_change_password=True,
                created_at=datetime.now(timezone.utc),
            )
        )
