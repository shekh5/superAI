"""Configuration for the optional Evidence Workspace production stack."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class WorkspaceSettings:
    enabled: bool = field(default_factory=lambda: _bool("EVIDENCE_WORKSPACE_ENABLED"))
    auth_required: bool = field(
        default_factory=lambda: _bool(
            "AUTH_REQUIRED", _bool("EVIDENCE_WORKSPACE_ENABLED")
        )
    )
    database_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", ""))
    s3_bucket: str = field(default_factory=lambda: os.environ.get("S3_BUCKET", ""))
    aws_region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "ap-south-1"))
    s3_endpoint_url: str = field(default_factory=lambda: os.environ.get("S3_ENDPOINT_URL", ""))
    session_cookie_name: str = "superai_session"
    session_ttl_seconds: int = 60 * 60 * 12
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
        )
    )
    embedding_dimensions: int = 384
    evidence_threshold: float = field(
        default_factory=lambda: _float("EVIDENCE_SUPPORT_THRESHOLD", 0.70)
    )
    retrieval_chunks: int = 8
    semantic_weight: float = 0.60
    lexical_weight: float = 0.40
    presigned_url_seconds: int = 300

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.auth_required:
            raise RuntimeError("Evidence Workspace requires AUTH_REQUIRED=true")
        missing = [
            name
            for name, value in (
                ("DATABASE_URL", self.database_url),
                ("S3_BUCKET", self.s3_bucket),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Evidence Workspace is enabled but configuration is missing: "
                + ", ".join(missing)
            )


workspace_settings = WorkspaceSettings()
