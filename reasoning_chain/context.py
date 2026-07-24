"""Conversation context selection and Redis-backed rolling memory."""

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from html import escape
from typing import Callable, Literal, Optional

from .context_compression import compress_text, normalize_whitespace

try:
    from redis.exceptions import WatchError
except ImportError:  # pragma: no cover - Redis is an application dependency
    WatchError = RuntimeError

Role = Literal["user", "model"]
TokenCounter = Callable[[list[dict], str], int]
Summarizer = Callable[[str, list["ContextMessage"]], str]


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ContextSettings:
    input_token_budget: int = field(
        default_factory=lambda: _env_int("CONTEXT_INPUT_TOKEN_BUDGET", 24_000)
    )
    output_token_reserve: int = field(
        default_factory=lambda: _env_int("CONTEXT_OUTPUT_TOKEN_RESERVE", 1_000)
    )
    token_safety_margin: int = field(
        default_factory=lambda: _env_int("CONTEXT_TOKEN_SAFETY_MARGIN", 1_000)
    )
    recent_messages: int = field(
        default_factory=lambda: _env_int("CONTEXT_RECENT_MESSAGES", 16)
    )
    recent_high_watermark: int = field(
        default_factory=lambda: _env_int("CONTEXT_RECENT_HIGH_WATERMARK", 20)
    )
    high_priority_messages: int = field(
        default_factory=lambda: _env_int("CONTEXT_HIGH_PRIORITY_MESSAGES", 4)
    )
    light_compression_ratio: float = field(
        default_factory=lambda: _env_float("CONTEXT_LIGHT_COMPRESSION_RATIO", 0.60)
    )
    strong_compression_ratio: float = field(
        default_factory=lambda: _env_float("CONTEXT_STRONG_COMPRESSION_RATIO", 0.80)
    )
    critical_compression_ratio: float = field(
        default_factory=lambda: _env_float("CONTEXT_CRITICAL_RATIO", 0.90)
    )
    tool_output_max_tokens: int = field(
        default_factory=lambda: _env_int("CONTEXT_TOOL_OUTPUT_MAX_TOKENS", 2_000)
    )
    session_max_messages: int = field(
        default_factory=lambda: _env_int("SESSION_MAX_MESSAGES", 200)
    )
    session_ttl_seconds: int = field(
        default_factory=lambda: _env_int("SESSION_TTL_SECONDS", 2_592_000)
    )

    @property
    def usable_input_tokens(self) -> int:
        reserved = self.output_token_reserve + self.token_safety_margin
        return max(1, self.input_token_budget - reserved)

    @property
    def compaction_threshold(self) -> int:
        return max(self.recent_high_watermark, self.recent_messages)

    @property
    def compression_thresholds(self) -> tuple[float, float, float]:
        light = self.light_compression_ratio
        strong = max(light, self.strong_compression_ratio)
        critical = max(strong, self.critical_compression_ratio)
        return light, strong, critical


class ContextPriority(IntEnum):
    LOW = 25
    MEDIUM = 50
    HIGH = 75
    CRITICAL = 100


@dataclass(frozen=True)
class ContextUsageStats:
    budget_tokens: int
    used_tokens: int
    utilization_percent: float
    recent_messages_available: int
    recent_messages_included: int
    messages_dropped: int
    high_priority_included: int
    medium_priority_included: int
    summary_included: bool
    compression_level: int
    tool_results_compressed: int = 0
    document_context_included: bool = False
    document_chunks_included: int = 0


@dataclass(frozen=True)
class ContextSelection:
    contents: list[dict]
    usage: ContextUsageStats


@dataclass(frozen=True)
class ContextMessage:
    role: Role
    text: str
    timestamp: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {"role": self.role, "text": self.text, "timestamp": self.timestamp},
            separators=(",", ":"),
        )

    @classmethod
    def from_raw(cls, raw: str | dict) -> Optional["ContextMessage"]:
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            sender = str(data.get("role", data.get("sender", ""))).lower()
            role: Role = "model" if sender in {"agent", "assistant", "model"} else "user"
            if sender not in {"agent", "assistant", "model", "user"}:
                return None
            text = str(data.get("text", "")).strip()
            if not text:
                return None
            return cls(role=role, text=text, timestamp=str(data.get("timestamp", "")))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return None


@dataclass(frozen=True)
class ContextBundle:
    summary: str = ""
    recent: list[ContextMessage] = field(default_factory=list)
    document_context: str = ""
    document_citations: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    document_chunks: int = 0
    document_passages: list[dict] = field(default_factory=list)


def estimate_tokens(contents: list[dict], system_instruction: str) -> int:
    """Conservative local fallback when Gemini token counting is unavailable."""
    serialized = system_instruction + json.dumps(contents, ensure_ascii=False)
    return max(1, math.ceil(len(serialized) / 3.5))


def _content(role: Role, text: str) -> dict:
    return {"role": role, "parts": [{"text": text}]}


def _summary_text(summary: str, compression_level: int, budget_tokens: int) -> str:
    if compression_level >= 2:
        summary = compress_text(summary, max_tokens=max(128, budget_tokens // 8))
    return (
        '<conversation_summary trust="untrusted" priority="high">\n'
        f"{escape(summary, quote=False)}\n"
        "</conversation_summary>"
    )


def _compression_level(utilization: float, settings: ContextSettings) -> int:
    light, strong, critical = settings.compression_thresholds
    if utilization >= critical:
        return 3
    if utilization >= strong:
        return 2
    if utilization >= light:
        return 1
    return 0


def _safe_count(counter: TokenCounter, contents: list[dict], system_instruction: str) -> int:
    try:
        count = counter(contents, system_instruction)
        if isinstance(count, int) and count > 0:
            return count
    except Exception:
        pass
    return estimate_tokens(contents, system_instruction)


def select_budgeted_contents(
    bundle: Optional[ContextBundle],
    current_prompt: str,
    system_instruction: str,
    settings: Optional[ContextSettings] = None,
    token_counter: Optional[TokenCounter] = None,
) -> ContextSelection:
    """Select context by priority and adapt compression to context-window utilization."""
    settings = settings or ContextSettings()
    counter = token_counter or estimate_tokens
    bundle = bundle or ContextBundle()
    budget = settings.usable_input_tokens
    current = _content("user", current_prompt)
    total_recent_available = len(bundle.recent)
    available = bundle.recent[-settings.recent_messages :]
    full_contents = [_content(message.role, message.text) for message in available]
    if bundle.summary:
        full_contents.insert(0, _content("user", _summary_text(bundle.summary, 0, budget)))
    if bundle.document_context:
        full_contents.append(_content("user", bundle.document_context))
    full_contents.append(current)
    full_tokens = _safe_count(counter, full_contents, system_instruction)
    level = _compression_level(full_tokens / budget, settings)

    if level == 0:
        recent_limit = settings.recent_messages
    elif level == 1:
        recent_limit = min(settings.recent_messages, 12)
    elif level == 2:
        recent_limit = min(settings.recent_messages, 8)
    else:
        recent_limit = min(settings.recent_messages, settings.high_priority_messages)

    selected = list(available[-recent_limit:])
    high_count = min(settings.high_priority_messages, len(selected))
    priorities = [ContextPriority.MEDIUM] * (len(selected) - high_count)
    priorities.extend([ContextPriority.HIGH] * high_count)
    medium_count = priorities.count(ContextPriority.MEDIUM)
    if level >= 1 and medium_count:
        selected = [
            ContextMessage(
                role=message.role,
                text=normalize_whitespace(message.text),
                timestamp=message.timestamp,
            )
            if index < medium_count
            else message
            for index, message in enumerate(selected)
        ]

    summary = _summary_text(bundle.summary, level, budget) if bundle.summary else ""
    document_context = bundle.document_context

    def assemble() -> list[dict]:
        contents = [_content(message.role, message.text) for message in selected]
        if summary:
            contents.insert(0, _content("user", summary))
        if document_context:
            contents.append(_content("user", document_context))
        contents.append(current)
        return contents

    contents = assemble()
    used_tokens = _safe_count(counter, contents, system_instruction)
    while used_tokens > budget and medium_count > 0:
        selected.pop(0)
        priorities.pop(0)
        medium_count -= 1
        contents = assemble()
        used_tokens = _safe_count(counter, contents, system_instruction)

    if used_tokens > budget and summary:
        summary = ""
        contents = assemble()
        used_tokens = _safe_count(counter, contents, system_instruction)

    while used_tokens > budget and selected:
        selected.pop(0)
        priorities.pop(0)
        contents = assemble()
        used_tokens = _safe_count(counter, contents, system_instruction)

    if used_tokens > budget and document_context:
        document_context = ""
        contents = assemble()
        used_tokens = _safe_count(counter, contents, system_instruction)

    included = len(selected)
    usage = ContextUsageStats(
        budget_tokens=budget,
        used_tokens=used_tokens,
        utilization_percent=round(used_tokens / budget * 100, 2),
        recent_messages_available=total_recent_available,
        recent_messages_included=included,
        messages_dropped=total_recent_available - included,
        high_priority_included=priorities.count(ContextPriority.HIGH),
        medium_priority_included=priorities.count(ContextPriority.MEDIUM),
        summary_included=bool(summary),
        compression_level=level,
        document_context_included=bool(document_context),
        document_chunks_included=bundle.document_chunks if document_context else 0,
    )
    return ContextSelection(contents=contents, usage=usage)


def build_budgeted_contents(
    bundle: Optional[ContextBundle],
    current_prompt: str,
    system_instruction: str,
    settings: Optional[ContextSettings] = None,
    token_counter: Optional[TokenCounter] = None,
) -> list[dict]:
    """Backward-compatible wrapper for callers that need only Gemini contents."""
    return select_budgeted_contents(
        bundle,
        current_prompt,
        system_instruction,
        settings=settings,
        token_counter=token_counter,
    ).contents


class RedisContextStore:
    """Stores clean model context independently from rich UI/trace records."""

    def __init__(self, client, settings: Optional[ContextSettings] = None):
        self.client = client
        self.settings = settings or ContextSettings()

    @staticmethod
    def _keys(session_id: str) -> dict[str, str]:
        prefix = f"session:{session_id}"
        return {
            "display": f"{prefix}:messages",
            "recent": f"{prefix}:context:recent",
            "summary": f"{prefix}:context:summary",
            "meta": f"{prefix}:context:meta",
            "metadata": f"{prefix}:metadata",
        }

    def load(self, session_id: str) -> ContextBundle:
        keys = self._keys(session_id)
        recent_exists = self.client.exists(keys["recent"])
        meta_exists = self.client.exists(keys["meta"])
        if self._exists(recent_exists) or self._exists(meta_exists):
            raw_messages = self.client.lrange(keys["recent"], 0, -1)
        else:
            raw_messages = self._migrate_legacy_display_history(keys)
        messages = [ContextMessage.from_raw(raw) for raw in raw_messages]
        summary = self.client.get(keys["summary"]) or ""
        return ContextBundle(summary=str(summary), recent=[msg for msg in messages if msg])

    @staticmethod
    def _exists(value) -> bool:
        return value is True or isinstance(value, int) and value > 0

    def _migrate_legacy_display_history(self, keys: dict[str, str]) -> list[str]:
        legacy = self.client.lrange(keys["display"], 0, -1)
        clean = [message for raw in legacy if (message := ContextMessage.from_raw(raw))]
        pipe = self.client.pipeline()
        if clean:
            pipe.rpush(keys["recent"], *(message.to_json() for message in clean))
        pipe.set(
            keys["meta"],
            json.dumps({"version": 1, "migrated": True}),
            ex=self.settings.session_ttl_seconds,
        )
        pipe.expire(keys["recent"], self.settings.session_ttl_seconds)
        pipe.execute()
        return [message.to_json() for message in clean]

    def append(self, session_id: str, message: ContextMessage) -> None:
        keys = self._keys(session_id)
        pipe = self.client.pipeline()
        pipe.rpush(keys["recent"], message.to_json())
        pipe.expire(keys["recent"], self.settings.session_ttl_seconds)
        pipe.expire(keys["summary"], self.settings.session_ttl_seconds)
        pipe.execute()

    def append_display(self, session_id: str, *records: dict) -> None:
        keys = self._keys(session_id)
        pipe = self.client.pipeline()
        pipe.rpush(keys["display"], *(json.dumps(record) for record in records))
        pipe.ltrim(keys["display"], -self.settings.session_max_messages, -1)
        pipe.expire(keys["display"], self.settings.session_ttl_seconds)
        pipe.execute()

    def compact(self, session_id: str, summarizer: Summarizer) -> bool:
        """Summarize old messages. Leave Redis unchanged if generation fails."""
        keys = self._keys(session_id)
        snapshot = self.client.lrange(keys["recent"], 0, -1)
        if len(snapshot) <= self.settings.compaction_threshold:
            return False
        existing_summary = self.client.get(keys["summary"]) or ""
        keep_count = min(self.settings.recent_messages, len(snapshot))
        old_raw, retained = snapshot[:-keep_count], snapshot[-keep_count:]
        old_messages = [ContextMessage.from_raw(raw) for raw in old_raw]
        old_messages = [message for message in old_messages if message]
        if not old_messages:
            return False
        new_summary = summarizer(str(existing_summary), old_messages).strip()
        if not new_summary:
            return False

        # Optimistic concurrency: compact only if no request appended during summarization.
        pipe = self.client.pipeline()
        try:
            pipe.watch(keys["recent"], keys["summary"])
            current_summary = pipe.get(keys["summary"]) or ""
            if pipe.lrange(keys["recent"], 0, -1) != snapshot:
                pipe.unwatch()
                return False
            if str(current_summary) != str(existing_summary):
                pipe.unwatch()
                return False
            pipe.multi()
            pipe.delete(keys["recent"])
            if retained:
                pipe.rpush(keys["recent"], *retained)
            pipe.set(keys["summary"], new_summary, ex=self.settings.session_ttl_seconds)
            pipe.set(
                keys["meta"],
                json.dumps(
                    {
                        "version": 1,
                        "summarized_at": datetime.now(timezone.utc).isoformat(),
                        "summarized_messages": len(old_messages),
                    }
                ),
                ex=self.settings.session_ttl_seconds,
            )
            pipe.expire(keys["recent"], self.settings.session_ttl_seconds)
            pipe.execute()
            return True
        except WatchError:
            return False
        finally:
            reset = getattr(pipe, "reset", None)
            if reset:
                reset()
