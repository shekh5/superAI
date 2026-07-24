"""
Structured data contracts for the reasoning chain.

Production note:
Everything passed between stages (decompose -> execute -> verify) is a
validated Pydantic model, never raw text. This is the single biggest
difference between a "prompt chain demo" and something you'd trust in
production: if the LLM's JSON doesn't match the schema, Pydantic raises
immediately at the boundary instead of a malformed value silently
propagating three steps downstream where it's much harder to debug.
"""

from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
BriefText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]


class ToolName(str, Enum):
    calculator = "calculator"
    get_time = "get_time"
    weather = "weather"
    web_search = "web_search"


class ActionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["action"]
    reason: BriefText
    tool: ToolName
    tool_input: dict


class FinalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["final"]
    satisfied: bool
    final_summary: NonEmptyText
    claims: list["AnswerClaim"] = Field(default_factory=list)


class AnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: BriefText
    evidence_chunk_ids: list[str] = Field(default_factory=list, max_length=8)


class EvidenceReference(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    document_type: str = ""
    location: str
    page: Optional[int] = None
    citation: str
    excerpt: str
    retrieval_score: float = 0.0


class VerifiedClaim(BaseModel):
    text: str
    status: Literal["supported", "evidence_missing"]
    support_score: float = 0.0
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: NonEmptyText


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: NonEmptyText


class GetTimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone_name: NonEmptyText = "UTC"


class WebSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=2, max_length=500),
    ]


class CorrectionType(str, Enum):
    empty_response = "empty_response"
    json_parse_error = "json_parse_error"
    schema_validation_error = "schema_validation_error"
    unknown_tool = "unknown_tool"
    invalid_tool_input = "invalid_tool_input"
    duplicate_failed_action = "duplicate_failed_action"
    unsupported_final_answer = "unsupported_final_answer"
    missing_citations = "missing_citations"


class CorrectionRecord(BaseModel):
    step_number: int
    attempt: int
    correction_type: CorrectionType
    validation_error: str
    successful: bool = False
    result_error: Optional[str] = None


class PlanStep(BaseModel):
    step_id: int
    tool: ToolName
    tool_input: dict = Field(default_factory=dict)
    reason: str = Field(..., description="Why this step is needed, for tracing/debugging")


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]


class ContextUsage(BaseModel):
    budget_tokens: int = 0
    used_tokens: int = 0
    utilization_percent: float = 0.0
    recent_messages_available: int = 0
    recent_messages_included: int = 0
    messages_dropped: int = 0
    high_priority_included: int = 0
    medium_priority_included: int = 0
    summary_included: bool = False
    compression_level: int = 0
    tool_results_compressed: int = 0
    document_context_included: bool = False
    document_chunks_included: int = 0


class ModelCall(BaseModel):
    stage: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    temperature: float = 0.0
    prompt_version: str = ""
    context_usage: ContextUsage = Field(default_factory=ContextUsage)


class StepResult(BaseModel):
    step_id: int
    tool: ToolName
    tool_input: dict
    output: Optional[str] = None
    success: bool
    error: Optional[str] = None
    attempt: int = 1
    latency_ms: float = 0.0
    api_calls: list[dict] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class VerifyResult(BaseModel):
    satisfied: bool
    missing: list[str] = Field(default_factory=list)
    repair_steps: list[PlanStep] = Field(default_factory=list)
    final_summary: str
    claims: list[VerifiedClaim] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class ChainTrace(BaseModel):
    """The full record of one run. This is what you'd log to Redis / your
    observability tool -- one document per request_id, every stage included."""

    request_id: str
    goal: str
    plan: Plan
    results: list[StepResult]
    verify: VerifyResult
    repair_rounds: int
    start_time: str = ""
    end_time: str = ""
    total_latency_ms: float = 0.0
    llm_calls: list[ModelCall] = Field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    temperature: float = 0.1
    context_usage: ContextUsage = Field(default_factory=ContextUsage)
    corrections: list[CorrectionRecord] = Field(default_factory=list)
    total_corrections: int = 0
    session_id: Optional[str] = None
    document_ids: list[str] = Field(default_factory=list)


class SessionMessage(BaseModel):
    sender: str
    text: str
    timestamp: str
    trace: Optional[dict] = None


class SessionMetadata(BaseModel):
    id: str
    title: str
    created_at: str
