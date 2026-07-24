"""
Reasoning orchestration with a bounded ReAct loop and reusable planning helpers.

Production note:
The active run_chain() flow asks the model for one action at a time, executes
that action with bounded retries, and stops after eight actions. The standalone
decompose_goal() and verify_and_repair() helpers support plan inspection and
experimentation but are not stages in run_chain().
No step trusts the previous step's output blindly -- every boundary is a
Pydantic model (see schemas.py), and every stage is logged with a
request_id so a failed run can be replayed/debugged like a CI run.
"""

import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from .context import (
    ContextBundle,
    ContextMessage,
    ContextSettings,
    estimate_tokens,
    select_budgeted_contents,
)
from .context_compression import compress_step_results
from .decisions import (
    DecisionValidationError,
    action_fingerprint,
    extract_json,
    parse_agent_decision,
)
from .prompts import (
    DECOMPOSE_PROMPT_VERSION,
    REACT_PROMPT_VERSION,
    SUMMARY_SYSTEM_PROMPT,
    VERIFY_PROMPT_VERSION,
    build_correction_request,
    build_decompose_system_prompt,
    build_goal_request,
    build_react_request,
    build_react_system_prompt,
    build_summary_request,
    build_verify_request,
    build_verify_system_prompt,
)
from .schemas import (
    ActionDecision,
    ChainTrace,
    ContextUsage,
    CorrectionRecord,
    CorrectionType,
    FinalDecision,
    ModelCall,
    Plan,
    PlanStep,
    StepResult,
    ToolName,
    VerifyResult,
)
from .tools import TOOL_REGISTRY, ToolError
from .workspace_config import workspace_settings

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_STEPS = 6
MAX_REPAIR_ROUNDS = 1
MAX_TOOL_RETRIES = 1
MAX_DECISION_CORRECTIONS = 2
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0
WEB_SOURCE_PATTERN = re.compile(r"\]\((https?://[^)\s]+)\)")


def _temperature_from_env(name: str, default: float) -> float:
    """Read a temperature without allowing bad deployment config to break startup."""
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(MIN_TEMPERATURE, min(value, MAX_TEMPERATURE))


REACT_TEMPERATURE = _temperature_from_env("REACT_TEMPERATURE", 0.1)
SUMMARY_TEMPERATURE = _temperature_from_env("SUMMARY_TEMPERATURE", 0.2)

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None  # type: ignore
    types = None  # type: ignore

_client = None


def _get_client():
    global _client
    if _client is None:
        if genai is None:
            raise RuntimeError(
                "google-genai is not installed; install dependencies or patch "
                "decompose_goal/verify_and_repair in tests"
            )
        _client = genai.Client()
    return _client


def _count_tokens(contents: list[dict], system_instruction: str) -> int:
    """Use Gemini's tokenizer when available and a conservative local fallback otherwise."""
    try:
        response = _get_client().models.count_tokens(
            model=MODEL,
            contents=contents,
            config=types.CountTokensConfig(system_instruction=system_instruction),
        )
        count = getattr(response, "total_tokens", None)
        if isinstance(count, int) and count > 0:
            return count
    except Exception:
        pass
    return estimate_tokens(contents, system_instruction)


def summarize_context(existing_summary: str, messages: list[ContextMessage]) -> str:
    """Merge older conversation turns into compact, instruction-safe memory."""
    payload = build_summary_request(
        existing_summary,
        [{"role": message.role, "text": message.text} for message in messages],
    )
    response = _get_client().models.generate_content(
        model=MODEL,
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=SUMMARY_SYSTEM_PROMPT,
            max_output_tokens=600,
            temperature=SUMMARY_TEMPERATURE,
        ),
    )
    return (response.text or "").strip()


def _extract_json(text: str) -> dict:
    """Backward-compatible JSON extraction wrapper for non-ReAct helper stages."""
    return extract_json(text)


def _validate_runtime_decision(
    decision: ActionDecision | FinalDecision,
    results: list[StepResult],
    failed_action_fingerprints: set[str],
    document_citations: Optional[list[str]] = None,
) -> None:
    if isinstance(decision, ActionDecision):
        if action_fingerprint(decision) in failed_action_fingerprints:
            raise DecisionValidationError(
                CorrectionType.duplicate_failed_action,
                "The same tool action already failed. Correct its input, choose another "
                "registered action, or return an honest final response.",
            )
        return
    if decision.satisfied and results and not any(result.success for result in results):
        raise DecisionValidationError(
            CorrectionType.unsupported_final_answer,
            "All executed tools failed, so satisfied=true is not supported by a successful "
            "tool result.",
        )
    web_sources = {
        url
        for result in results
        if result.success and result.tool == ToolName.web_search and result.output
        for url in WEB_SOURCE_PATTERN.findall(result.output)
    }
    if decision.satisfied and web_sources and not any(
        source in decision.final_summary for source in web_sources
    ):
        raise DecisionValidationError(
            CorrectionType.missing_citations,
            "The final answer uses a successful web search but does not preserve any returned "
            "source URL. Add at least one exact source link from the web_search output.",
        )
    document_citations = document_citations or []
    if decision.satisfied and document_citations and not any(
        citation in decision.final_summary for citation in document_citations
    ):
        raise DecisionValidationError(
            CorrectionType.missing_citations,
            "The answer uses retrieved document passages but does not preserve a returned "
            "citation. Add at least one exact [filename, locator] citation from document_context.",
        )
    if (
        decision.satisfied
        and document_citations
        and workspace_settings.enabled
        and (
            not decision.claims
            or not any(claim.evidence_chunk_ids for claim in decision.claims)
        )
    ):
        raise DecisionValidationError(
            CorrectionType.missing_citations,
            "Evidence Workspace document answers require atomic claims with supplied "
            "evidence_chunk_ids.",
        )


def decompose_goal(goal: str, model_calls: list = None) -> Plan:
    """Stage 1: ask the model to break the goal into tool calls, returned
    as structured JSON only -- not prose."""
    system = build_decompose_system_prompt(MAX_STEPS)
    user_prompt = build_goal_request(goal)
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1000,
            temperature=REACT_TEMPERATURE,
        ),
    )
    raw = resp.text
    if model_calls is not None:
        p_tok = 0
        c_tok = 0
        t_tok = 0
        if resp.usage_metadata:
            p_tok = resp.usage_metadata.prompt_token_count or 0
            c_tok = resp.usage_metadata.candidates_token_count or 0
            t_tok = resp.usage_metadata.total_token_count or 0
        model_calls.append(
            ModelCall(
                stage="decompose",
                system_prompt=system,
                user_prompt=user_prompt,
                raw_response=raw,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=t_tok,
                temperature=REACT_TEMPERATURE,
                prompt_version=DECOMPOSE_PROMPT_VERSION,
            )
        )
    data = _extract_json(raw)
    return Plan.model_validate(data)


def _resolve_references(tool_input: dict, results_history: list[StepResult]) -> dict:
    """
    Scans values in tool_input for step reference patterns like [id] (e.g. '[1]') 
    and replaces them with the parsed numeric output of that step from results_history.
    """
    resolved = {}
    for k, v in tool_input.items():
        if isinstance(v, str):
            def replacer(match):
                ref_id = int(match.group(1))
                for res in results_history:
                    if res.step_id == ref_id and res.success and res.output:
                        # Extract all numbers from the target step output
                        nums = re.findall(r"[-+]?\d*\.\d+|\d+", res.output)
                        if nums:
                            return nums[-1]  # Take the last parsed number
                return match.group(0)
            
            resolved[k] = re.sub(r"\[(\d+)\]", replacer, v)
        else:
            resolved[k] = v
    return resolved


def execute_plan(
    plan: Plan,
    results_history: Optional[list[StepResult]] = None,
    disabled_tools: Optional[set[str]] = None,
) -> list[StepResult]:
    """Stage 2: run each step against real tools with retry + circuit
    breaker. A tool that fails twice in a row gets skipped for the rest
    of the run instead of retried forever."""
    results: list[StepResult] = []
    prior_results = results_history or []
    disabled_tools = disabled_tools if disabled_tools is not None else set()

    for step in plan.steps[:MAX_STEPS]:
        if step.tool in disabled_tools:
            results.append(
                StepResult(
                    step_id=step.step_id,
                    tool=step.tool,
                    tool_input=step.tool_input,
                    success=False,
                    error="skipped: tool disabled after repeated failures",
                )
            )
            continue

        resolved_input = _resolve_references(step.tool_input, prior_results + results)
        fn = TOOL_REGISTRY[step.tool]
        last_error = None
        last_api_calls = []
        step_latency = 0.0

        # A grounded search can be billable. Do not repeat it automatically and surprise the
        # operator with duplicate calls; the agent may issue a corrected query explicitly.
        max_attempts = 1 if step.tool == ToolName.web_search else MAX_TOOL_RETRIES + 1
        for attempt in range(1, max_attempts + 1):
            step_start = time.perf_counter()
            try:
                res = fn(**resolved_input)
                step_latency = (time.perf_counter() - step_start) * 1000
                if isinstance(res, tuple):
                    output, api_calls = res
                else:
                    output, api_calls = res, []

                results.append(
                    StepResult(
                        step_id=step.step_id,
                        tool=step.tool,
                        tool_input=resolved_input,
                        output=output,
                        success=True,
                        attempt=attempt,
                        latency_ms=round(step_latency, 2),
                        api_calls=api_calls,
                    )
                )
                break
            except ToolError as e:
                step_latency = (time.perf_counter() - step_start) * 1000
                last_error = str(e)
                last_api_calls = getattr(e, "api_calls", [])
            except TypeError as e:
                step_latency = (time.perf_counter() - step_start) * 1000
                last_error = f"bad tool_input: {e}"
                last_api_calls = []
                results.append(
                    StepResult(
                        step_id=step.step_id,
                        tool=step.tool,
                        tool_input=resolved_input,
                        success=False,
                        error=last_error,
                        attempt=attempt,
                        latency_ms=round(step_latency, 2),
                    )
                )
                break
        else:
            results.append(
                StepResult(
                    step_id=step.step_id,
                    tool=step.tool,
                    tool_input=resolved_input,
                    success=False,
                    error=last_error,
                    attempt=max_attempts,
                    latency_ms=round(step_latency, 2),
                    api_calls=last_api_calls,
                )
            )
            disabled_tools.add(step.tool)

    return results


def verify_and_repair(
    goal: str, plan: Plan, results: list[StepResult], model_calls: list = None
) -> VerifyResult:
    """Stage 3: ask the model to check tool outputs against the goal. It
    can either declare victory, ask for a small number of repair steps,
    or -- critically -- admit what it couldn't determine rather than
    hallucinating a confident-sounding answer."""
    system = build_verify_system_prompt()
    payload = build_verify_request(goal, [r.model_dump() for r in results])
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1000,
            temperature=REACT_TEMPERATURE,
        ),
    )
    raw = resp.text
    if model_calls is not None:
        model_calls.append(
            ModelCall(
                stage="verify",
                system_prompt=system,
                user_prompt=payload,
                raw_response=raw,
                temperature=REACT_TEMPERATURE,
                prompt_version=VERIFY_PROMPT_VERSION,
            )
        )
    data = _extract_json(raw)
    return VerifyResult.model_validate(data)


def run_chain(
    goal: str,
    conversation: Optional[ContextBundle] = None,
    temperature: Optional[float] = None,
) -> ChainTrace:
    """The full orchestrator using a step-by-step ReAct loop."""
    effective_temperature = REACT_TEMPERATURE if temperature is None else temperature
    effective_temperature = max(MIN_TEMPERATURE, min(effective_temperature, MAX_TEMPERATURE))
    start_time = datetime.now(timezone.utc).isoformat()
    chain_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    model_calls = []
    results = []
    plan_steps = []
    disabled_tools: set[str] = set()
    failed_action_fingerprints: set[str] = set()
    corrections: list[CorrectionRecord] = []

    satisfied = False
    final_summary = ""
    final_claims = []
    missing_reason = "Goal not fully met or stopped prematurely"
    step_count = 0
    max_steps = 8
    context_settings = ContextSettings()

    system = build_react_system_prompt()

    while step_count < max_steps:
        # Build history payload for this step
        steps_taken = [
            {
                "step_id": r.step_id,
                "tool": r.tool,
                "tool_input": r.tool_input,
                "output": r.output,
                "success": r.success,
                "error": r.error,
            }
            for r in results
        ]
        compact_steps, compressed_fields = compress_step_results(
            steps_taken,
            max_tokens=context_settings.tool_output_max_tokens,
        )
        base_prompt = build_react_request(goal, compact_steps)
        correction_prompt = ""
        active_correction: Optional[CorrectionRecord] = None
        decision: Optional[ActionDecision | FinalDecision] = None
        p_tok = c_tok = t_tok = 0

        for correction_attempt in range(MAX_DECISION_CORRECTIONS + 1):
            current_prompt = base_prompt
            if correction_prompt:
                current_prompt = f"{base_prompt}\n{correction_prompt}"
            selection = select_budgeted_contents(
                conversation,
                current_prompt,
                system,
                settings=context_settings,
                token_counter=_count_tokens,
            )
            context_usage = ContextUsage(
                **{
                    **selection.usage.__dict__,
                    "tool_results_compressed": compressed_fields,
                }
            )
            response = _get_client().models.generate_content(
                model=MODEL,
                contents=selection.contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=1000,
                    temperature=effective_temperature,
                ),
            )
            raw = response.text or ""
            p_tok = c_tok = t_tok = 0
            if response.usage_metadata:
                p_tok = response.usage_metadata.prompt_token_count or 0
                c_tok = response.usage_metadata.candidates_token_count or 0
                t_tok = response.usage_metadata.total_token_count or 0

            stage = f"step_{step_count + 1}"
            if correction_attempt:
                stage += f"_correction_{correction_attempt}"
            model_calls.append(
                ModelCall(
                    stage=stage,
                    system_prompt=system,
                    user_prompt=current_prompt,
                    raw_response=raw,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=t_tok,
                    temperature=effective_temperature,
                    prompt_version=REACT_PROMPT_VERSION,
                    context_usage=context_usage,
                )
            )

            try:
                candidate = parse_agent_decision(raw)
                _validate_runtime_decision(
                    candidate,
                    results,
                    failed_action_fingerprints,
                    conversation.document_citations
                    if conversation and selection.usage.document_context_included
                    else None,
                )
            except DecisionValidationError as error:
                if active_correction is not None:
                    active_correction.result_error = error.message
                if correction_attempt >= MAX_DECISION_CORRECTIONS:
                    break
                active_correction = CorrectionRecord(
                    step_number=step_count + 1,
                    attempt=correction_attempt + 1,
                    correction_type=error.correction_type,
                    validation_error=error.message,
                )
                corrections.append(active_correction)
                correction_prompt = build_correction_request(
                    raw,
                    error.correction_type.value,
                    error.message,
                )
                continue

            if active_correction is not None:
                active_correction.successful = True
            decision = candidate
            break

        if decision is None:
            final_summary = (
                "The agent could not produce a valid decision after "
                f"{MAX_DECISION_CORRECTIONS} correction attempts."
            )
            missing_reason = "Valid agent decision"
            break

        if isinstance(decision, FinalDecision):
            satisfied = decision.satisfied
            final_summary = decision.final_summary
            final_claims = decision.claims
            if not satisfied:
                missing_reason = "Agent reported that the goal could not be fully satisfied"
            break

        step_id = step_count + 1
        plan_step = PlanStep(
            step_id=step_id,
            tool=decision.tool,
            tool_input=decision.tool_input,
            reason=decision.reason,
        )
        plan_steps.append(plan_step)
        single_plan = Plan(goal=goal, steps=[plan_step])
        step_results = execute_plan(
            single_plan,
            results_history=results,
            disabled_tools=disabled_tools,
        )
        
        if step_results:
            res = step_results[0]
            res.prompt_tokens = p_tok
            res.completion_tokens = c_tok
            res.total_tokens = t_tok
            results.append(res)
            if not res.success:
                failed_action_fingerprints.add(action_fingerprint(decision))

        step_count += 1
    else:
        satisfied = False
        final_summary = f"Stopped after reaching maximum of {max_steps} steps."

    total_prompt_tokens = sum(c.prompt_tokens for c in model_calls)
    total_completion_tokens = sum(c.completion_tokens for c in model_calls)
    total_tokens = sum(c.total_tokens for c in model_calls)

    total_latency_ms = (time.perf_counter() - chain_start) * 1000
    end_time = datetime.now(timezone.utc).isoformat()

    verified_claims = []
    evidence = []
    if (
        workspace_settings.enabled
        and conversation
        and conversation.document_passages
        and final_claims
    ):
        try:
            from .evidence import verify_claim_evidence

            verified_claims, evidence = verify_claim_evidence(
                final_claims, conversation.document_passages
            )
        except Exception:
            verified_claims = [
                {
                    "text": claim.text,
                    "status": "evidence_missing",
                    "support_score": 0.0,
                    "evidence_chunk_ids": [],
                }
                for claim in final_claims
            ]

    return ChainTrace(
        request_id=request_id,
        goal=goal,
        plan=Plan(goal=goal, steps=plan_steps),
        results=results,
        verify=VerifyResult(
            satisfied=satisfied,
            missing=[] if satisfied else [missing_reason],
            repair_steps=[],
            final_summary=final_summary,
            claims=verified_claims,
            evidence=evidence,
        ),
        repair_rounds=0,
        start_time=start_time,
        end_time=end_time,
        total_latency_ms=round(total_latency_ms, 2),
        llm_calls=model_calls,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        temperature=effective_temperature,
        context_usage=model_calls[-1].context_usage if model_calls else ContextUsage(),
        corrections=corrections,
        total_corrections=len(corrections),
        document_ids=conversation.document_ids if conversation else [],
    )
