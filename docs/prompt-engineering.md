# Prompt engineering in the reasoning chain

The project applies system prompts, safe reasoning summaries, few-shot examples, and XML
delimiters as production behavior rather than documentation-only concepts.

## Versioned system prompts

`reasoning_chain/prompts.py` owns the prompt contracts for ReAct, decomposition, verification,
and rolling summaries. Each contract has a version such as `react-v8`. New `ModelCall` traces
record that version so prompt changes can be compared with temperature, token usage, tool choices,
and outcomes. Historical traces remain compatible and show an empty or `N/A` version.

The ReAct system prompt separates its role, tools, reference policy, rules, reasoning policy,
few-shot examples, and output contract with XML tags. User goals and conversation messages are
never interpolated into this trusted system instruction.

## Shared assistant behavior

`ASSISTANT_BEHAVIOR_POLICY` is a separately versioned XML policy shared by the ReAct and
verification prompts, because those are the paths that can generate a user-facing final answer.
It covers SuperAI's truthful product identity, concise refusals, child safety, harmful
capabilities, legal and financial caution, tone, wellbeing, even-handedness, knowledge freshness,
and resistance to instructions embedded in untrusted content.

The policy is adapted to this application rather than copying provider-specific identity claims.
SuperAI does not claim to be Claude, Anthropic, Gemini, Google, or a particular underlying model
unless trusted runtime metadata supplies that identity. It also does not advertise hard-coded
model names, prices, quotas, documentation links, or knowledge cutoffs that could become false.
Current facts are instead routed to `web_search` when available.

The policy is intentionally omitted from decomposition and rolling-summary prompts. Those calls
produce internal structured data rather than user-facing answers, so repeating the full policy
would consume context without improving their output. Their existing trust boundaries and strict
output contracts remain in force.

## Safe reasoning instead of exposed chain-of-thought

The model is instructed to reason internally and return only a brief action-selection `reason`:

```json
{
  "reason": "The request requires arithmetic.",
  "tool": "calculator",
  "tool_input": {"expression": "500 * 0.20"}
}
```

The trace records this decision rationale, the selected tool, validated input, real output, and
final answer. It does not intentionally request a detailed hidden reasoning transcript. The parser
still accepts the previous `thought` field so older responses and tests remain compatible.

## Few-shot examples

The ReAct prompt includes three curated examples:

1. Selecting the calculator for arithmetic.
2. Referencing a completed step with `[1]`.
3. Stopping with a supported final answer.

Examples are kept small because they consume input tokens on every ReAct call. Tests ensure they
remain JSON serializable, use only registered tools, and match either the tool-action or completion
contract. The context manager counts the complete system prompt—including examples—when enforcing
the input budget.

## XML-delimited untrusted input

Dynamic ReAct input is sent as user content:

```xml
<request_context trust="untrusted">
  <current_goal>...</current_goal>
  <steps_taken>...</steps_taken>
</request_context>
```

Rolling summaries use a similar `conversation_summary` element. Reserved XML characters in goals,
summaries, and tool results are escaped before insertion, so input containing closing tags remains
text instead of changing the document structure.

XML tags improve structure but are not a security boundary. Protection also depends on native
`user`/`model` roles, server-side tool validation, strict JSON parsing, bounded execution, and the
rule that untrusted content cannot override system instructions.

## Observability

The dashboard's model-call view displays prompt version, temperature, and token usage. These fields
support controlled comparisons when a future prompt version is introduced. Prompt changes should
be evaluated against the existing deterministic tests and a reviewed set of representative goals
before deployment.
