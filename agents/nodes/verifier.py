from __future__ import annotations

import json
import os

import anthropic

from agents.state import AgentState

VERIFIER_SYSTEM_PROMPT = """You are a groundedness verifier for a recipe assistant.
Given a question, retrieved context, and an answer, score how well the answer
is grounded in the retrieved context.

Scale:
5 = Fully grounded — every claim is directly supported by the retrieved context.
4 = Mostly grounded — minor additions that are harmless common knowledge.
3 = Partially grounded — some claims not explicitly in context but plausible.
2 = Weakly grounded — significant unsupported or contradictory claims.
1 = Not grounded — answer ignores the context or makes up information.

Respond with valid JSON only, no markdown:
{"score": <1-5>, "feedback": "<one sentence explaining the score>"}"""


def _format_retrieved_context(state: AgentState) -> str:
    results = state.get("fused_results", [])
    if not results:
        return "No context retrieved."
    parts = []
    for i, r in enumerate(results, 1):
        name = r.get("metadata", {}).get("name", r.get("recipe_id", "?"))
        content = r.get("text") or r.get("description", "")
        parts.append(f"[Recipe {i}: {name}]\n{content}")
    return "\n\n---\n\n".join(parts)


def verifier_node(state: AgentState, llm_client: anthropic.Anthropic) -> dict:
    retrieved_context = _format_retrieved_context(state)
    user_msg = (
        f"Question: {state.get('query', '')}\n\n"
        f"Context: {retrieved_context}\n\n"
        f"Answer: {state.get('answer', '')}"
    )

    response = llm_client.messages.create(
        model=os.getenv("DEFAULT_LLM", "claude-sonnet-4-6"),
        max_tokens=150,
        system=[
            {
                "type": "text",
                "text": VERIFIER_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    try:
        parsed = json.loads(raw)
        score = int(parsed["score"])
        feedback = parsed.get("feedback", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 3
        feedback = "Could not parse verifier response."

    usage = response.usage
    token_usage = dict(state.get("token_usage", {}))
    token_usage["input_tokens"] = token_usage.get("input_tokens", 0) + usage.input_tokens
    token_usage["output_tokens"] = token_usage.get("output_tokens", 0) + usage.output_tokens
    token_usage["cache_read_tokens"] = token_usage.get("cache_read_tokens", 0) + getattr(
        usage, "cache_read_input_tokens", 0
    )

    return {
        "verification_passed": score >= 3,
        "verification_score": score,
        "verification_feedback": feedback,
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        "token_usage": token_usage,
    }


def route_after_verifier(state: AgentState) -> str:
    if not state.get("verification_passed") and state.get("retry_count", 0) < 1:
        return "synthesizer"
    return "memory_update"
