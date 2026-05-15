from __future__ import annotations

import os

import anthropic

from agents.nodes.memory import build_memory_context
from agents.state import AgentState

KB_OVERVIEW = """You are a personal recipe assistant with access to a curated collection
of 20 recipes spanning Japanese, Italian, Indian, American, Vietnamese, Thai,
Middle Eastern, Greek, French, Chinese, Malaysian, and Turkish cuisines.

Rules:
- Always ground your answer in the retrieved context provided below.
- Cite recipe names explicitly when referencing them.
- If the retrieved context does not contain the answer, say so clearly rather than guessing.
- For dietary queries, check the Dietary field of each recipe before recommending it.
- Be concise but complete."""


def _format_results(fused_results: list[dict]) -> str:
    if not fused_results:
        return "No relevant recipes were retrieved."
    parts = []
    for i, r in enumerate(fused_results, 1):
        meta = r.get("metadata", {})
        name = meta.get("name", r.get("recipe_id", "Unknown"))
        text = r.get("text") or r.get("description", "")
        sources = r.get("sources", ["text"])
        parts.append(f"[Recipe {i}: {name} | Source: {', '.join(sources)}]\n{text}")
    return "\n\n---\n\n".join(parts)


def build_synthesis_prompt(state: AgentState, memory_context: str) -> list[dict]:
    messages: list[dict] = []

    # Memory context (not cached — changes every turn)
    if memory_context:
        messages.append(
            {
                "role": "user",
                "content": f"[User context and conversation history]\n{memory_context}",
            }
        )
        messages.append(
            {"role": "assistant", "content": "Understood. I will use this context to answer."}
        )

    # Build the user question, optionally with retry feedback
    user_content = state["query"]
    if state.get("retry_count", 0) > 0 and state.get("verification_feedback"):
        user_content = (
            f"{state['query']}\n\n"
            f"[Note: A previous answer was rejected for the following reason: "
            f"{state['verification_feedback']}. Please correct this in your response.]"
        )

    messages.append({"role": "user", "content": user_content})
    return messages


def synthesizer_node(state: AgentState, llm_client: anthropic.Anthropic) -> dict:
    model = (
        os.getenv("COMPLEX_LLM", "claude-opus-4-7")
        if state.get("query_type") == "ANALYTICAL"
        else os.getenv("DEFAULT_LLM", "claude-sonnet-4-6")
    )

    memory_ctx = build_memory_context(state)
    messages = build_synthesis_prompt(state, memory_ctx)

    retrieved_context = _format_results(state.get("fused_results", []))

    response = llm_client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": KB_OVERVIEW,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"[Retrieved context]\n{retrieved_context}",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=messages,
    )

    answer = response.content[0].text.strip()
    usage = response.usage
    token_usage = dict(state.get("token_usage", {}))
    token_usage["input_tokens"] = token_usage.get("input_tokens", 0) + usage.input_tokens
    token_usage["output_tokens"] = token_usage.get("output_tokens", 0) + usage.output_tokens
    token_usage["cache_read_tokens"] = token_usage.get("cache_read_tokens", 0) + getattr(
        usage, "cache_read_input_tokens", 0
    )

    return {
        "answer": answer,
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        "token_usage": token_usage,
    }
