from __future__ import annotations

import json
import os

import anthropic

from agents.state import AgentState, QueryType

ROUTER_SYSTEM_PROMPT = """You are a query classifier for a recipe knowledge base assistant.
Classify the user query into exactly one of four categories:

FACTUAL - Direct fact retrieval from text (cooking time, ingredients, steps, dietary info).
CROSS_MODAL - Requires searching by visual appearance (e.g., "find a dish that looks like...",
  "show me a recipe whose image shows...").
ANALYTICAL - Combines information from multiple recipes or requires reasoning
  (comparing, listing options given ingredients, multi-hop questions).
CONVERSATIONAL - Depends on prior conversation, stated preferences, or personal constraints
  (allergies, dietary restrictions, follow-up questions).

Respond with valid JSON only, no markdown:
{"query_type": "<FACTUAL|CROSS_MODAL|ANALYTICAL|CONVERSATIONAL>", "rationale": "<one sentence>"}"""


def router_node(state: AgentState, llm_client: anthropic.Anthropic) -> dict:
    response = llm_client.messages.create(
        model=os.getenv("DEFAULT_LLM", "claude-sonnet-4-6"),
        max_tokens=100,
        system=[
            {
                "type": "text",
                "text": ROUTER_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": state["query"]}],
    )
    raw = response.content[0].text.strip()
    try:
        parsed = json.loads(raw)
        query_type: QueryType = parsed["query_type"]
    except (json.JSONDecodeError, KeyError):
        query_type = "FACTUAL"

    usage = response.usage
    token_usage = dict(state.get("token_usage", {}))
    token_usage["input_tokens"] = token_usage.get("input_tokens", 0) + usage.input_tokens
    token_usage["output_tokens"] = token_usage.get("output_tokens", 0) + usage.output_tokens
    token_usage["cache_read_tokens"] = token_usage.get("cache_read_tokens", 0) + getattr(
        usage, "cache_read_input_tokens", 0
    )

    return {
        "query_type": query_type,
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        "token_usage": token_usage,
    }


def route_after_router(state: AgentState) -> str:
    mapping = {
        "FACTUAL": "text_retriever",
        "CROSS_MODAL": "image_retriever",
        "ANALYTICAL": "hybrid_retriever",
        "CONVERSATIONAL": "memory_retriever",
    }
    return mapping.get(state.get("query_type", "FACTUAL"), "text_retriever")
