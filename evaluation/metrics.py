from __future__ import annotations

import json
import os

import anthropic

from agents.state import AgentState

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a recipe question-answering system.
Given a user question, expected keywords/phrases, and a system answer,
score the system answer on a scale of 1 to 5:

5 = Correct and complete — all key information is addressed, no factual errors.
4 = Mostly correct — minor omissions or imprecision but overall accurate.
3 = Partially correct — addresses some expected keywords but misses important information.
2 = Weak — minimal relevant content, significant gaps or inaccuracies.
1 = Incorrect or irrelevant — fails to address the question.

Respond with valid JSON only, no markdown:
{"score": <1-5>, "rationale": "<one sentence>"}"""


def compute_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 3,
) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k & relevant_set) / len(relevant_set)


def compute_mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_set:
            return 1.0 / i
    return 0.0


def compute_llm_judge_score(
    query: str,
    answer: str,
    expected_keywords: list[str],
    llm_client: anthropic.Anthropic,
) -> tuple[int, str]:
    keywords_str = ", ".join(f'"{kw}"' for kw in expected_keywords)
    user_msg = (
        f"Question: {query}\n\n"
        f"Expected keywords/phrases that a correct answer should address: {keywords_str}\n\n"
        f"System answer: {answer}"
    )
    response = llm_client.messages.create(
        model=os.getenv("DEFAULT_LLM", "claude-sonnet-4-6"),
        max_tokens=150,
        system=[
            {
                "type": "text",
                "text": JUDGE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    try:
        parsed = json.loads(raw)
        return int(parsed["score"]), parsed.get("rationale", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        return 3, "Parse error"


def extract_retrieved_ids(state: AgentState | dict) -> list[str]:
    fused = state.get("fused_results", [])
    if fused:
        return [r["recipe_id"] for r in fused]
    text = state.get("retrieved_text", [])
    if text:
        return [r["recipe_id"] for r in text]
    images = state.get("retrieved_images", [])
    if images:
        return [r["recipe_id"] for r in images]
    return []
