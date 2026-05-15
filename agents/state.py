from __future__ import annotations

from typing import Literal, TypedDict

QueryType = Literal["FACTUAL", "CROSS_MODAL", "ANALYTICAL", "CONVERSATIONAL"]


class AgentState(TypedDict, total=False):
    # Input
    query: str

    # Router output
    query_type: QueryType | None

    # Retrieval outputs
    retrieved_text: list[dict]
    retrieved_images: list[dict]
    fused_results: list[dict]

    # Generation outputs
    answer: str
    verification_passed: bool
    verification_score: int
    verification_feedback: str
    retry_count: int

    # Memory
    conversation_history: list[dict]
    user_profile: dict

    # Telemetry
    tool_calls_count: int
    start_time: float
    token_usage: dict


def initial_state(query: str) -> AgentState:
    import time

    return AgentState(
        query=query,
        query_type=None,
        retrieved_text=[],
        retrieved_images=[],
        fused_results=[],
        answer="",
        verification_passed=False,
        verification_score=0,
        verification_feedback="",
        retry_count=0,
        conversation_history=[],
        user_profile={},
        tool_calls_count=0,
        start_time=time.time(),
        token_usage={"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
    )
