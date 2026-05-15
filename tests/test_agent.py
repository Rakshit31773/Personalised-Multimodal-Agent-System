"""Tests for agent nodes. LLM calls are mocked to avoid API usage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents.nodes.memory import memory_load_node, memory_update_node
from agents.nodes.router import route_after_router, router_node
from agents.nodes.verifier import route_after_verifier
from agents.state import initial_state


def _make_llm_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0)
    return msg


# ── Router ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("query_type", ["FACTUAL", "CROSS_MODAL", "ANALYTICAL", "CONVERSATIONAL"])
def test_router_node_returns_correct_type(query_type: str) -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_llm_response(
        json.dumps({"query_type": query_type, "rationale": "test"})
    )
    state = initial_state("test query")
    result = router_node(state, mock_client)
    assert result["query_type"] == query_type


def test_router_node_increments_tool_calls() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_llm_response(
        json.dumps({"query_type": "FACTUAL", "rationale": "test"})
    )
    state = initial_state("test")
    state["tool_calls_count"] = 2
    result = router_node(state, mock_client)
    assert result["tool_calls_count"] == 3


def test_router_node_fallback_on_bad_json() -> None:
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_llm_response("not json at all")
    state = initial_state("test")
    result = router_node(state, mock_client)
    assert result["query_type"] == "FACTUAL"  # default fallback


@pytest.mark.parametrize(
    "query_type,expected_node",
    [
        ("FACTUAL", "text_retriever"),
        ("CROSS_MODAL", "image_retriever"),
        ("ANALYTICAL", "hybrid_retriever"),
        ("CONVERSATIONAL", "memory_retriever"),
    ],
)
def test_route_after_router(query_type: str, expected_node: str) -> None:
    state = initial_state("test")
    state["query_type"] = query_type
    assert route_after_router(state) == expected_node


# ── Verifier ──────────────────────────────────────────────────────────────────


def test_route_after_verifier_pass() -> None:
    state = initial_state("test")
    state["verification_passed"] = True
    state["retry_count"] = 0
    assert route_after_verifier(state) == "memory_update"


def test_route_after_verifier_fail_first_time() -> None:
    state = initial_state("test")
    state["verification_passed"] = False
    state["retry_count"] = 0
    assert route_after_verifier(state) == "synthesizer"


def test_route_after_verifier_fail_max_retries() -> None:
    state = initial_state("test")
    state["verification_passed"] = False
    state["retry_count"] = 1  # already retried once
    assert route_after_verifier(state) == "memory_update"


# ── Memory ────────────────────────────────────────────────────────────────────


def test_memory_load_node_missing_file(tmp_path: Path) -> None:
    with patch("agents.nodes.memory.USER_PROFILE_PATH", tmp_path / "profile.json"):
        state = initial_state("test")
        result = memory_load_node(state)
    profile = result["user_profile"]
    assert "allergies" in profile
    assert "dietary_restrictions" in profile


def test_memory_update_detects_peanut_allergy(tmp_path: Path) -> None:
    with patch("agents.nodes.memory.USER_PROFILE_PATH", tmp_path / "profile.json"):
        state = initial_state("I'm allergic to peanuts.")
        state["user_profile"] = {
            "allergies": [],
            "dietary_restrictions": [],
            "preferred_cuisines": [],
            "past_queries": [],
        }
        state["answer"] = "Noted."
        result = memory_update_node(state)
    assert "peanuts" in result["user_profile"]["allergies"]


def test_memory_update_appends_turn(tmp_path: Path) -> None:
    with patch("agents.nodes.memory.USER_PROFILE_PATH", tmp_path / "profile.json"):
        state = initial_state("What is miso soup?")
        state["user_profile"] = {
            "allergies": [],
            "dietary_restrictions": [],
            "preferred_cuisines": [],
            "past_queries": [],
        }
        state["answer"] = "Miso soup is a Japanese dish."
        state["conversation_history"] = []
        result = memory_update_node(state)
    history = result["conversation_history"]
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_memory_update_persists_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    with patch("agents.nodes.memory.USER_PROFILE_PATH", profile_path):
        state = initial_state("I'm vegan")
        state["user_profile"] = {
            "allergies": [],
            "dietary_restrictions": [],
            "preferred_cuisines": [],
            "past_queries": [],
        }
        state["answer"] = "Understood."
        memory_update_node(state)
    assert profile_path.exists()
