"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from agents.state import initial_state
from evaluation.metrics import compute_mrr, compute_recall_at_k, extract_retrieved_ids


def test_recall_at_k_perfect() -> None:
    assert compute_recall_at_k(["R01", "R02", "R03"], ["R01"], k=3) == 1.0


def test_recall_at_k_partial() -> None:
    result = compute_recall_at_k(["R02", "R03", "R04"], ["R01", "R02"], k=3)
    assert result == pytest.approx(0.5)


def test_recall_at_k_miss() -> None:
    assert compute_recall_at_k(["R05", "R06", "R07"], ["R01"], k=3) == 0.0


def test_recall_at_k_multiple_relevant() -> None:
    result = compute_recall_at_k(["R01", "R02", "R03"], ["R01", "R02", "R04"], k=3)
    assert result == pytest.approx(2 / 3)


def test_recall_at_k_empty_retrieved() -> None:
    assert compute_recall_at_k([], ["R01"], k=3) == 0.0


def test_recall_at_k_empty_relevant() -> None:
    assert compute_recall_at_k(["R01"], [], k=3) == 0.0


def test_mrr_first_hit() -> None:
    assert compute_mrr(["R01", "R02"], ["R01"]) == 1.0


def test_mrr_second_hit() -> None:
    assert compute_mrr(["R02", "R01"], ["R01"]) == pytest.approx(0.5)


def test_mrr_no_hit() -> None:
    assert compute_mrr(["R03", "R04"], ["R01"]) == 0.0


def test_extract_retrieved_ids_from_fused() -> None:
    state = initial_state("test")
    state["fused_results"] = [
        {"recipe_id": "A", "rrf_score": 0.9},
        {"recipe_id": "B", "rrf_score": 0.5},
    ]
    assert extract_retrieved_ids(state) == ["A", "B"]


def test_extract_retrieved_ids_fallback_to_text() -> None:
    state = initial_state("test")
    state["fused_results"] = []
    state["retrieved_text"] = [{"recipe_id": "C"}, {"recipe_id": "D"}]
    assert extract_retrieved_ids(state) == ["C", "D"]


def test_extract_retrieved_ids_fallback_to_images() -> None:
    state = initial_state("test")
    state["fused_results"] = []
    state["retrieved_text"] = []
    state["retrieved_images"] = [{"recipe_id": "E"}]
    assert extract_retrieved_ids(state) == ["E"]


def test_extract_retrieved_ids_empty() -> None:
    state = initial_state("test")
    assert extract_retrieved_ids(state) == []
