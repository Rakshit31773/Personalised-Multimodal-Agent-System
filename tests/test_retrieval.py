"""Tests for retrieval components. Uses tmp_path for isolated ChromaDB instances."""

from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_base.text_store import TextStore
from retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from retrieval.image_retriever import ImageRetriever
from retrieval.text_retriever import TextRetriever

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"


@pytest.fixture()
def text_store(tmp_path: Path) -> TextStore:
    store = TextStore(str(tmp_path / "chroma"), EMBED_MODEL)
    store.add_recipe(
        "miso_soup",
        "Name: Miso Soup\nTime: 10 minutes\nDietary: vegan, gluten-free\nIngredients: tofu, miso",
        {
            "name": "Miso Soup",
            "cuisine": "Japanese",
            "time_minutes": 10,
            "dietary_tags": "vegan, gluten-free",
            "image_path": "",
            "recipe_id": "miso_soup",
        },
    )
    store.add_recipe(
        "pad_thai",
        "Name: Pad Thai\nTime: 20 minutes\nDietary: gluten-free, contains-peanuts",
        {
            "name": "Pad Thai",
            "cuisine": "Thai",
            "time_minutes": 20,
            "dietary_tags": "gluten-free, contains-peanuts",
            "image_path": "",
            "recipe_id": "pad_thai",
        },
    )
    store.add_recipe(
        "shakshuka",
        "Name: Shakshuka\nTime: 25 minutes\nDietary: vegetarian, gluten-free",
        {
            "name": "Shakshuka",
            "cuisine": "Middle Eastern",
            "time_minutes": 25,
            "dietary_tags": "vegetarian, gluten-free",
            "image_path": "",
            "recipe_id": "shakshuka",
        },
    )
    return store


def test_text_store_add_and_search(text_store: TextStore) -> None:
    assert text_store.count() == 3
    results = text_store.search("miso soup with tofu", n_results=3)
    assert len(results) > 0
    ids = [r["recipe_id"] for r in results]
    assert "miso_soup" in ids


def test_text_retriever_dietary_filter(text_store: TextStore) -> None:
    retriever = TextRetriever(text_store, top_k=5)
    results = retriever.retrieve(
        "recipe",
        dietary_filter={"exclude_keywords": ["peanuts", "contains-peanuts"]},
    )
    ids = [r["recipe_id"] for r in results]
    assert "pad_thai" not in ids
    assert "miso_soup" in ids or "shakshuka" in ids


def test_text_store_idempotent_add(text_store: TextStore) -> None:
    count_before = text_store.count()
    text_store.add_recipe(
        "miso_soup",
        "duplicate text",
        {
            "name": "Miso Soup",
            "cuisine": "Japanese",
            "time_minutes": 10,
            "dietary_tags": "vegan",
            "image_path": "",
            "recipe_id": "miso_soup",
        },
    )
    assert text_store.count() == count_before


def test_rrf_two_lists() -> None:
    list_a = [
        {"recipe_id": "A", "rank": 1, "metadata": {}},
        {"recipe_id": "B", "rank": 2, "metadata": {}},
        {"recipe_id": "C", "rank": 3, "metadata": {}},
    ]
    list_b = [
        {"recipe_id": "B", "rank": 1, "metadata": {}},
        {"recipe_id": "A", "rank": 2, "metadata": {}},
        {"recipe_id": "D", "rank": 3, "metadata": {}},
    ]
    fused = reciprocal_rank_fusion([("text", list_a), ("image", list_b)], k=60)
    ids_ordered = [r["recipe_id"] for r in fused]
    # A: 1/61 + 1/62 ≈ 0.0326; B: 1/62 + 1/61 ≈ 0.0326 (tied, order may vary); C, D below
    assert set(ids_ordered[:2]) == {"A", "B"}
    assert "C" in ids_ordered
    assert "D" in ids_ordered


def test_rrf_single_list_passthrough() -> None:
    list_a = [
        {"recipe_id": "X", "rank": 1, "metadata": {}},
        {"recipe_id": "Y", "rank": 2, "metadata": {}},
    ]
    fused = reciprocal_rank_fusion([("text", list_a)], k=60)
    # With a single list RRF scores should still be computed correctly
    assert fused[0]["recipe_id"] == "X"


def test_rrf_sources_field() -> None:
    list_a = [{"recipe_id": "A", "rank": 1, "metadata": {}}]
    list_b = [{"recipe_id": "A", "rank": 1, "metadata": {}}]
    fused = reciprocal_rank_fusion([("text", list_a), ("image", list_b)], k=60)
    assert "text" in fused[0]["sources"]
    assert "image" in fused[0]["sources"]


def test_text_retriever_wraps_store(text_store: TextStore) -> None:
    retriever = TextRetriever(text_store, top_k=2)
    results = retriever.retrieve("Japanese soup tofu")
    assert len(results) <= 2
    assert all("recipe_id" in r for r in results)


def test_hybrid_retriever_text_only(text_store: TextStore) -> None:
    text_retriever = TextRetriever(text_store, top_k=3)
    # Mock image retriever to ensure it is not called
    mock_image_retriever = ImageRetriever.__new__(ImageRetriever)
    hybrid = HybridRetriever(text_retriever, mock_image_retriever, top_k=3)
    results = hybrid.retrieve("miso soup", modalities=("text",))
    assert len(results) > 0
    assert all("recipe_id" in r for r in results)
