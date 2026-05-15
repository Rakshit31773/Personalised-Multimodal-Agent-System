"""Benchmark test cases for the evaluation suite.

12 test cases across 4 required query families (3 per family).
recipe_id values use slugs matching the knowledge base (e.g., "miso_soup").
"""

from __future__ import annotations

TEST_CASES: list[dict] = [
    # ── FACTUAL ────────────────────────────────────────────────────────────────
    {
        "id": "F1",
        "query": "How long does miso soup take to make?",
        "family": "FACTUAL",
        "relevant_recipe_ids": ["miso_soup"],
        "expected_keywords": ["10 minutes", "10", "miso"],
        "turns": None,
    },
    {
        "id": "F2",
        "query": "What are the ingredients in shakshuka?",
        "family": "FACTUAL",
        "relevant_recipe_ids": ["shakshuka"],
        "expected_keywords": ["eggs", "tomatoes", "peppers", "cumin"],
        "turns": None,
    },
    {
        "id": "F3",
        "query": "Is pad thai a gluten-free dish?",
        "family": "FACTUAL",
        "relevant_recipe_ids": ["pad_thai"],
        "expected_keywords": ["gluten-free", "rice noodles"],
        "turns": None,
    },
    # ── CROSS-MODAL ────────────────────────────────────────────────────────────
    {
        "id": "CM1",
        "query": "Find a recipe whose image shows a soup with cubes of tofu floating in broth.",
        "family": "CROSS_MODAL",
        "relevant_recipe_ids": ["miso_soup"],
        "expected_keywords": ["miso", "tofu", "soup"],
        "turns": None,
    },
    {
        "id": "CM2",
        "query": "Show me a recipe that looks like a red tomato-based stew with eggs baked on top.",
        "family": "CROSS_MODAL",
        "relevant_recipe_ids": ["shakshuka"],
        "expected_keywords": ["shakshuka", "eggs", "tomato"],
        "turns": None,
    },
    {
        "id": "CM3",
        "query": (
            "Find a recipe whose image shows noodles in a clear aromatic broth"
            " with fresh herbs on top."
        ),
        "family": "CROSS_MODAL",
        "relevant_recipe_ids": ["pho_bo", "tom_yum"],
        "expected_keywords": ["broth", "noodles"],
        "turns": None,
    },
    # ── ANALYTICAL ─────────────────────────────────────────────────────────────
    {
        "id": "A1",
        "query": (
            "I have eggs, tomatoes, and onions at home."
            " What can I cook with them, and which option is the fastest?"
        ),
        "family": "ANALYTICAL",
        "relevant_recipe_ids": ["shakshuka", "omelette", "fried_rice"],
        "expected_keywords": ["shakshuka", "omelette", "fastest", "minutes"],
        "turns": None,
    },
    {
        "id": "A2",
        "query": "Which vegan recipes in the collection can I make in under 15 minutes?",
        "family": "ANALYTICAL",
        "relevant_recipe_ids": ["avocado_toast", "bruschetta"],
        "expected_keywords": ["avocado toast", "bruschetta", "vegan"],
        "turns": None,
    },
    {
        "id": "A3",
        "query": (
            "Compare the cooking times and difficulty of the two Japanese"
            " recipes in the collection."
        ),
        "family": "ANALYTICAL",
        "relevant_recipe_ids": ["miso_soup", "ramen"],
        "expected_keywords": ["miso soup", "ramen", "10 minutes", "60 minutes"],
        "turns": None,
    },
    # ── CONVERSATIONAL ─────────────────────────────────────────────────────────
    {
        "id": "V1",
        "query": "Suggest something quick for dinner tonight.",
        "family": "CONVERSATIONAL",
        "relevant_recipe_ids": [
            "miso_soup",
            "avocado_toast",
            "omelette",
            "fried_rice",
            "bruschetta",
            "banana_oat_pancakes",
        ],
        "expected_keywords": ["quick", "minutes"],
        "turns": [
            {
                "role": "user",
                "content": "I'm allergic to peanuts. Please keep this in mind for all suggestions.",
            },
            {
                "role": "assistant",
                "content": (
                    "Understood — I will avoid any recipes containing peanuts or peanut products."
                ),
            },
        ],
    },
    {
        "id": "V2",
        "query": "What can I make for a weeknight dinner?",
        "family": "CONVERSATIONAL",
        "relevant_recipe_ids": ["spaghetti_carbonara", "caesar_salad", "bruschetta"],
        "expected_keywords": ["Italian", "vegetarian"],
        "turns": [
            {
                "role": "user",
                "content": "I'm vegetarian and I love Italian food.",
            },
            {
                "role": "assistant",
                "content": (
                    "Got it — I will focus on vegetarian recipes and favour Italian cuisine."
                ),
            },
        ],
    },
    {
        "id": "V3",
        "query": "Which of those can I make in under 30 minutes?",
        "family": "CONVERSATIONAL",
        "relevant_recipe_ids": ["tom_kha", "satay_chicken"],
        "expected_keywords": ["coconut milk", "30 minutes", "25"],
        "turns": [
            {
                "role": "user",
                "content": "I've been craving something with coconut milk lately.",
            },
            {
                "role": "assistant",
                "content": (
                    "Great choice! We have several coconut milk recipes:"
                    " Tom Kha Gai (25 min) and Chicken Satay (30 min)."
                    " Would you like to explore them?"
                ),
            },
        ],
    },
]


def get_test_cases(family: str | None = None) -> list[dict]:
    if family is None:
        return TEST_CASES
    return [tc for tc in TEST_CASES if tc["family"] == family]
