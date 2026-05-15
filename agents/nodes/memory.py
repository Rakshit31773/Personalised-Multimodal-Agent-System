from __future__ import annotations

import json
import re
from pathlib import Path

from agents.state import AgentState

USER_PROFILE_PATH = Path("data/user_profile.json")

_ALLERGY_PATTERNS = [
    (r"allergic to (\w+)", 1),
    (r"i don['’]t eat (\w+)", 1),
    (r"no (\w+)", 1),
    (r"i['’]m vegan", None),
    (r"i['’]m vegetarian", None),
]

_CUISINE_PATTERN = re.compile(r"(?:love|prefer|like|favourite|favorite)\s+(\w+)\s+food", re.I)


def load_user_profile() -> dict:
    if USER_PROFILE_PATH.exists():
        try:
            return json.loads(USER_PROFILE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "dietary_restrictions": [],
        "allergies": [],
        "preferred_cuisines": [],
        "past_queries": [],
    }


def save_user_profile(profile: dict) -> None:
    USER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = USER_PROFILE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(USER_PROFILE_PATH)


def memory_load_node(state: AgentState) -> dict:
    profile = load_user_profile()
    return {
        "user_profile": profile,
        "conversation_history": state.get("conversation_history", []),
    }


def memory_update_node(state: AgentState) -> dict:
    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": state.get("query", "")})
    history.append({"role": "assistant", "content": state.get("answer", "")})

    profile = dict(state.get("user_profile", {}))
    profile.setdefault("dietary_restrictions", [])
    profile.setdefault("allergies", [])
    profile.setdefault("preferred_cuisines", [])
    profile.setdefault("past_queries", [])

    query_lower = state.get("query", "").lower()

    # Detect allergies
    allergy_match = re.search(r"allergic to (\w+)", query_lower)
    if allergy_match:
        allergen = allergy_match.group(1)
        if allergen not in profile["allergies"]:
            profile["allergies"].append(allergen)

    # Detect dietary preferences
    if re.search(r"i['’]?m vegan", query_lower):
        if "vegan" not in profile["dietary_restrictions"]:
            profile["dietary_restrictions"].append("vegan")
    if re.search(r"i['’]?m vegetarian", query_lower):
        if "vegetarian" not in profile["dietary_restrictions"]:
            profile["dietary_restrictions"].append("vegetarian")

    # Detect cuisine preferences
    cuisine_match = _CUISINE_PATTERN.search(state.get("query", ""))
    if cuisine_match:
        cuisine = cuisine_match.group(1).lower()
        if cuisine not in profile["preferred_cuisines"]:
            profile["preferred_cuisines"].append(cuisine)

    profile["past_queries"].append(state.get("query", ""))
    profile["past_queries"] = profile["past_queries"][-50:]  # cap at 50

    save_user_profile(profile)
    return {"conversation_history": history, "user_profile": profile}


def build_memory_context(state: AgentState) -> str:
    profile = state.get("user_profile", {})
    parts: list[str] = []

    if profile.get("allergies"):
        parts.append(f"User allergies: {', '.join(profile['allergies'])}.")
    if profile.get("dietary_restrictions"):
        parts.append(f"Dietary restrictions: {', '.join(profile['dietary_restrictions'])}.")
    if profile.get("preferred_cuisines"):
        parts.append(f"Preferred cuisines: {', '.join(profile['preferred_cuisines'])}.")

    history = state.get("conversation_history", [])
    recent = history[-10:]  # last 5 turns (user+assistant pairs)
    if recent:
        parts.append("\nRecent conversation:")
        for turn in recent:
            role = turn.get("role", "user").capitalize()
            parts.append(f"  {role}: {turn.get('content', '')}")

    return "\n".join(parts) if parts else ""
