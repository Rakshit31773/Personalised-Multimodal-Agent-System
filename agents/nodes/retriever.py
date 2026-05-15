from __future__ import annotations

from agents.nodes.memory import build_memory_context
from agents.state import AgentState
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.image_retriever import ImageRetriever
from retrieval.text_retriever import TextRetriever


def _allergy_filter(state: AgentState) -> dict | None:
    profile = state.get("user_profile", {})
    allergies = profile.get("allergies", [])
    if not allergies:
        return None
    # Build keyword list for post-retrieval filtering in TextRetriever
    keywords = []
    for a in allergies:
        keywords.append(a)
        keywords.append(f"contains-{a}")
    return {"exclude_keywords": keywords}


def text_retriever_node(state: AgentState, retriever: TextRetriever) -> dict:
    results = retriever.retrieve(
        state["query"],
        dietary_filter=_allergy_filter(state),
    )
    return {
        "retrieved_text": results,
        "retrieved_images": [],
        "fused_results": results,
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
    }


def image_retriever_node(state: AgentState, retriever: ImageRetriever) -> dict:
    results = retriever.retrieve(state["query"])
    return {
        "retrieved_text": [],
        "retrieved_images": results,
        "fused_results": results,
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
    }


def hybrid_retriever_node(state: AgentState, retriever: HybridRetriever) -> dict:
    results = retriever.retrieve(
        state["query"],
        dietary_filter=_allergy_filter(state),
        modalities=("text", "image"),
    )
    text_results = [r for r in results if "text" in r.get("sources", [])]
    image_results = [r for r in results if "image" in r.get("sources", [])]
    return {
        "retrieved_text": text_results,
        "retrieved_images": image_results,
        "fused_results": results,
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
    }


def memory_retriever_node(state: AgentState, retriever: HybridRetriever) -> dict:
    memory_ctx = build_memory_context(state)
    if memory_ctx:
        augmented_query = f"{memory_ctx}\n\nUser question: {state['query']}"
    else:
        augmented_query = state["query"]
    results = retriever.retrieve(
        augmented_query,
        dietary_filter=_allergy_filter(state),
        modalities=("text", "image"),
    )
    text_results = [r for r in results if "text" in r.get("sources", [])]
    image_results = [r for r in results if "image" in r.get("sources", [])]
    return {
        "retrieved_text": text_results,
        "retrieved_images": image_results,
        "fused_results": results,
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
    }
