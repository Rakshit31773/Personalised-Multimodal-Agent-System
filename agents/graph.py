from __future__ import annotations

from functools import partial
from typing import Any

import anthropic
from langgraph.graph import END, START, StateGraph

from agents.nodes.memory import memory_load_node, memory_update_node
from agents.nodes.retriever import (
    hybrid_retriever_node,
    image_retriever_node,
    memory_retriever_node,
    text_retriever_node,
)
from agents.nodes.router import route_after_router, router_node
from agents.nodes.synthesizer import synthesizer_node
from agents.nodes.verifier import route_after_verifier, verifier_node
from agents.state import AgentState
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.image_retriever import ImageRetriever
from retrieval.text_retriever import TextRetriever


def _increment_retry(state: AgentState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_full_graph(
    text_retriever: TextRetriever,
    image_retriever: ImageRetriever,
    hybrid_retriever: HybridRetriever,
    llm_client: anthropic.Anthropic,
) -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("memory_load", memory_load_node)
    graph.add_node("router", partial(router_node, llm_client=llm_client))
    graph.add_node("text_retriever", partial(text_retriever_node, retriever=text_retriever))
    graph.add_node("image_retriever", partial(image_retriever_node, retriever=image_retriever))
    graph.add_node("hybrid_retriever", partial(hybrid_retriever_node, retriever=hybrid_retriever))
    graph.add_node("memory_retriever", partial(memory_retriever_node, retriever=hybrid_retriever))
    graph.add_node("synthesizer", partial(synthesizer_node, llm_client=llm_client))
    graph.add_node("verifier", partial(verifier_node, llm_client=llm_client))
    graph.add_node("retry_increment", _increment_retry)
    graph.add_node("memory_update", memory_update_node)

    graph.add_edge(START, "memory_load")
    graph.add_edge("memory_load", "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "text_retriever": "text_retriever",
            "image_retriever": "image_retriever",
            "hybrid_retriever": "hybrid_retriever",
            "memory_retriever": "memory_retriever",
        },
    )
    graph.add_edge("text_retriever", "synthesizer")
    graph.add_edge("image_retriever", "synthesizer")
    graph.add_edge("hybrid_retriever", "synthesizer")
    graph.add_edge("memory_retriever", "synthesizer")
    graph.add_edge("synthesizer", "verifier")
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"synthesizer": "retry_increment", "memory_update": "memory_update"},
    )
    graph.add_edge("retry_increment", "synthesizer")
    graph.add_edge("memory_update", END)

    return graph.compile()


def create_graph_for_variant(
    variant: str,
    text_retriever: TextRetriever,
    image_retriever: ImageRetriever,
    hybrid_retriever: HybridRetriever,
    llm_client: anthropic.Anthropic,
) -> Any:
    """Build the appropriate graph for a given ablation variant.

    A: plain LLM — returns None (caller handles directly)
    B: text RAG without router/memory/verifier (linear pipeline)
    C: full graph but forces FACTUAL routing (text only)
    D: full graph but forces CROSS_MODAL routing (image only)
    E: full hybrid graph
    """
    if variant == "A":
        return None

    if variant == "B":
        graph = StateGraph(AgentState)
        graph.add_node("text_retriever", partial(text_retriever_node, retriever=text_retriever))
        graph.add_node("synthesizer", partial(synthesizer_node, llm_client=llm_client))
        graph.add_edge(START, "text_retriever")
        graph.add_edge("text_retriever", "synthesizer")
        graph.add_edge("synthesizer", END)
        return graph.compile()

    if variant == "C":
        # Full graph but router is bypassed — always routes to text_retriever
        def force_text_router(state: AgentState) -> dict:
            return {"query_type": "FACTUAL"}

        graph = StateGraph(AgentState)
        graph.add_node("memory_load", memory_load_node)
        graph.add_node("router", force_text_router)
        graph.add_node("text_retriever", partial(text_retriever_node, retriever=text_retriever))
        graph.add_node("image_retriever", partial(image_retriever_node, retriever=image_retriever))
        graph.add_node(
            "hybrid_retriever", partial(hybrid_retriever_node, retriever=hybrid_retriever)
        )
        graph.add_node(
            "memory_retriever", partial(memory_retriever_node, retriever=hybrid_retriever)
        )
        graph.add_node("synthesizer", partial(synthesizer_node, llm_client=llm_client))
        graph.add_node("verifier", partial(verifier_node, llm_client=llm_client))
        graph.add_node("retry_increment", _increment_retry)
        graph.add_node("memory_update", memory_update_node)
        graph.add_edge(START, "memory_load")
        graph.add_edge("memory_load", "router")
        graph.add_conditional_edges(
            "router",
            route_after_router,
            {
                "text_retriever": "text_retriever",
                "image_retriever": "image_retriever",
                "hybrid_retriever": "hybrid_retriever",
                "memory_retriever": "memory_retriever",
            },
        )
        graph.add_edge("text_retriever", "synthesizer")
        graph.add_edge("image_retriever", "synthesizer")
        graph.add_edge("hybrid_retriever", "synthesizer")
        graph.add_edge("memory_retriever", "synthesizer")
        graph.add_edge("synthesizer", "verifier")
        graph.add_conditional_edges(
            "verifier",
            route_after_verifier,
            {"synthesizer": "retry_increment", "memory_update": "memory_update"},
        )
        graph.add_edge("retry_increment", "synthesizer")
        graph.add_edge("memory_update", END)
        return graph.compile()

    if variant == "D":
        # Full graph but router always routes to image_retriever
        def force_image_router(state: AgentState) -> dict:
            return {"query_type": "CROSS_MODAL"}

        graph = StateGraph(AgentState)
        graph.add_node("memory_load", memory_load_node)
        graph.add_node("router", force_image_router)
        graph.add_node("text_retriever", partial(text_retriever_node, retriever=text_retriever))
        graph.add_node("image_retriever", partial(image_retriever_node, retriever=image_retriever))
        graph.add_node(
            "hybrid_retriever", partial(hybrid_retriever_node, retriever=hybrid_retriever)
        )
        graph.add_node(
            "memory_retriever", partial(memory_retriever_node, retriever=hybrid_retriever)
        )
        graph.add_node("synthesizer", partial(synthesizer_node, llm_client=llm_client))
        graph.add_node("verifier", partial(verifier_node, llm_client=llm_client))
        graph.add_node("retry_increment", _increment_retry)
        graph.add_node("memory_update", memory_update_node)
        graph.add_edge(START, "memory_load")
        graph.add_edge("memory_load", "router")
        graph.add_conditional_edges(
            "router",
            route_after_router,
            {
                "text_retriever": "text_retriever",
                "image_retriever": "image_retriever",
                "hybrid_retriever": "hybrid_retriever",
                "memory_retriever": "memory_retriever",
            },
        )
        graph.add_edge("text_retriever", "synthesizer")
        graph.add_edge("image_retriever", "synthesizer")
        graph.add_edge("hybrid_retriever", "synthesizer")
        graph.add_edge("memory_retriever", "synthesizer")
        graph.add_edge("synthesizer", "verifier")
        graph.add_conditional_edges(
            "verifier",
            route_after_verifier,
            {"synthesizer": "retry_increment", "memory_update": "memory_update"},
        )
        graph.add_edge("retry_increment", "synthesizer")
        graph.add_edge("memory_update", END)
        return graph.compile()

    # variant == "E"
    return build_full_graph(text_retriever, image_retriever, hybrid_retriever, llm_client)
