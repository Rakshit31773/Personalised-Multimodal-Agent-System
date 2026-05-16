"""Interactive CLI for the Personalised Multimodal Agent System.

Usage:
    python -m app.main [--variant {A,B,C,D,E}] [--persist-dir PATH] [--verbose]

Commands during session:
    quit / exit    — end the session
    profile        — display current user profile
    reset memory   — clear conversation history for this session
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def load_dependencies(persist_dir: str | None = None) -> tuple[Any, Any, Any, Any]:
    import anthropic

    from knowledge_base.image_store import ImageStore
    from knowledge_base.text_store import TextStore
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.image_retriever import ImageRetriever
    from retrieval.text_retriever import TextRetriever

    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    embed_model = os.getenv("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    clip_model = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    top_k = int(os.getenv("TOP_K", "5"))
    rrf_k = int(os.getenv("RRF_K", "60"))

    print("Loading text embedding model...")
    text_store = TextStore(persist_dir, embed_model)
    print("Loading CLIP model...")
    image_store = ImageStore(persist_dir, clip_model)

    text_retriever = TextRetriever(text_store, top_k=top_k)
    image_retriever = ImageRetriever(image_store, top_k=top_k)
    hybrid_retriever = HybridRetriever(text_retriever, image_retriever, rrf_k=rrf_k, top_k=top_k)
    llm_client = anthropic.Anthropic()

    return text_retriever, image_retriever, hybrid_retriever, llm_client


def run_interactive_cli(graph: Any, variant: str, verbose: bool = False) -> None:
    from agents.nodes.memory import load_user_profile
    from agents.state import initial_state

    profile = load_user_profile()
    print(f"\n{'=' * 60}")
    print("  Personalised Multimodal Recipe Agent")
    print(f"  Variant: {variant} | Type 'quit' to exit, 'profile' to view preferences")
    print(f"{'=' * 60}")
    if profile.get("allergies"):
        print(f"  Known allergies: {', '.join(profile['allergies'])}")
    if profile.get("dietary_restrictions"):
        print(f"  Dietary: {', '.join(profile['dietary_restrictions'])}")
    print()

    conversation_history: list[dict] = []

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "profile":
            import json

            print(json.dumps(load_user_profile(), indent=2))
            continue

        if user_input.lower() == "reset memory":
            conversation_history = []
            print("Conversation history cleared.")
            continue

        state = initial_state(user_input)
        state["conversation_history"] = conversation_history

        t0 = time.time()

        if variant == "A":
            # Plain LLM — no graph
            import anthropic as ant

            client = ant.Anthropic()
            response = client.messages.create(
                model=os.getenv("DEFAULT_LLM", "claude-sonnet-4-6"),
                max_tokens=1024,
                messages=[{"role": "user", "content": user_input}],
            )
            answer = response.content[0].text.strip()
            latency = (time.time() - t0) * 1000
            print(f"\n{answer}\n")
            if verbose:
                print(f"[Variant A | Latency: {latency:.0f}ms]")
        else:
            final_state = graph.invoke(state)
            latency = (time.time() - t0) * 1000
            answer = final_state.get("answer", "No answer generated.")
            print(f"\n{answer}\n")
            conversation_history = final_state.get("conversation_history", [])
            if verbose:
                print(
                    f"[Variant {variant} | Type: {final_state.get('query_type')} | "
                    f"Retrieved: {len(final_state.get('fused_results', []))} | "
                    f"Verified: {final_state.get('verification_score')}/5 | "
                    f"Tokens: {final_state.get('token_usage', {})} | "
                    f"Latency: {latency:.0f}ms]"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Personalised Multimodal Recipe Agent")
    parser.add_argument(
        "--variant",
        choices=["A", "B", "C", "D", "E"],
        default="E",
        help="System variant (A=plain LLM, B=text RAG, C=text agent, D=image agent, E=full hybrid)",
    )
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        help="ChromaDB directory",
    )
    parser.add_argument("--verbose", action="store_true", help="Show metadata after each response")
    args = parser.parse_args()

    text_r, image_r, hybrid_r, llm_client = load_dependencies(persist_dir=args.persist_dir)

    from agents.graph import create_graph_for_variant

    graph = create_graph_for_variant(args.variant, text_r, image_r, hybrid_r, llm_client)
    run_interactive_cli(graph, variant=args.variant, verbose=args.verbose)


if __name__ == "__main__":
    main()
