"""Run the full evaluation across all 5 system variants × 12 benchmark queries.

Usage:
    python -m evaluation.run_eval --persist-dir ./chroma_db
    python -m evaluation.run_eval --variant E --family FACTUAL
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
from dotenv import load_dotenv

from evaluation.benchmark import get_test_cases
from evaluation.metrics import compute_llm_judge_score, compute_recall_at_k, extract_retrieved_ids

load_dotenv()

VARIANTS = ["A", "B", "C", "D", "E"]


def _plain_llm_answer(query: str, llm_client: anthropic.Anthropic) -> dict:
    t0 = time.time()
    response = llm_client.messages.create(
        model=os.getenv("DEFAULT_LLM", "claude-sonnet-4-6"),
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
    )
    latency = (time.time() - t0) * 1000
    usage = response.usage
    return {
        "answer": response.content[0].text.strip(),
        "latency_ms": latency,
        "token_count": usage.input_tokens + usage.output_tokens,
        "fused_results": [],
        "retrieved_text": [],
        "retrieved_images": [],
        "query_type": "N/A",
        "verification_score": None,
    }


def run_variant_on_case(
    variant: str,
    test_case: dict,
    graph: Any,
    llm_client: anthropic.Anthropic,
) -> dict:
    from agents.state import initial_state

    query = test_case["query"]
    turns = test_case.get("turns") or []

    if variant == "A":
        result = _plain_llm_answer(query, llm_client)
    else:
        state = initial_state(query)
        # Pre-populate conversation history for multi-turn CONVERSATIONAL cases
        if turns:
            state["conversation_history"] = list(turns)

        t0 = time.time()
        final_state = graph.invoke(state)
        latency = (time.time() - t0) * 1000

        token_usage = final_state.get("token_usage", {})
        result = {
            "answer": final_state.get("answer", ""),
            "latency_ms": latency,
            "token_count": token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0),
            "fused_results": final_state.get("fused_results", []),
            "retrieved_text": final_state.get("retrieved_text", []),
            "retrieved_images": final_state.get("retrieved_images", []),
            "query_type": final_state.get("query_type", ""),
            "verification_score": final_state.get("verification_score"),
        }

    retrieved_ids = extract_retrieved_ids(result)
    recall = compute_recall_at_k(retrieved_ids, test_case["relevant_recipe_ids"], k=3)
    judge_score, judge_rationale = compute_llm_judge_score(
        query, result["answer"], test_case["expected_keywords"], llm_client
    )

    return {
        "variant": variant,
        "query_id": test_case["id"],
        "family": test_case["family"],
        "query": query,
        "answer": result["answer"][:200],  # truncate for display
        "recall_at_3": recall,
        "llm_judge_score": judge_score,
        "judge_rationale": judge_rationale,
        "latency_ms": round(result["latency_ms"], 1),
        "token_count": result["token_count"],
        "query_type_detected": result["query_type"],
        "verification_score": result["verification_score"],
    }


def run_full_evaluation(
    persist_dir: str,
    output_csv: str = "results/eval_results.csv",
    variants: list[str] | None = None,
    family_filter: str | None = None,
) -> pd.DataFrame:
    from agents.graph import create_graph_for_variant
    from knowledge_base.image_store import ImageStore
    from knowledge_base.text_store import TextStore
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.image_retriever import ImageRetriever
    from retrieval.text_retriever import TextRetriever

    embed_model = os.getenv("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    clip_model = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    top_k = int(os.getenv("TOP_K", "5"))
    rrf_k = int(os.getenv("RRF_K", "60"))

    print("Loading models...")
    text_store = TextStore(persist_dir, embed_model)
    image_store = ImageStore(persist_dir, clip_model)
    text_retriever = TextRetriever(text_store, top_k=top_k)
    image_retriever = ImageRetriever(image_store, top_k=top_k)
    hybrid_retriever = HybridRetriever(text_retriever, image_retriever, rrf_k=rrf_k, top_k=top_k)
    llm_client = anthropic.Anthropic()

    run_variants = variants or VARIANTS
    test_cases = get_test_cases(family_filter)

    rows = []
    for variant in run_variants:
        print(f"\n{'─' * 50}")
        print(f"Running Variant {variant}...")
        graph = create_graph_for_variant(
            variant, text_retriever, image_retriever, hybrid_retriever, llm_client
        )
        for tc in test_cases:
            print(f"  [{tc['id']}] {tc['query'][:60]}...")
            row = run_variant_on_case(variant, tc, graph, llm_client)
            rows.append(row)
            r3 = row["recall_at_3"]
            jd = row["llm_judge_score"]
            lt = row["latency_ms"]
            print(f"       Recall@3={r3:.2f} | Judge={jd}/5 | {lt:.0f}ms")

    df = pd.DataFrame(rows)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    print_results_table(df)
    return df


def print_results_table(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print("RECALL@3 by Variant × Family")
    print("=" * 60)
    pivot_recall = df.pivot_table(
        values="recall_at_3", index="family", columns="variant", aggfunc="mean"
    )
    print(pivot_recall.round(2).to_string())

    print(f"\n{'=' * 60}")
    print("LLM JUDGE SCORE (1-5) by Variant × Family")
    print("=" * 60)
    pivot_judge = df.pivot_table(
        values="llm_judge_score", index="family", columns="variant", aggfunc="mean"
    )
    print(pivot_judge.round(2).to_string())

    print(f"\n{'=' * 60}")
    print("EFFICIENCY by Variant (mean latency ms, mean tokens)")
    print("=" * 60)
    efficiency = df.groupby("variant")[["latency_ms", "token_count"]].mean().round(1)
    print(efficiency.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation across system variants")
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        help="ChromaDB directory",
    )
    parser.add_argument(
        "--output",
        default="results/eval_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=VARIANTS,
        help="Restrict to specific variants (default: all)",
    )
    parser.add_argument(
        "--family",
        choices=["FACTUAL", "CROSS_MODAL", "ANALYTICAL", "CONVERSATIONAL"],
        help="Restrict to a specific query family",
    )
    args = parser.parse_args()

    run_full_evaluation(
        persist_dir=args.persist_dir,
        output_csv=args.output,
        variants=args.variant,
        family_filter=args.family,
    )


if __name__ == "__main__":
    main()
