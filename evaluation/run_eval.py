"""Run the full evaluation across all 5 system variants x 12 benchmark queries.

Usage:
    python -m evaluation.run_eval --persist-dir ./chroma_db
    python -m evaluation.run_eval --variant E --family FACTUAL
    python -m evaluation.run_eval --runs 5 # repeat 5x, report mean ± std
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
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
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
        # Variant B is a linear pipeline with no verifier node; report None rather
        # than the leftover initial_state default (0), which reads as a real score.
        verification_score = None if variant == "B" else final_state.get("verification_score")
        result = {
            "answer": final_state.get("answer", ""),
            "latency_ms": latency,
            "token_count": token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0),
            "fused_results": final_state.get("fused_results", []),
            "retrieved_text": final_state.get("retrieved_text", []),
            "retrieved_images": final_state.get("retrieved_images", []),
            "query_type": final_state.get("query_type", ""),
            "verification_score": verification_score,
            "cache_read_tokens": token_usage.get("cache_read_tokens", 0),
            "cache_creation_tokens": token_usage.get("cache_creation_tokens", 0),
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
        "cache_read_tokens": result["cache_read_tokens"],
        "cache_creation_tokens": result["cache_creation_tokens"],
        "query_type_detected": result["query_type"],
        "verification_score": result["verification_score"],
    }


def run_full_evaluation(
    persist_dir: str,
    output_csv: str = "results/eval_results.csv",
    variants: list[str] | None = None,
    family_filter: str | None = None,
    runs: int = 1,
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

    # Build each variant's graph once; reuse across all repeat runs.
    graphs = {
        v: create_graph_for_variant(
            v, text_retriever, image_retriever, hybrid_retriever, llm_client
        )
        for v in run_variants
    }

    rows = []
    for run_idx in range(runs):
        if runs > 1:
            print(f"\n{'═' * 50}")
            print(f" RUN {run_idx + 1} / {runs}")
            print("═" * 50)
        for variant in run_variants:
            print(f"\n{'─' * 50}")
            print(f"Running Variant {variant}...")
            for tc in test_cases:
                print(f"  [{tc['id']}] {tc['query'][:60]}...")
                row = run_variant_on_case(variant, tc, graphs[variant], llm_client)
                row["run_index"] = run_idx
                rows.append(row)
                r3 = row["recall_at_3"]
                jd = row["llm_judge_score"]
                lt = row["latency_ms"]
                print(f"       Recall@3={r3:.2f} | Judge={jd}/5 | {lt:.0f}ms")

    df = pd.DataFrame(rows)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nRaw per-run results saved to {output_csv}")

    if runs > 1:
        summary = aggregate_runs(df)
        summary_csv = str(
            Path(output_csv).with_name(Path(output_csv).stem + "_summary" + Path(output_csv).suffix)
        )
        summary.to_csv(summary_csv, index=False)
        print(f"Per-query mean ± std summary saved to {summary_csv}")

    print_results_table(df, runs=runs)
    return df


# Numeric columns that vary run-to-run and are worth aggregating.
_METRIC_COLS = [
    "recall_at_3",
    "llm_judge_score",
    "latency_ms",
    "token_count",
    "cache_read_tokens",
    "cache_creation_tokens",
    "verification_score",
]


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeat runs into per-(variant, query) mean and std for each metric.

    Each (variant, query_id) group holds exactly `runs` rows — one per repeat —
    so the std here is the clean run-to-run variability for that query.
    """
    n_runs = df["run_index"].nunique()
    numeric = df.copy()
    numeric[_METRIC_COLS] = numeric[_METRIC_COLS].apply(pd.to_numeric, errors="coerce")
    grouped = numeric.groupby(["variant", "query_id", "family"], sort=False)
    out = grouped[_METRIC_COLS].agg(["mean", "std"])
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]
    out.insert(0, "n_runs", n_runs)
    return out.reset_index()


def _family_table(df: pd.DataFrame, value: str, runs: int, decimals: int = 2) -> str:
    """Family × variant table. With repeat runs, cells show mean ± std, where the
    std is taken across the per-run family means (i.e. run-to-run noise)."""
    per_run = df.groupby(["run_index", "family", "variant"])[value].mean().reset_index()
    mean = per_run.pivot_table(values=value, index="family", columns="variant", aggfunc="mean")
    if runs <= 1:
        return mean.round(decimals).to_string()
    std = per_run.pivot_table(values=value, index="family", columns="variant", aggfunc="std")
    combined = mean.round(decimals).astype(str) + " ± " + std.round(decimals).astype(str)
    return combined.to_string()


def _variant_table(df: pd.DataFrame, values: list[str], runs: int) -> str:
    """Per-variant efficiency table. With repeat runs, cells show mean ± std taken
    across the per-run variant means."""
    per_run = df.groupby(["run_index", "variant"])[values].mean()
    mean = per_run.groupby("variant").mean()
    if runs <= 1:
        return mean.round(1).to_string()
    std = per_run.groupby("variant").std()
    combined = mean.round(1).astype(str) + " ± " + std.round(1).astype(str)
    return combined.to_string()


def print_results_table(df: pd.DataFrame, runs: int = 1) -> None:
    suffix = f"  (mean ± std over {runs} runs)" if runs > 1 else ""

    print(f"\n{'=' * 60}")
    print(f"RECALL@3 by Variant × Family{suffix}")
    print("=" * 60)
    print(_family_table(df, "recall_at_3", runs))

    print(f"\n{'=' * 60}")
    print(f"LLM JUDGE SCORE (1-5) by Variant × Family{suffix}")
    print("=" * 60)
    print(_family_table(df, "llm_judge_score", runs))

    print(f"\n{'=' * 60}")
    print(f"EFFICIENCY by Variant — latency ms, tokens{suffix}")
    print("=" * 60)
    print(_variant_table(df, ["latency_ms", "token_count"], runs))


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
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeat the whole evaluation N times and report mean ± std (default: 1)",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")

    run_full_evaluation(
        persist_dir=args.persist_dir,
        output_csv=args.output,
        variants=args.variant,
        family_filter=args.family,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
