from __future__ import annotations

from retrieval.image_retriever import ImageRetriever
from retrieval.text_retriever import TextRetriever


def reciprocal_rank_fusion(
    result_lists: list[tuple[str, list[dict]]],
    k: int = 60,
) -> list[dict]:
    """Fuse ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: list of (modality_name, results) tuples;
            each result dict must have 'recipe_id' and 'rank'.
        k: RRF constant (default 60 per Cormack et al. 2009).

    Returns:
        Merged, re-ranked list with rrf_score and sources fields.
    """
    scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}

    for modality, results in result_lists:
        for item in results:
            rid = item["recipe_id"]
            rank = item["rank"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
            if rid not in doc_data:
                doc_data[rid] = {
                    "recipe_id": rid,
                    "metadata": item.get("metadata", {}),
                    "text": item.get("text"),
                    "description": item.get("description"),
                    "sources": [],
                }
            if modality not in doc_data[rid]["sources"]:
                doc_data[rid]["sources"].append(modality)

    merged = sorted(doc_data.values(), key=lambda x: scores[x["recipe_id"]], reverse=True)
    for i, item in enumerate(merged):
        item["rrf_score"] = scores[item["recipe_id"]]
        item["rank"] = i + 1
    return merged


class HybridRetriever:
    def __init__(
        self,
        text_retriever: TextRetriever,
        image_retriever: ImageRetriever,
        rrf_k: int = 60,
        top_k: int = 5,
    ) -> None:
        self._text = text_retriever
        self._image = image_retriever
        self._rrf_k = rrf_k
        self._default_top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        dietary_filter: dict | None = None,
        modalities: tuple[str, ...] = ("text", "image"),
    ) -> list[dict]:
        k = top_k if top_k is not None else self._default_top_k

        result_lists = []
        if "text" in modalities:
            text_results = self._text.retrieve(query, top_k=k, dietary_filter=dietary_filter)
            result_lists.append(("text", text_results))
        if "image" in modalities:
            image_results = self._image.retrieve(query, top_k=k)
            result_lists.append(("image", image_results))

        if len(result_lists) == 1:
            # Single modality — skip fusion, return directly
            return result_lists[0][1][:k]

        fused = reciprocal_rank_fusion(result_lists, k=self._rrf_k)
        return fused[:k]
