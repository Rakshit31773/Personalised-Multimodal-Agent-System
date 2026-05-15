from __future__ import annotations

from knowledge_base.text_store import TextStore


class TextRetriever:
    def __init__(self, store: TextStore, top_k: int = 5) -> None:
        self._store = store
        self._default_top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        dietary_filter: dict | None = None,
    ) -> list[dict]:
        k = top_k if top_k is not None else self._default_top_k
        # Fetch more than needed so post-filtering has enough candidates
        fetch_k = min(self._store.count() or k, max(k * 4, 20))
        results = self._store.search(query, n_results=fetch_k)
        if dietary_filter:
            results = self._apply_filter(results, dietary_filter)
        # Re-rank after filtering
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results[:k]

    def _apply_filter(self, results: list[dict], dietary_filter: dict) -> list[dict]:
        """Post-filter results based on a dietary restriction dict.

        Supported key: "exclude_keywords" — list of strings that must NOT
        appear in the dietary_tags metadata field.
        """
        exclude = dietary_filter.get("exclude_keywords", [])
        if not exclude:
            return results
        filtered = []
        for r in results:
            tags = r.get("metadata", {}).get("dietary_tags", "").lower()
            if not any(kw.lower() in tags for kw in exclude):
                filtered.append(r)
        return filtered
