from __future__ import annotations

from knowledge_base.image_store import ImageStore


class ImageRetriever:
    def __init__(self, store: ImageStore, top_k: int = 5) -> None:
        self._store = store
        self._default_top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k if top_k is not None else self._default_top_k
        return self._store.search_by_text(query, n_results=k)
