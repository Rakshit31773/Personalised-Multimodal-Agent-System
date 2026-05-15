from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions


class TextStore:
    COLLECTION_NAME = "recipes_text"

    def __init__(self, persist_dir: str, embed_model: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_recipe(self, recipe_id: str, text: str, metadata: dict) -> None:
        existing = self._collection.get(ids=[recipe_id])
        if existing["ids"]:
            return
        self._collection.add(
            ids=[recipe_id],
            documents=[text],
            metadatas=[metadata],
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        kwargs: dict = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        output = []
        for i, (doc_id, doc, meta, dist) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            output.append(
                {
                    "recipe_id": doc_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "rank": i + 1,
                }
            )
        return output

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._get_or_create_collection()
