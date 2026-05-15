from __future__ import annotations

import chromadb
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageStore:
    COLLECTION_NAME = "recipes_image"

    def __init__(self, persist_dir: str, clip_model: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._model = CLIPModel.from_pretrained(clip_model)
        self._processor = CLIPProcessor.from_pretrained(clip_model)
        self._model.eval()
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _encode_image(self, image_path: str) -> list[float]:
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self._model.get_image_features(**inputs).pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    def _encode_text(self, text: str) -> list[float]:
        import torch

        inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs).pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    def add_recipe(
        self,
        recipe_id: str,
        image_path: str,
        description: str,
        metadata: dict,
    ) -> None:
        existing = self._collection.get(ids=[recipe_id])
        if existing["ids"]:
            return
        embedding = self._encode_image(image_path)
        self._collection.add(
            ids=[recipe_id],
            embeddings=[embedding],
            documents=[description],
            metadatas=[metadata],
        )

    def search_by_text(self, text_query: str, n_results: int = 5) -> list[dict]:
        embedding = self._encode_text(text_query)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )
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
                    "description": doc,
                    "metadata": meta,
                    "distance": dist,
                    "rank": i + 1,
                }
            )
        return output

    def search_by_image(self, image_path: str, n_results: int = 5) -> list[dict]:
        embedding = self._encode_image(image_path)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )
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
                    "description": doc,
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
