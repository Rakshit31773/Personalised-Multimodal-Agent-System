"""Ingest recipe text and images into ChromaDB.

Usage:
    python -m knowledge_base.ingest \\
        --texts-dir knowledge_base/data/recipes/texts \\
        --images-dir knowledge_base/data/recipes/images \\
        --persist-dir ./chroma_db
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from knowledge_base.image_store import ImageStore
from knowledge_base.text_store import TextStore

load_dotenv()


def parse_recipe_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    recipe: dict = {"full_text": text, "recipe_id": path.stem}

    ingredients: list[str] = []
    instructions: list[str] = []
    section = None

    for line in lines:
        line = line.strip()
        if line.startswith("Name:"):
            recipe["name"] = line.removeprefix("Name:").strip()
        elif line.startswith("Cuisine:"):
            recipe["cuisine"] = line.removeprefix("Cuisine:").strip()
        elif line.startswith("Time:"):
            raw = line.removeprefix("Time:").strip()
            recipe["time_str"] = raw
            # Extract integer minutes
            import re

            m = re.search(r"(\d+)", raw)
            recipe["time_minutes"] = int(m.group(1)) if m else 0
        elif line.startswith("Difficulty:"):
            recipe["difficulty"] = line.removeprefix("Difficulty:").strip()
        elif line.startswith("Dietary:"):
            raw_tags = line.removeprefix("Dietary:").strip()
            # Split on comma, normalise whitespace
            recipe["dietary_tags"] = [t.strip() for t in raw_tags.split(",") if t.strip()]
        elif line.startswith("Notes:"):
            recipe["notes"] = line.removeprefix("Notes:").strip()
        elif line == "Ingredients:":
            section = "ingredients"
        elif line == "Instructions:":
            section = "instructions"
        elif section == "ingredients" and line.startswith("-"):
            ingredients.append(line.lstrip("- ").strip())
        elif section == "instructions" and line and line[0].isdigit():
            instructions.append(line.split(".", 1)[-1].strip())

    recipe["ingredients"] = ingredients
    recipe["instructions"] = instructions
    return recipe


def ingest_text(recipes: list[dict], store: TextStore, images_dir: Path) -> int:
    added = 0
    for r in recipes:
        image_path = str(images_dir / f"{r['recipe_id']}.jpg")
        metadata = {
            "name": r.get("name", r["recipe_id"]),
            "cuisine": r.get("cuisine", ""),
            "time_minutes": r.get("time_minutes", 0),
            "dietary_tags": ", ".join(r.get("dietary_tags", [])),
            "image_path": image_path,
            "recipe_id": r["recipe_id"],
        }
        before = store.count()
        store.add_recipe(r["recipe_id"], r["full_text"], metadata)
        if store.count() > before:
            added += 1
    return added


def ingest_images(recipes: list[dict], store: ImageStore, images_dir: Path) -> int:
    added = 0
    for r in recipes:
        image_path = images_dir / f"{r['recipe_id']}.jpg"
        if not image_path.exists():
            print(f"  [SKIP] No image found for {r['recipe_id']}")
            continue
        name = r.get("name", r["recipe_id"])
        cuisine = r.get("cuisine", "")
        description = f"A photograph of {name}, a {cuisine} dish."
        metadata = {
            "name": r.get("name", r["recipe_id"]),
            "cuisine": r.get("cuisine", ""),
            "time_minutes": r.get("time_minutes", 0),
            "dietary_tags": ", ".join(r.get("dietary_tags", [])),
            "image_path": str(image_path),
            "recipe_id": r["recipe_id"],
        }
        before = store.count()
        store.add_recipe(r["recipe_id"], str(image_path), description, metadata)
        if store.count() > before:
            added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest recipe data into ChromaDB")
    parser.add_argument(
        "--texts-dir",
        default="knowledge_base/data/recipes/texts",
        help="Directory containing recipe .txt files",
    )
    parser.add_argument(
        "--images-dir",
        default="knowledge_base/data/recipes/images",
        help="Directory containing recipe .jpg images",
    )
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        help="ChromaDB persistence directory",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate collections before ingesting",
    )
    args = parser.parse_args()

    texts_dir = Path(args.texts_dir)
    images_dir = Path(args.images_dir)
    persist_dir = args.persist_dir

    embed_model = os.getenv("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    clip_model = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

    print(f"Loading text embed model: {embed_model}")
    text_store = TextStore(persist_dir, embed_model)

    print(f"Loading CLIP model: {clip_model}")
    image_store = ImageStore(persist_dir, clip_model)

    if args.reset:
        print("Resetting collections...")
        text_store.reset()
        image_store.reset()

    recipe_files = sorted(texts_dir.glob("*.txt"))
    if not recipe_files:
        print(f"No .txt files found in {texts_dir}")
        return

    print(f"\nParsing {len(recipe_files)} recipe files...")
    recipes = [parse_recipe_file(f) for f in recipe_files]

    print("Ingesting text...")
    text_added = ingest_text(recipes, text_store, images_dir)

    print("Ingesting images...")
    image_added = ingest_images(recipes, image_store, images_dir)

    print(f"\n{'=' * 50}")
    print(f"Text collection:  {text_store.count()} docs ({text_added} newly added)")
    print(f"Image collection: {image_store.count()} docs ({image_added} newly added)")
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
