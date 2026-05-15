"""Fetch one food image per recipe for the knowledge base.

Sources (via --source):
  wikimedia   — Wikimedia Commons, CC/public-domain images, no API key (default)
  huggingface — food101 dataset (~5GB download)
  placeholder — generated coloured images, fully offline
  manual      — use images you place in the images dir yourself

Any recipe that can't be fetched falls back to a generated placeholder
(except in manual mode, which only reports what is present vs. missing).

Usage:
    python scripts/download_images.py [--source wikimedia] [--output-dir ...]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia's User-Agent policy (https://w.wiki/4wJS) requires a contact URL.
USER_AGENT = (
    "PersonalisedMultimodalAgentSystem/1.0 "
    "(https://github.com/Rakshit31773/Personalised-Multimodal-Agent-System; "
    "recipe knowledge-base image fetch)"
)

# Maps recipe slug → food101 class name (closest visual proxy)
RECIPE_IMAGE_MAP: dict[str, str] = {
    "miso_soup": "miso_soup",
    "spaghetti_carbonara": "spaghetti_carbonara",
    "chicken_tikka_masala": "chicken_curry",
    "avocado_toast": "guacamole",  # closest in food101
    "pho_bo": "pho",
    "pad_thai": "pad_thai",
    "shakshuka": "huevos_rancheros",  # closest: eggs in tomato sauce
    "caesar_salad": "caesar_salad",
    "dahl": "dal_makhani",
    "ramen": "ramen",
    "greek_salad": "greek_salad",
    "omelette": "omelette",
    "tom_yum": "hot_and_sour_soup",  # closest clear broth soup
    "satay_chicken": "chicken_wings",  # closest grilled chicken on skewer
    "fried_rice": "fried_rice",
    "bruschetta": "bruschetta",
    "lentil_soup": "lentil_soup",
    "beef_stir_fry": "beef_and_broccoli",  # some food101 variants include this
    "banana_oat_pancakes": "pancakes",
    "tom_kha": "coconut_soup",  # proxy for coconut milk soup
}


def download_via_huggingface(slug: str, class_name: str, output_dir: Path) -> bool:
    """Attempt to download image using HuggingFace datasets."""
    try:
        from datasets import load_dataset

        # food101 class names use underscores; normalise
        normalised = class_name.replace(" ", "_")
        ds = load_dataset("food101", split="validation", trust_remote_code=True)

        # Find first example matching this class
        label_names = ds.features["label"].names
        # Find label index; fall back to similar name
        label_idx = None
        for i, name in enumerate(label_names):
            if name == normalised or name.replace("_", " ") == normalised.replace("_", " "):
                label_idx = i
                break

        if label_idx is None:
            return False

        for example in ds:
            if example["label"] == label_idx:
                img = example["image"]
                out_path = output_dir / f"{slug}.jpg"
                img.convert("RGB").save(str(out_path), "JPEG", quality=85)
                return True
        return False
    except Exception:
        return False


def _get_with_retry(url: str, *, max_retries: int = 4, **kwargs):
    """HTTP GET with exponential backoff when Wikimedia returns 429 (rate limited)."""
    import requests

    delay = 1.0
    resp = None
    for _ in range(max_retries):
        resp = requests.get(url, timeout=30, **kwargs)
        if resp.status_code != 429:
            break
        time.sleep(delay)
        delay *= 2
    resp.raise_for_status()
    return resp


def download_via_wikimedia(slug: str, output_dir: Path) -> bool:
    """Download a food photo from Wikimedia Commons (CC / public domain)."""
    try:
        from io import BytesIO

        from PIL import Image

        query = slug.replace("_", " ")
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,  # File namespace
            "gsrlimit": 10,
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": 600,  # request a 600px-wide thumbnail
        }
        resp = _get_with_retry(WIKIMEDIA_API, params=params, headers={"User-Agent": USER_AGENT})
        pages = resp.json().get("query", {}).get("pages", {})

        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            if not info.get("mime", "").startswith("image/"):
                continue
            img_url = info.get("thumburl") or info.get("url")
            if not img_url:
                continue
            img_resp = _get_with_retry(img_url, headers={"User-Agent": USER_AGENT})
            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            img.save(str(output_dir / f"{slug}.jpg"), "JPEG", quality=85)
            return True
        return False
    except Exception as e:
        print(f"  [warn] {slug}: Wikimedia fetch failed ({type(e).__name__}: {e})")
        return False


def generate_placeholder_image(slug: str, output_dir: Path) -> None:
    """Generate a coloured placeholder image with the recipe name."""
    import hashlib

    from PIL import Image, ImageDraw

    # Deterministic colour from slug hash
    h = int(hashlib.md5(slug.encode()).hexdigest()[:6], 16)
    r = (h >> 16) & 0xFF
    g = (h >> 8) & 0xFF
    b = h & 0xFF
    # Ensure not too dark or too light
    r = max(80, min(200, r))
    g = max(80, min(200, g))
    b = max(80, min(200, b))

    img = Image.new("RGB", (400, 300), color=(r, g, b))
    draw = ImageDraw.Draw(img)
    label = slug.replace("_", " ").title()
    # Draw text in white
    draw.text((20, 130), label, fill=(255, 255, 255))
    out_path = output_dir / f"{slug}.jpg"
    img.save(str(out_path), "JPEG", quality=85)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download food images for the recipe knowledge base"
    )
    parser.add_argument(
        "--output-dir",
        default="knowledge_base/data/recipes/images",
        help="Directory to save images",
    )
    parser.add_argument(
        "--source",
        choices=["wikimedia", "huggingface", "placeholder", "manual"],
        default="wikimedia",
        help="Image source: wikimedia (CC images, no API key), huggingface "
        "(food101, ~5GB), placeholder (generated, offline), or manual "
        "(use images you place in the images dir yourself)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "manual":
        present, missing = [], []
        for slug in RECIPE_IMAGE_MAP:
            if (output_dir / f"{slug}.jpg").exists():
                present.append(slug)
            else:
                missing.append(slug)
        print(f"\nManual mode — {len(present)} image(s) found in {output_dir}")
        if missing:
            print(f"{len(missing)} recipe(s) still need an image (add as <slug>.jpg):")
            for slug in missing:
                print(f"  - {slug}.jpg")
            print(
                "\nDrop the files in, or re-run with --source wikimedia "
                "(or placeholder) to auto-fill the gaps."
            )
        else:
            print("All recipes have an image.")
        return

    downloaded = 0
    placeholders = 0

    for slug, class_name in tqdm(RECIPE_IMAGE_MAP.items(), desc="Fetching images"):
        out_path = output_dir / f"{slug}.jpg"
        if out_path.exists():
            continue

        success = False
        if args.source == "wikimedia":
            success = download_via_wikimedia(slug, output_dir)
            time.sleep(1.0)  # be polite to the Wikimedia servers between requests
        elif args.source == "huggingface":
            success = download_via_huggingface(slug, class_name, output_dir)

        if success:
            downloaded += 1
        else:
            generate_placeholder_image(slug, output_dir)
            placeholders += 1

    print(f"\nDone: {downloaded} downloaded, {placeholders} placeholders generated.")
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
