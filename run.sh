#!/usr/bin/env bash
#
# run.sh — one-command setup and run for the Personalised Multimodal Agent System.
#
# Usage:
#   ./run.sh agent    Set up (if needed), then launch the interactive agent (default)
#   ./run.sh eval     Set up (if needed), then run the 5-variant evaluation suite
#   ./run.sh test     Set up (if needed), then run the test suite
#
# Setup steps — each is skipped if already done:
#   1. Install Python dependencies
#   2. Verify .env exists and has a real API key
#   3. Download food images if fewer than 20 are present
#   4. Ingest recipes into ChromaDB if not already ingested
#
# Note: uses the currently-active Python environment. Activate your venv first
# if you use one (see notes/instructions.txt).

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-agent}"
TEXTS_DIR="knowledge_base/data/recipes/texts"
IMG_DIR="knowledge_base/data/recipes/images"
PERSIST_DIR="./chroma_db"

# Validate the mode up front, before doing any setup work.
case "$MODE" in
    agent|eval|test) ;;
    *)
        echo "Usage: ./run.sh [agent|eval|test]"
        exit 1
        ;;
esac

echo "=== 1/4 Dependencies ==="
if python -c "import anthropic, langgraph, chromadb, sentence_transformers" 2>/dev/null; then
    echo "Already installed — skipping."
else
    echo "Installing..."
    pip install -e ".[dev]"
fi

echo "=== 2/4 API key (.env) ==="
if [ ! -f .env ]; then
    echo "ERROR: .env not found."
    echo "Run:  cp .env.example .env   then put your Anthropic API key in it."
    exit 1
fi
if grep -q "your_key_here" .env; then
    echo "ERROR: .env still contains the placeholder key."
    echo "Edit .env and set a real ANTHROPIC_API_KEY."
    exit 1
fi
echo "OK."

echo "=== 3/4 Food images ==="
if [ -d "$IMG_DIR" ]; then
    IMG_COUNT=$(find "$IMG_DIR" -name '*.jpg' | wc -l | tr -d '[:space:]')
else
    IMG_COUNT=0
fi
if [ "$IMG_COUNT" -lt 20 ]; then
    echo "Found $IMG_COUNT/20 images — downloading..."
    python scripts/download_images.py
else
    echo "20/20 images present — skipping download."
fi

echo "=== 4/4 ChromaDB ingestion ==="
if [ -d "$PERSIST_DIR" ]; then
    echo "ChromaDB present — skipping ingestion."
else
    echo "Ingesting recipes..."
    python -m knowledge_base.ingest \
        --texts-dir "$TEXTS_DIR" \
        --images-dir "$IMG_DIR" \
        --persist-dir "$PERSIST_DIR"
fi

echo
echo "=== Setup complete — running: $MODE ==="
case "$MODE" in
    agent)
        python -m app.main
        ;;
    eval)
        python -m evaluation.run_eval --persist-dir "$PERSIST_DIR"
        ;;
    test)
        pytest
        ;;
esac
