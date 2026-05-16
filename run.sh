#!/usr/bin/env bash
#
# run.sh — one-command setup and run for the Personalised Multimodal Agent System.
#
# Usage:
#   ./run.sh agent [A-E]   Set up (if needed), then launch the interactive agent (default)
#   ./run.sh eval  [A-E]   Set up (if needed), then run the 5-variant evaluation suite
#   ./run.sh test          Set up (if needed), then run the test suite
#
# The optional second arg restricts agent/eval to a single variant (A, B, C, D
# or E). Omit it to use the defaults (agent -> E, eval -> all five variants).
#
# Setup steps — each is skipped if already done:
#   1. Install Python dependencies
#   2. Verify .env exists and has a real API key
#   3. Verify at least one recipe text file is present
#   4. Download food images only if none are present (any non-zero count is OK)
#   5. Ingest recipes into ChromaDB if not already ingested
#
# Note: uses the currently-active Python environment. Activate your venv first
# if you use one (see notes/instructions.txt).

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-agent}"
VARIANT="${2:-}"
TEXTS_DIR="knowledge_base/data/recipes/texts"
IMG_DIR="knowledge_base/data/recipes/images"
PERSIST_DIR="./chroma_db"

# Validate the mode up front, before doing any setup work.
case "$MODE" in
    agent|eval|test) ;;
    *)
        echo "Usage: ./run.sh [agent|eval|test] [A-E]"
        exit 1
        ;;
esac

if [ -n "$VARIANT" ]; then
    case "$VARIANT" in
        A|B|C|D|E) ;;
        *)
            echo "ERROR: variant must be one of A, B, C, D, E (got '$VARIANT')."
            exit 1
            ;;
    esac
    if [ "$MODE" = "test" ]; then
        echo "ERROR: 'test' mode does not accept a variant argument."
        exit 1
    fi
fi

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

echo "=== 3/5 Recipe text files ==="
if [ -d "$TEXTS_DIR" ]; then
    TXT_COUNT=$(find "$TEXTS_DIR" -name '*.txt' | wc -l | tr -d '[:space:]')
else
    TXT_COUNT=0
fi
if [ "$TXT_COUNT" -lt 1 ]; then
    echo "ERROR: no .txt recipes found in $TEXTS_DIR."
    echo "Add at least one recipe text file before running."
    exit 1
fi
echo "$TXT_COUNT recipe(s) present."

echo "=== 4/5 Food images ==="
if [ -d "$IMG_DIR" ]; then
    IMG_COUNT=$(find "$IMG_DIR" -name '*.jpg' | wc -l | tr -d '[:space:]')
else
    IMG_COUNT=0
fi
if [ "$IMG_COUNT" -lt 1 ]; then
    echo "No images found — downloading..."
    python scripts/download_images.py
else
    echo "$IMG_COUNT image(s) present — skipping download."
fi

echo "=== 5/5 ChromaDB ingestion ==="
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
VARIANT_ARG=()
if [ -n "$VARIANT" ]; then
    VARIANT_ARG=(--variant "$VARIANT")
fi

case "$MODE" in
    agent)
        python -m app.main "${VARIANT_ARG[@]}"
        ;;
    eval)
        python -m evaluation.run_eval --persist-dir "$PERSIST_DIR" "${VARIANT_ARG[@]}"
        ;;
    test)
        pytest
        ;;
esac
