# Personalised Multimodal Agent System

Building an intelligent agent backed by your own multimodal knowledge base. Utilise LangGraph and Large Language Models (LLMs) to produce reliable, grounded, and domain-specific answers.

## Design Hypothesis

> Does hybrid multimodal retrieval (sentence-transformers text + CLIP ViT-B/32 image, fused via Reciprocal Rank Fusion) with LangGraph query routing and two-tier session memory outperform text-only RAG and plain LLM baselines for personalised recipe knowledge QA?

## System Variants

| Variant | Description |
|---------|-------------|
| A | Plain LLM — no retrieval, no agent |
| B | Text-only RAG — static pipeline |
| C | Text-only Agent — LangGraph + text retrieval + memory |
| D | Image-only Agent — LangGraph + CLIP retrieval + memory |
| E | **Full Hybrid Agent** — LangGraph + RRF fusion + memory *(default)* |

---

## Quick Start

The `run.sh` script installs dependencies, downloads the food images, and ingests
them into ChromaDB — skipping any step already done — then runs the project.

```bash
# Clone, enter, and create a virtual environment
git clone https://github.com/Rakshit31773/Personalised-Multimodal-Agent-System.git
cd Personalised-Multimodal-Agent-System
python3.11 -m venv .venv && source .venv/bin/activate

# Add your Anthropic API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

# Set up everything and run
./run.sh agent    # interactive agent (default)
./run.sh eval     # 5-variant evaluation suite
./run.sh test     # test suite
```

For manual, step-by-step setup, follow the sections below.

---

## Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

---

## Installation

```bash
# 1. Clone and enter the project
git clone https://github.com/Rakshit31773/Personalised-Multimodal-Agent-System.git
cd Personalised-Multimodal-Agent-System

# 2. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=your_key_here
```

---

## Data Setup

```bash
# 5. Get food images (~20 images, one per recipe)
#    Default source is Wikimedia Commons (CC images, no API key).
python scripts/download_images.py

# Other sources: huggingface (food101, ~5GB), placeholder (offline),
# or manual (supply your own <slug>.jpg files in the images dir).
python scripts/download_images.py --source placeholder

# 6. Ingest all recipe text and images into ChromaDB
python -m knowledge_base.ingest \
    --texts-dir knowledge_base/data/recipes/texts \
    --images-dir knowledge_base/data/recipes/images \
    --persist-dir ./chroma_db
```

Expected output:
```
Text collection:  20 docs (20 newly added)
Image collection: 20 docs (20 newly added)
Ingestion complete.
```

---

## Running the Agent

```bash
# Full hybrid agent (Variant E — default)
python -m app.main

# Specific variant
python -m app.main --variant A   # Plain LLM
python -m app.main --variant B   # Text-only RAG
python -m app.main --variant C   # Text-only Agent
python -m app.main --variant D   # Image-only Agent

# With verbose metadata output
python -m app.main --verbose
```

### Example session

```
> How long does miso soup take?
Miso Soup takes only 10 minutes to prepare. It is an Easy Japanese recipe...

> I'm allergic to peanuts. What can I make quickly?
Given your peanut allergy, here are some quick options (all nut-free): ...

> profile
{"allergies": ["peanuts"], "dietary_restrictions": [], ...}
```

---

## Running the Evaluation

```bash
# Full evaluation: all 5 variants × 12 benchmark queries
python -m evaluation.run_eval --persist-dir ./chroma_db

# Filter by variant or query family
python -m evaluation.run_eval --variant E --family FACTUAL

# Results saved to results/eval_results.csv
```

The evaluation outputs three summary tables:
1. Recall@3 by Variant × Query Family
2. LLM Judge Score (1–5) by Variant × Query Family
3. Mean latency and token count per Variant

---

## Running Tests

```bash
pytest
pytest --cov=. --cov-report=term-missing
```

---

## Linting

```bash
ruff check .
ruff format --check .
```

---

## Architecture Overview

```
User Query
    │
[Memory Load] ← loads user_profile.json + conversation history
    │
[Router] ← Claude classifies query type (FACTUAL/CROSS_MODAL/ANALYTICAL/CONVERSATIONAL)
    │
    ├─ FACTUAL        → [Text Retriever]   (sentence-transformers + ChromaDB)
    ├─ CROSS_MODAL    → [Image Retriever]  (CLIP ViT-B/32 + ChromaDB)
    ├─ ANALYTICAL     → [Hybrid Retriever] (Text + Image → RRF fusion)
    └─ CONVERSATIONAL → [Memory Retriever] (memory-augmented query + Hybrid)
                               │
                        [Synthesizer] ← Claude generates grounded answer
                               │
                        [Verifier] ← groundedness check (retries once if score < 3)
                               │
                        [Memory Update] → saves profile + conversation history
                               │
                         Final Answer
```

---

## Project Structure

```
knowledge_base/          Vector store wrappers (TextStore, ImageStore) + ingestion
  data/recipes/texts/    20 recipe .txt files
  data/recipes/images/   Food photos (downloaded or placeholder)
retrieval/               TextRetriever, ImageRetriever, HybridRetriever (RRF)
agents/                  LangGraph agent: state, nodes, graph
  nodes/                 memory, router, retriever, synthesizer, verifier
evaluation/              Benchmark (12 test cases), metrics, run_eval
app/                     CLI entry point
scripts/                 Image download script
tests/                   Unit tests for retrieval, agent nodes, metrics
```
