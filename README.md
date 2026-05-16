# Personalised Multimodal Agent System

A LangGraph-based recipe assistant with hybrid text + image retrieval, query routing, and two-tier memory.

---

## Quick Start

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/Rakshit31773/Personalised-Multimodal-Agent-System.git
cd Personalised-Multimodal-Agent-System
python3.11 -m venv .venv && source .venv/bin/activate

# 2. Add your Anthropic API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 3. One-command setup + run
./run.sh agent      # launch the interactive agent (default: Variant E)
```

`run.sh` installs dependencies, downloads food images (only if none are
present), and ingests the knowledge base into ChromaDB — each step is skipped
if already done, so subsequent runs start instantly.

### Three modes

```bash
./run.sh agent      # interactive agent CLI
./run.sh eval       # run the 5-variant evaluation suite
./run.sh test       # run the pytest test suite
```

### Selecting a variant

`agent` and `eval` accept an optional second argument restricting the run to
a single variant (`A`, `B`, `C`, `D`, or `E`):

```bash
./run.sh agent E    # full hybrid agent (default)
./run.sh agent A    # plain LLM baseline (no retrieval)
./run.sh agent B    # text-only RAG (linear pipeline)
./run.sh agent C    # text-only agent (full graph, text retrieval)
./run.sh agent D    # image-only agent (full graph, CLIP retrieval)

./run.sh eval E     # evaluate Variant E only
./run.sh eval       # evaluate all 5 variants × 12 queries
```

### Using the agent

Once `./run.sh agent` is running, type a question at the `>` prompt. The agent
classifies it, retrieves grounded context, and synthesises an answer.

```
> How long does miso soup take to make?
Miso Soup takes 10 minutes to prepare. It is an Easy Japanese recipe ...

> I'm allergic to peanuts. What can I make quickly?
Given your peanut allergy, here are some quick options (all nut-free): ...

> profile
{"allergies": ["peanuts"], "dietary_restrictions": [], "preferred_cuisines": [], ...}

> quit
```

Session commands: `profile` (show stored preferences), `reset memory` (clear
conversation history for the session), `quit` / `exit` (end the session).

### Evaluation output

Running `./run.sh eval` writes results to:

```
results/eval_results.csv            # one row per (variant, query, run)
results/eval_results_summary.csv    # per-query mean ± std (only with --runs > 1)
```

The terminal also prints three summary tables: Recall@3, LLM-judge score, and
mean latency / token count — each broken down by variant × query family.

---

## What the system is

A personalised recipe-knowledge assistant grounded in a curated cookbook. It
answers four categories of questions:

- **FACTUAL** — "How long does miso soup take?"
- **CROSS-MODAL** — "Show me a recipe that looks like a red stew with eggs on top."
- **ANALYTICAL** — "What can I cook with eggs and tomatoes, fastest?"
- **CONVERSATIONAL** — "I'm allergic to peanuts — suggest something quick."

It does this by combining four ideas:

1. **Two ChromaDB collections** — recipe text indexed with sentence-transformers
   and food photos indexed with CLIP ViT-B/32 image features.
2. **Reciprocal Rank Fusion** — when both modalities are queried, the ranked
   lists are merged with RRF (k = 60), which is parameter-free and robust to
   the scale mismatch between cosine similarity and CLIP scores.
3. **LangGraph query routing** — a Claude-based router classifies each query
   into one of the four families above and dispatches it to exactly one
   retriever (text / image / hybrid / memory-augmented hybrid).
4. **Two-tier memory** — session memory (recent conversation in graph state)
   and persistent memory (allergies, dietary restrictions, preferred cuisines
   in `data/user_profile.json`).

### System variants

| Variant | Description |
|---------|-------------|
| A | Plain LLM — no retrieval, no graph |
| B | Text-only RAG — linear pipeline, no router or memory |
| C | Text-only Agent — full LangGraph, text retrieval forced |
| D | Image-only Agent — full LangGraph, CLIP retrieval forced |
| E | **Full Hybrid Agent** — LangGraph + RRF fusion + memory *(default)* |

---

## Preparing your own data

The agent reads from two folders. **Both must use the same `slug` filename**
so a text file and its image are linked.

### Text recipes

Path: `knowledge_base/data/recipes/texts/<slug>.txt`

Each file follows a fixed key-value layout. The parser reads the labelled
fields, the `Ingredients:` block (each line prefixed `- `), and the numbered
`Instructions:` block.

Example — `knowledge_base/data/recipes/texts/miso_soup.txt`:

```
Name: Miso Soup
Cuisine: Japanese
Time: 10 minutes
Difficulty: Easy
Dietary: vegan, gluten-free, dairy-free
Ingredients:
- 4 cups dashi stock (kombu-based for vegan)
- 3 tablespoons white miso paste
- 150g soft tofu, cut into small cubes
- 2 tablespoons dried wakame seaweed
Instructions:
1. Bring dashi stock to a gentle simmer over medium heat. Do not boil.
2. Add the dried wakame seaweed and let it rehydrate for 1-2 minutes.
3. Add the cubed tofu and warm through for 2 minutes.
Notes: Never boil miso after adding it — boiling destroys the probiotic cultures.
```

Required fields: `Name:`, `Cuisine:`, `Time:`, `Difficulty:`, `Dietary:`,
`Ingredients:`, `Instructions:`. `Notes:` is optional.

### Food images

Path: `knowledge_base/data/recipes/images/<slug>.jpg`

The image filename must match the text filename: `miso_soup.txt` pairs with
`miso_soup.jpg`. If you don't have your own photos, run
`python scripts/download_images.py` and the bundled script will fetch one
per recipe from Wikimedia Commons (or supply your own and skip the script).

### Re-ingesting after adding recipes

ChromaDB persists between runs and the setup script skips ingestion if a
populated `chroma_db/` already exists. To pick up newly added recipes,
delete the directory and re-run:

```bash
rm -rf chroma_db && ./run.sh agent
```

---

## Personalisation: allergies and preferences

Persistent preferences live in `data/user_profile.json`:

```json
{
  "dietary_restrictions": [],
  "allergies": ["peanuts", "shellfish"],
  "preferred_cuisines": ["Japanese", "Italian"],
  "past_queries": []
}
```

You can edit this file directly, or simply tell the agent in conversation —
it detects allergies and dietary preferences from natural-language input and
writes them to the profile:

```
> I'm allergic to peanuts.
Noted — I'll avoid recipes containing peanuts.

> profile
{"allergies": ["peanuts"], ...}
```

At query time the text retriever post-filters out any recipe whose
`Dietary:` tags contain an allergen keyword (e.g. `contains-peanuts`).

---

## Architecture

```
User Query
    │
[Memory Load] ← loads data/user_profile.json + conversation history
    │
[Router] ← Claude classifies query into one of four families
    │
    ├─ FACTUAL        → [Text Retriever]    (sentence-transformers + ChromaDB)
    ├─ CROSS_MODAL    → [Image Retriever]   (CLIP ViT-B/32 + ChromaDB)
    ├─ ANALYTICAL     → [Hybrid Retriever]  (Text + Image → RRF fusion)
    └─ CONVERSATIONAL → [Memory Retriever]  (memory-augmented query → Hybrid)
                                │
                         [Synthesizer] ← Claude generates grounded answer
                                │
                          [Verifier] ← groundedness 1–5; one retry if < 3
                                │
                       [Memory Update] → persists profile + conversation
                                │
                         Final Answer
```

Prompt caching (`cache_control: ephemeral`) is applied to the router,
synthesizer, and verifier system prompts, and to the retrieved context in
the synthesizer.

---

## Project structure

```
agents/                   LangGraph agent code
  state.py                AgentState TypedDict
  graph.py                build_full_graph + create_graph_for_variant
  nodes/                  one file per graph node
    memory.py             memory_load + memory_update + profile detection
    router.py             query classifier (Claude call)
    retriever.py          text / image / hybrid / memory_augmented retrievers
    synthesizer.py        grounded answer generation
    verifier.py           groundedness check + retry routing
retrieval/                Retriever wrappers around the stores
  text_retriever.py       sentence-transformer search + dietary filter
  image_retriever.py      CLIP text-to-image search
  hybrid_retriever.py     reciprocal_rank_fusion + HybridRetriever
knowledge_base/           ChromaDB wrappers and ingestion
  text_store.py           recipes_text collection (MiniLM embeddings)
  image_store.py          recipes_image collection (CLIP embeddings)
  ingest.py               parse .txt recipes and populate both collections
  data/recipes/texts/     recipe text files (one .txt per recipe)
  data/recipes/images/    recipe photographs (one .jpg per recipe)
evaluation/               Benchmark suite and metrics
  benchmark.py            12 test cases across 4 query families
  metrics.py              Recall@k, LLM-as-judge, ID extraction
  run_eval.py             runs all variants × all cases, writes CSV
app/                      Interactive CLI entry point
  main.py                 python -m app.main
scripts/                  Helper scripts
  download_images.py      fetch food images from Wikimedia / food101
tests/                    Unit tests
  test_agent.py           router, verifier, memory nodes
  test_retrieval.py       stores, retrievers, RRF
  test_metrics.py         recall@k, retrieved-id extraction
data/                     Runtime data
  user_profile.json       persistent allergies / cuisines / dietary preferences
run.sh                    one-command setup + run script
pyproject.toml            package metadata, dependencies, ruff/pytest config
AI_declaration.md         declaration of AI assistance used in development
```

---

## Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

---

## Manual setup (alternative to `./run.sh`)

If you'd rather run each step yourself:

```bash
pip install -e ".[dev]"

cp .env.example .env
# set ANTHROPIC_API_KEY

python scripts/download_images.py
python -m knowledge_base.ingest \
    --texts-dir knowledge_base/data/recipes/texts \
    --images-dir knowledge_base/data/recipes/images \
    --persist-dir ./chroma_db

python -m app.main                      # agent (Variant E)
python -m app.main --variant A          # agent (specific variant)
python -m evaluation.run_eval           # full evaluation
python -m evaluation.run_eval --runs 3  # 3 repeats, mean ± std
pytest                                  # tests
ruff check . && ruff format --check .   # lint
```
