# Declaration on Use of Generative AI

This file accompanies the source code submission for the *Personalised
Multimodal Agent System* project. It records how generative AI was used
during development, in keeping with academic-integrity expectations on AI
assistance.

## Tool

Generative AI was used as a coding and writing assistant, accessed via
**Claude Code** with the underlying model **Claude Opus 4.7**
(`claude-opus-4-7`, Anthropic).

This is distinct from the LLM calls that are *part of the system itself* —
the agent's router, synthesizer, verifier, and the LLM-as-judge evaluator
all call `claude-sonnet-4-6` / `claude-opus-4-7` at runtime, which is the
subject of study, not assistance.

## What AI was used for

Each AI-assisted change below was prompted by an explicit author instruction
and reviewed before being applied.

- **Code review of the evaluation pipeline.** Identified two defects:
  (i) Variant B's `verification_score` was being written to the CSV as
  `0` rather than blank, because the `initial_state` default leaked through
  the no-verifier pipeline and could be misread as a real score; and
  (ii) the CSV did not report prompt-cache tokens at all, so
  `token_count` understated true input usage.
- **Eval harness extensions.** Added prompt-cache reporting
  (`cache_read_tokens` and `cache_creation_tokens` columns); added a
  `--runs N` flag to `evaluation/run_eval.py` with a per-(variant, query)
  mean ± std summary CSV, run-tagged raw rows, and mean ± std versions of
  the printed comparison tables. Touched `evaluation/run_eval.py`,
  `agents/state.py`, `agents/nodes/router.py`, `agents/nodes/synthesizer.py`
  and `agents/nodes/verifier.py`.
- **Statistical interpretation.** Helped aggregate and read the 3-run
  evaluation data (e.g. confirming Recall@3 is deterministic across runs,
  and that the Variant B vs. Variant E judge-score gap of 0.08 is smaller
  than its run-to-run std of 0.13 — i.e. within noise).
- **Report drafting.** Helped draft and refine sections of the project
  report from the results CSV and the author's own design-decision notes.

## What was the author's own work

- The design hypothesis, the choice of recipe-collection domain, the
  five-variant ablation ladder (A–E), the agent graph shape, the choice of
  Reciprocal Rank Fusion, the two-tier memory design, the benchmark
  (12 test cases across 4 query families), and the metric selection.
- The original implementation of the knowledge base
  (`knowledge_base/`), retrievers (`retrieval/`), agent graph
  (`agents/`), original evaluation harness (`evaluation/`),
  and CLI (`app/`) — all of which pre-date the AI-assisted work
  described above.
- The decision to make each change above, and the testing and review of
  AI-suggested code before committing it.

## Reference

Anthropic (2025) *Claude* (claude-opus-4-7) [Large language model].
Available at: https://claude.ai (Accessed: 15 May 2026).
