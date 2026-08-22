---
name: mt-pipelines
description: >-
  Runs or modifies LAIT machine-translation pipelines P1/P2 (direct) and P3
  (agentic). Use when working on mt_pipeline.py, agents_pipeline, books/MT,
  chunking, prompts, translation runs, or when the user mentions P1, P2, P3,
  agentic MT, or pipeline.
---

# MT pipelines

Public outputs live under `books/MT/`. Full reruns need controlled-access **source** files (`docs/DATA_ACCESS.md`). Python: `use-project-venv`.

## Which pipeline

| Pipeline | Code | Public outputs | First doc |
| --- | --- | --- | --- |
| P1 / P2 direct | `mt_pipeline.py` | `books/MT/pipeline1/`, `pipeline2/` (by model) | `mt_pipeline.py --help`, `books/MT/README.md` |
| P3 agentic | `agents_pipeline/runner.py` | `books/MT/pipeline3/` (`extern/` = appendix multilingual) | `agents_pipeline/README.md` |

Five reported systems: P1 Gemini, P1 GPT-5.4 High, P2 Gemini, P2 GPT-5.4 High, P3 Agents. Scope: `docs/release/PREPRINT_SCOPE_AUDIT.md`.

## Procedure

1. Confirm whether the task is **direct** (P1/P2) or **agentic** (P3). Do not share chunkers or configs across them.
2. Read the matching first doc above. For P3, read `agents_pipeline/README.md` before editing stages, prompts, or gates.
3. Smoke-check CLI: `python mt_pipeline.py --help` or `python agents_pipeline/runner.py --help`.
4. If source files are missing, restore per `docs/DATA_ACCESS.md`. Do not recreate sources from MT.
5. Copy `.env.example` → `.env` only for the providers you need. Never commit `.env` or run workspaces that contain source chunks.

## Traps

- P3 chunks with `agents_pipeline/book.py` (tiktoken `o200k_base`, paragraph-preserving, ~1000 tokens). Root `book.py` is for P1/P2 / eval helpers — not P3.
- P3 prompts live in `agents_pipeline/prompts/`. Root `prompts/` is for P1/P2 (e.g. post-editing). Stage models: `agents_pipeline/config.json` — change the config, not a hard-coded stage table in docs, unless the checked-in config actually changed.
- `eval_pipeline.py` is **not** an MT pipeline (legacy LiTransProQA wrapper). Scoring is `mt-eval` / skill `eval-metrics`.

After significant pipeline/CLI/path changes, follow skill `update-docs`.
