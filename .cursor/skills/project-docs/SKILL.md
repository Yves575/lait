---
name: project-docs
description: >-
  Maps LAIT topics to canonical docs and paths (architecture, data layout,
  pipelines, eval, analysis, conventions). Use when looking something up in
  the project, or when the user mentions docs, documentation, where is, how
  does this project, architecture, pipeline, eval, analysis, conventions,
  find in project, reproducibility, or data access.
---

# Project docs map

Read this skill first. Open **only** the files that match the question. Do not dump docs into chat.

Python: follow `use-project-venv`. Do not follow README's `.venv` name.

## Start here

| Need | Open |
| --- | --- |
| Fastest reader path | `START_HERE.md` |
| Preprint → public path | `docs/NAVIGATION.md` |
| What each top-level dir is | `docs/REPO_MAP.md` |
| Setup / CLI `--help` | `docs/REPRODUCIBILITY.md`, root `README.md` |
| Missing source/HT, restore | `docs/DATA_ACCESS.md` |
| Directory before opening files | `README.md` inside `books/`, `books/MT/`, `book_stats/`, `human_eval/`, `analysis/`, `results_*` |

## Topic → doc

| Topic | Canonical file(s) |
| --- | --- |
| Public vs withheld; restore gated data | `docs/DATA_ACCESS.md` |
| Exact withheld/sanitized manifests | `docs/release/WITHHELD_FILES.md`, `docs/release/withheld-files.tsv`, `docs/release/sanitized-files.tsv` |
| Why a path was kept or dropped | `docs/release/PREPRINT_SCOPE_AUDIT.md` |
| Release packaging / sanitizer | `docs/release/CHANGE_AUDIT.md`, `docs/RELEASE_CHECKLIST.md` |
| P1/P2 direct MT | `mt_pipeline.py --help`, `books/MT/README.md` |
| P3 agentic MT | `agents_pipeline/README.md`, `agents_pipeline/runner.py --help` |
| MT output locations | `books/MT/pipeline1/`, `pipeline2/`, `pipeline3/` |
| Automatic metrics | `mt_eval.py --help`, `results_all_metrics/README.md` |
| Older LiTransProQA wrapper | `eval_pipeline.py --help` (legacy; prefer `mt_eval.py`) |
| Chunk-review / mapped metrics | `results_chunk_review_eval/README.md`, `results_mapped_metrics/README.md` |
| Human evaluation (reader study) | `human_eval/README.md`, `analysis/human_eval/README.md`, `docs/paper_supplement/` |
| Comment-coding schemas | `docs/COMMENT_CODING_SCHEMA_GUIDELINE.md`; labels in `analysis/manuscript_tables/tex/` |
| Analysis tables / scripts | `analysis/README.md`, `analysis/scripts/` |
| Book/chunk counts (no text) | `book_stats/README.md` |
| HT vs MT statistical reruns | `statistical/README.md` |
| LiTransProQA support code | `LiTransProQA/README.md` |
| Public-release publish steps | `docs/RELEASE_CHECKLIST.md` |

There is no `AGENTS.md`. Coding style is `.cursor/rules/ponytail.mdc` (already always-on).

## Data boundary (every lookup)

This checkout is the **public** branch: P1/P2/P3 MT under `books/MT/`, aggregate metrics/tables, code. **Not** on GitHub: source texts, human translations, raw human-eval exports, run workspaces with source chunks.

- Marker in retained files: `[withheld from public GitHub release]`
- Restore: `scripts/restore_controlled_access_data.py` (dry-run default; `--apply` writes). Details: `docs/DATA_ACCESS.md`
- Do not invent `books/dev/`, `books/eval/`, or `books/HT/` from MT outputs. Do not commit restored or unsanitized text.

## Workflow stubs (then stop)

**Run or change MT** → skill `mt-pipelines`, then `agents_pipeline/README.md` (P3) or `mt_pipeline.py --help` (P1/P2).

**Score or find eval** → skill `eval-metrics`. Do not treat `human_eval/` as `mt_eval.py`.

**Rebuild tables** → `analysis/README.md`, then the script under `analysis/scripts/`. Many need restored source/HT.

**After a significant change** → skill `update-docs`.
