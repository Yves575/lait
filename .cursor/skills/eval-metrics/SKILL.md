---
name: eval-metrics
description: >-
  Finds the right automatic-metric or human-evaluation docs, CLIs, and result
  paths in LAIT. Use when working on mt_eval.py, eval_pipeline.py, human_eval,
  results_all_metrics, chunk-review eval, mapped metrics, LiTransProQA
  scoring, comment coding, or when the user mentions eval, metrics, COMET,
  MetricX, or human evaluation.
---

# Eval and metrics

Four different "eval" surfaces. Pick one, then open only its docs. Python: `use-project-venv`. Source/HT text in result files is often redacted; scores and IDs remain.

## Which surface

| Surface | Meaning | CLI / code | Outputs / docs |
| --- | --- | --- | --- |
| Automatic metrics | COMET-22, COMETKiwi, MetricX, MetricX-QE, LiTransProQA | `mt_eval.py` | `results_all_metrics/` (`all_results.csv`, `summary.json`, `dev/`, `eval/`) |
| Chunk-review metrics | Chunk-aligned metric surface | helpers under `scripts/` | `results_chunk_review_eval/README.md` |
| Paragraph→chunk maps | How aggregates were assembled | `scripts/map_paragraph_metrics_to_chunk_review.py` | `results_mapped_metrics/README.md` |
| Human evaluation | Reader study (not automatic scores) | `human_eval/*.R` (needs controlled-access inputs) | `human_eval/README.md`, `analysis/human_eval/README.md`, `docs/paper_supplement/` |
| Legacy wrapper | Older LiTransProQA file eval | `eval_pipeline.py` | Prefer `mt_eval.py` unless the task is this wrapper |

LiTransProQA **code** (no text datasets on GitHub): `LiTransProQA/README.md`. Weights used by `mt_eval.py`: `LiTransProQA/config/question_weights.csv`.

## Procedure

1. Name the surface from the table. "Eval" alone is ambiguous — ask or infer from paths.
2. Open the matching README. For preprint alignment, `docs/NAVIGATION.md`.
3. Rerun automatic metrics: `python mt_eval.py --help`. Inputs are paragraph pickles or chunk-review JSONL with SRC/HT/MT; those inputs are controlled-access.
4. Human-eval aggregates and figures are public; **row-level comments and annotation exports are withheld**. Schema development: `docs/COMMENT_CODING_SCHEMA_GUIDELINE.md`. Final labels: `analysis/manuscript_tables/tex/annotation_scheme_Q1_Q2_Q3_Q5.tex` and `annotation_scheme_Q4.tex`.
5. Derived tables: `analysis/README.md` → `analysis/scripts/` or `analysis/manuscript_tables/`.

## Traps

- Do not put GEMBA / GPTZero / `results_ai_detection/` back without a scope decision (`docs/release/PREPRINT_SCOPE_AUDIT.md`).
- Sanitized fields use `[withheld from public GitHub release]`. Restoring text: `docs/DATA_ACCESS.md`. Do not commit unredacted result files.
- `statistical/` is HT-vs-MT research plots (`statistical/README.md`), not the public metric dump in `results_all_metrics/`.

After significant eval-protocol or result-path changes, follow skill `update-docs`.
