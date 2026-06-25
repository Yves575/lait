# Repository Map

## First-Order Public Materials

| Path | Purpose |
| --- | --- |
| `docs/NAVIGATION.md` | Reader routes through the repo, including a preprint-to-path map. |
| `docs/paper_supplement/` | Human-evaluation interface screenshots and participant-facing guidelines referenced by the preprint. |
| `books/MT/pipeline1/` | P1 machine translation outputs by model. |
| `books/MT/pipeline2/` | P2 machine translation outputs by model. |
| `books/MT/pipeline3/` | P3 agentic-pipeline machine translation outputs, including appendix multilingual examples under `extern/`. |
| `book_stats/` | Aggregate count statistics only; directory names distinguish source-text, human-translation, machine-translation, and pipeline counts. |
| `human_eval/` | Public human-evaluation figures, aggregate outputs, and count summaries. |
| `analysis/manuscript_tables/` | Derived LaTeX/CSV tables retained for public inspection. |
| `results_all_metrics/` | Automatic metric outputs and summaries. |
| `results_chunk_review_eval/` | Chunk-review evaluation outputs with text fields redacted. |
| `results_mapped_metrics/` | Chunk/paragraph mapped metrics with text fields redacted. |

## Code

| Path | Purpose |
| --- | --- |
| `mt_pipeline.py` | Direct MT pipeline. |
| `agents_pipeline/` | Agentic MT pipeline and its docs/config. |
| `mt_eval.py` | Automatic MT metric scoring. |
| `eval_pipeline.py` | Older evaluation wrapper. |
| `analysis/scripts/` | Scripts used to build derived analysis artifacts. |
| `scripts/` | Utility scripts, including release sanitization. |
| `tests/` | Python tests for selected pipeline and utility behavior. |

## Supporting Projects

| Path | Purpose |
| --- | --- |
| `LiTransProQA/` | Evaluation framework code. |
| `metricx/` | MetricX code; retains its own license. |
| `par3/` | PAR3-related code/docs; data-bearing examples are withheld. |
| `tools/` | Public-safe local tools, if present. |

## Release Audit

| Path | Purpose |
| --- | --- |
| `docs/release/WITHHELD_FILES.md` | Human summary of withheld files. |
| `docs/release/CHANGE_AUDIT.md` | Human-readable audit of release refactor changes. |
| `docs/release/PREPRINT_SCOPE_AUDIT.md` | Scope decisions from the preprint review. |
| `docs/release/withheld-files.tsv` | Exact removed-file manifest. |
| `docs/release/sanitized-files.tsv` | Exact sanitized-file manifest. |
| `docs/release/human-eval-and-litransproqa-removals.txt` | Final pass of row-level human-evaluation and LiTransProQA dataset removals. |

Source texts and human translations are intentionally absent from the public
branch. See `docs/DATA_ACCESS.md` for controlled-access instructions.
