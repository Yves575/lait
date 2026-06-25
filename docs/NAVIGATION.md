# Navigation Guide

This guide is for readers arriving with different levels of context. It points
to the public files that correspond to the preprint while keeping source texts
and human translations behind controlled access.

## If You Have Not Read The Preprint

Start with these public materials:

| Question | Go to |
| --- | --- |
| What did the MT systems output? | `books/MT/pipeline1/`, `books/MT/pipeline2/`, `books/MT/pipeline3/` |
| What are the five reported MT systems? | `books/MT/pipeline1/gemini/`, `books/MT/pipeline1/gpt54_high/`, `books/MT/pipeline2/gemini/`, `books/MT/pipeline2/gpt54_high/`, `books/MT/pipeline3/` |
| How were the systems compared? | `results_all_metrics/`, `results_chunk_review_eval/`, `results_mapped_metrics/` |
| What did readers say in the human evaluation? | `human_eval/figures/`, `analysis/human_eval/`, `analysis/manuscript_tables/` |
| What did the human-evaluation interface and guidelines look like? | `docs/paper_supplement/` |
| What data is missing from GitHub? | `docs/DATA_ACCESS.md`, `docs/release/WITHHELD_FILES.md` |

## If You Skimmed The Abstract

The shortest path through the repository is:

1. Open `books/MT/` to inspect the public MT outputs.
2. Open `results_all_metrics/all_results.csv` and
   `results_all_metrics/summary.json` for automatic metric summaries.
3. Open `human_eval/figures/` and `analysis/manuscript_tables/` for retained
   human-evaluation summaries, coding tables, and model-output tables.
4. Open `docs/DATA_ACCESS.md` before looking for source texts or human
   translations; those files are intentionally absent from this public branch.

## If You Have The Preprint Open

Use this map to move from the preprint into the repository.

| Preprint area | Public repository paths |
| --- | --- |
| Dataset tables and book inventories | `book_stats/`, `book_stats/README.md`, `docs/DATA_ACCESS.md`, `docs/release/WITHHELD_FILES.md` |
| P1 and P2 direct MT outputs | `books/MT/pipeline1/`, `books/MT/pipeline2/` |
| P3 agentic MT output and workflow | `books/MT/pipeline3/`, `agents_pipeline/`, `prompts/` |
| Agent pipeline configs and artifact summaries | `agents_pipeline/config/`, `agents_pipeline/README.md`, `book_stats/pipelines/` |
| Automatic metric results | `results_all_metrics/`, `results_chunk_review_eval/`, `results_mapped_metrics/` |
| Human-evaluation figures and aggregate summaries | `human_eval/README.md`, `human_eval/figures/`, `human_eval/analysis_outputs/`, `analysis/human_eval/` |
| Human-evaluation interface screenshots and participant-facing guidelines | `docs/paper_supplement/` |
| Annotation schemes and model summaries | `analysis/manuscript_tables/`, `analysis/scripts/` |
| Public-release scope decisions | `docs/release/PREPRINT_SCOPE_AUDIT.md`, `docs/release/withheld-files.tsv`, `docs/release/sanitized-files.tsv` |

## Common Reader Tasks

| Task | Path |
| --- | --- |
| Compare one book across public MT systems | `books/MT/pipeline1/`, `books/MT/pipeline2/`, `books/MT/pipeline3/` |
| Check which systems are included | `docs/release/PREPRINT_SCOPE_AUDIT.md` |
| Find metric scores for a system | `results_all_metrics/all_results.csv` |
| Inspect per-book metric summaries | `results_all_metrics/dev/`, `results_all_metrics/eval/` |
| Find human-evaluation visual summaries | `human_eval/figures/` |
| View human-evaluation interface screenshots and participant-facing guidelines | `docs/paper_supplement/` |
| Understand a top-level directory before opening files | The `README.md` inside `books/`, `books/MT/`, `book_stats/`, `human_eval/`, `analysis/`, and `results_*` directories |
| Rebuild retained tables from aggregate inputs | `analysis/scripts/` |
| Learn why source/HT files are absent | `docs/DATA_ACCESS.md` |

## Controlled-Access Boundary

The preprint refers to source texts, human translations, chunk alignments, and
human-evaluation materials that cannot be redistributed in this public GitHub
branch. When a path would require those materials, the public repo either omits
the file or keeps a sanitized derivative with text fields redacted. The exact
audit trail is in `docs/release/withheld-files.tsv` and
`docs/release/sanitized-files.tsv`.
