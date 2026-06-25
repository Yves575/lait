# Release Change Audit

This audit documents the public-release refactor at a level suitable for
author review. It does not replace the Git diff; use it with
`withheld-files.tsv`, `sanitized-files.tsv`, and the branch diff.

## Review Principle

The refactor is intended to change repository packaging, navigation, and data
exposure boundaries. It is not intended to recompute reported results or change
scientific conclusions.

## Navigation And Documentation Changes

Added or rewrote reader-facing documentation:

- `README.md`
- `START_HERE.md`
- `docs/NAVIGATION.md`
- `docs/DATA_ACCESS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/REPO_MAP.md`
- `docs/RELEASE_CHECKLIST.md`
- Directory README files under `books/`, `books/MT/`, `book_stats/`,
  `human_eval/`, `analysis/`, `results_all_metrics/`,
  `results_chunk_review_eval/`, and `results_mapped_metrics/`

Purpose: make the public branch navigable for readers who have read the
preprint, skimmed only the abstract, or opened the repository directly.

## Data Boundary Changes

Removed source texts, human translations, row-level human-evaluation exports,
intermediate run workspaces, and out-of-scope generated artifacts.

Exact review files:

- `docs/release/withheld-files.tsv`: file-by-file removal manifest.
- `docs/release/WITHHELD_FILES.md`: summarized removal counts.
- `docs/release/all-withheld-removals.txt`: path list suitable for command-line review.
- `docs/release/PREPRINT_SCOPE_AUDIT.md`: high-level include/exclude rationale.

The `LiTransProQA/datasets/` directory was removed because it contained
source/target benchmark tables and local run inputs. The non-text question
weights from the old dataset location were retained separately at
`LiTransProQA/config/question_weights.csv`.

## Sanitized Result Files

Some derived outputs were retained but had text-bearing fields redacted.
Numeric metric scores, identifiers, and aggregate summaries were retained where
possible.

Exact review file:

- `docs/release/sanitized-files.tsv`

Sanitization was performed by `scripts/sanitize_public_release_outputs.py`.
The script redacts known source/HT-bearing fields such as `source`,
`hypothesis`, `reference`, `chunk_source`, `chunk_hypothesis`, and free-text
comment fields in derived result directories.

## Runtime Code Changes

Runtime code changes were kept narrow and tied to the release boundary or
public smoke-testability:

| File | Change | Reason |
| --- | --- | --- |
| `mt_eval.py` | Removed unreported GEMBA-MQM CLI/code paths; kept reported metrics and LiTransProQA judge path. | Keep public metric surface aligned with the preprint scope. |
| `mt_eval.py` | Moved LiTransProQA weights from the removed dataset path to `LiTransProQA/config/question_weights.csv`. | Preserve non-text metric configuration without restoring text-bearing datasets. |
| `mt_eval.py` | Normalized nested dataset names to POSIX-style separators. | Make output names stable across Windows and Unix. |
| `eval_pipeline.py` | Changed default scratch input/output paths from removed local folders to ignored `request/` paths. | Avoid recreating public data directories when controlled-access users run the wrapper. |
| `LiTransProQA/prompting_method/build_dataset_reviwed.py` | Allows an explicit CSV path before falling back to the historical dataset-relative path. | Support the new scratch path while preserving backward compatibility. |
| `agents_pipeline/runner.py` | Added direct-script import bootstrap and ASCII help text. | Make `python agents_pipeline/runner.py --help` work in a clean public checkout on Windows. |
| `agents_pipeline/core/executor.py` | Hardened relative-path validation for Windows/POSIX absolute paths. | Prevent workspace path escape on Windows and Unix. |
| `agents_pipeline/core/executor.py` | Added `--bare` to the Claude command builder. | Match the existing restricted-tool test expectation. |
| `api_model.py` | Added explicit handling for empty OpenRouter response content. | Avoid silent failures when a provider returns an unusable response. |
| `analysis/scripts/*` and `book_stats/calculate_stats.py` | Updated paths after renaming count directories. | Keep scripts aligned with clearer public directory names. |

## Test-Only Changes

| File | Change | Reason |
| --- | --- | --- |
| `pytest.ini` | Limits default collection to `tests/`. | Avoid collecting third-party/subproject tests that require unavailable optional dependencies. |
| `tests/agents_pipeline/test_gate.py` | Updated one fixture to use a full 25-question LiTransProQA response. | Keep production parser strict while preserving the test's intended chunk-selection assertion. |

## Directory Naming Changes

Renamed count directories for reader clarity:

- `book_stats/HT/` -> `book_stats/human_translation_counts/`
- `book_stats/MT/` -> `book_stats/machine_translation_counts/`
- `book_stats/SRC/` -> `book_stats/source_text_counts/`

The original source-text and human-translation count summaries remain aggregate
counts only; source/HT text files are not included.

## Verification To Review

Latest local checks performed during the refactor:

- `python -m pytest`
- `python -m compileall mt_eval.py mt_pipeline.py eval_pipeline.py LiTransProQA\prompting_method\build_dataset_reviwed.py scripts\sanitize_public_release_outputs.py analysis\scripts book_stats agents_pipeline`
- `python mt_pipeline.py --help`
- `python mt_eval.py --help`
- `python eval_pipeline.py --help`
- `python agents_pipeline\runner.py --help`
- `python -m agents_pipeline.runner --help`
- Markdown link scan across tracked documentation.
- Restricted-path scan for source/HT/internal paths.
- Structured sensitive-field scan for retained JSON/JSONL/CSV outputs.

Before publishing, authors should review this file, the branch diff, and both
release manifests.
