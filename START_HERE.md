# Start Here

This repository is organized as a public research release. If you are arriving
from the study, the most useful public materials are the P1/P2/P3 MT outputs,
aggregate metric results, and derived analysis tables/figures.

If you have the preprint open, or if you have only skimmed the abstract, use
[docs/NAVIGATION.md](docs/NAVIGATION.md) as the path-by-path guide.

## 1. Inspect The MT Outputs

Start with:

- `books/MT/pipeline1/`
- `books/MT/pipeline2/`
- `books/MT/pipeline3/`

P1 and P2 are grouped by model. P3 is the agentic pipeline output, with the
appendix multilingual target-language examples under `books/MT/pipeline3/extern/`.

## 2. Inspect Analysis Results

Use:

- `analysis/manuscript_tables/`
- `human_eval/figures/`
- `human_eval/analysis_outputs/`
- `results_all_metrics/`
- `results_chunk_review_eval/`
- `results_mapped_metrics/`
- `docs/paper_supplement/`

The public branch redacts source and human-translation text fields in derived
tables, while retaining aggregate metrics and provenance fields. Row-level
participant comments and annotation exports are withheld.

## 3. Understand What Is Withheld

Read [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md). Source texts and human
translations are not included on GitHub. They can be requested from the authors
for research use, and will later be available through a gated Hugging Face
dataset with terms acceptance.

For exact audit trails:

- `docs/release/withheld-files.tsv`
- `docs/release/sanitized-files.tsv`
- `docs/release/PREPRINT_SCOPE_AUDIT.md`

## 4. Run Or Reproduce Code

Read [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), then inspect:

- `mt_pipeline.py`
- `agents_pipeline/runner.py`
- `mt_eval.py`
- `analysis/scripts/`

Some commands require controlled-access source or HT files and cannot be fully
rerun from the public GitHub checkout alone.
