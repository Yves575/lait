# Reproducibility

This branch supports public inspection and partial reproduction. Full reruns
that require source texts or human translations need controlled-access data.

## Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` only when running model-backed commands. Never
commit `.env`.

## Publicly Inspectable Outputs

- `books/MT/pipeline1/`
- `books/MT/pipeline2/`
- `books/MT/pipeline3/`
- `analysis/manuscript_tables/`
- `results_all_metrics/`
- `results_chunk_review_eval/`
- `results_mapped_metrics/`

Derived result files may contain redacted text fields. Scores, system labels,
book IDs, and aggregate quantities remain available where possible.

## Common Commands

```bash
python mt_pipeline.py --help
python agents_pipeline/runner.py --help
python mt_eval.py --help
python eval_pipeline.py --help
```

Analysis scripts live under `analysis/scripts/`. Some expect local
controlled-access inputs and will not fully rerun from public GitHub data alone.

## Release Verification

Before publishing, run:

```bash
python -m pytest
python scripts/sanitize_public_release_outputs.py
```

Then verify that no source/HT paths or raw text fields have been reintroduced.
The release checklist in `docs/RELEASE_CHECKLIST.md` records the manual review
steps.
