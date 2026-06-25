# LAIT <br><sub><sup>Literary AI Translation</sup></sub>

<p align="left">
  <img src="./lait_white.png" width="80" alt="LAIT Logo" />
</p>

[![arxiv](https://img.shields.io/badge/arXiv-2606.26040-b31b1b.svg)](http://arxiv.org/abs/2606.26040) [![Website](https://img.shields.io/website?url=https%3A%2F%2Flait.cs.sfu.ca%2F)](https://lait.cs.sfu.ca/)

LAIT is a public research release for a reader-centered evaluation study of
literary machine translation. The repository centers the three MT pipelines
discussed in the study and the aggregate analyses used to compare machine
translations with human translations.

This public GitHub release does **not** include source texts or human
translations. Those materials are available for research access by emailing the
authors, and will later be distributed through a gated Hugging Face dataset
that requires users to agree to access terms.

## Start Here

- [START_HERE.md](START_HERE.md): fastest path through the repository.
- [docs/NAVIGATION.md](docs/NAVIGATION.md): routes for readers with the
  preprint open, the abstract skimmed, or no prior context.
- [docs/REPO_MAP.md](docs/REPO_MAP.md): what each top-level directory is for.
- [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md): what is public, what is withheld,
  and how to request controlled access.
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): setup and rerun notes.
- [docs/paper_supplement/](docs/paper_supplement/): human-evaluation interface
  screenshots and participant-facing guidelines referenced by the preprint.
- [docs/release/WITHHELD_FILES.md](docs/release/WITHHELD_FILES.md): exact
  manifest of files withheld from this public branch.
- [docs/release/PREPRINT_SCOPE_AUDIT.md](docs/release/PREPRINT_SCOPE_AUDIT.md):
  summary of the paper-scope review decisions.

## Main Public Outputs

If you are trying to match the repository to the preprint, use
[docs/NAVIGATION.md](docs/NAVIGATION.md) first. It maps the dataset, MT
pipeline, automatic-metric, and human-evaluation materials to public paths.

The primary public MT outputs are:

| Pipeline | Path                  | Description                                                                                                |
| -------- | --------------------- | ---------------------------------------------------------------------------------------------------------- |
| P1       | `books/MT/pipeline1/` | First MT pipeline outputs, grouped by model.                                                               |
| P2       | `books/MT/pipeline2/` | Second MT pipeline outputs, grouped by model.                                                              |
| P3       | `books/MT/pipeline3/` | Agentic MT pipeline outputs, including the appendix multilingual target-language examples under `extern/`. |

Secondary baseline outputs are kept where they do not expose source or human
translation text, but the release navigation prioritizes P1, P2, and P3.

## Analysis Outputs

- `analysis/manuscript_tables/`: derived LaTeX/CSV tables retained for the
  public release. Text-bearing source/HT fields are redacted where needed.
- `human_eval/`: public human-evaluation figures, aggregate model outputs, and
  count summaries. Row-level comments and annotation exports are withheld.
- `results_all_metrics/`, `results_chunk_review_eval/`, and
  `results_mapped_metrics/`: derived metric outputs with source/HT text fields
  redacted where needed.

## Code Entry Points

| Task                                    | Entry point                               |
| --------------------------------------- | ----------------------------------------- |
| Run the direct MT pipeline              | `python mt_pipeline.py --help`            |
| Run the agentic MT pipeline             | `python agents_pipeline/runner.py --help` |
| Score MT outputs with automatic metrics | `python mt_eval.py --help`                |
| Run the older evaluation wrapper        | `python eval_pipeline.py --help`          |
| Inspect analysis scripts                | `analysis/scripts/`                       |
| Inspect utility scripts                 | `scripts/`                                |

Some commands require controlled-access source or HT files. The public branch
keeps the code and aggregate outputs, but not those restricted inputs.

## Setup

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

Copy `.env.example` to `.env` and fill only the provider credentials needed for
the command you plan to run. Never commit `.env`.

## Public-Release Data Policy

Included on GitHub:

- P1/P2/P3 MT outputs under `books/MT/`.
- Aggregate metrics, tables, and derived analysis outputs.
- Code, prompts, configs, tests, and documentation.

Withheld from GitHub:

- Source-language texts.
- Human translations.
- Raw source/HT chunk-review inputs.
- Run workspaces that contain source chunks.
- Raw human-evaluation exports and internal study dumps.
- Row-level participant comments, annotation exports, and disagreement viewers.

See `docs/release/withheld-files.tsv` and
`docs/release/sanitized-files.tsv` for the exact release audit manifests.

## License

Repository code and documentation are released under the MIT License unless a
file or subdirectory states a different license. Third-party components retain
their own licenses, including `par3/`, `metricx/`, and `LiTransProQA/`.
