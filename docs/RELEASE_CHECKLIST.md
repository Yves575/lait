# Public Release Checklist

Use this checklist before publishing the GitHub branch.

## Branch And License

- [ ] Branch is `release/public-preprint`.
- [ ] Root `LICENSE` is MIT.
- [ ] Third-party licenses remain in place for `par3/`, `metricx/`, and `LiTransProQA/`.

## Data Boundary

- [ ] `docs/release/withheld-files.tsv` has been reviewed by the authors.
- [ ] `docs/release/sanitized-files.tsv` has been reviewed by the authors.
- [ ] `docs/release/CHANGE_AUDIT.md` has been reviewed against the branch diff.
- [ ] No source-language books are present in public Git.
- [ ] No human translations are present in public Git.
- [ ] Raw human-evaluation exports and internal dumps are absent.
- [ ] P1/P2/P3 MT outputs remain present under `books/MT/`.

## Reader Navigation

- [ ] Root README links to P1/P2/P3, data access, repo map, and reproducibility.
- [ ] `START_HERE.md` is understandable for readers who only skimmed the paper.
- [ ] `docs/DATA_ACCESS.md` explains email/gated Hugging Face access.
- [ ] `docs/REPO_MAP.md` matches the actual public tree.

## Verification

- [ ] `python -m pytest` passes in a prepared environment.
- [ ] CLI smoke checks pass:
  - [ ] `python mt_pipeline.py --help`
  - [ ] `python agents_pipeline/runner.py --help`
  - [ ] `python mt_eval.py --help`
  - [ ] `python eval_pipeline.py --help`
- [ ] Leakage checks find no unexpected source/HT files or fields.
- [ ] The release branch diff has been reviewed by at least one coauthor.
