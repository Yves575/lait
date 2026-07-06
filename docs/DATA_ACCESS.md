# Data Access

This public GitHub branch is designed to be useful for paper readers without
redistributing source texts or human translations.

## Included On GitHub

- Machine translation outputs for P1, P2, and P3 under `books/MT/`.
- Aggregate metric outputs and derived tables/figures.
- Code, prompts, configs, tests, and documentation.
- Sanitized derived files where source/HT text fields have been replaced with
  `[withheld from public GitHub release]`.

## Withheld From GitHub

- Source-language texts.
- Human translations.
- Chunked source or HT review inputs.
- Alignment views and datasets that embed source or HT text.
- Run workspaces containing source chunks.
- Raw human-evaluation exports and internal study dumps.
- Row-level participant comments, annotation exports, disagreement viewers, and
  local example tables that can expose restricted text or fine-grained study
  records.

The exact withheld-file manifest is `docs/release/withheld-files.tsv`.

## Requesting Controlled Access

Researchers who need source texts or human translations should email the
authors with a short description of the intended research use. The controlled
files are distributed through a gated Hugging Face dataset,
`<HUGGING_FACE_DATASET_URL>`; users must accept the access terms before
downloading restricted materials.

## Restoring Controlled-Access Text Locally

After cloning the public GitHub repository, download the gated Hugging Face
dataset files from `<HUGGING_FACE_DATASET_URL>` into `controlled_access/` at
the repository root:

```text
controlled_access/
  lait_books_controlled_access.jsonl
  withheld_file_replacements.jsonl
```

Preview the restoration plan:

```bash
python3 scripts/restore_controlled_access_data.py
```

Apply the restoration:

```bash
python3 scripts/restore_controlled_access_data.py --apply
```

The script uses the gated JSONL files to create the withheld source and human
translation files under `books/dev/`, `books/eval/`, and `books/HT/`, and to
replace sanitized metric/alignment files that contain
`[withheld from public GitHub release]` with their controlled-access text.
The script does not restore out-of-scope internal workspaces or raw private
study records that are not part of the gated release.

## Public Release Audit

Two manifests support manual inspection:

- `docs/release/withheld-files.tsv`: files removed from the public branch.
- `docs/release/sanitized-files.tsv`: files retained with text fields redacted.
- `docs/release/human-eval-and-litransproqa-removals.txt`: row-level
  human-evaluation and LiTransProQA dataset files removed in the final
  public-safety pass.

If a withheld file should become public, restore it deliberately and record why
it does not expose source text, human translation text, private participant
data, or internal-only material.
