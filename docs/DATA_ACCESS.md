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
authors with a short description of the intended research use. A gated Hugging
Face dataset is planned; once available, it will require users to accept access
terms before downloading restricted materials.

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
