# Human-Evaluation Outputs

This directory contains public human-evaluation aggregate outputs, model
summaries, figures, and scripts.

Interface screenshots and participant-facing guidelines are in
[`docs/paper_supplement/`](../docs/paper_supplement/).

For the manual process used to develop and calibrate the comment-coding
schemas, see
[`docs/COMMENT_CODING_SCHEMA_GUIDELINE.md`](../docs/COMMENT_CODING_SCHEMA_GUIDELINE.md).
The final category definitions are retained under
`analysis/manuscript_tables/tex/`.

Included:

- `figures/`: public visual summaries.
- `analysis_outputs/`: retained statistical model outputs and descriptive
  summaries.
- `comments_word_token_stats/`: aggregate count summaries for participant
  free-text responses.
- `plot_study_results.R`, `analyze_study_data.R`, and
  `generate_analysis_summaries.R`: analysis scripts that require
  controlled-access inputs for full reruns.

Not included in public GitHub:

- Raw study exports.
- Participant-level annotation exports.
- Row-level comments and text snippets.

## Controlled-access inputs

The controlled-access bundle includes these three analysis inputs:

- `data/part1-study-data-full.csv`
- `data/part2-study-data-full.csv`
- `data/part2-span-study-data-full.csv`

From the repository root, restore a locally shared bundle with:

```bash
python3 scripts/restore_controlled_access_data.py --dir controlled_access
python3 scripts/restore_controlled_access_data.py --dir controlled_access --apply
```

For downloads in `hf_dataset/`, omit `--dir controlled_access`. Restoration
creates `human_eval/data/` automatically; these CSVs remain ignored by Git.
The bundle does not include `events.csv` or `study-data-full.json`.
See [data access](../docs/DATA_ACCESS.md) for access terms and setup.
