# Preprint Scope Audit

This note records the release-scope review used for this public branch. The
goal is to keep the GitHub repository aligned with the preprint while
withholding source texts, human translations, and internal/raw materials.

## Review Passes

1. The preprint PDF was parsed for section headings, appendix headings, table
   and figure references, and named systems/tools.
2. A second text pass checked appendix-heavy terms against the retained tree,
   including the five MT systems, automatic metrics, human-evaluation outputs,
   agent pipeline summaries, and supplementary materials.
3. The retained file tree was then checked against those findings, with removed
   paths recorded in `docs/release/withheld-files.tsv`.

## In Scope For GitHub

- P1/P2/P3 MT outputs for the five public systems discussed in the preprint:
  P1 Gemini, P1 GPT-5.4 High, P2 Gemini, P2 GPT-5.4 High, and P3 Agents.
- Aggregate metric outputs for the metrics reported in the preprint:
  COMET-22, COMETKiwi, MetricX, MetricX-QE, and LiTransProQA.
- Human-evaluation aggregate outputs, coding summaries, and retained
  manuscript tables, with source/HT text fields redacted where applicable.
- Agent-pipeline code, prompts, configs, and public documentation.
- Third-party or supporting code for retained methods, including `par3/`,
  `metricx/`, and `LiTransProQA/`, subject to their own licenses.

## Out Of Scope For GitHub

- Source-language texts and human translations.
- Raw human-evaluation exports, internal study dumps, run workspaces, and
  binary aligned data that may embed withheld text.
- Row-level participant comments, annotation exports, disagreement viewers, and
  case-study/example tables that can expose restricted text or fine-grained
  study records.
- LiTransProQA local datasets and notebooks that embed source/target text or
  out-of-scope metric experiments.
- Out-of-preprint MT systems and their derived metric outputs.
- Generated figures/tables from earlier broader analyses that do not match the
  preprint's five-system comparison.
- GEMBA/GPTZero detector workflows and outputs; these are not reported in the
  preprint.
- Scratch/internal alignment, HT chunk-review, standalone Q5 agreement, and
  superseded vocabulary-comparison subprojects.

## Appendix-Sensitive Decisions

- The preprint mentions Pangram in the context of participant-comment quality
  checks, but the tracked `results_ai_detection/` files are detector outputs
  over book text and are not the participant-comment QA described there.
- The preprint discusses Q5 coding and agreement, but the standalone
  `q5_agreement/` directory duplicated scratch/raw-comment workflow material.
  The retained human-evaluation outputs carry the public aggregate analysis.
- The preprint discusses n-gram convergence, but older generated n-gram
  artifacts were produced for a broader system inventory. They were removed
  rather than shipped as stale supporting material.
- Agent-run summary TeX files that reported older run counts were removed to
  avoid contradicting the appendix.
