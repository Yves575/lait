# MT Undetectability Category-Interval Analysis

## What Was Counted

Each Q1-Q3 coded open-ended response was reduced to counts of broad qualitative-code families: A = language-level features, B = narrative-level features, C = reader experience, and D = meta-translation. Q4 responses use their own M-style origin-cue codebook. Counts are per response, not global label totals. For example, two comma-separated A labels in one response contribute `A_count = 2` for that response.

Q3 comparison rationales were split into `q3_pos` and `q3_neg`: POS labels describe why the preferred translation was good, and NEG labels describe why the rejected translation was bad. Q4 AI-origin rationales were split into `q4_ai_telling` and `q4_human_telling`; because the Q4 export uses M-style origin-cue labels, Q4 plots and tests use `M1`-`M13` counts instead of A/B/C/D counts.

The intervals in the summary table and plots are empirical distribution summaries: median, Q1, Q3, min/max, and 5th/95th percentiles. They are not confidence intervals.

## Scenario Definitions

- Isolated undetected MT: actual `current_trans == MT`, guessed `thought_ai == HT`.
- Isolated detected MT: actual `current_trans == MT`, guessed `thought_ai == MT`.
- Isolated HT baselines: `HT` recognized as `HT` or misclassified as `MT`.
- Comparison undetected/preferred MT: `preferred_continue == MT` and `thought_ai == HT`.
- Comparison normal/preferred HT: `preferred_continue == HT` and `thought_ai == MT`.
- Additional exploratory groups include preference-only and AI-attribution-only splits.

## Generated Files

- `category_counts_by_response.csv`: tidy per-response category counts.
- `category_interval_summary.csv`: empirical interval summaries by scope and group (A/B/C/D for Q1-Q3; M1-M13 for Q4).
- `scenario_category_deltas.csv`: focal-minus-baseline differences and Cliff's delta.
- `category_scenario_tests.csv`: Mann-Whitney U tests with BH adjustment within scope.
- `*.png` and `*.pdf`: required and exploratory interval/point plots.

## Scope Sizes

- `q1_pos_isolated`: 60 response rows
- `q2_neg_isolated`: 60 response rows
- `q3_neg`: 30 response rows
- `q3_pos`: 30 response rows
- `q4_ai_telling`: 30 response rows
- `q4_human_telling`: 30 response rows

## Scenario Counts

`isolated_detection_group`:
- `isolated_ht_recognized`: 36
- `isolated_detected_mt`: 32
- `isolated_undetected_mt`: 28
- `isolated_ht_misclassified`: 24

`comparison_detection_group`:
- `comparison_normal_preferred_ht`: 68
- `comparison_undetected_preferred_mt`: 44
- `comparison_ht_preferred_ht_judged_ai`: 8

## Strongest Required-Comparison Deltas

These are descriptive focal-minus-baseline differences sorted by absolute mean difference. Small sample sizes, especially for `comparison_undetected_preferred_mt`, make these hypothesis-generating rather than confirmatory.

| Scope | Comparison | Category | Focal n | Baseline n | Mean delta | Median delta | Cliff's delta | p | BH q |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q1_pos_isolated` | `undetected_mt_vs_detected_mt` | `A_count` | 14 | 16 | 0.78 | 1 | 0.50 | 0.014 | 0.070 |
| `q1_pos_isolated` | `undetected_mt_vs_detected_mt` | `C_count` | 14 | 16 | 0.67 | 0 | 0.46 | 0.023 | 0.076 |
| `q2_neg_isolated` | `undetected_mt_vs_detected_mt` | `C_count` | 14 | 16 | -0.61 | 0 | -0.29 | 0.151 | 0.251 |
| `q3_neg` | `preferred_mt_thought_ht_vs_preferred_ht_thought_mt` | `C_count` | 11 | 17 | -0.52 | -1 | -0.24 | 0.298 | 0.767 |
| `q2_neg_isolated` | `undetected_mt_vs_detected_mt` | `D_count` | 14 | 16 | -0.43 | 0 | -0.31 | 0.052 | 0.174 |
| `q4_ai_telling` | `preferred_mt_thought_ht_vs_thought_mt` | `M9_count` | 11 | 17 | 0.34 | 0 | 0.34 | 0.052 | 0.574 |
| `q3_pos` | `preferred_mt_thought_ht_vs_preferred_ht_thought_mt` | `B_count` | 11 | 17 | 0.32 | 1 | 0.22 | 0.307 | 0.929 |
| `q1_pos_isolated` | `undetected_mt_vs_detected_mt` | `B_count` | 14 | 16 | 0.29 | 1 | 0.22 | 0.278 | 0.463 |
| `q3_pos` | `preferred_mt_thought_ht_vs_preferred_ht_thought_mt` | `C_count` | 11 | 17 | 0.27 | 0 | 0.07 | 0.740 | 0.929 |
| `q4_ai_telling` | `preferred_mt_thought_ht_vs_thought_mt` | `M5_count` | 11 | 17 | -0.26 | 0 | -0.26 | 0.132 | 0.618 |
| `q1_pos_isolated` | `undetected_mt_vs_detected_mt` | `D_count` | 14 | 16 | 0.23 | 0 | 0.10 | 0.500 | 0.715 |
| `q4_ai_telling` | `preferred_mt_thought_ht_vs_thought_mt` | `M10_count` | 11 | 17 | 0.19 | 0 | 0.19 | 0.287 | 0.821 |

## BH-Adjusted Results

The following primary category tests have BH q <= 0.05 within scope:
- `q2_neg_isolated` / `undetected_mt_vs_isolated_ht_recognized` / `A_count`: mean delta 1.32, BH q 0.036

## Caveats

- The primary purpose is visualization and hypothesis generation, not definitive inference.
- Some focal groups are very small; point overlays should be read before p-values.
- Q1/Q2 origin guesses are recovered from Part 1 because the coding exports leave `thought_ai` blank for isolated-reading rows.
- Q4 is not directly comparable to Q1-Q3 category families because it uses a separate M-style AI/human-origin cue codebook.
