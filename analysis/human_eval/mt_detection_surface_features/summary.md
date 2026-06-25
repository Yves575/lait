# MT Detection Surface/Lexical Feature Analysis

## What This Tests

This analysis compares actual MT excerpts that participants guessed were `HT` against actual MT excerpts that participants guessed were `MT` in isolated reading.

The participant-level rows are descriptive because the same MT excerpt can be judged differently by different readers. Book-level correlations with `undetected_rate` are therefore the cleaner text-feature view, though n is only 15 books.

## Data Checks

- Isolated MT judgment rows: 30
- MT guessed HT: 14
- MT guessed MT: 16
- Canonical books with MT text: 15
- Canonical books with paired HT eval text: 15
- Books with mixed reader judgments: 10

## Generated Files

- `mt_judgment_feature_table.csv`: one row per isolated MT judgment.
- `book_level_feature_table.csv`: one row per canonical book with undetected rate.
- `feature_test_summary.csv`: participant-level Mann-Whitney tests and Cliff's delta.
- `book_level_feature_correlations.csv`: book-level Spearman correlations.
- Paired features use `mt_minus_ht_*`, `abs_mt_minus_ht_*`, and `mt_div_ht_*` columns.
- Chunk-aligned features aggregate local MT-vs-HT chunk differences from aligned JSONL chunks.
- `*.png` and `*.pdf`: boxplots, effect summary, book scatterplots, and heatmap.

## Largest Participant-Level Effects

| Feature | Mean delta | Cliff's delta | p | BH q |
|---|---:|---:|---:|---:|
| `mt_div_ht_slash_rate` | -0.642 | -0.667 | 0.617 | 1 |
| `mt_minus_ht_dash_style_count` | -0.562 | -0.438 | 0.025 | 0.837 |
| `mt_div_ht_dash_style_count` | -0.256 | -0.429 | 0.031 | 0.837 |
| `q2_SG_A_sentence_count` | -0.402 | -0.402 | 0.033 | 0.837 |
| `mt_div_ht_quote_rate` | -0.120 | -0.393 | 0.070 | 0.837 |
| `mean_signed_chunk_delta_en_dash_rate` | -1.017 | -0.339 | 0.090 | 0.837 |
| `mt_minus_ht_en_dash_rate` | -0.997 | -0.339 | 0.090 | 0.837 |
| `quote_rate` | -6.026 | -0.339 | 0.118 | 0.837 |
| `dash_style_count` | -0.402 | -0.321 | 0.092 | 0.837 |
| `q1_SG_C_smoothness_count` | 0.402 | 0.321 | 0.090 | 0.837 |

## Strongest Book-Level Correlations

| Feature | Spearman rho | permutation p | BH q |
|---|---:|---:|---:|
| `mt_minus_ht_dash_style_count` | -0.732 | 0.004 | 0.152 |
| `mt_div_ht_dash_style_count` | -0.707 | 0.004 | 0.152 |
| `slash_rate` | -0.651 | 0.039 | 0.268 |
| `mt_div_ht_quote_rate` | -0.588 | 0.021 | 0.371 |
| `en_dash_rate` | -0.567 | 0.044 | 0.371 |
| `mean_signed_chunk_delta_en_dash_rate` | -0.561 | 0.026 | 0.371 |
| `mt_minus_ht_en_dash_rate` | -0.561 | 0.026 | 0.371 |
| `dash_style_count` | -0.555 | 0.052 | 0.371 |
| `abs_mt_minus_ht_slash_rate` | -0.528 | 0.069 | 0.406 |
| `share_chunks_mt_more_en_dash` | -0.528 | 0.067 | 0.406 |

## Strongest Paired MT-vs-HT Book-Level Correlations

| Feature | Spearman rho | permutation p | BH q |
|---|---:|---:|---:|
| `mt_minus_ht_dash_style_count` | -0.732 | 0.004 | 0.152 |
| `mt_div_ht_dash_style_count` | -0.707 | 0.004 | 0.152 |
| `mt_div_ht_quote_rate` | -0.588 | 0.021 | 0.371 |
| `mt_minus_ht_en_dash_rate` | -0.561 | 0.026 | 0.371 |
| `abs_mt_minus_ht_slash_rate` | -0.528 | 0.069 | 0.406 |
| `mt_div_ht_en_dash_rate` | -0.500 | 0.539 | 0.985 |
| `abs_mt_minus_ht_quote_rate` | 0.347 | 0.202 | 0.985 |
| `mt_minus_ht_quote_rate` | -0.347 | 0.202 | 0.985 |
| `abs_mt_minus_ht_dash_rate` | -0.297 | 0.288 | 0.985 |
| `mt_minus_ht_std_sentence_words` | -0.287 | 0.304 | 0.985 |

## Strongest Chunk-Aligned MT-vs-HT Correlations

| Feature | Spearman rho | permutation p | BH q |
|---|---:|---:|---:|
| `mean_signed_chunk_delta_en_dash_rate` | -0.561 | 0.026 | 0.371 |
| `share_chunks_mt_more_en_dash` | -0.528 | 0.067 | 0.406 |
| `share_chunks_dash_style_mismatch` | -0.364 | 0.182 | 0.985 |
| `mean_signed_chunk_delta_quote_rate` | -0.321 | 0.238 | 0.985 |
| `mean_abs_chunk_delta_quote_rate` | 0.295 | 0.282 | 0.985 |
| `mean_abs_chunk_delta_hyphen_rate` | 0.282 | 0.307 | 0.985 |
| `mean_signed_chunk_delta_em_dash_rate` | 0.276 | 0.321 | 0.985 |
| `mean_abs_chunk_delta_mtld` | 0.265 | 0.345 | 0.985 |
| `share_chunks_mt_more_quote` | -0.261 | 0.337 | 0.985 |
| `aligned_chunk_count` | 0.254 | 0.360 | 0.985 |

## Caveats

- This is hypothesis-generating; do not treat the p-values as confirmatory.
- Text features repeat for participants who read the same book.
- A mixed book means the exact same MT text was guessed HT by one reader and MT by another.
- Paired MT-vs-HT deltas are more interpretable than absolute MT features for comparison-style judgments.
- Chunk-aligned scores are better local surface-style comparisons, but they still aggregate to only 15 book-level points.
- Qualitative subgroup features come from participants' comments, not from the MT text itself.
