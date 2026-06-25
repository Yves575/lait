# Part 2 Chunk Paired HT-vs-MT Features

## What Was Counted

- Input responses from the controlled-access human-evaluation export were
  filtered to HT/MT chunk preferences.
- Coded rationales were joined by participant base, canonical book ID, and
  Part 2 chunk ID.
- Span highlights were aggregated by participant/book/chunk/version.
- Chunk texts came from controlled-access HT chunks and public MT chunks.
- Part 2 chunk IDs are zero-based in the CSVs; JSONL chunk IDs are one-based here. The output keeps `chunk_id`, `jsonl_chunk_id`, and `chunk_id_indexing_note`.

The paired feature columns use `mt_minus_ht_*`, `abs_mt_minus_ht_*`, and `mt_div_ht_*`.
Positive `mt_minus_ht_*` values mean the MT chunk has more of that feature than the paired HT chunk.

## Exploratory Cautions

This is a descriptive screen, not a confirmatory model. Participant responses are not independent:
chunks are nested in books, and many chunks have two participant responses. Mann-Whitney and Spearman
p-values are included to rank patterns, but effect sizes and plots should carry more weight.

## Row Counts

- Chunk-pair feature rows: 386
- Participant-response rows: 772
- Books in book-level summary: 15

## Strongest Participant-Level Feature Deltas

| feature                               | n_focal | n_baseline | mean_delta_focal_minus_baseline | cliffs_delta | p_value | q_value |
| ------------------------------------- | ------- | ---------- | ------------------------------- | ------------ | ------- | ------- |
| mt_minus_ht_mean_sentence_words       | 250     | 522        | -1.344                          | -0.225       | 0.000   | 0.000   |
| mt_minus_ht_character_count           | 250     | 522        | -60.773                         | -0.223       | 0.000   | 0.000   |
| mt_minus_ht_median_sentence_words     | 250     | 522        | -1.490                          | -0.211       | 0.000   | 0.000   |
| mt_minus_ht_token_count               | 250     | 522        | -9.985                          | -0.181       | 0.000   | 0.000   |
| mt_minus_ht_word_count                | 250     | 522        | -10.027                         | -0.178       | 0.000   | 0.001   |
| mt_minus_ht_punctuation_density       | 250     | 522        | 0.001                           | 0.167        | 0.000   | 0.001   |
| mt_minus_ht_contraction_rate          | 250     | 522        | 3.747                           | 0.164        | 0.000   | 0.001   |
| mt_minus_ht_max_sentence_words        | 250     | 522        | -2.721                          | -0.164       | 0.000   | 0.001   |
| mt_minus_ht_repeated_blank_line_count | 250     | 522        | -1.003                          | -0.162       | 0.000   | 0.000   |
| mt_minus_ht_hyphen_rate               | 250     | 522        | 1.625                           | 0.161        | 0.000   | 0.002   |

## Strongest Book-Level Correlations

| feature                         | n_books | spearman_rho | p_value | permutation_p_value | permutation_q_value |
| ------------------------------- | ------- | ------------ | ------- | ------------------- | ------------------- |
| mt_minus_ht_char_span_balance   | 15      | 0.917        | 0.000   | 0.000               | 0.013               |
| mt_minus_ht_count_span_balance  | 15      | 0.917        | 0.000   | 0.000               | 0.013               |
| mt_div_ht_hyphen_rate           | 15      | 0.731        | 0.002   | 0.003               | 0.111               |
| mt_div_ht_long_sentence_share   | 15      | -0.706       | 0.003   | 0.003               | 0.111               |
| mt_minus_ht_hyphen_rate         | 15      | 0.636        | 0.011   | 0.013               | 0.330               |
| mt_div_ht_median_sentence_words | 15      | -0.617       | 0.014   | 0.017               | 0.346               |
| mt_div_ht_mean_sentence_words   | 15      | -0.608       | 0.016   | 0.021               | 0.346               |
| mt_minus_ht_mean_sentence_words | 15      | -0.597       | 0.019   | 0.024               | 0.346               |
| abs_mt_minus_ht_hyphen_rate     | 15      | 0.574        | 0.025   | 0.030               | 0.360               |
| mt_minus_ht_contraction_rate    | 15      | 0.567        | 0.028   | 0.026               | 0.346               |

Row-level participant-response tables and case-study exports are withheld from
the public GitHub branch because they can contain participant comments or
fine-grained evaluation records.
