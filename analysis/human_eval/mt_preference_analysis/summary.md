# MT Preference And AI Undetectability Analysis

This analysis focuses on whole-excerpt comparison cases where participants preferred MT
for continued reading and judged HT as more likely AI-translated.

## Core Counts

- Part 1 comparison rows: 30
- MT-preference cases: 11
- MT-preference cases that judged HT more AI-like: 11

Preference by AI-attribution crosstab:

| Preferred to continue | Thought HT AI | Thought MT AI |
| --- | ---: | ---: |
| HT | 2 | 17 |
| MT | 11 | 0 |

## Scenario Counts

| Scenario | Cases |
| --- | ---: |
| Relative MT advantage | 9 |
| MT strength | 2 |

## Strongest Q3 Label Differences Against HT-Preference Norm

Positive values mean the label appears in a larger share of MT-preference cases than
HT-preference cases. Fisher p-values and BH q-values are exploratory because n is small.

| Polarity | Label | MT rate | HT rate | Delta | Fisher p | BH q |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| POS | Enjoyment / overall positive affect | 45% | 5% | +40 pp | 0.016 | 0.778 |
| NEG | Word choice: clarity | 55% | 16% | +39 pp | 0.042 | 0.925 |
| POS | Smoothness / reading effort | 45% | 79% | -33 pp | 0.108 | 0.925 |
| POS | Sentence structure | 0% | 32% | -32 pp | 0.061 | 0.925 |
| POS | Engagement / immersion | 45% | 21% | +24 pp | 0.225 | 0.925 |
| POS | Character voice & portrayal | 27% | 5% | +22 pp | 0.126 | 0.925 |
| POS | Word choice: richness vs. blandness | 36% | 16% | +21 pp | 0.372 | 0.925 |
| POS | Faithfulness to original | 18% | 0% | +18 pp | 0.126 | 0.925 |

## Output Files

- `q3_preference_norm_label_lift.csv`: Q3 MT-preference vs HT-preference label rates and tests.
- `q3_preference_norm_family_rates.csv`: Q3 family-level baseline rates by preference group.
- `q5_chunk_preference_norm_label_lift.csv`: Q5 chunk-level MT-preferred vs HT-preferred label rates and tests.
- `q5_chunk_preference_norm_family_rates.csv`: Q5 family-level baseline rates by chunk preference.
- `book_level_mt_favorability.csv`: book-level comparison and chunk preference summary.
- `*.png` / `*.pdf`: plots for the analysis.

Row-level case files and label-record exports are withheld from the public
GitHub branch because they contain participant comments or fine-grained
evaluation records.
