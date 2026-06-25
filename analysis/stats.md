# Statistical Analysis Plan

This document defines a practical statistics plan for the study data in this repository.
It is optimized for:

- repeated measures (same readers across stages),
- nested structure (chunks within participant-book assignments),
- small sample size with some missingness/partial completion,
- potential telemetry quality concerns (e.g., `p009`, `p012` sensitivity).

## 1) Data Structure

### Units

- **Participant**: one user account (e.g., `p013_02`).
- **Assignment**: one participant on one book.
- **Single reading row**: questionnaire response after reading HT or MT alone.
- **Comparison row**: final side-by-side preference/AI-guess questionnaire.
- **Chunk row**: one annotated chunk in side-by-side task.

### Canonical person collapse (critical)

Before modeling, collapse aliases that represent the same real reader into one `person_id` (canonical person identifier), then use `person_id` for random effects.

At minimum, apply the known merges:

- `p0013_01` and `p013_02` -> same person
- `lauren_p1` and `p001_02` -> same person

If the full reader-pair mapping is available, use it for all pairs and define one canonical `person_id` per pair.

This avoids treating the same reader as two independent participants.

### Key grouping factors

- `person_id` (canonical reader identity after merge)
- `book_id` (canonicalized)
- `language` (book language)
- `order` (`HT-first` vs `MT-first`)

### Canonical coding rules

- Always decode T1/T2 to actual `HT`/`MT` before agreement or preference models.
- Keep `No meaningful difference` explicitly for comparison Q1/Q2 (and additionally run a no-difference-removed variant).
- For comparison Q3 preference, collapse to nominal `HT` vs `MT` for main inference.

## 2) Research Questions And Models

## RQ1. What do readers prefer overall (HT vs MT)?

Primary endpoint: comparison Q3 collapsed to `prefer_HT` (1=yes, 0=no).

Recommended model:

```r
glmer(prefer_HT ~ order + language +
      (1 + order | person_id) +
      (1 | book_id),
      family = binomial, data = comparison_df)
```

Notes:

- This directly answers "overall HT vs MT preference" while accounting for:
  - `order` (fixed effect and random slope by person),
  - `book` (random intercept),
  - `source/language` (fixed effect),
  - `participant` (random intercept via `person_id`).
- Because N is small, this may become singular. Fallback sequence:
  1. `(1 + order | person_id) + (1 | book_id)`
  2. `(1 | person_id) + (1 | book_id)` (if random slope fails)

Report:

- HT/MT proportions (descriptive),
- odds ratio for HT preference,
- 95% CI (or 95% CrI if Bayesian).

## RQ2. Can readers identify MT?

Analyze as two distinct tasks:

1. **Single-reading origin guess** (Q7, decoded correctness vs actual version).
2. **Comparison AI guess** (Q5, decoded correctness whether guessed version is MT).

Model template:

```r
glmer(correct_guess ~ stage + order + language + actual_version +
      (1 + order | person_id) + (1 | book_id),
      family = binomial, data = guess_df)
```

Also report exact binomial tests against chance (`p=0.5`) per stage.

## RQ3. Do readers prefer MT when they know it is MT?

Outcome: `prefer_MT` from comparison Q3.
Key predictor: `correctly_identified_MT` from comparison Q5.

```r
glmer(prefer_MT ~ correctly_identified_MT + order + language +
      (1 + order | person_id) + (1 | book_id),
      family = binomial, data = comparison_df)
```

Also provide simple count table (interpretable with small N):

- preferred HT/MT x correctly identified MT yes/no.

## RQ4. Are single-reading quality ratings higher for HT or MT?

Outcomes: Q1 Acceptability, Q2 Smoothness, Q3 Immersion, Q4 Continue (ordinal 1-5).

Preferred:

```r
ordinal::clmm(rating ~ version + order + language +
              (1 + order | person_id) + (1 | book_id), data = long_ratings_df)
```

Alternative robust option:

```r
brms::brm(rating ~ version + order + language +
          (1 + order | person_id) + (1 | book_id),
          family = cumulative())
```

## RQ5. Does chunk-level preference align with full-text preference?

Two views:

1. Assignment-level transition (comparison preference vs chunk majority HT/MT/tie).
2. Chunk-level mixed model:

```r
glmer(chunk_pref_HT ~ comparison_pref_HT + difficulty + order + language +
      (1 + order | person_id) + (1 | book_id) + (1 | chunk_id),
      family = binomial, data = chunk_df)
```

## RQ6. Does order (HT-first vs MT-first) shift outcomes?

Include `order` in all primary models above.
Also report descriptive split:

- HT-first: % prefer HT,
- MT-first: % prefer HT,
- and same split for MT-identification accuracy.

## 3) Agreement (IAA) Framework

Treat IAA as reliability/audit, not sole evidence of effect.

## Single reading

- Q1 Acceptable for reader (ordinal 1-5)
- Q1 collapsed (1-2 / 3 / 4-5)
- Q2 Smoothness (ordinal 1-5)
- Q2 collapsed (1-2 / 3 / 4-5)
- Q3 Immersion (ordinal 1-5)
- Q3 collapsed (1-2 / 3 / 4-5)
- Q4 Want to continue this version? (ordinal 1-5)
- Q4 collapsed (1-2 / 3 / 4-5)
- Q7 MT guess (single-read; nominal HT/MT decoded)

## Comparison

- Q1 Better dialogue (nominal HT/MT/no-difference decoded)
- Q1 Better dialogue (excluding no-difference)
- Q2 Better word choice (nominal HT/MT/no-difference decoded)
- Q2 Better word choice (excluding no-difference)
- Q3 Preferred to continue (collapsed HT/MT decoded)
- Q5 Which is AI? (nominal HT/MT decoded)

## Chunks

- Step 2 Preference (nominal HT/MT decoded)
- Step 1 Any good span on HT (presence)
- Step 1 Any bad span on HT (presence)
- Step 1 Any good span on MT (presence)
- Step 1 Any bad span on MT (presence)

Metrics to report per item:

- pairwise agreement (%),
- Krippendorff alpha,
- Gwet AC1/AC2,
- `n` overlapping items and pairs.

## 4) Sensitivity / Robustness

Run all key descriptive and inferential outputs under:

1. all participants,
2. exclude `p009`,
3. exclude `p012`,
4. exclude both `p009` and `p012`.

Optional targeted sensitivity:

- exclude only `p012` MT single-reading session (instead of removing all her data).

For write-up, show whether conclusions flip or remain directionally stable.

## 5) Qualitative Analysis (Open-Ended Responses)

Use the extracted CSVs in `analysis/questions/`.

Codebook (recommended):

- fluency/naturalness,
- awkward literalness,
- lexical choice richness,
- sentence structure,
- dialogue handling,
- imagery/vividness,
- emotional tone,
- punctuation/formatting issues,
- clarity/ease,
- mistranslation/wrong term.

Deliverables:

- frequency table by code and condition (HT vs MT praised/disliked),
- representative quotes per major code,
- optional cross-tabs by preference group.

## 6) Reporting Standards

For each RQ include:

- effect size (OR or probability difference),
- uncertainty interval,
- denominator (`n`) and overlap constraints where relevant,
- sensitivity panel (all vs exclusions),
- one plain-language interpretation sentence with caveat.

Avoid over-claiming population inference given sample size and partial missingness.

## 7) Minimal Reproducible Workflow

1. Regenerate analysis data:

```bash
npm run reports -- --input "/path/to/study-data-full.json" --out analysis
npm run iaa -- --input "/path/to/study-data-full.json" --out analysis/iaa-report.md
```

2. Export open-ended question datasets:

```bash
npm run questions -- --input "/path/to/study-data-full.json"
```

3. Run statistical models in R from exported/derived tables.

---

This plan is intentionally conservative and emphasizes decoded HT/MT outcomes, mixed-effects modeling, and explicit sensitivity analysis over single-point summaries.
