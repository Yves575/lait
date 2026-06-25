# Human eval analysis outputs — concise summary

Source: [analyze_study_data.R](../analyze_study_data.R) → `human_eval/analysis_outputs/` (n ≈ 60 single-reading rows, 30 Part 1 assignments, 772 Part 2 chunk rows, 15 readers).

Paper summaries: [generate_analysis_summaries.R](../generate_analysis_summaries.R) → `part_1/single_reading/summaries/`.

**Convention:** HT = human translation, MT = machine translation; `typeMT` / `versionMT` = MT vs HT reference; `orderMT-first` = presentation order. \* = p < .05 unless noted.

**Test abbreviations:** CLMM = cumulative link mixed model (ordinal); CLM = cumulative link model (ordinal, no RE); LMER = linear mixed model; GLMER = generalized LMER (binomial logit); GLM = generalized linear model (binomial or quasibinomial logit); exact binomial = one-sample `binom.test` vs 50%; Spearman/Kendall/Pearson = rank/linear correlation (`cor.test`).

Every output file begins with a `Test:` header line stating the inferential test (or `Test: None (descriptive)`).

---

## `_shared/`

Cross-part documentation and reference files.

| File | Test | Result | Interpretation |
|------|------|--------|----------------|
| `preference_descriptives.txt` | None (counts, proportions) | Chunk-level **68% HT** (522/772); assignment majority **HT 21 / MT 7 / tie 2**; mean prop HT **0.68** | Descriptive HT majority at chunk and assignment level. |
| `response_variables.txt` | None (reference map) | — | Maps each response variable to intended model type and output path. |

---

## `part_1/`

Part 1 single-reading and side-by-side comparison tasks.

### `part_1/single_reading/`

#### `part_1/single_reading/ordinal/`

Single-reading Likert ratings (1–5): `q* ~ type + order + (1 | reader) + (1 | book)`.

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `q1_acceptability_clmm.txt` | **CLMM** (ordinal logit) | Acceptability | **HT > MT** (typeMT β = −1.38, **p = .007**); order ns | HT excerpts rated clearly more acceptable. |
| `q2_smoothness_clmm.txt` | **CLMM** (ordinal logit) | Smoothness | **HT > MT** (β = −1.49, **p = .003**); singular RE; order ns | HT rated smoother; reader/book variance not estimable. |
| `q2_smoothness_clm.txt` | **CLM** fallback (ordinal logit): `q2 ~ type + order` | Smoothness | Same fixed effects; emmeans mean class **HT 4.23 vs MT 3.33** | Confirms HT smoothness edge without random effects. |
| `q3_immersion_clmm.txt` | **CLMM** (ordinal logit) | Immersion | typeMT β = −0.83, **p = .082** (marginal); singular RE; order ns | Weak HT advantage on immersion, not significant at α = .05. |
| `q3_immersion_clm.txt` | **CLM** fallback (ordinal logit): `q3 ~ type + order` | Immersion | Same; emmeans **HT 3.91 vs MT 3.32** | Same marginal HT tilt on immersion. |
| `q4_continue_reading_clmm.txt` | **CLMM** (ordinal logit) | Continue reading | typeMT p = .115; order ns | No difference in willingness to continue reading. |

#### `part_1/single_reading/word_count/`

Open-ended response lengths: `q*_nbr_words ~ version + order + (1 | reader) + (1 | book)`.

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `q5_open_response_lmer.txt` | **LMER** (Gaussian) | Open response length (q5) | versionMT −3.5 words, p = .72; order ns; singular book RE | Open-ended answers equally long for HT and MT. |
| `q6_follow_up_lmer.txt` | **LMER** (Gaussian) | Follow-up word count (q6) | versionMT +6.2, p = .87; orderMT-first −8.7, p = .22 | Word count similar; slight shorter answers after MT-first, ns. |

#### `part_1/single_reading/origin_guess/`

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `origin_guess_glmer_binomial.txt` | **GLMER** (binomial logit): `origin_guess ~ version + order + (1\|reader) + (1\|book)` | P(guess MT) | versionMT OR ns (p = .30); order ns; singular RE | Origin guesses not tied to which version was read. |

#### `part_1/single_reading/ai_identification/`

Single-reading origin guess and Q8 confidence.

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `confusion_matrix.txt` | None (confusion matrix, row proportions) | Single-reading origin guess | **Accuracy 56.7%**; sensitivity MT 53.3%; specificity HT 60%; bias toward guessing MT 46.7% | Modest discrimination, near chance overall. |
| `correct_guess_binom.txt` | **Exact binomial test** (`binom.test`, H₀: p = 0.5) | Single-reading accuracy vs chance | 34/60 correct, **p = .37** | Single-reading guesses indistinguishable from chance. |
| `guessed_MT_glm_binomial.txt` | **GLM** (binomial logit): `guessed_MT ~ actual_version + single_read_position` | P(guess MT) | actual MT ↑ guess MT (p = .30); position ns; emmeans P(MT guess) **HT .40, MT .53** | Slight MT-label bias when text is MT, not significant. |
| `guessed_MT_glmer_binomial.txt` | **GLMER** (binomial logit): `guessed_MT ~ actual_version + single_read_position + (1\|person_id) + (1\|book_id)` | P(guess MT) | Identical fixed effects; **singular** RE | Random effects add nothing over the fixed GLM. |
| `confidence_clm.txt` | **CLM** (ordinal logit): `confidence_ord ~ correct_guess_factor + actual_version` | Q8 confidence (1–5) | Both ns; emmeans confidence **~3.6** either way | Confidence unrelated to correctness or version. |

#### `part_1/single_reading/summaries/`

Manuscript summaries from [generate_analysis_summaries.R](../generate_analysis_summaries.R) (derived from CLMM/CLM outputs above).

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `q1_summary.txt` | None (paper summary) | Q1 — Acceptability | See file | Plain-language summary for manuscript. |
| `q2_summary.txt` | None (paper summary) | Q2 — Smoothness | See file | Uses CLM fallback where CLMM is singular. |
| `q3_summary.txt` | None (paper summary) | Q3 — Immersion | See file | Uses CLM fallback where CLMM is singular. |
| `q4_summary.txt` | None (paper summary) | Q4 — Continue reading | See file | Plain-language summary for manuscript. |
| `all_questions_summary.txt` | None (combined) | Q1–Q4 | — | Combined paper summaries. |

### `part_1/comparison/`

Side-by-side excerpt comparison. Q1/Q2 are three-level ordinal (MT < NO DIFF < HT), fit with CLMM/CLM — not binomial GLMER. `group` = source language (French baseline); n = 30 assignments.

Subfolders: `ordinal/`, `word_count/`, `ai_identification/`, `excerpt_preference/`, `preference_confidence/` (Q3 strength × Q6 confidence), `perceived_ht_preference/` (Q3 preference × Q5 AI tag).

#### `part_1/comparison/ordinal/`

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `preferred_overall_clmm.txt` | **CLMM** (ordinal logit): levels MT < NO DIFF < HT | Q1 — Overall winner | orderMT-first β = 1.40, **p = .10**; group ns; RE warning (n = 30) | Marginal trend: MT-first → higher HT preference on ordered scale. |
| `preferred_overall_clm.txt` | **CLM** fallback (ordinal logit) | Q1 — Overall winner | orderMT-first **p = .084**; emmeans mean class **French 2.11, Japanese 2.22, Polish 2.52** | Confirms marginal order effect; mean class ≈ between NO DIFF and HT. |
| `smoother_clmm.txt` | **CLMM** (ordinal logit): levels MT < NO DIFF < HT | Q2 — Smoother translation | order **p = .19**; group ns; **singular** RE | No reliable smoother preference; RE not estimable. |
| `smoother_clm.txt` | **CLM** fallback (ordinal logit) | Q2 — Smoother translation | Same fixed effects; emmeans mean class **French 2.39, Japanese 2.00, Polish 2.42** | Same null results without random effects. |

#### `part_1/comparison/word_count/`

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `q4_explanation_lmer.txt` | **LMER** (Gaussian): `comparison_q4_nbr_words ~ group + order + (1\|reader) + (1\|book)` | Q4 — Open explanation length | Japanese/Polish vs French ns; order ns; singular book RE | Comparison explanations similar across languages. |
| `q7_second_response_lmer.txt` | **LMER** (Gaussian): `comparison_q7_nbr_words ~ group + order + (1\|reader) + (1\|book)` | Q7 — Second open response length | All fixed effects ns | Second open response length unaffected by language or order. |

#### `part_1/comparison/ai_identification/`

Comparison-stage MT identification and Q6 confidence.

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `descriptives.txt` | None (counts, mean proportion) | Comparison-stage MT identification | 17/30 correct; mean **0.567** | Side-by-side identification also ~57% correct. |
| `correct_guess_binom.txt` | **Exact binomial test** (`binom.test`, H₀: p = 0.5) | Comparison accuracy vs chance | **p = .58** | Comparison accuracy not above chance. |
| `ai_guess_glm_binomial.txt` | **GLM** (binomial logit): `ai_guess_correct ~ order` | P(correct MT id) | order ns; overall emmean **0.57** | Presentation order does not aid MT detection. |
| `stage_accuracy_glm_quasibinomial.txt` | **GLM** (quasibinomial logit): `cbind(n_correct, n_wrong) ~ stage + order` | Accuracy by task stage | stage ns; order ns; emmeans **both stages ~0.57** | Neither task stage is easier than the other. |
| `confidence_clm.txt` | **CLM** (ordinal logit): `confidence_ord ~ correct_guess_factor + order` | Q6 confidence (1–5) | correct trend **p = .096**; emmeans **wrong 3.41 vs correct 4.05**; order ns | Correct identifiers somewhat more confident, borderline. |

Q6 is also modeled against Q3 preference strength in `preference_confidence/` (below).

#### `part_1/comparison/excerpt_preference/`

Continue-reading preference from comparison Q3 (excerpt-level HT vs MT).

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `prefer_HT_glm_binomial.txt` | **GLM** (binomial logit): `excerpt_pref_HT ~ order` | P(prefer HT excerpt) | order ns; emmeans **HT-first .60, MT-first .67**; overall **.63** prefer HT | ~63% pick HT; order does not shift preference. |
| `prefer_HT_glmer_binomial.txt` | **GLMER** (binomial logit): `excerpt_pref_HT ~ order + (1\|reader) + (1\|book)` | P(prefer HT excerpt) | Same fixed effects; **singular** RE | Reader/book variation not estimable here. |
| `preference_strength_glm_binomial.txt` | **GLM** (binomial logit): `excerpt_clear ~ excerpt_preference + order` | P(clear vs slight preference) | HT preference → clearer (p = .22); emmeans P(clear) **MT .64, HT .84** | HT choices look stronger but effect is not significant. |

#### `part_1/comparison/preference_confidence/`

**Research question:** Is how strongly a reader prefers one translation (Q3) associated with how confident they are in their AI-identification choice (Q6, regarding Q5)?

**Coding (from comparison questionnaire):**

- **Q3** (continue-reading preference): raw 1–4 = Translation 1/2 × clearly/slightly better → `excerpt_strength` = **slight** (2, 3) vs **clear** (1, 4); `excerpt_clear` = 1 when clear.
- **Q6** (confidence in Q5): ordinal **1–5** (not at all → extremely confident); stored as `comparison_q6` / `ai_confidence`.

**Synthesis (n = 30):** Descriptives and regression models point the same way—clearer Q3 choices go with higher Q6—but correlation tests do not reach α = .05, likely limited by sample size. Treat the **GLM** (*p* = .097) and **Pearson** (*p* = .08) as marginal evidence, not confirmation.

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `descriptives.txt` | None (crosstabs, means) | Strength × confidence | Mean Q6 **3.14** (slight, n = 7) vs **3.96** (clear, n = 23); crosstab shows most high-Q6 responses (4–5) among **clear** preferences | Clear preferences co-occur with higher identification confidence; low-Q6 scores almost only when preference is slight. |
| `strength_confidence_correlation.txt` | **Spearman** (`cor.test`, exact = FALSE) | Monotonic association: binary clear (0/1) vs Q6 (1–5) | **ρ = 0.24**, **p = .21** | Positive rank association; not significant at α = .05. |
| `strength_confidence_correlation.txt` | **Kendall** (`cor.test`) | Same pair | **τ = 0.21**, **p = .21** | Concordant with Spearman; same conclusion. |
| `strength_confidence_correlation.txt` | **Pearson** (`cor.test`) | Linear association: clear (0/1) vs Q6 | **r = 0.33**, 95% CI **−0.04 to 0.61**, **p = .08** | Marginal linear trend toward higher confidence with clearer preference. |
| `strength_confidence_correlation.txt` | **Spearman** (reference) | Raw Q3 (1–4) vs Q6 | **ρ = −0.15**, **p = .43** | Not a clean strength-only measure (direction confounded); ignore for strength–confidence claim. |
| `confidence_by_strength_clm.txt` | **CLM** (ordinal logit): `confidence_ord ~ excerpt_strength_factor + order` | Expected Q6 level by Q3 strength | clear vs slight **p = .15**; order **p = .93**; emmeans mean class **3.13** (slight) vs **3.90** (clear) | Ordinal model: ~0.8 point higher confidence when preference is clear; borderline significance. |
| `strength_by_confidence_glm_binomial.txt` | **GLM** (binomial logit): `excerpt_clear ~ ai_confidence + order` | P(clear vs slight Q3) per unit Q6 | ai_confidence β = **0.73**, **p = .097**; order **p = .67**; emmean P(clear) ≈ **0.79** at mean Q6 | Each step up on Q6 increases odds of a clear preference (~2× per point on logit scale); marginal. |

#### `part_1/comparison/perceived_ht_preference/`

**Research question:** Do readers prefer the translation they believe is human (perceived HT), regardless of whether that excerpt is actually HT?

**Coding:** Q5 marks which excerpt is AI → `ai_guess_version`; perceived human = the other excerpt; `prefers_perceived_HT` = Q3 continue-reading choice matches perceived human.

**Synthesis (n = 30):** Preference aligns with perceived human translation far above chance. The **63%** actual-HT rate is lower because wrong AI tags sometimes pair with preferring actual MT; when MT identification is correct, alignment is **100%**.

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `descriptives.txt` | None (crosstabs, proportions) | Prefer perceived vs actual HT | **93%** prefer perceived HT vs **63%** actual HT; when MT id correct **100%** (17/17) vs **85%** (11/13) when wrong | Readers overwhelmingly pick the passage they believe is human; gap vs actual-HT rate driven partly by identification errors. |
| `prefer_perceived_HT_binom.txt` | **Exact binomial test** (`binom.test`, H₀: p = 0.5) | P(prefer excerpt reader treats as human) | **28/30**, **p &lt; .001** (95% CI **.78–.99**); estimate **0.93** | Preference tracks perceived human translation far above chance. |
| `prefer_perceived_HT_glm_binomial.txt` | **GLM** (binomial logit): `prefers_perceived_HT ~ order` | P(prefer perceived HT) | Intercept **p = .014**; order ns (separation: MT-first emmean **1.0**) | Strong baseline alignment; order not estimable reliably. |
| `prefer_perceived_HT_glmer_binomial.txt` | **GLMER** (binomial logit): `prefers_perceived_HT ~ order + (1\|reader) + (1\|book)` | Same | Intercept **p = .014**; order ns; RE variance ~0; convergence warning | Same pattern; random effects negligible at assignment level. |

---

## `part_2/`

Part 2 chunk-level comparison tasks. Binary chunk preference uses binomial GLMER; relative quality (`difficulty`) uses ordinal CLMM/CLM (levels: similar_quality < better < significantly_better).

### `part_2/chunk_preference/`

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `preferred_translation_glmer_binomial.txt` | **GLMER** (binomial logit): `preferred_translation ~ group + order + (1\|reader) + (1\|book)` | Preferred translation (HT vs MT) | Intercept **p = .044** (baseline odds); Japanese/Polish and order **ns** | Default chunk preference is HT; language and order do not shift binary choice. |
| `assignment_prop_HT_glm_quasibinomial.txt` | **GLM** (quasibinomial logit): `cbind(n_HT_chunks, n_MT_chunks) ~ order` | Proportion HT chunks per assignment | Intercept **p = .018** (baseline ~**70%** HT); order ns (emmeans **.70 vs .65**) | Assignments mostly HT-majority; order effect absent. |
| `chunk_level_glmer_binomial.txt` | **GLMER** (binomial logit): `chunk_pref_HT ~ order + chunk_index_z + (1\|assignment_id) + (1\|chunk_uid)` | Chunk-level HT preference | order and chunk_index **ns**; intercept **p = .003**; **singular** chunk RE | Strong baseline HT preference; position and order irrelevant. |
| `reader_book_glmer_binomial.txt` | **GLMER** (binomial logit): `chunk_pref_HT ~ 1 + (1\|reader) + (1\|book)` | Baseline chunk HT preference (772 chunks) | Intercept **p = .011**; emmeans **76%** HT (95% CI **.57–.89**); reader/book RE SD **1.03 / 1.39**; **not singular** | HT preferred after accounting for reader and book variation; aligns with descriptive ~68% HT. |

### `part_2/difficulty/`

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `difficulty_clmm.txt` | **CLMM** (ordinal logit): `difficulty ~ group + order + (1\|reader) + (1\|book)` | Strength of preference | groupJapanese **p = .082**; groupPolish and order ns | Japanese chunks rated marginally higher on quality scale than French baseline. |
| `difficulty_clm.txt` | **CLM** fallback (ordinal logit): `difficulty ~ group + order` | Strength of preference | groupJapanese **p < .001**; groupPolish **p = .012**; order ns; emmeans mean class **French 1.99, Japanese 2.28, Polish 1.82** | Without RE, Japanese skews “better”, Polish “similar”; order still ns. |

### `part_2/word_count/`

| File | Test | Question | Result | Interpretation |
|------|------|----------|--------|----------------|
| `justification_lmer.txt` | **LMER** (Gaussian): `justification_nbr_words ~ group + order + (1\|reader) + (1\|book)` | Justification word count | **Polish −8.7 words** (p = .068); Japanese and order ns | Polish justifications marginally shorter than French. |

---

## Cross-cutting takeaways

- **Strongest signal:** HT beats MT on single-reading **acceptability** and **smoothness** (`part_1/single_reading/ordinal/` CLMM); immersion weakly favors HT (CLMM marginal, CLM confirms); no effect on willingness to continue (CLMM).
- **Presentation order:** Mostly null across LMER/GLMER/CLMM models. Exception: side-by-side **overall preference** shows a marginal shift toward HT after MT-first (`part_1/comparison/ordinal/` CLMM *p* = .10; CLM *p* = .084).
- **AI detection:** ~57% accuracy at single-reading and comparison (`part_1/single_reading/ai_identification/`, `part_1/comparison/ai_identification/`); **exact binomial tests** do not reject 50% chance; GLM/GLMER fixed effects mostly ns.
- **Direct preference:** Descriptively **~63–68% HT** at excerpt/chunk level (`_shared/preference_descriptives.txt`, `part_1/comparison/excerpt_preference/`, `part_2/chunk_preference/`). **GLMER/GLM** binary models confirm baseline HT majority; **CLMM/CLM** on three-level comparison and difficulty outcomes show language/order rarely move the full ordinal scale (except marginal Japanese quality vs French).
- **Perceived human vs actual HT (`perceived_ht_preference/`):** **93%** of continue-reading choices match the excerpt readers treat as human (not the one tagged AI on Q5), **p &lt; .001** vs 50%; only **63%** prefer actual HT. Readers choose the translation they believe is human even when that belief is wrong.
- **Preference strength × AI confidence (`preference_confidence/`):** Clear Q3 choices (clearly better) pair with higher Q6 confidence (mean **3.96 vs 3.14**). **Spearman/Kendall** correlations are positive but **ns** (ρ ≈ .24, *p* ≈ .21); **Pearson** *p* = .08; **CLM** and **GLM** show the same direction marginally (*p* ≈ .10–.15). Not definitive at n = 30, but consistent with “stronger preference when more confident in the AI call.”
- **Reliability caveats:** Many **GLMER/CLMM** fits are **singular** (reader/book RE ≈ 0). Comparison **CLMMs** have only **n = 30**, so use **CLM fallbacks** in `part_1/comparison/ordinal/` and `part_1/single_reading/ordinal/` for fixed-effect inference on Q1/Q2 and q2/q3; correlation and Q3×Q6 models are similarly power-limited.

---

## Deprecated outputs

The script no longer writes flat `comparison_preferred_overall_glmer.txt`, `comparison_smoother_glmer.txt`, or `part2_difficulty_glmer.txt`. Three-level comparison and difficulty outcomes are under the `ordinal/` folders as `*_clmm.txt` / `*_clm.txt`.
