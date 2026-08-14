# Authors' Guidelines for Developing Comment-Coding Schemas

## Goal

This document describes how the authors developed two coding schemas using an
inductive coding approach: one for translation quality (29 categories) and one
for why a translation was judged AI-generated (10 categories).

## Comment Sources in LAIT

The authors developed the schemas from five types of open-ended responses:

- `Q1`: positive aspects reported after reading one translation in isolation
  (60 comments).
- `Q2`: negative aspects reported after reading one translation in isolation
  (60 comments).
- `Q3`: reasons for preferring one of the two complete translations after
  reading both (30 comments).
- `Q4`: reasons for judging one complete translation as AI-generated (30
  comments).
- `Q5`: reasons for preferring one translation of a shorter aligned chunk
  (approximately 10% of 772 comments, or 77 comments).

The quality coding schema was developed from `Q1`, `Q2`, `Q3`, and `Q5`. The
AI-likeness coding schema was developed separately from `Q4`.

## Step-by-Step Development Guidelines

### 1. Read Comments and Propose an Initial Inventory

Inspect the reader comments and list recurring reasons for praise (`Q1`),
criticism (`Q2`), preference (`Q3` and `Q5`), and AI-likeness judgments (`Q4`).

At this stage, prioritize coverage rather than creating a fixed inventory. If
a comment expresses a concept that does not fit the provisional labels, create
a candidate category instead of forcing the comment into the closest existing
one.

Record representative examples alongside each candidate label. Use these
examples to determine whether similarly named categories capture the same
phenomenon or should remain separate.

### 2. Write a Definition for Every Candidate Label

Give each candidate label a name and a concise operational definition. Check
that the definition explains what evidence should trigger the label and
distinguishes it from nearby categories.

A useful label entry contains:

- A stable identifier and descriptive name.
- A definition of the phenomenon.
- Positive and negative examples, when both polarities are meaningful.
- Inclusion criteria describing evidence that should receive the label.
- Exclusion criteria or a contrast with easily confused labels.
- Representative phrases from reader comments.

Revise definitions whenever researchers could reasonably assign the same
phrase to two labels or interpret the scope of a label differently.

### 3. Pilot One Response Source at a Time

Begin with a smaller immersive-reading response set instead of attempting to
code all comments at once. First refine the labels against `Q1`, then repeat
the process for `Q2` and the comparative responses.

Have multiple researchers read the comments and iteratively add or revise
labels. Once the provisional inventory appears to cover the comments, have two
researchers apply the same schema to a shared subset. Compare their annotations
after roughly ten comments so that unclear definitions can be detected before
the full set is coded.

### 4. Compare the Two Researchers' Annotations

Compare the complete sets of labels assigned by the two researchers to each
shared comment. For every difference, ask:

- Did one researcher overlook an explicitly stated reason?
- Was one researcher coding at a finer level of detail than the other?
- Was the definition too broad, too narrow, or ambiguous?
- Did two categories overlap enough that they should be clarified or merged?
- Did one category contain distinct concepts that should be split?
- Was the apparent disagreement actually a polarity or response-scope error?
- Did the comment express a new concept that required another category?

Do not expect an exact initial match. Use the comparison to reveal how the
coding schema behaves in practice and to align the researchers' interpretation
of its categories.

### 5. Refine Definitions and Category Boundaries

Discuss disagreements comment by comment and revise the coding schema when
necessary. Correct missed labels as part of this review. Pay particular
attention to annotation density: one researcher may identify more explicitly
stated aspects in a long comment while another assigns a smaller, more general
set. Distinguish these omissions from fundamental disagreement about the
meaning of a category.

When a disagreement comes from different levels of granularity, check both the
fine-grained label and its higher-level family. Determine whether the
researchers agree on the general phenomenon but differ only in the specific
label.

After discussion, produce an agreed annotation for each reviewed comment. Use
the resulting consensus to clarify how the final schema should be applied.

### 6. Repeat the Process Across Response Sources

After calibrating the schema on one response source, repeat the
read--annotate--compare cycle on the other quality-comment sources. Reuse the
same quality coding schema when readers discuss related qualities while
praising a translation, criticizing it, or explaining a preference.

Allow new response sources to expose missing distinctions. In comparative
comments, explicitly separate qualities attributed to the preferred
translation from weaknesses attributed to the rejected translation.

Handle `Q4` comments with a separate coding schema because readers may name
surface cues or assumptions about AI-generated prose that are not general
assessments of translation quality.

### 7. Organize the Quality Labels into Higher-Level Families

As the quality inventory stabilizes, group related labels into four families:

- **A. Language-level features:** grammar, wording, register, sentence
  structure, consistency, and other local linguistic properties.
- **B. Narrative-level features:** dialogue, character voice, description,
  emotional conveyance, narrative organization, point of view, and pacing.
- **C. Reader experience:** comprehension, reading effort, engagement,
  humanness, and enjoyment.
- **D. Meta-translation:** whether the text reads as translated, literalness or
  adaptation, perceived faithfulness, and explicitly named MT/AI impressions.

Use the families to provide structure without replacing the fine-grained
labels. Allow a comment to receive multiple labels within or across families.

### 8. Finalize the Coding Schemas

Consider a coding schema ready for full application when:

- The reviewed comments are covered without routinely forcing evidence into
  unsuitable labels.
- Every label has a usable definition and representative examples.
- Commonly confused labels have an explicit distinction.
- Two researchers can apply the schema with a shared understanding after
  calibration.
- Remaining disagreements can be resolved through the written definitions and
  discussion.
- No recurring concept requires an additional label.

The resulting quality coding schema contains 29 categories in the four
families above. The separate AI-likeness coding schema contains 10 categories.
The paper's appendix tables are the authoritative list of the final labels and
definitions.

## Coding Rules Established During Development

### Evidence Must Be Explicit

Annotators assign a label only when the reader's comment supports it.
Preference alone does not justify a quality label. For example, selecting T1
does not by itself establish that T1 was smoother or more natural.

The translation excerpts may help resolve what a pronoun or comparison in the
comment refers to, but they should not be used to invent reasons that the
reader did not state.

### Comments Are Multilabel

Annotators should code every relevant, explicitly stated reason. A long
comment can discuss word choice, comprehension, dialogue, and formatting at
the same time. Assigning one broad label should not suppress other supported
labels.

At the same time, annotators should not create redundant labels for the same
evidence merely because several definitions contain related words.

### Apply the Correct Polarity for Each Response Source

- For `Q1`, code only positive aspects. Do not code a negative aside included
  in the response.
- For `Q2`, code only negative aspects.
- For `Q3` and `Q5`, record positive and negative evidence separately:
  - POS labels describe explicitly stated strengths of the preferred
    translation.
  - NEG labels describe explicitly stated weaknesses of the rejected
    translation.
- For `Q4`, use the separate AI-likeness coding schema.

### Do Not Mirror Comparative Labels Automatically

A positive statement about the preferred translation does not automatically
constitute a negative statement about the rejected translation, and vice
versa.

For example, "T1 is more natural" supports a positive naturalness label for
T1. It supports a negative naturalness label for T2 only if the reader also
explicitly characterizes T2 as unnatural, awkward, or otherwise deficient.
Conversely, if the reader only calls T2 awkward, annotate that negative reason
without inventing positive naturalness for T1.

### Treat Source Accuracy Cautiously

The comments capture reader perception. Unless the source was consulted as
part of a separate verification, annotators should not treat a reader's
uncertainty as proof that one translation is factually inaccurate. They may
code the perceived clarity, consistency, faithfulness, or mismatch that the
reader actually described.

### Human Judgment Remains Authoritative During Coding Schema Development

Models may suggest candidate categories or possible labels for a difficult
comment. During schema development, the researchers inspect the original
wording and decide whether a suggestion is supported. Bulk automatic labeling
must not be used to define or validate the coding schema itself.

## Recommended Record for Each Coding Schema Revision

For each revision, retain:

- The coding schema version or date.
- The response sources inspected.
- The comments used for calibration.
- The researchers involved in annotation or review.
- Categories added, removed, merged, split, or renamed.
- Definitions or examples changed after disagreement.
- Unresolved edge cases.
- The location of the final annotation sheet or exported file.

This record separates the development history from the final coding schema and
makes later analyses traceable to the definitions used at the time.

## Completion Checklist

Before applying a coding schema to the complete collection, confirm that:

- [ ] Every label is grounded in observed reader comments.
- [ ] Every label has a definition and at least one example.
- [ ] Commonly confused labels have documented boundaries.
- [ ] Multilabel and polarity rules are understood by all researchers.
- [ ] Two researchers have applied the provisional schema to the same subset.
- [ ] Differences in annotation density have been discussed.
- [ ] Disagreements have been resolved or recorded as unresolved edge cases.
- [ ] The schema has been checked against all relevant response sources.
- [ ] The quality and AI-likeness coding schemas remain separate.
- [ ] The final coding schema version and annotation files have been recorded.

## Relationship to the Paper Appendix

This file explains how the label inventories were created and calibrated. The
paper appendix separately reports how the finalized coding schemas were
applied in this study, including manual coding coverage, reviewer roles,
consensus construction, the sampled `Q5` annotations, the exact GPT-5.4 prompt
and output format, model--human agreement, and automatic coding of the
remaining comments.

Repository copies of the final schemas are available for the
[translation-quality categories](../analysis/manuscript_tables/tex/annotation_scheme_Q1_Q2_Q3_Q5.tex)
and the separate
[AI-likeness categories](../analysis/manuscript_tables/tex/annotation_scheme_Q4.tex).
