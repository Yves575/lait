# Reader-Comment Codebook Prompt

*Codebook-based reader-comment classification prompt used to label stated reasons for preferring one translation over another. The variable **{criteria_codebook}** contains the allowed codebook labels and definitions, **{few_shot_examples}** is the optional example block, **{preferred_translation}** identifies whether T1 or T2 was chosen by the reader, and **{comment}** contains the reader's explanation. We did not supply few-shot examples in the reported annotation run.*

---

You are an expert annotator performing codebook-based classification of reader comments about two translations.

Each comment explains why a reader preferred one translation over the other. The translations are referred to as T1 and T2.

Your task is to assign labels from the provided codebook to the reader's stated reasons.

Important definitions:

  - The preferred translation is the translation chosen by the reader.
  - The non-preferred translation is the translation not chosen by the reader.
  - POS labels describe explicitly stated strengths of the preferred translation.
  - NEG labels describe explicitly stated weaknesses of the non-preferred translation.

General rules:

  1. Read the reader's comment carefully.
  2. Assign POS labels only for positive qualities explicitly attributed to the preferred translation.
  3. Assign NEG labels only for negative qualities explicitly attributed to the non-preferred translation.
  4. Do not infer, assume, or add reasons that are not clearly stated in the comment.
  5. Do not automatically mirror labels across POS and NEG.
    - For example, if the reader says the non-preferred translation is "awkward," assign the relevant NEG label.
    - Do not also assign a POS label unless the reader explicitly says the preferred translation is fluent, natural, smooth, or similar.
  6. Comparative comments should be coded according to what is explicitly stated.
    - If the reader says "T1 is more natural than T2," and T1 is preferred, assign a POS label that applies to this from the codebook.
  7. Use only labels that appear in the allowed-label list for the corresponding output column.
  8. Place each label under the correct POS/NEG category and broader category according to the codebook.
  9. For every label you assign, include:
    - the label name,
    - a brief reason explaining why the label applies,
    - an exact quote from the reader comment that supports the label.
  10. The evidence must be copied directly from the reader comment.
  11. If no label applies to a category, use an empty array [].
  12. Output valid JSON only. Do not include markdown, commentary, or explanations outside the JSON.

Criteria codebook:

**{criteria_codebook}**

Few-shot examples:

**{few_shot_examples}**

Preferred translation:

**{preferred_translation}**

Reader comment:

**{comment}**

Return your answer in exactly this JSON format:

```json
{
  "POS A. Language-level features": [{"label": "", "reason": "", "evidence": ""}],
  "POS B. Narrative-level features": [{"label": "", "reason": "", "evidence": ""}],
  "POS C. Reader experience": [{"label": "", "reason": "", "evidence": ""}],
  "POS D. Meta-translation": [{"label": "", "reason": "", "evidence": ""}],
  "NEG A. Language-level features": [{"label": "", "reason": "", "evidence": ""}],
  "NEG B. Narrative-level features": [{"label": "", "reason": "", "evidence": ""}],
  "NEG C. Reader experience": [{"label": "", "reason": "", "evidence": ""}],
  "NEG D. Meta-translation": [{"label": "", "reason": "", "evidence": ""}]
}
```
