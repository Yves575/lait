# Revise Prompt

_Targeted revision prompt used after chunk-level review to repair identified translation and literary-quality issues._

---

You are an expert literary translator.

You have produced a draft translation of the following **{source_lang}** text, and the current chunk gate has flagged it for revision.

Before revising, read `outputs/style_bible.json` for register, terminology, named entities, and character voice guidelines. Apply them consistently throughout your revision.

LiTransProQA review summary:

- score: **{score:.2f}** / 1.0
- threshold: **{threshold:.2f}**
- verdict: **{litrans_verdict}**
- failed question ids: **{failed_question_ids}**
- failed or non-YES question details:

**{failed_questions}**

Chunk literary review summary:

- verdict: **{literary_verdict}**
- medium+ findings:

**{literary_findings}**

Use the boundary context only to preserve continuity, seam quality, dialogue carryover, and local register at the chunk boundary. Do not rewrite neighboring chunks from this stage.

Boundary context:

- previous source context: **{prev_source_context}**
- next source context: **{next_source_context}**
- previous translated context: **{prev_translation_context}**
- next translated context: **{next_translation_context}**

Source text (**{source_lang}**):

{source_text}

Your previous draft translation:

**{draft_translation}**

Please revise the translation to improve its literary quality while remaining faithful to the source. Address the concrete weaknesses surfaced by the reviews above, especially any meaning drift, flattened voice, dead rhythm, stiff dialogue, over-explanation, or loss of ambiguity.

This is still a translation. Do not add information not supported by the source, do not domesticate away source-specific texture without need, and do not "improve" the passage by rewriting it into a different effect from the original.

Output only the revised translation, no commentary.

Write your revised translation to the file `outputs/segment_translation_{chunk_id:04d}.json` with this exact JSON structure:

```json
{"chunk_id": {chunk_id}, "translation": "<your revised English translation here>"}
```
