# Final Revise Prompt

_Final revision prompt used to incorporate review findings into the completed machine-translated excerpt._

---

You are an expert literary translator performing a targeted revision.

Before revising, read `outputs/style_bible.json` for register, terminology, named entities, and character voice guidelines. Apply them consistently throughout your revision.

Excerpt-level review and cross-chunk audit have identified the following grouped issues for chunk **{chunk_id}** of the translation:

**{findings}**

Use the boundary context only to preserve continuity, seam quality, dialogue carryover, and local register at the chunk boundary. Do not rewrite neighboring chunks from this stage.

Boundary context:

- previous source context: **{prev_source_context}**
- next source context: **{next_source_context}**
- previous translated context: **{prev_translation_context}**
- next translated context: **{next_translation_context}**

Source text (**{source_lang}**):

**{source_text}**

Current translation:

**{draft_translation}**

Revise the translation to address all findings listed above. The findings are grouped by review source so you can distinguish excerpt-level literary issues from seam/consistency issues. Preserve the literary quality, voice, and style established in the rest of the excerpt. Make only the changes needed to fix the identified issues. This is still a translation: do not add unsupported information, erase source-specific texture, or rewrite the passage into a different effect from the original.

Write your revised translation to `outputs/segment_translation_{chunk_id:04d}.json` with this exact JSON structure:

```json
{"chunk_id": {chunk_id}, "translation": "<your revised English translation here>"}
```

Also write a revision report to `outputs/final_revise_{chunk_id:04d}.json` with this exact JSON structure:

```json
{"chunk_id": {chunk_id}, "findings_addressed": [{"finding_id": "<id>", "status_after_revision": "FIXED"}]}
```

Valid values for status_after_revision: "FIXED", "PARTIAL", "UNRESOLVED".
