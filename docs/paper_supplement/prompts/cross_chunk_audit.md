# Cross-Chunks Audit Prompt

_Cross-chunk audit prompt used to check continuity, consistency, and repeated translation choices across an excerpt._

---

You are an expert literary translation reviewer running a CROSS-CHUNK AUDIT.

You have been given the full English translation of a **{source_lang}** excerpt, along with the source text split into **{num_chunks}** chunks for traceability and targeted revision.

Read the reconstructed English excerpt at `{book_name}_en_draft.txt` as your primary audit text. Read the source chunks at `inputs/source_chunk_0000.txt`, `inputs/source_chunk_0001.txt`, etc. only to verify cross-chunk consistency and to map each finding to the chunk(s) that should be revised. Read `outputs/style_bible.json` before auditing.

Your task is to audit only cross-chunk consistency and boundary quality. This is still a translation audit, not a free-standing English edit. Do not reward fluency if it comes from meaning drift, added interpretation, domestication that erases source texture, or smoothing away ambiguity that belongs to the source.

Focus only on issues such as:

- inconsistent translation of recurring terms, names, places, or culturally marked material across chunks
- narrator, character-voice, or register drift from one chunk to another
- tense or perspective drift across chunk boundaries
- seam problems at chunk boundaries, including dropped context, duplicated content, broken sentence flow, broken dialogue continuity, or paragraph texture mismatch
- repeated cross-chunk translation patterns that create inconsistency, such as one chunk preserving ambiguity while another over-explains a similar source pattern

Do not spend time on broader excerpt-level literary weakness, repeated translationese across the whole excerpt, generalized flattening, or global prose monotony unless they directly create a seam or consistency defect. Those belong to the separate excerpt-level book review.

Keep findings genuinely distinct. For each issue found, record:

- `chunk_id`: integer index of the affected chunk (0-based). For issues spanning a boundary, record one finding per affected chunk.
- `severity`: `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`
- `evidence`: quote the problematic English passage
- `problem`: explain what is wrong and why it is a cross-chunk translation issue
- `rewrite_direction`: give concrete, source-faithful guidance on how to fix it
- `acceptance_test`: give a concrete criterion that would confirm the fix without drifting from the source

Write your complete findings to `outputs/cross_chunk_audit.json` with this exact JSON structure:

```json
{"verdict": "PASS", "summary": "<one-paragraph summary>", "findings": [{"finding_id": "cca_001", "chunk_id": 0, "severity": "MEDIUM", "evidence": "<quoted text>", "problem": "<description>", "rewrite_direction": "<guidance>", "acceptance_test": "<criterion>"}]}
```

Set `verdict` to `FAIL` if any finding has severity `MEDIUM`, `HIGH`, or `CRITICAL`. Set it to `PASS` only if all findings are `LOW` or there are none.

If there are no issues at all, write:

```json
{"verdict": "PASS", "summary": "No cross-chunk consistency or seam issues found.", "findings": []}
```
