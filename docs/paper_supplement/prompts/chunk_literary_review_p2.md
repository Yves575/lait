# Chunks Reviewing Prompt (part 2)

_Second chunk-level literary review prompt, used to guide targeted revision after local literary-quality issues are identified._

---

Flag issues such as:

- over-explanation that kills subtext
- flattened or generic diction where the source is sharp or strange
- dialogue that is semantically correct but stiff, composed, or unnaturally formal
- rhythm or sentence movement that has gone dead through literalism
- euphemizing or sanitizing difficult material that the source renders directly
- unnecessary upgrading into more elegant or more quotable English than the source warrants

For each finding, include:

- `finding_id`
- `source`: one of `accuracy`, `voice`, `dialogue`, `prose`
- `severity`: `MEDIUM`, `HIGH`, or `CRITICAL`
- `chunk_id`: must equal **{chunk_id}**
- `evidence`: quote the problematic text from the English translation
- `problem`: explain what is wrong and why it is a translation problem, not just a matter of taste
- `rewrite_direction`: give a concrete, source-faithful revision direction
- `acceptance_test`: give a concrete condition that would show the issue is fixed while staying faithful to the source

Write your review to `outputs/chunk_literary_review_{chunk_id:04d}.json` with this exact JSON structure:

```json
{
    "chunk_id": {chunk_id},
    "verdict": "PASS",
    "verdicts": {
        "accuracy": "PASS",
        "voice": "PASS",
        "dialogue": "PASS",
        "prose": "PASS"
    },
    "findings": [
        {
            "finding_id": "clr_001",
            "source": "prose",
            "severity": "MEDIUM",
            "chunk_id": {chunk_id},
            "evidence": "<quoted English passage>",
            "problem": "<what is wrong and why>",
            "rewrite_direction": "<source-faithful fix guidance>",
            "acceptance_test": "<concrete pass/fail condition>"
        }
    ],
    "summary": "<brief summary>"
}
```

Set top-level `verdict` to `FAIL` if any lens fails or any finding is `MEDIUM+`. Set it to `PASS` only when the chunk is source-faithful and literarily strong in English.
