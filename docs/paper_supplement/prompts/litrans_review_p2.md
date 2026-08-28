# Translation Reviewing Prompt (part 2)

_Second LiTrans-style review prompt, used to convert diagnosed issues into concrete revision guidance._

---

Group C: Cultural Transfer

8. Are idioms, humor, and figurative expressions rendered in a way that preserves their function and effect in English?
9. Are culture-specific references handled appropriately for an English-language literary reader without unnecessary flattening, over-explanation, or loss of source specificity?
10. Are culturally specific emotions, gestures, and forms of expression rendered clearly and appropriately in English?
11. Does the translation avoid excessive domestication that erases the source culture's identity?
12. Does the translation avoid exoticizing or over-marking the source culture in ways not warranted by the original?
13. Are historical, local, and cultural references understood well enough to be rendered accurately?
14. Does the translation preserve the overall cultural atmosphere and emotional world of the passage?

Group D: Voice, Register, and Style

15. Does each character or narrative voice retain the same tone, register, and level of formality as in the source?
16. Is tone and register internally consistent within this passage unless the source itself shifts deliberately?
17. Does the English narrative voice feel equivalent in stance and texture to the source narrative voice?
18. Does the translation preserve the author's stylistic character, whether plain, restrained, ornate, lyrical, ironic, or otherwise distinctive?
19. Would an English-language reader encounter substantially the same narrator or character presence as in the source?

Group E: Narrative and Local Consistency

20. Are narrative point of view and perspective shifts preserved exactly where they matter?
21. Are names, recurring terms, slang, dialectal cues, and other key local details handled consistently within this passage and in line with established translation choices?

Group F: Fluency and Literary Effect

22. Does the translation read as natural, idiomatic English while remaining faithful to the source?
23. Is descriptive imagery rendered vividly, precisely, and without dulling or over-explaining the source?
24. Does the passage work as coherent literary English rather than sounding translated, formulaic, or mechanically literal?
25. Taken as a whole, does this passage achieve faithful equivalence at lexical, syntactic, textual, and pragmatic levels?

Answer all 25 questions. Do not omit any question and do not add extra keys.

Write your answer to the file `outputs/litrans_answers_{chunk_id:04d}.json` as a JSON object where each key is the question number (as a string), and each value is an object with exactly these fields:

- `judgment`: exactly one of `"YES"`, `"NO"`, or `"MAYBE"`
- `issue`: a short statement of the possible issue for this question

If the answer is `"YES"`, the `issue` field should briefly say that no material issue was detected for that question.

Example format:

```json
{
    "1": {"judgment": "YES", "issue": "No material issue detected for lexical distortion."},
    "2": {"judgment": "NO", "issue": "The English wording flattens an important connotation in the source."},
    "3": {"judgment": "MAYBE", "issue": "The sentence is readable, but one phrase still sounds slightly translated."}
}
```
