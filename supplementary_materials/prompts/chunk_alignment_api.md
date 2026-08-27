# Chunk Alignment (API) Prompt

_API prompt used to align whole-paragraph chunks between two English translations._

---

You are aligning two English translations of the same novel, paragraph by paragraph.

You will receive:

- a CANDIDATE CHUNK from one translation, made up of one or more paragraphs separated by blank lines;
- a larger TARGET WINDOW from a second translation.

Your task: pick the contiguous run of whole paragraphs from the TARGET WINDOW whose combined content corresponds — semantically, scene-for-scene — to the CANDIDATE CHUNK.

---CANDIDATE CHUNK---

**{source_chunk}**

---END CANDIDATE CHUNK---

---TARGET WINDOW---

**{target_window}**

---END TARGET WINDOW---

Rules:

- The match MUST be a single contiguous span of paragraphs from the TARGET WINDOW. Do not skip paragraphs in the middle.
- Paragraph boundaries are blank lines. Begin at the start of a paragraph and end at the end of a paragraph; do not break mid-paragraph.
- The two translations may differ in paragraph count, phrasing, or minor content — translators sometimes add, omit, or merge sentences. Choose the smallest contiguous run of target paragraphs that covers the same narrative content as the candidate chunk.
- Use the FIRST plausible match in the window — do not jump ahead to a later, similar passage.
- Copy the matched text VERBATIM from the TARGET WINDOW: same characters, same whitespace, same blank lines. Do not paraphrase or add commentary.
- There should always be a match. If you are certain that there is no match, please return "no match".

Return ONLY the matched text itself, copied verbatim from the TARGET WINDOW.

Do not add any explanations or change the text in any way.
