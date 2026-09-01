# Chunk Alignment (MT to MT) Prompt

_Prompt for Chunk Alignment (MT to MT)._

---

Look at books/MT/pipeline1/gemini/**{book_name}**.

This file contains chunking tags `<chunk>` and `</chunk>`.

Also look at books/MT/pipeline2/gemini/**{book_name}**.

This file is another machine translation of the same source text, but it does not contain chunking tags.

Your task is to insert `<chunk>` and `</chunk>` into books/MT/pipeline2/gemini/**{book_name}** so that its chunk boundaries correspond to the chunks in books/MT/pipeline1/gemini/**{book_name}**.

Rules:

- Preserve the text of books/MT/pipeline2/gemini/**{book_name}** exactly.
- Only insert `<chunk>` and `</chunk>`.
- `<chunk>` and `</chunk>` should always be at the beginning of the line.
- Do not rewrite, translate, delete, or reorder the MT text.
- You may insert line breaks ('\n' one or more) if needed to properly place tags, but do not alter the text itself.
- Each chunk in the pipeline 2 MT file should correspond to the same passage as the matching chunk in the pipeline 1 MT file.

Output:

- Return the full content of books/MT/pipeline2/gemini/**{book_name}** with the inserted chunk tags.
