# Chunk Alignment (HT to SRC) Prompt

_Prompt for Chunk Alignment (HT to SRC)._

---

Look at books/HT/**{book_ht_path}**.

This English file contains chunking tags `<chunk>` and `</chunk>`.

In books/test/**{book_src_path}**, you will find the **{lang}** translation of the same text.

However, this file does not contain any chunking tags.

Your task is to insert `<chunk>` and `</chunk>` tags into **{book_src_path}** so that the chunk structure matches the one in **{book_ht_path}**.

Rules:

- Each chunk in the English file corresponds to the same translated content in the **{lang}** file.
- You must place `<chunk>` and `</chunk>` around the **{lang}** text that corresponds to each English chunk.
- Do NOT modify, translate, or rewrite any **{lang}** text.
- Only insert `<chunk>` and `</chunk>` tags.
- You may insert line breaks (\n) if needed to properly place tags, but do not alter the text itself.

Output:

- Return the full updated content of **{book_src_path}** with the inserted chunk tags.
