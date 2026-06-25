"""Reduce each book in books/dev to approximately 10% of its tokens.

Uses the Gemini tokenizer to count tokens and splits at paragraph boundaries
(\n\n) so we never cut mid-sentence.

The final text respects two caps (whichever is stricter):
  - 10% of the book's total tokens
  - MAX_WORDS words (whitespace-split)

- If OUTPUT_DIR is empty, reduced books are generated and saved.
- If OUTPUT_DIR already contains files, stats are printed from the existing files.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

INPUT_DIR = Path("books/all/")
OUTPUT_DIR = Path("books/HT/")
RATIO = 0.10
MAX_WORDS = 8_000
MODEL = "gemini-3.1-pro-preview"


def count_tokens(client: genai.Client, text: str) -> int:
    """Return the number of tokens in *text* as counted by the Gemini tokenizer."""
    response = client.models.count_tokens(model=MODEL, contents=text)
    return response.total_tokens


def reduce_book(client: genai.Client, text: str, total_tokens: int, token_limit: int) -> str:
    """Return the longest prefix of *text* ending on a paragraph boundary that
    satisfies both caps: token_limit tokens and MAX_WORDS words.

    Strategy:
    1. Estimate the cut paragraph index from the token/char ratio.
    2. Also find the paragraph index where the word count first exceeds MAX_WORDS,
       and use whichever index is stricter (smaller).
    3. Verify with one count_tokens call; nudge backward one paragraph at a time
       if still over the token limit.
    """
    paragraphs = text.split("\n\n")

    # --- Token-based estimated cut ---
    char_ratio = len(text) / total_tokens
    target_chars = int(token_limit * char_ratio)

    cumulative_chars = 0
    token_cut_index = len(paragraphs)
    for i, para in enumerate(paragraphs):
        cumulative_chars += len(para) + 2  # +2 for the "\n\n" separator
        if cumulative_chars >= target_chars:
            token_cut_index = i + 1
            break

    # --- Word-based cut ---
    cumulative_words = 0
    word_cut_index = len(paragraphs)
    for i, para in enumerate(paragraphs):
        cumulative_words += len(para.split())
        if cumulative_words > MAX_WORDS:
            word_cut_index = i  # stop before this paragraph
            break

    # Take whichever cap is stricter.
    cut_index = min(token_cut_index, word_cut_index)

    # Nudge backward until within the token limit.
    while cut_index > 0:
        candidate = "\n\n".join(paragraphs[:cut_index])
        if count_tokens(client, candidate) <= token_limit:
            return candidate
        cut_index -= 1

    return ""


def print_stats(name: str, total_tokens: int, limit: int, final_tokens: int, final_words: int) -> None:
    """Print token and word statistics for one book."""
    final_pct = final_tokens / total_tokens * 100
    print(
        f"{name}: "
        f"total={total_tokens:,} tokens | "
        f"10% limit={limit:,} | "
        f"final={final_tokens:,} tokens ({final_pct:.1f}%) | "
        f"final={final_words:,} words"
    )


TSV_HEADER = "book\ttotal_tokens\tlimit_10pct\tfinal_tokens\tfinal_pct\tfinal_words"
TSV_PATH = OUTPUT_DIR.parent / "reduce_stats.tsv"


def main() -> None:
    """Process every .txt file in INPUT_DIR.

    If OUTPUT_DIR is empty, generate and save reduced books.
    If OUTPUT_DIR already contains files, only print stats from existing files.

    In both cases, writes a TSV file to TSV_PATH for easy copy-paste into Google Sheets.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in the environment.")

    client = genai.Client(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    already_done = list(OUTPUT_DIR.glob("*.txt"))
    rows: list[tuple] = []

    if already_done:
        print(f"OUTPUT_DIR is not empty ({len(already_done)} files). Printing stats only.\n")
        for out_path in sorted(already_done):
            in_path = INPUT_DIR / out_path.name
            if not in_path.exists():
                print(f"{out_path.name}: source file not found in {INPUT_DIR}, skipping.")
                continue
            original_text = in_path.read_text(encoding="utf-8")
            reduced_text = out_path.read_text(encoding="utf-8")
            reduced_text_clean = reduced_text.replace("<chunk>", "").replace("</chunk>", "")
            total_tokens = count_tokens(client, original_text)
            final_tokens = count_tokens(client, reduced_text_clean)
            limit = max(1, int(total_tokens * RATIO))
            final_words = len(reduced_text_clean.split())
            print_stats(out_path.name, total_tokens, limit, final_tokens, final_words)
            rows.append((out_path.name, total_tokens, limit, final_tokens,
                         round(final_tokens / total_tokens * 100, 1), final_words))
    else:
        txt_files = sorted(INPUT_DIR.glob("*.txt"))
        if not txt_files:
            print(f"No .txt files found in {INPUT_DIR}")
            return

        for book_path in txt_files:
            text = book_path.read_text(encoding="utf-8")
            total_tokens = count_tokens(client, text)
            limit = max(1, int(total_tokens * RATIO))

            reduced = reduce_book(client, text, total_tokens, limit)
            final_tokens = count_tokens(client, reduced)
            final_words = len(reduced.split())

            print_stats(book_path.name, total_tokens, limit, final_tokens, final_words)
            rows.append((book_path.name, total_tokens, limit, final_tokens,
                         round(final_tokens / total_tokens * 100, 1), final_words))

            out_path = OUTPUT_DIR / book_path.name
            out_path.write_text(reduced, encoding="utf-8")

    if rows:
        lines = [TSV_HEADER] + ["\t".join(str(v) for v in row) for row in rows]
        TSV_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nStats saved to {TSV_PATH}")


if __name__ == "__main__":
    main()
