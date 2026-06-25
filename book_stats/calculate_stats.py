"""Calculate book- and chunk-level word count and token count stats for JSONL books."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import tiktoken


def is_empty_chunk(text: str, word_count: int, token_count: int) -> bool:
    """True when a chunk has no substantive text or counts."""
    return not str(text).strip() and word_count <= 0 and token_count <= 0


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        # Fallback to simple approximation if tiktoken fails
        return len(text.split())


def load_chunk_counts(folder: Path) -> pd.DataFrame:
    """Load chunk word counts and token counts from every JSONL file in folder.

    Records with empty text and zero word/token counts are skipped.
    """
    rows = []
    skipped = 0

    for path in sorted(folder.glob("*.jsonl")):
        if not path.is_file():
            continue

        try:
            df = pd.read_json(path, lines=True)
        except ValueError as exc:
            print(f"Error reading {path}: {exc}", file=sys.stderr)
            raise SystemExit(1)

        if "word_count" not in df.columns:
            print(f"Error: {path} has no word_count column", file=sys.stderr)
            raise SystemExit(1)

        word_counts = pd.to_numeric(df["word_count"], errors="coerce")
        if word_counts.isna().any():
            bad_rows = word_counts[word_counts.isna()].index + 1
            bad_rows_text = ", ".join(str(row) for row in bad_rows[:10])
            print(
                f"Error: {path} has non-numeric word_count values on row(s): {bad_rows_text}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        # Find text column (try common names)
        text_column = None
        for col_name in ["text", "content", "chunk_text", "body"]:
            if col_name in df.columns:
                text_column = col_name
                break

        chunk_ids = df["chunk_id"] if "chunk_id" in df.columns else pd.Series(df.index + 1)
        for idx, (chunk_id, word_count) in enumerate(zip(chunk_ids, word_counts)):
            row = {
                "book": path.stem,
                "file": str(path),
                "chunk_id": chunk_id,
                "word_count": int(word_count),
            }
            
            text = ""
            if text_column is not None and idx < len(df):
                text = str(df.iloc[idx][text_column]) if pd.notna(df.iloc[idx][text_column]) else ""
                row["token_count"] = count_tokens(text)
            else:
                row["token_count"] = 0

            if is_empty_chunk(text, row["word_count"], row["token_count"]):
                skipped += 1
                continue

            rows.append(row)

    if skipped:
        print(f"Skipped {skipped} empty chunk(s) in {folder}", file=sys.stderr)

    return pd.DataFrame(rows)


def calculate_stats(counts: pd.Series, metric_name: str = "word_count") -> pd.DataFrame:
    """Return max, min, std, mean, median, and total for a count series."""
    stats = counts.agg(["max", "min", "std", "mean", "median", "sum"])
    return stats.rename_axis("stat").reset_index(name=metric_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate book and chunk word-count and token-count statistics from JSONL files."
    )

    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("books/HT/eval"),
        help="Directory containing JSONL book chunks (default: books/HT/eval).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("book_stats/human_translation_counts"),
        help=(
            "Directory for aggregate count CSVs "
            "(default: book_stats/human_translation_counts)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output_dir)

    if not chunks_dir.is_dir():
        print(f"Error: {chunks_dir} is not a directory", file=sys.stderr)
        raise SystemExit(1)

    chunk_counts = load_chunk_counts(chunks_dir)
    if chunk_counts.empty:
        print(f"Error: no JSONL chunks found in {chunks_dir}", file=sys.stderr)
        raise SystemExit(1)

    # Word count stats
    book_word_counts = (
        chunk_counts.groupby("book", as_index=False)["word_count"]
        .sum()
        .sort_values("book")
    )
    book_word_stats = calculate_stats(book_word_counts["word_count"], "word_count")
    chunk_word_stats = calculate_stats(chunk_counts["word_count"], "word_count")

    # Token count stats
    book_token_counts = (
        chunk_counts.groupby("book", as_index=False)["token_count"]
        .sum()
        .sort_values("book")
    )
    book_token_stats = calculate_stats(book_token_counts["token_count"], "token_count")
    chunk_token_stats = calculate_stats(chunk_counts["token_count"], "token_count")

    # Create output directories and save files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    book_word_stats.to_csv(output_dir / "book_word_count_stats.csv", index=False)
    chunk_word_stats.to_csv(output_dir / "chunk_word_count_stats.csv", index=False)
    book_token_stats.to_csv(output_dir / "book_token_count_stats.csv", index=False)
    chunk_token_stats.to_csv(output_dir / "chunk_token_count_stats.csv", index=False)

    print(f"Wrote book word count stats to {output_dir / "book_word_count_stats.csv"}")
    print(f"Wrote chunk word count stats to {output_dir / "chunk_word_count_stats.csv"}")
    print(f"Wrote book token count stats to {output_dir / "book_token_count_stats.csv"}")
    print(f"Wrote chunk token count stats to {output_dir / "chunk_token_count_stats.csv"}")


if __name__ == "__main__":
    main()
