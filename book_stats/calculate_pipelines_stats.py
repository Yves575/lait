"""Calculate book- and chunk-level word and token count stats for MT pipeline outputs.

Walks every subdirectory of ``books/MT`` and, for each directory that contains ``.txt``
files, writes book- and chunk-level word- and token-count statistics to a mirrored path
under ``book_stats/pipelines``. The whole ``books/MT`` tree is processed in a single run.

Child directories named ``before_PE`` or ``extern`` (and any of their descendants) are
skipped, so per-model pre-post-edit and external folders do not contribute to the stats.

For these plain-text outputs there is no explicit chunk structure, so each paragraph
(separated by one or more blank lines) is treated as a chunk, mirroring the granularity
of the JSONL chunk files consumed by ``book_stats/calculate_stats.py``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tiktoken

# Directories that are siblings/children of pipeline outputs but should not be counted
# as part of the final translated text (e.g. pre-post-edit drafts, external references).
EXCLUDED_DIR_NAMES = frozenset({"before_PE", "extern"})

PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def count_tokens(text: str, encoding) -> int:
    """Count tokens in text using the supplied tiktoken encoding."""
    try:
        return len(encoding.encode(text))
    except Exception:
        return len(text.split())


def split_into_chunks(text: str) -> list[str]:
    """Split a book's text into paragraph-sized chunks (separated by blank lines)."""
    return [chunk.strip() for chunk in PARAGRAPH_SPLIT_RE.split(text) if chunk.strip()]


def iter_text_directories(root: Path) -> list[Path]:
    """Yield directories under ``root`` that directly contain ``.txt`` files.

    Any directory whose name is in ``EXCLUDED_DIR_NAMES`` is pruned, along with all of
    its descendants. Returned directories are sorted for deterministic output.
    """
    directories: set[Path] = set()
    stack: list[Path] = [root]

    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError as exc:
            print(f"Warning: cannot read {current}: {exc}", file=sys.stderr)
            continue

        for entry in entries:
            if entry.is_dir():
                if entry.name in EXCLUDED_DIR_NAMES:
                    continue
                stack.append(entry)
            elif entry.is_file() and entry.suffix.lower() == ".txt":
                directories.add(current)

    return sorted(directories)


def load_chunk_counts(directory: Path, encoding) -> pd.DataFrame:
    """Load per-chunk word and token counts for every ``.txt`` file in ``directory``."""
    rows = []

    for path in sorted(directory.glob("*.txt")):
        if not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"Error reading {path}: {exc}", file=sys.stderr)
            raise SystemExit(1)

        chunks = split_into_chunks(text)
        if not chunks:
            # Treat empty files as a single empty chunk so the book still appears.
            chunks = [""]

        for chunk_id, chunk_text in enumerate(chunks, start=1):
            rows.append(
                {
                    "book": path.stem,
                    "file": str(path),
                    "chunk_id": chunk_id,
                    "word_count": len(chunk_text.split()),
                    "token_count": count_tokens(chunk_text, encoding),
                }
            )

    return pd.DataFrame(rows)


def calculate_stats(counts: pd.Series, metric_name: str) -> pd.DataFrame:
    """Return max, min, std, mean, median, and sum for a count series."""
    stats = counts.agg(["max", "min", "std", "mean", "median", "sum"])
    return stats.rename_axis("stat").reset_index(name=metric_name)


def write_stats_for_directory(
    chunk_counts: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Compute the four stat CSVs for one directory's chunk counts and write them."""
    output_dir.mkdir(parents=True, exist_ok=True)

    book_word_counts = (
        chunk_counts.groupby("book", as_index=False)["word_count"].sum().sort_values("book")
    )
    book_token_counts = (
        chunk_counts.groupby("book", as_index=False)["token_count"].sum().sort_values("book")
    )

    outputs = {
        "book_word_count_stats.csv": calculate_stats(book_word_counts["word_count"], "word_count"),
        "chunk_word_count_stats.csv": calculate_stats(chunk_counts["word_count"], "word_count"),
        "book_token_count_stats.csv": calculate_stats(book_token_counts["token_count"], "token_count"),
        "chunk_token_count_stats.csv": calculate_stats(chunk_counts["token_count"], "token_count"),
    }

    written: list[Path] = []
    for filename, frame in outputs.items():
        destination = output_dir / filename
        frame.to_csv(destination, index=False)
        written.append(destination)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate book and chunk word- and token-count statistics for every MT "
            "pipeline output directory under books/MT."
        )
    )
    parser.add_argument(
        "--mt-dir",
        type=Path,
        default=Path("books/MT"),
        help="Root directory containing MT pipeline outputs (default: books/MT).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("book_stats/pipelines"),
        help="Directory in which to mirror the input structure (default: book_stats/pipelines).",
    )
    parser.add_argument(
        "--encoding",
        default="o200k_base",
        help="tiktoken encoding name to use for token counts (default: o200k_base).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mt_dir: Path = args.mt_dir
    output_root: Path = args.output_dir

    if not mt_dir.is_dir():
        print(f"Error: {mt_dir} is not a directory", file=sys.stderr)
        raise SystemExit(1)

    try:
        encoding = tiktoken.get_encoding(args.encoding)
    except Exception as exc:
        print(f"Error loading tiktoken encoding '{args.encoding}': {exc}", file=sys.stderr)
        raise SystemExit(1)

    text_directories = iter_text_directories(mt_dir)
    if not text_directories:
        print(f"Error: no .txt files found under {mt_dir}", file=sys.stderr)
        raise SystemExit(1)

    summary: dict[Path, int] = defaultdict(int)

    for directory in text_directories:
        chunk_counts = load_chunk_counts(directory, encoding)
        if chunk_counts.empty:
            continue

        relative = directory.relative_to(mt_dir)
        target_dir = output_root / relative
        written = write_stats_for_directory(chunk_counts, target_dir)
        summary[directory] = len(chunk_counts["book"].unique())

        print(f"[{relative}] {summary[directory]} book(s) -> {target_dir}")
        for path in written:
            print(f"  wrote {path}")

    total_dirs = len(summary)
    total_books = sum(summary.values())
    print(
        f"Processed {total_dirs} directory(ies), {total_books} book(s) total under {mt_dir}."
    )


if __name__ == "__main__":
    main()
