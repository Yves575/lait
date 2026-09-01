"""Smoke-test average whitespace-delimited words per page of a PDF.

Uses that average to estimate how many pages the eval HT+MT word total fills.
Word counts use ``str.split()``, same as ``calculate_stats.py``.
Eval totals are the ``sum`` rows in existing ``chunk_word_count_stats.csv`` files.
Requires pypdf (``python -m pip install pypdf``). Does not print page text.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    print("Error: pypdf is required. Install with: python -m pip install pypdf", file=sys.stderr)
    raise SystemExit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = REPO_ROOT / "books" / "SRC" / "smoke_test.pdf"
HT_WORD_STATS = (
    REPO_ROOT / "book_stats" / "human_translation_counts" / "chunk_word_count_stats.csv"
)
MT_WORD_STATS = (
    REPO_ROOT / "book_stats" / "machine_translation_counts" / "chunk_word_count_stats.csv"
)


def page_word_counts(pdf: Path) -> list[int]:
    """Return whitespace-delimited word counts, one per PDF page."""
    reader = PdfReader(str(pdf))
    return [len((page.extract_text() or "").split()) for page in reader.pages]


def word_sum(path: Path) -> int:
    """Return the ``sum`` row from a chunk_word_count_stats.csv."""
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("stat") == "sum" and row.get("word_count") not in ("", None):
                return int(float(row["word_count"]))
    raise SystemExit(f"Error: no sum row in {path.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print average whitespace-delimited words per PDF page, "
            "then estimate eval HT+MT pages from existing word-count CSVs."
        )
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF,
        help="PDF path (default: books/SRC/smoke_test.pdf)",
    )
    return parser.parse_args()


def main() -> None:
    pdf = parse_args().pdf
    if not pdf.is_file():
        print(f"Error: PDF not found: {pdf}", file=sys.stderr)
        raise SystemExit(1)

    counts = page_word_counts(pdf)
    assert counts, f"no pages extracted from {pdf}"
    pdf_words = sum(counts)
    assert pdf_words > 0, f"no words extracted from {pdf}"
    n_pages = len(counts)
    avg = pdf_words / n_pages
    print(f"pdf: {pdf.name}")
    print(f"pages: {n_pages}")
    print(f"words: {pdf_words:,}")
    print(f"avg words/page: {avg:.1f}")

    for path in (HT_WORD_STATS, MT_WORD_STATS):
        if not path.is_file():
            print(
                f"Error: {path.relative_to(REPO_ROOT)} not found. "
                "Regenerate via book_stats/calculate_stats.py.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    ht_words = word_sum(HT_WORD_STATS)
    mt_words = word_sum(MT_WORD_STATS)
    eval_words = ht_words + mt_words
    eval_pages = eval_words / avg
    print(f"HT eval words: {ht_words:,}")
    print(f"MT eval words: {mt_words:,}")
    print(f"eval HT+MT words: {eval_words:,}")
    print(f"eval pages: {eval_pages:.1f}")
    print(f"TOTAL eval pages (factoring in 2 readings per excerpt): {eval_pages * 2:.1f}")


if __name__ == "__main__":
    main()
