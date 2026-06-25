"""Export unified CSV for the eval books/chunks compact summary table.

Reads ``book_stats/{human_translation_counts,machine_translation_counts,source_text_counts}/{book,chunk}_{word,token}_count_stats.csv`` and
writes one table-ready CSV with HTR (HT), MTR (MT), and pooled SRC columns.

Optionally refreshes stats via ``book_stats/calculate_stats.py`` (SRC reads
``books/eval/chunks``).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOK_STATS_ROOT = REPO_ROOT / "book_stats"
VERSION_STATS_DIRS = {
    "HT": "human_translation_counts",
    "MT": "machine_translation_counts",
    "SRC": "source_text_counts",
}
HT_BOOKS_DIR = REPO_ROOT / "books" / "HT" / "eval"
MT_CHUNKS_DIR = REPO_ROOT / "books" / "MT_chunks"
SOURCE_CHUNKS_DIR = REPO_ROOT / "books" / "eval" / "chunks"
CALCULATE_STATS_SCRIPT = REPO_ROOT / "book_stats" / "calculate_stats.py"
OUTPUT_PATH = (
    REPO_ROOT
    / "analysis"
    / "manuscript_tables"
    / "tables"
    / "csv"
    / "eval_books_chunks_compact_summary_stats.csv"
)

SECTIONS = ("books", "chunks")
STATS_ORDER = ("mean", "std", "max", "min")
FIELDNAMES = (
    "section",
    "stat",
    "n",
    "htr_tokens",
    "htr_words",
    "mtr_tokens",
    "mtr_words",
    "src_tokens",
    "src_words",
)


def read_simple_stats(path: Path, metric: str) -> dict[str, float]:
    """Read a stats CSV with rows ``stat,<metric>_count``."""
    column = f"{metric}_count"
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stat = row.get("stat", "")
            value = row.get(column, "")
            if stat and value not in ("", None):
                out[stat] = float(value)
    return out


def count_eval_chunks(books_dir: Path) -> int:
    """Count non-empty JSONL lines across all eval books."""
    total = 0
    for path in sorted(books_dir.glob("*.jsonl")):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as fh:
            total += sum(1 for line in fh if line.strip())
    return total


def count_eval_books(books_dir: Path) -> int:
    return len([p for p in books_dir.glob("*.jsonl") if p.is_file()])


def refresh_book_stats() -> None:
    """Re-run calculate_stats for HT, MT, and pooled SRC (eval source chunks)."""
    commands = [
        [
            sys.executable,
            str(CALCULATE_STATS_SCRIPT),
            "--chunks-dir",
            str(HT_BOOKS_DIR),
            "--output-dir",
            str(BOOK_STATS_ROOT / VERSION_STATS_DIRS["HT"]),
        ],
        [
            sys.executable,
            str(CALCULATE_STATS_SCRIPT),
            "--chunks-dir",
            str(MT_CHUNKS_DIR),
            "--output-dir",
            str(BOOK_STATS_ROOT / VERSION_STATS_DIRS["MT"]),
        ],
        [
            sys.executable,
            str(CALCULATE_STATS_SCRIPT),
            "--chunks-dir",
            str(SOURCE_CHUNKS_DIR),
            "--output-dir",
            str(BOOK_STATS_ROOT / VERSION_STATS_DIRS["SRC"]),
        ],
    ]
    for cmd in commands:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def load_section_stats(version: str, section: str) -> dict[str, dict[str, float]]:
    """Load token and word stats for books or chunks."""
    prefix = "book" if section == "books" else "chunk"
    version_dir = BOOK_STATS_ROOT / VERSION_STATS_DIRS[version]
    return {
        "token": read_simple_stats(version_dir / f"{prefix}_token_count_stats.csv", "token"),
        "word": read_simple_stats(version_dir / f"{prefix}_word_count_stats.csv", "word"),
    }


def build_rows(n_books: int, n_chunks: int) -> list[dict[str, object]]:
    """Build export rows for books and chunks sections."""
    htr = {section: load_section_stats("HT", section) for section in SECTIONS}
    mtr = {section: load_section_stats("MT", section) for section in SECTIONS}
    src = {section: load_section_stats("SRC", section) for section in SECTIONS}
    n_by_section = {"books": n_books, "chunks": n_chunks}

    rows: list[dict[str, object]] = []
    for section in SECTIONS:
        n = n_by_section[section]
        for stat in STATS_ORDER:
            rows.append(
                {
                    "section": section,
                    "stat": stat,
                    "n": n,
                    "htr_tokens": htr[section]["token"][stat],
                    "htr_words": htr[section]["word"][stat],
                    "mtr_tokens": mtr[section]["token"][stat],
                    "mtr_words": mtr[section]["word"][stat],
                    "src_tokens": src[section]["token"][stat],
                    "src_words": src[section]["word"][stat],
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unified eval books/chunks summary stats CSV."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Re-run calculate_stats for HT, MT, and SRC "
            "(books/eval/chunks) before export."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output CSV path (default: {OUTPUT_PATH.relative_to(REPO_ROOT)}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not HT_BOOKS_DIR.is_dir():
        print(f"Error: {HT_BOOKS_DIR} is not a directory", file=sys.stderr)
        raise SystemExit(1)

    if not SOURCE_CHUNKS_DIR.is_dir():
        print(f"Error: {SOURCE_CHUNKS_DIR} is not a directory", file=sys.stderr)
        raise SystemExit(1)

    if args.refresh:
        if not CALCULATE_STATS_SCRIPT.is_file():
            print(f"Error: {CALCULATE_STATS_SCRIPT} not found", file=sys.stderr)
            raise SystemExit(1)
        refresh_book_stats()

    src_stats_dir = BOOK_STATS_ROOT / VERSION_STATS_DIRS["SRC"]
    if not src_stats_dir.is_dir():
        print(
            f"Error: {src_stats_dir.relative_to(REPO_ROOT)} not found. "
            "Run with --refresh to generate source stats.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    n_books = count_eval_books(HT_BOOKS_DIR)
    n_chunks = count_eval_chunks(HT_BOOKS_DIR)
    if n_books == 0:
        print(f"Error: no JSONL files in {HT_BOOKS_DIR}", file=sys.stderr)
        raise SystemExit(1)
    if n_chunks == 0:
        print(f"Error: no chunks found in {HT_BOOKS_DIR}", file=sys.stderr)
        raise SystemExit(1)

    rows = build_rows(n_books, n_chunks)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path.relative_to(REPO_ROOT)}")
    print(f"n_books={n_books}, n_chunks={n_chunks}")


if __name__ == "__main__":
    main()
