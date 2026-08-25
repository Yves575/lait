"""Print total spaCy sentence counts for eval HT and MT chunks combined.

Uses existing ``chunk_sentence_count_stats.csv`` ``sum`` rows produced by
``book_stats/calculate_stats.py --count-sentences`` (spaCy ``en_core_web_trf``).
Does not re-run the sentence pipeline unless ``--refresh`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CALCULATE_STATS_SCRIPT = REPO_ROOT / "book_stats" / "calculate_stats.py"
HT_STATS = (
    REPO_ROOT / "book_stats" / "human_translation_counts" / "chunk_sentence_count_stats.csv"
)
MT_STATS = (
    REPO_ROOT / "book_stats" / "machine_translation_counts" / "chunk_sentence_count_stats.csv"
)
HT_CHUNKS_DIR = REPO_ROOT / "books" / "HT" / "eval"
MT_CHUNKS_DIR = REPO_ROOT / "books" / "MT_chunks"


def sentence_sum(path: Path) -> int:
    """Return the ``sum`` row from a chunk_sentence_count_stats.csv."""
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("stat") == "sum" and row.get("sentence_count") not in ("", None):
                return int(float(row["sentence_count"]))
    raise SystemExit(f"Error: no sum row in {path.relative_to(REPO_ROOT)}")


def refresh_sentence_stats() -> None:
    """Re-run calculate_stats --count-sentences for eval HT and MT chunks."""
    jobs = (
        (HT_CHUNKS_DIR, HT_STATS.parent),
        (MT_CHUNKS_DIR, MT_STATS.parent),
    )
    for chunks_dir, output_dir in jobs:
        if not chunks_dir.is_dir():
            print(f"Error: {chunks_dir} is not a directory", file=sys.stderr)
            raise SystemExit(1)
        subprocess.run(
            [
                sys.executable,
                str(CALCULATE_STATS_SCRIPT),
                "--chunks-dir",
                str(chunks_dir),
                "--output-dir",
                str(output_dir),
                "--count-sentences",
            ],
            check=True,
            cwd=REPO_ROOT,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print total spaCy sentence counts for eval HT and MT chunks. "
            "Reads existing book_stats CSVs by default."
        )
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Re-run calculate_stats.py --count-sentences on books/HT/eval and "
            "books/MT_chunks before summing. Requires those chunk directories."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh:
        refresh_sentence_stats()

    for path in (HT_STATS, MT_STATS):
        if not path.is_file():
            print(
                f"Error: {path.relative_to(REPO_ROOT)} not found. "
                "Run with --refresh (needs books/HT/eval and books/MT_chunks), "
                "or regenerate via book_stats/calculate_stats.py --count-sentences.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    ht_total = sentence_sum(HT_STATS)
    mt_total = sentence_sum(MT_STATS)
    combined = ht_total + mt_total
    print(f"HT eval chunk sentences: {ht_total:,}")
    print(f"MT eval chunk sentences: {mt_total:,}")
    print(f"combined: {combined:,}")


if __name__ == "__main__":
    main()
