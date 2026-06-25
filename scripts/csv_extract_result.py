"""Count cited pipelines from an anonymized review CSV.

The CSV is expected to contain:
  - a "Book Name" column
  - a "Preferred Version" column
  - a "Source Lang" column

The mapping JSON is expected to map:
  {
    "<book_key>": {
      "<fruit>": "<pipeline>"
    }
  }

Example:
  python script/csv_extract_result.py dev_books/Books_Forms.csv mapping.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


BOOK_NAME_COLUMN = "Book Name"
PREFERRED_VERSION_COLUMN = "Preferred Version"
SOURCE_LANG_COLUMN = "Source Lang"
UNKNOWN_SOURCE_LANG = "Unknown"


def normalize_book_name(book_name: str) -> str:
    """Normalize a CSV book title to the mapping.json key format."""
    normalized = book_name.strip().lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def load_mapping(mapping_path: Path) -> dict[str, dict[str, str]]:
    """Load the book -> fruit -> pipeline mapping."""
    with mapping_path.open("r", encoding="utf-8") as handle:
        data: Any = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {mapping_path}")

    cleaned: dict[str, dict[str, str]] = {}
    for book_name, fruit_map in data.items():
        if not isinstance(book_name, str) or not isinstance(fruit_map, dict):
            raise ValueError(f"Invalid mapping entry for {book_name!r}")
        cleaned[book_name] = {}
        for fruit, pipeline in fruit_map.items():
            if not isinstance(fruit, str) or not isinstance(pipeline, str):
                raise ValueError(f"Invalid fruit mapping for {book_name!r}")
            cleaned[book_name][fruit.strip().lower()] = pipeline.strip()
    return cleaned


def csv_extract_result(
    csv_path: Path,
    mapping: dict[str, dict[str, str]],
) -> tuple[Counter[str], dict[str, Counter[str]], list[str]]:
    """Return overall and per-language pipeline counts plus non-fatal warnings."""
    counts: Counter[str] = Counter()
    counts_by_source_lang: dict[str, Counter[str]] = {}
    warnings: list[str] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        required_columns = {BOOK_NAME_COLUMN, PREFERRED_VERSION_COLUMN, SOURCE_LANG_COLUMN}
        missing_columns = required_columns.difference(fieldnames)
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV column(s): {missing_list}")

        for row_number, row in enumerate(reader, start=2):
            book_name_raw = (row.get(BOOK_NAME_COLUMN) or "").strip()
            fruit_raw = (row.get(PREFERRED_VERSION_COLUMN) or "").strip().lower()
            source_lang = (row.get(SOURCE_LANG_COLUMN) or "").strip()

            if not book_name_raw and not fruit_raw:
                continue
            if not fruit_raw:
                continue
            if not book_name_raw:
                warnings.append(
                    f"Line {row_number}: missing '{BOOK_NAME_COLUMN}' for preferred version {fruit_raw!r}"
                )
                continue

            book_key = normalize_book_name(book_name_raw)
            fruit_mapping = mapping.get(book_key)
            if fruit_mapping is None:
                warnings.append(
                    f"Line {row_number}: book {book_name_raw!r} normalized to {book_key!r} was not found in mapping.json"
                )
                continue

            pipeline_name = fruit_mapping.get(fruit_raw)
            if pipeline_name is None:
                warnings.append(
                    f"Line {row_number}: fruit {fruit_raw!r} was not found for book {book_key!r}"
                )
                continue

            if not source_lang:
                source_lang = UNKNOWN_SOURCE_LANG
                warnings.append(
                    f"Line {row_number}: missing '{SOURCE_LANG_COLUMN}' for valid citation"
                )

            counts[pipeline_name] += 1
            counts_by_source_lang.setdefault(source_lang, Counter())[pipeline_name] += 1

    return counts, counts_by_source_lang, warnings


def get_winners(counts: Counter[str]) -> tuple[list[str], int]:
    """Return pipeline names tied for first place and their count."""
    top_count = max(counts.values())
    winners = sorted(
        pipeline_name for pipeline_name, count in counts.items() if count == top_count
    )
    return winners, top_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deanonymize an evaluation CSV and count cited pipelines."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV file containing Book Name and Preferred Version columns.",
    )
    parser.add_argument(
        "mapping_path",
        type=Path,
        help="Path to the mapping JSON file.",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Print skipped-row warnings to stderr.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    mapping = load_mapping(args.mapping_path)
    counts, counts_by_source_lang, warnings = csv_extract_result(args.csv_path, mapping)

    total_citations = sum(counts.values())
    if total_citations == 0:
        print("No valid pipeline citations found.")
    else:
        print("Pipeline counts:")
        for pipeline_name, count in counts.most_common():
            print(f"{pipeline_name}: {count}")

        winners, top_count = get_winners(counts)

        print()
        print(f"Total valid citations: {total_citations}")
        if len(winners) == 1:
            print(f"Most cited pipeline: {winners[0]} ({top_count})")
        else:
            print(f"Most cited pipelines: {', '.join(winners)} ({top_count} each)")

        print()
        print(f"Most cited pipeline per {SOURCE_LANG_COLUMN}:")
        for source_lang in sorted(counts_by_source_lang):
            language_counts = counts_by_source_lang[source_lang]
            language_winners, language_top_count = get_winners(language_counts)
            language_total = sum(language_counts.values())
            if len(language_winners) == 1:
                print(
                    f"{source_lang}: {language_winners[0]} ({language_top_count} of {language_total})"
                )
            else:
                print(
                    f"{source_lang}: {', '.join(language_winners)} ({language_top_count} each of {language_total})"
                )

    if args.show_warnings and warnings:
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
