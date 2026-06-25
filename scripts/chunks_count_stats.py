"""Print descriptive statistics for word_count values in JSONL chunk files.

Example:
  python scripts/chunks_count_stats.py books/HT/eval
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("books/HT/eval")


def percentile(sorted_values: list[float], percent: float) -> float:
    """Return a linearly interpolated percentile from sorted numeric values."""
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty list.")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percent / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[int(position)]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def parse_word_count(value: Any) -> float | None:
    """Return a numeric word_count value, or None if the value is unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def collect_word_counts(input_dir: Path) -> tuple[list[float], list[str], int]:
    """Read all JSONL files in input_dir and return word counts plus warnings."""
    values: list[float] = []
    warnings: list[str] = []
    total_lines = 0

    for jsonl_path in sorted(input_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                total_lines += 1
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    record: Any = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    warnings.append(
                        f"{jsonl_path}:{line_number}: invalid JSON ({exc.msg})"
                    )
                    continue

                if not isinstance(record, dict):
                    warnings.append(f"{jsonl_path}:{line_number}: JSON value is not an object")
                    continue

                word_count = parse_word_count(record.get("word_count"))
                if word_count is None:
                    warnings.append(f"{jsonl_path}:{line_number}: missing or invalid word_count")
                    continue

                values.append(word_count)

    return values, warnings, total_lines


def format_number(value: float) -> str:
    """Format integers without decimals and non-integers to two decimals."""
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def print_stats(values: list[float], input_dir: Path, file_count: int, total_lines: int) -> None:
    """Print descriptive statistics for collected word_count values."""
    sorted_values = sorted(values)
    count = len(sorted_values)

    q1 = percentile(sorted_values, 25)
    median = percentile(sorted_values, 50)
    q3 = percentile(sorted_values, 75)

    print(f"Input folder: {input_dir}")
    print(f"JSONL files: {file_count:,}")
    print(f"Lines read: {total_lines:,}")
    print(f"Chunks with word_count: {count:,}")
    print()
    print("Word-count statistics:")
    print(f"min: {format_number(min(sorted_values))}")
    print(f"first quartile (Q1 / 25th percentile): {format_number(q1)}")
    print(f"median (Q2 / 50th percentile): {format_number(median)}")
    print(f"third quartile (Q3 / 75th percentile): {format_number(q3)}")
    print(f"max: {format_number(max(sorted_values))}")
    print(f"mean: {format_number(statistics.fmean(sorted_values))}")
    print(f"total words: {format_number(sum(sorted_values))}")
    print(f"interquartile range (IQR): {format_number(q3 - q1)}")
    print(f"population standard deviation: {format_number(statistics.pstdev(sorted_values))}")

    if count > 1:
        print(f"sample standard deviation: {format_number(statistics.stdev(sorted_values))}")

    print(f"10th percentile: {format_number(percentile(sorted_values, 10))}")
    print(f"90th percentile: {format_number(percentile(sorted_values, 90))}")
    print(f"95th percentile: {format_number(percentile(sorted_values, 95))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute descriptive statistics for word_count values in JSONL chunk files."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing .jsonl files. Defaults to {DEFAULT_INPUT_DIR}.",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Print malformed-line and missing-word-count warnings to stderr.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir

    if not input_dir.exists():
        print(f"Input folder does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"Input path is not a folder: {input_dir}", file=sys.stderr)
        return 1

    file_count = len(list(input_dir.glob("*.jsonl")))
    if file_count == 0:
        print(f"No .jsonl files found in {input_dir}", file=sys.stderr)
        return 1

    values, warnings, total_lines = collect_word_counts(input_dir)
    if not values:
        print(f"No valid word_count values found in {input_dir}", file=sys.stderr)
        return 1

    print_stats(values, input_dir, file_count, total_lines)

    if warnings:
        print()
        print(f"Skipped entries: {len(warnings):,}")
        if args.show_warnings:
            for warning in warnings:
                print(f"Warning: {warning}", file=sys.stderr)
        else:
            print("Run with --show-warnings to print details.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
