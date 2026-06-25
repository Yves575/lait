"""Compare non-empty line counts for matching .txt files in two folders.

Files are matched by their stem without the final language suffix. For example,
``cece_pl_en.txt`` matches ``cece_pl_pl.txt`` because both use the key
``cece_pl``.

Example:
  python scripts/compare_txt_line_counts.py books/msft/dev books/dev
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextFile:
    path: Path
    key: str
    line_count: int


def line_count(path: Path) -> int:
    """Count non-empty lines, treating repeated blank lines as zero lines."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def match_key(path: Path) -> str:
    """Return the filename key used to match source and target files."""
    stem = path.stem
    prefix, separator, _language = stem.rpartition("_")
    if not separator:
        return stem
    return prefix


def collect_files(folder: Path, recursive: bool) -> dict[str, list[TextFile]]:
    pattern = "**/*.txt" if recursive else "*.txt"
    files: dict[str, list[TextFile]] = {}

    for path in sorted(folder.glob(pattern)):
        if not path.is_file():
            continue
        key = match_key(path)
        files.setdefault(key, []).append(
            TextFile(path=path, key=key, line_count=line_count(path))
        )

    return files


def validate_folder(path: Path, label: str) -> bool:
    if not path.exists():
        print(f"{label} folder does not exist: {path}", file=sys.stderr)
        return False
    if not path.is_dir():
        print(f"{label} path is not a folder: {path}", file=sys.stderr)
        return False
    return True


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def print_duplicate_keys(label: str, files_by_key: dict[str, list[TextFile]]) -> int:
    duplicates = 0
    for key, files in files_by_key.items():
        if len(files) <= 1:
            continue
        duplicates += len(files)
        print(f"AMBIGUOUS {label} key {key!r}:")
        for file in files:
            print(f"  - {relative(file.path)}")
    return duplicates


def compare(source_dir: Path, target_dir: Path, recursive: bool, show_matches: bool) -> int:
    source_files = collect_files(source_dir, recursive)
    target_files = collect_files(target_dir, recursive)

    if not source_files:
        print(f"No .txt files found in source folder: {source_dir}", file=sys.stderr)
        return 1

    failures = 0
    failures += print_duplicate_keys("source", source_files)
    failures += print_duplicate_keys("target", target_files)

    if failures:
        return 1

    matched = 0
    mismatched = 0
    missing = 0

    print("status\tsource_lines\ttarget_lines\tsource\ttarget")
    for key in sorted(source_files):
        source = source_files[key][0]
        targets = target_files.get(key)

        if not targets:
            missing += 1
            print(f"MISSING\t{source.line_count}\t-\t{relative(source.path)}\t-")
            continue

        target = targets[0]
        matched += 1
        status = "OK" if source.line_count == target.line_count else "DIFF"
        if status == "DIFF":
            mismatched += 1

        if show_matches or status == "DIFF":
            print(
                f"{status}\t{source.line_count}\t{target.line_count}\t"
                f"{relative(source.path)}\t{relative(target.path)}"
            )

    extra_targets = sorted(set(target_files) - set(source_files))
    for key in extra_targets:
        target = target_files[key][0]
        print(f"EXTRA\t-\t{target.line_count}\t-\t{relative(target.path)}")

    print()
    print(f"Source files: {sum(len(files) for files in source_files.values()):,}")
    print(f"Matched files: {matched:,}")
    print(f"Mismatched line counts: {mismatched:,}")
    print(f"Missing target files: {missing:,}")
    print(f"Extra target files: {len(extra_targets):,}")

    return 1 if mismatched or missing or extra_targets else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare non-empty line counts for .txt files in a source folder "
            "against matching .txt files in a target folder."
        )
    )
    parser.add_argument("source_dir", type=Path, help="Folder containing source .txt files.")
    parser.add_argument("target_dir", type=Path, help="Folder containing target .txt files.")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search for .txt files recursively instead of only the folder root.",
    )
    parser.add_argument(
        "--show-matches",
        action="store_true",
        help="Print matching OK files too. By default, only differences are printed.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not validate_folder(args.source_dir, "Source"):
        return 1
    if not validate_folder(args.target_dir, "Target"):
        return 1

    return compare(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        recursive=args.recursive,
        show_matches=args.show_matches,
    )


if __name__ == "__main__":
    raise SystemExit(main())
