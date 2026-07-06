#!/usr/bin/env python3
"""Restore controlled-access LAIT data into the public-release tree.

The public GitHub release intentionally replaces source, human-translation, and
some segment-level text fields with a placeholder. This script consumes the
controlled-access JSONL files intended for the gated dataset release and
materializes the withheld data locally.

Dry-run is the default. Pass --apply to write files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PLACEHOLDER = "[withheld from public GitHub release]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the LAIT repository root.",
    )
    parser.add_argument(
        "--books-jsonl",
        type=Path,
        default=Path("controlled_access/lait_books_controlled_access.jsonl"),
        help="Controlled-access book-level JSONL, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--replacements-jsonl",
        type=Path,
        default=Path("controlled_access/withheld_file_replacements.jsonl"),
        help="Exact file-replacement JSONL, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write restored files. Without this flag, only report what would change.",
    )
    parser.add_argument(
        "--skip-book-files",
        action="store_true",
        help="Do not materialize books/dev, books/eval, and books/HT text files.",
    )
    parser.add_argument(
        "--skip-replacement-files",
        action="store_true",
        help="Do not restore exact sanitized metric/alignment files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite replacement-file targets even when they do not contain the placeholder.",
    )
    return parser.parse_args()


def resolve_under_repo(repo_root: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit(f"Refusing path outside repo root: {resolved}") from exc
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def write_text(path: Path, text: str, apply: bool) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return True


def restore_book_files(repo_root: Path, books_jsonl: Path, apply: bool) -> tuple[int, int]:
    rows = read_jsonl(books_jsonl)
    planned = 0
    changed = 0

    for row in rows:
        split = row["split"]
        if split not in {"dev", "eval"}:
            raise SystemExit(f"Unsupported split for {row['book_id']}: {split}")

        source_path = repo_root / "books" / split / row["source_filename"]
        ht_path = repo_root / "books" / "HT" / row["human_translation_filename"]

        for path, text in (
            (source_path, row["source_text"]),
            (ht_path, row["human_translation_text"]),
        ):
            planned += 1
            if write_text(path, text, apply):
                changed += 1

    return planned, changed


def restore_replacement_files(
    repo_root: Path,
    replacements_jsonl: Path,
    apply: bool,
    force: bool,
) -> tuple[int, int]:
    rows = read_jsonl(replacements_jsonl)
    planned = 0
    changed = 0

    for row in rows:
        rel_path = Path(row["path"])
        target = resolve_under_repo(repo_root, rel_path)
        text = row["text"]
        expected_sha = row.get("sha256")
        actual_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if expected_sha and actual_sha != expected_sha:
            raise SystemExit(f"SHA-256 mismatch in replacement payload for {rel_path}")

        if target.exists() and not force:
            existing = target.read_text(encoding="utf-8")
            if PLACEHOLDER not in existing and existing != text:
                raise SystemExit(
                    f"Refusing to overwrite non-sanitized file without --force: {rel_path}"
                )

        planned += 1
        if write_text(target, text, apply):
            changed += 1

    return planned, changed


def count_remaining_placeholders(repo_root: Path) -> int:
    count = 0
    for path in repo_root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            count += path.read_text(encoding="utf-8").count(PLACEHOLDER)
        except UnicodeDecodeError:
            continue
    return count


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    books_jsonl = resolve_under_repo(repo_root, args.books_jsonl)
    replacements_jsonl = resolve_under_repo(repo_root, args.replacements_jsonl)

    total_planned = 0
    total_changed = 0

    if not args.skip_book_files:
        planned, changed = restore_book_files(repo_root, books_jsonl, args.apply)
        total_planned += planned
        total_changed += changed
        print(f"book files: planned={planned} changed={changed}")

    if not args.skip_replacement_files:
        planned, changed = restore_replacement_files(
            repo_root,
            replacements_jsonl,
            args.apply,
            args.force,
        )
        total_planned += planned
        total_changed += changed
        print(f"replacement files: planned={planned} changed={changed}")

    mode = "applied" if args.apply else "dry-run"
    print(f"{mode}: planned={total_planned} changed={total_changed}")

    if args.apply:
        remaining = count_remaining_placeholders(repo_root)
        print(f"remaining placeholder occurrences: {remaining}")


if __name__ == "__main__":
    main()
