"""Compare JSONL chunk word counts against matching parent-folder .txt books."""

import argparse
import json
import sys
from pathlib import Path


def count_words(text: str) -> int:
    """Return the number of whitespace-separated words in text."""
    return len(text.split())


def sum_chunk_word_counts(path: Path) -> tuple[int, int]:
    """Return the sum of word_count fields and the number of invalid chunks."""
    chunk_total = 0
    invalid_chunks = 0

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"{path}:{line_number}: invalid JSON: {exc}", file=sys.stderr)
                invalid_chunks += 1
                continue

            chunk_id = chunk.get("chunk_id", line_number)
            expected = chunk.get("word_count")

            if not isinstance(expected, int):
                print(
                    f"{path}:{line_number}: chunk {chunk_id}: missing/non-integer word_count",
                    file=sys.stderr,
                )
                invalid_chunks += 1
                continue

            chunk_total += expected

    return chunk_total, invalid_chunks


def matching_txt_path(jsonl_path: Path) -> Path:
    """Return the expected .txt book path for a JSONL file in an eval folder."""
    return jsonl_path.parent.parent / f"{jsonl_path.stem}.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare summed JSONL chunk word_count fields to matching .txt books."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("books/HT/eval"),
        help="Folder containing .jsonl books (default: books/HT/eval).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.folder.is_dir():
        print(f"Error: {args.folder} is not a directory", file=sys.stderr)
        raise SystemExit(1)

    files = sorted(path for path in args.folder.glob("*.jsonl") if path.is_file())
    if not files:
        print(f"No .jsonl files found in {args.folder}")
        return

    total_txt_words = 0
    total_chunk_words = 0
    failures = 0

    print("txt_words\tchunk_words\tdifference\tinvalid_chunks\tstatus\tjsonl_file\ttxt_file")
    for jsonl_path in files:
        txt_path = matching_txt_path(jsonl_path)
        chunk_words, invalid_chunks = sum_chunk_word_counts(jsonl_path)
        total_chunk_words += chunk_words

        if not txt_path.is_file():
            print(f"0\t{chunk_words}\tNA\t{invalid_chunks}\tMISSING_TXT\t{jsonl_path}\t{txt_path}")
            failures += 1
            continue

        txt_words = count_words(txt_path.read_text(encoding="utf-8"))
        difference = txt_words - chunk_words
        status = "OK" if difference == 0 and invalid_chunks == 0 else "MISMATCH"

        total_txt_words += txt_words
        if status != "OK":
            failures += 1

        print(
            f"{txt_words}\t{chunk_words}\t{difference}\t{invalid_chunks}\t"
            f"{status}\t{jsonl_path}\t{txt_path}"
        )

    total_difference = total_txt_words - total_chunk_words
    print(f"{total_txt_words}\t{total_chunk_words}\t{total_difference}\t-\tTOTAL\t-\t-")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
