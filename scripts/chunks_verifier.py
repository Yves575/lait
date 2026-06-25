#!/usr/bin/env python3
"""
Compare chunked `.jsonl` text against reference `.txt` files.

The `.jsonl` side is reconstructed by concatenating each record's `text` field.
The comparison is whitespace-tolerant so chunk boundary formatting differences
do not cause false mismatches.
"""

import sys
import os
import json
import re


def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_jsonl_as_text(path: str) -> str:
    """Read a .jsonl file and join each record's `text` field."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "text" not in record:
                raise ValueError(
                    f"{path}:{line_number} is missing the required `text` field"
                )
            texts.append(record["text"])
    return "\n\n".join(texts)


def load_side(path: str) -> str:
    """Load a comparison input side."""
    if path.endswith(".jsonl"):
        return load_jsonl_as_text(path)
    return strip_chunks(load_file(path))


def strip_chunks(text: str) -> str:
    """Remove <chunk> and </chunk> tags and any \n they introduced."""
    # Remove opening tags: the tag itself plus any trailing newline it added
    text = re.sub(r"<chunk>\n?", "", text)
    # Remove closing tags: any leading newline it added plus the tag itself
    text = re.sub(r"\n?</chunk>", "", text)
    return text


def normalize(text: str) -> str:
    """
    Normalize for comparison:
    - All Unicode/exotic whitespace (\u3000, \u00a0, etc.) → plain space
    - \n between two CJK characters → removed (no space in Japanese/Chinese)
    - \n elsewhere → single space
    - Collapse runs of spaces to one
    """
    # Replace exotic Unicode spaces with plain space
    text = re.sub("[　  -​  ]", " ", text)
    # Remove newline between two CJK characters (no word boundary needed)
    text = re.sub(r"(?<=[　-鿿＀-￯])\s+(?=[　-鿿＀-￯])", "", text)
    # All remaining whitespace runs → single space
    text = re.sub(r"[ \t\r\n]+", " ", text)
    return text.strip()


def find_mismatch(a: str, b: str):
    """
    Find the first position where two strings diverge.
    Returns (position, line_a, col_a, line_b, col_b, context_a, context_b).
    """
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return _build_mismatch_report(a, b, i)

    # One string is a prefix of the other
    if len(a) != len(b):
        return _build_mismatch_report(a, b, min_len)

    return None


def _build_mismatch_report(a: str, b: str, pos: int):
    def line_col(text, idx):
        idx = min(idx, len(text))
        line = text[:idx].count("\n") + 1
        col = idx - text[:idx].rfind("\n")
        return line, col

    line_a, col_a = line_col(a, pos)
    line_b, col_b = line_col(b, pos)

    ctx = 40
    snippet_a = repr(a[max(0, pos - ctx): pos + ctx])
    snippet_b = repr(b[max(0, pos - ctx): pos + ctx])

    char_a = repr(a[pos]) if pos < len(a) else "<END OF FILE>"
    char_b = repr(b[pos]) if pos < len(b) else "<END OF FILE>"

    return {
        "position": pos,
        "file1": {"line": line_a, "col": col_a, "char": char_a, "context": snippet_a},
        "file2": {"line": line_b, "col": col_b, "char": char_b, "context": snippet_b},
    }


def compare_files(chunked_path: str, plain_path: str) -> bool:
    """Compare a `.jsonl` chunked file against a reference `.txt` file."""
    print(f"\n--- Comparing: {chunked_path}  ↔  {plain_path}")

    cleaned = normalize(load_side(chunked_path))
    plain   = normalize(load_side(plain_path))

    if cleaned == plain:
        print("✅  True — files match.")
        return True

    # Files differ — find and report the mismatch
    report = find_mismatch(cleaned, plain)
    print("❌  False — files do NOT match.\n")

    if report:
        print(f"First mismatch at character position {report['position']}:")
        print(
            f"  Chunked file (after cleanup) — "
            f"line {report['file1']['line']}, col {report['file1']['col']}: "
            f"char {report['file1']['char']}"
        )
        print(f"    Context : {report['file1']['context']}")
        print(
            f"  Plain file  — "
            f"line {report['file2']['line']}, col {report['file2']['col']}: "
            f"char {report['file2']['char']}"
        )
        print(f"    Context : {report['file2']['context']}")
    else:
        # Lengths differ but content matches up to the shorter one
        print(
            f"  Content matches up to character {min(len(cleaned), len(plain))}, "
            f"but lengths differ: "
            f"chunked (cleaned)={len(cleaned)}, plain={len(plain)}"
        )

    return False


def compare_folders(chunked_dir: str, plain_dir: str) -> bool:
    """Compare `.jsonl` files in one folder against `.txt` files in another."""
    chunked_index = {}
    for f in os.listdir(chunked_dir):
        stem, ext = os.path.splitext(f)
        if ext == ".jsonl":
            chunked_index[stem] = f
    plain_index = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(plain_dir) if f.endswith(".txt")
    }

    common = sorted(set(chunked_index) & set(plain_index))
    only_chunked = sorted(set(chunked_index) - set(plain_index))
    only_plain = sorted(set(plain_index) - set(chunked_index))

    if only_chunked:
        print(f"⚠  Files only in {chunked_dir}: {', '.join(only_chunked)}")
    if only_plain:
        print(f"⚠  Files only in {plain_dir}: {', '.join(only_plain)}")

    if not common:
        print("❌  No matching filenames found between the two folders.")
        return False

    print(f"Comparing {len(common)} file(s)...\n")

    all_match = True
    match_count = 0
    for stem in common:
        chunked_path = os.path.join(chunked_dir, chunked_index[stem])
        plain_path = os.path.join(plain_dir, plain_index[stem])
        if compare_files(chunked_path, plain_path):
            match_count += 1
        else:
            all_match = False

    print(f"\n{'=' * 60}")
    print(f"Results: {match_count}/{len(common)} files match.")
    return all_match


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python chunks_verifier.py "
            "<chunked_file_or_dir (.jsonl)> <plain_file_or_dir (.txt)>"
        )
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]

    if os.path.isfile(path_a) and not path_a.endswith(".jsonl"):
        print("Error: the first file must be a .jsonl file.")
        sys.exit(1)
    if os.path.isfile(path_b) and not path_b.endswith(".txt"):
        print("Error: the second file must be a .txt file.")
        sys.exit(1)

    if os.path.isdir(path_a) and os.path.isdir(path_b):
        result = compare_folders(path_a, path_b)
    elif os.path.isfile(path_a) and os.path.isfile(path_b):
        result = compare_files(path_a, path_b)
    else:
        print("Error: both arguments must be files or both must be directories.")
        sys.exit(1)

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
