"""Convert txt files with <chunk> tags into jsonl files.

Reads every `.txt` file in the input directory, extracts the text inside each
`<chunk>...</chunk>` block, and writes one JSON object per chunk to
`{output_dir}/{name}.jsonl` with fields: `chunk_id`, `word_count`, `text`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CHUNK_PATTERN = re.compile(r"<chunk>(.*?)</chunk>", re.DOTALL)


def extract_chunks(text: str) -> list[str]:
    """Return the contents of every <chunk>...</chunk> block, stripped."""
    return [m.group(1).strip("\n") for m in CHUNK_PATTERN.finditer(text)]


def convert_file(src: Path, dst: Path) -> int:
    """Convert one txt file to jsonl. Return the number of chunks written."""
    text = src.read_text(encoding="utf-8")
    chunks = extract_chunks(text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for chunk_id, chunk_text in enumerate(chunks, start=1):
            record = {
                "chunk_id": chunk_id,
                "word_count": len(chunk_text.split()),
                "text": chunk_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing .txt files with <chunk> tags.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write .jsonl files to.",
    )
    args = parser.parse_args()

    txt_files = sorted(args.input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {args.input_dir}")
        return

    for src in txt_files:
        dst = args.output_dir / (src.stem + ".jsonl")
        n = convert_file(src, dst)
        print(f"{src.name}: {n} chunks -> {dst}")


if __name__ == "__main__":
    main()
