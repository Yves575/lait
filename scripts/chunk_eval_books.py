#!/usr/bin/env python3
"""Chunk eval books into JSONL files using paragraph-aware word limits.

Reads every ``.txt`` file in ``books/eval`` by default and writes one matching
``.jsonl`` file per input into ``books/HT/eval``. Chunk boundaries only occur
between paragraphs so the concatenated chunk text is identical to the original
source text.

Chunking rules:
- target a soft limit of 300 words per chunk
- allow up to 350 words when appending the next paragraph
- if a single paragraph exceeds 350 words, keep it as one chunk

Each JSONL record contains ``chunk_id``, ``word_count``, and ``text``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "books" / "eval"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "books" / "HT" / "eval"
PARAGRAPH_BREAK_RE = re.compile(r"(?:\r?\n)(?:[ \t\f\v]*(?:\r?\n))+")
WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class ParagraphSpan:
    text: str
    word_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--soft-limit", type=int, default=300)
    parser.add_argument("--hard-limit", type=int, default=350)
    return parser


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_paragraph_spans(text: str) -> list[ParagraphSpan]:
    matches = list(PARAGRAPH_BREAK_RE.finditer(text))
    if not matches:
        return [ParagraphSpan(text=text, word_count=count_words(text))] if text else []

    starts = [0, *[match.end() for match in matches]]
    spans: list[ParagraphSpan] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        span_text = text[start:end]
        if span_text:
            spans.append(ParagraphSpan(text=span_text, word_count=count_words(span_text)))
    return spans


def chunk_spans(
    spans: list[ParagraphSpan],
    *,
    soft_limit: int,
    hard_limit: int,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    current_text_parts: list[str] = []
    current_word_count = 0

    def flush() -> None:
        nonlocal current_text_parts, current_word_count
        if not current_text_parts:
            return
        chunks.append(
            {
                "chunk_id": len(chunks) + 1,
                "word_count": current_word_count,
                "text": "".join(current_text_parts),
            }
        )
        current_text_parts = []
        current_word_count = 0

    for span in spans:
        if not current_text_parts:
            current_text_parts.append(span.text)
            current_word_count = span.word_count
            continue

        proposed_word_count = current_word_count + span.word_count
        if current_word_count >= soft_limit or proposed_word_count > hard_limit:
            flush()

        current_text_parts.append(span.text)
        current_word_count = span.word_count if current_word_count == 0 else current_word_count + span.word_count

    flush()
    return chunks


def output_path_for(source_path: Path, output_dir: Path) -> Path:
    parts = source_path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Could not derive output filename from {source_path.name}")
    parts[-1] = "en"
    return output_dir / f"{'_'.join(parts)}.jsonl"


def process_book(source_path: Path, output_dir: Path, soft_limit: int, hard_limit: int) -> tuple[Path, int]:
    output_path = output_path_for(source_path, output_dir)
    if output_path.exists():
        return output_path, 0

    text = source_path.read_text(encoding="utf-8")
    spans = split_paragraph_spans(text)
    chunks = chunk_spans(spans, soft_limit=soft_limit, hard_limit=hard_limit)

    reconstructed = "".join(chunk["text"] for chunk in chunks)
    if reconstructed != text:
        raise RuntimeError(f"Reconstruction mismatch for {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return output_path, len(chunks)


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if args.soft_limit <= 0 or args.hard_limit <= 0:
        print("ERROR: limits must be positive integers.", file=sys.stderr)
        return 1
    if args.soft_limit > args.hard_limit:
        print("ERROR: soft limit cannot exceed hard limit.", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    source_paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
    if not source_paths:
        print(f"ERROR: no .txt files found in {input_dir}", file=sys.stderr)
        return 1

    for source_path in source_paths:
        output_path, chunk_count = process_book(source_path, output_dir, args.soft_limit, args.hard_limit)
        if chunk_count == 0:
            print(f"{source_path.name} -> {output_path} (skipped: already exists)")
            continue
        print(f"{source_path.name} -> {output_path} ({chunk_count} chunks)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
