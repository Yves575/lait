#!/usr/bin/env python3
"""Recompute count fields for JSONL chunk files.

Updates each record's ``word_count`` from its ``text`` field. If a record
already contains ``token_count``, that field is also recomputed using the same
``o200k_base`` tokenizer used elsewhere in the repository.

The input may be a single ``.jsonl`` file or a directory. When a directory is
provided, all ``.jsonl`` files under it are processed recursively.

Files are rewritten in place unless ``--output`` is provided. When processing a
directory, ``--output`` must point to a directory and the input structure is
mirrored there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to an input .jsonl file or a directory containing .jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. If omitted, the input file is updated in place.",
    )
    parser.add_argument(
        "--skip-token-count",
        action="store_true",
        help="Only update word_count and leave any existing token_count values unchanged.",
    )
    return parser.parse_args()


def build_token_counter():
    try:
        import tiktoken
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tiktoken is required to recompute token_count. "
            "Install requirements or rerun with --skip-token-count."
        ) from exc

    encoding = tiktoken.get_encoding("o200k_base")
    return lambda text: len(encoding.encode(text))


def update_record(
    record: Any,
    path: Path,
    line_number: int,
    token_counter,
    skip_token_count: bool,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: expected a JSON object per line.")

    text = record.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{path}:{line_number}: missing or invalid 'text' field.")

    updated = dict(record)
    updated["word_count"] = len(text.split())
    if "token_count" in updated and not skip_token_count:
        updated["token_count"] = token_counter(text)
    return updated


def rewrite_jsonl(input_path: Path, output_path: Path, *, skip_token_count: bool) -> tuple[int, int]:
    updated_records = 0
    updated_token_counts = 0
    token_counter = None

    with input_path.open("r", encoding="utf-8") as src:
        if output_path == input_path:
            with NamedTemporaryFile("w", encoding="utf-8", dir=input_path.parent, delete=False) as tmp:
                temp_path = Path(tmp.name)
                for line_number, line in enumerate(src, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "token_count" in record and not skip_token_count and token_counter is None:
                        token_counter = build_token_counter()
                    updated = update_record(
                        record,
                        input_path,
                        line_number,
                        token_counter,
                        skip_token_count,
                    )
                    tmp.write(json.dumps(updated, ensure_ascii=False) + "\n")
                    updated_records += 1
                    if "token_count" in record and not skip_token_count:
                        updated_token_counts += 1
            temp_path.replace(input_path)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as dst:
                for line_number, line in enumerate(src, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "token_count" in record and not skip_token_count and token_counter is None:
                        token_counter = build_token_counter()
                    updated = update_record(
                        record,
                        input_path,
                        line_number,
                        token_counter,
                        skip_token_count,
                    )
                    dst.write(json.dumps(updated, ensure_ascii=False) + "\n")
                    updated_records += 1
                    if "token_count" in record and not skip_token_count:
                        updated_token_counts += 1

    return updated_records, updated_token_counts


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Input file must be .jsonl: {input_path}")
        return [input_path]

    if input_path.is_dir():
        files = sorted(path for path in input_path.rglob("*.jsonl") if path.is_file())
        if not files:
            raise FileNotFoundError(f"No .jsonl files found under directory: {input_path}")
        return files

    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    output_path = args.output.resolve() if args.output else None
    input_files = collect_input_files(input_path)

    if input_path.is_file():
        destination = output_path if output_path else input_path
        updated_records, updated_token_counts = rewrite_jsonl(
            input_path,
            destination,
            skip_token_count=args.skip_token_count,
        )
        print(
            f"Updated {updated_records} record(s) in {destination}. "
            f"Recomputed token_count for {updated_token_counts} record(s)."
        )
        return

    if output_path and output_path.exists() and not output_path.is_dir():
        raise ValueError(f"--output must be a directory when input is a directory: {output_path}")

    total_files = 0
    total_records = 0
    total_token_counts = 0

    for source_path in input_files:
        destination = source_path
        if output_path:
            destination = output_path / source_path.relative_to(input_path)

        updated_records, updated_token_counts = rewrite_jsonl(
            source_path,
            destination,
            skip_token_count=args.skip_token_count,
        )
        total_files += 1
        total_records += updated_records
        total_token_counts += updated_token_counts
        print(
            f"Updated {updated_records} record(s) in {destination}. "
            f"Recomputed token_count for {updated_token_counts} record(s)."
        )

    print(
        f"Processed {total_files} file(s), updated {total_records} record(s) total. "
        f"Recomputed token_count for {total_token_counts} record(s)."
    )


if __name__ == "__main__":
    main()
