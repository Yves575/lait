#!/usr/bin/env python3
"""Merge source, HT, and MT chunk JSONL files into review-friendly files.

For each ``.jsonl`` file in ``--mt-dir``, this script looks for a file with the
same name in ``--ht-dir`` and writes a merged file to ``--output-dir``. Output
records are keyed by ``chunk_id`` and contain:

- ``chunk_id``
- ``SRC``: text from the optional source file, or ``null`` if missing
- ``HT``: text from the HT file, or ``null`` if missing
- ``MT``: text from the MT file, or ``null`` if missing

Chunk order follows the MT file first, then any HT-only chunks are appended in
their original HT order, followed by any SRC-only chunks when ``--src-dir`` is
provided.

When ``--src-dir`` is provided, the script first looks for an exact filename
match in that directory. If none exists, it replaces the final language suffix
in the MT filename with ``--src-lang``. When ``--src-lang`` is not provided, the
source language is inferred from the penultimate suffix. For example,
``mona_s_eyes_fr_en.jsonl`` maps to ``mona_s_eyes_fr_fr.jsonl``.

For easier manual review, each record is written as pretty-printed JSON with a
blank line between records, even though the output filenames keep the ``.jsonl``
extension.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mt-dir",
        default="books/DEL1/",
        help="Directory containing MT chunk JSONL files.",
    )
    parser.add_argument(
        "--ht-dir",
        default="books/HT/eval/",
        help="Directory containing HT chunk JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        default="books/chunk_review/",
        help="Directory where merged JSONL files will be written.",
    )
    parser.add_argument(
        "--src-dir",
        dest="src_dir",
        help="Optional directory containing source chunk JSONL files.",
    )
    parser.add_argument(
        "--en-dir",
        dest="src_dir",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--src-lang",
        dest="src_lang",
        help=(
            "Language suffix to use when resolving files from --src-dir. "
            "Defaults to inferring the source language from the MT filename."
        ),
    )
    parser.add_argument(
        "--en-lang",
        dest="src_lang",
        help=argparse.SUPPRESS,
    )
    return parser


def load_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc

            if "chunk_id" not in record or "text" not in record:
                raise ValueError(f"Missing required keys in {path} at line {line_number}")
            records.append(record)
    return records


def source_path_for(mt_path: Path, src_dir: Path, src_lang: str | None = None) -> Path:
    exact_path = src_dir / mt_path.name
    if exact_path.is_file():
        return exact_path

    stem_prefix, separator, _target_lang = mt_path.stem.rpartition("_")
    if not separator:
        return exact_path

    resolved_src_lang = src_lang
    if resolved_src_lang is None:
        _book_prefix, source_separator, inferred_src_lang = stem_prefix.rpartition("_")
        if not source_separator:
            return exact_path
        resolved_src_lang = inferred_src_lang

    return src_dir / f"{stem_prefix}_{resolved_src_lang}{mt_path.suffix}"


def merge_records(
    mt_records: list[dict],
    ht_records: list[dict],
    src_records: list[dict] | None = None,
) -> tuple[list[dict], int, int, int, int]:
    mt_by_chunk_id = {record["chunk_id"]: record for record in mt_records}
    ht_by_chunk_id = {record["chunk_id"]: record for record in ht_records}
    src_by_chunk_id = (
        {record["chunk_id"]: record for record in src_records}
        if src_records is not None
        else {}
    )

    merged: list[dict] = []
    seen_chunk_ids: set = set()
    matched = 0
    mt_only = 0
    src_missing = 0

    for record in mt_records:
        chunk_id = record["chunk_id"]
        ht_record = ht_by_chunk_id.get(chunk_id)
        src_record = src_by_chunk_id.get(chunk_id)
        merged_record = {"chunk_id": chunk_id}
        if src_records is not None:
            merged_record["SRC"] = None if src_record is None else src_record.get("text")
            if src_record is None:
                src_missing += 1
        merged_record["HT"] = None if ht_record is None else ht_record.get("text")
        merged_record["MT"] = record.get("text")
        merged.append(merged_record)
        seen_chunk_ids.add(chunk_id)
        if ht_record is None:
            mt_only += 1
        else:
            matched += 1

    ht_only = 0
    for record in ht_records:
        chunk_id = record["chunk_id"]
        if chunk_id in seen_chunk_ids:
            continue
        src_record = src_by_chunk_id.get(chunk_id)
        merged_record = {"chunk_id": chunk_id}
        if src_records is not None:
            merged_record["SRC"] = None if src_record is None else src_record.get("text")
            if src_record is None:
                src_missing += 1
        merged_record["HT"] = record.get("text")
        merged_record["MT"] = None
        merged.append(merged_record)
        seen_chunk_ids.add(chunk_id)
        ht_only += 1

    if src_records is not None:
        for record in src_records:
            chunk_id = record["chunk_id"]
            if chunk_id in seen_chunk_ids:
                continue
            merged.append(
                {
                    "chunk_id": chunk_id,
                    "SRC": record.get("text"),
                    "HT": None,
                    "MT": None,
                }
            )

    return merged, matched, mt_only, ht_only, src_missing


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(json.dumps(record, ensure_ascii=False, indent=2))
            handle.write("\n")
            if index < len(records) - 1:
                handle.write("\n")


def main() -> int:
    args = build_parser().parse_args()

    mt_dir = Path(args.mt_dir)
    ht_dir = Path(args.ht_dir)
    output_dir = Path(args.output_dir)
    src_dir = Path(args.src_dir) if args.src_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    mt_paths = sorted(path for path in mt_dir.glob("*.jsonl") if path.is_file())
    if not mt_paths:
        raise FileNotFoundError(f"No .jsonl files found in MT directory: {mt_dir}")

    for mt_path in mt_paths:
        output_path = output_dir / f"{mt_path.stem}{mt_path.suffix}"
        if output_path.exists():
            print(f"{mt_path.name}: skipped existing output at {output_path}")
            continue

        ht_path = ht_dir / mt_path.name
        if not ht_path.is_file():
            raise FileNotFoundError(f"Missing matching HT file for {mt_path.name}: {ht_path}")

        mt_records = load_jsonl_records(mt_path)
        ht_records = load_jsonl_records(ht_path)
        src_records = None
        src_path = None
        if src_dir is not None:
            src_path = source_path_for(mt_path, src_dir, args.src_lang)
            if not src_path.is_file():
                raise FileNotFoundError(f"Missing matching SRC file for {mt_path.name}: {src_path}")
            src_records = load_jsonl_records(src_path)

        merged_records, matched, mt_only, ht_only, src_missing = merge_records(
            mt_records,
            ht_records,
            src_records,
        )

        write_jsonl(output_path, merged_records)

        stats = f"matched={matched} mt_only={mt_only} ht_only={ht_only}"
        if src_path is not None:
            stats += f" src={src_path.name} src_missing={src_missing}"
        print(f"{mt_path.name}: {stats}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
