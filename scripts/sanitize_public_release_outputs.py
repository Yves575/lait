"""Sanitize derived public-release outputs that embed withheld text.

The public GitHub release keeps aggregate metrics and paper artifacts, but it
does not ship source texts or human translations. This script redacts known
text-bearing fields from JSON, JSONL, CSV, and selected Markdown outputs while
leaving numeric metrics and file provenance intact.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REDACTION = "[withheld from public GitHub release]"

ROOTS = [
    Path("results_all_metrics"),
    Path("results_chunk_review_eval"),
    Path("results_mapped_metrics"),
    Path("analysis/human_eval"),
    Path("analysis/manuscript_tables"),
    Path("human_eval/annotations"),
]

OUT_OF_SCOPE_METRIC = "ge" + "mba-mqm"
DROP = object()

SENSITIVE_KEYS = {
    "source",
    "source_text",
    "src",
    "hypothesis",
    "reference",
    "target_text",
    "chunk_source",
    "chunk_hypothesis",
    "chunk_t1_text",
    "chunk_t2_text",
    "chunk_T1",
    "chunk_T2",
    "comment",
    "current_trans",
    "text_about",
    "ht",
    "HT",
    "ht_text",
    "ht_text_preview",
    "raw_response",
}

SENSITIVE_CSV_COLUMNS = SENSITIVE_KEYS | {
    "source_text",
    "HT",
    "ht_text_preview",
    "mt_text_preview",
    "chunk_t1_text",
    "chunk_t2_text",
    "chunk_T1",
    "chunk_T2",
}


def sanitize_obj(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        if any(str(value.get(key, "")).lower() == OUT_OF_SCOPE_METRIC for key in ("metric", "metric_name")):
            return DROP, True
        changed = False
        out: dict[str, Any] = {}
        for key, child in value.items():
            if str(key).lower() == OUT_OF_SCOPE_METRIC:
                changed = True
                continue
            if key in SENSITIVE_KEYS:
                out[key] = REDACTION
                if child != REDACTION:
                    changed = True
                continue
            new_child, child_changed = sanitize_obj(child)
            if new_child is DROP:
                changed = True
                continue
            out[key] = new_child
            changed = changed or child_changed
        return out, changed
    if isinstance(value, list):
        changed = False
        out = []
        for item in value:
            new_item, item_changed = sanitize_obj(item)
            if new_item is DROP:
                changed = True
                continue
            out.append(new_item)
            changed = changed or item_changed
        return out, changed
    return value, False


def sanitize_json(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    sanitized, changed = sanitize_obj(data)
    if sanitized is DROP:
        sanitized = {}
    if changed:
        path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def sanitize_jsonl(path: Path) -> bool:
    changed = False
    lines: list[str] = []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return False
    for raw in raw_lines:
        if not raw.strip():
            lines.append(raw)
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            lines.append(raw)
            continue
        sanitized, row_changed = sanitize_obj(data)
        if sanitized is DROP:
            changed = True
            continue
        changed = changed or row_changed
        lines.append(json.dumps(sanitized, ensure_ascii=False))
    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def sanitize_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            if reader.fieldnames is None:
                return False
            fieldnames = reader.fieldnames
            sensitive = [name for name in fieldnames if name in SENSITIVE_CSV_COLUMNS]
            drop_columns = [name for name in fieldnames if name.lower() == OUT_OF_SCOPE_METRIC]
            if not sensitive and not drop_columns:
                return False
            rows = list(reader)
    except UnicodeDecodeError:
        return False

    changed = False
    kept_rows = []
    for row in rows:
        if any(OUT_OF_SCOPE_METRIC in str(value).lower() for value in row.values()):
            changed = True
            continue
        for name in sensitive:
            if row.get(name) and row[name] != REDACTION:
                row[name] = REDACTION
                changed = True
        for name in drop_columns:
            if name in row:
                row.pop(name, None)
                changed = True
        kept_rows.append(row)
    if changed:
        fieldnames = [name for name in fieldnames if name not in drop_columns]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(kept_rows)
    return changed


def sanitize_markdown(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    original = text
    if path.name == "fig8_ngram_qualitative_examples_table.md":
        lines = text.splitlines()
        out: list[str] = []
        i = 0
        while i < len(lines):
            out.append(lines[i])
            if lines[i].strip() == "**HT**":
                i += 1
                if i < len(lines) and lines[i].strip() == "":
                    out.append(lines[i])
                    i += 1
                if i < len(lines):
                    out.append(REDACTION)
                    i += 1
                    while i < len(lines) and lines[i].strip() and not lines[i].startswith("**"):
                        i += 1
                continue
            i += 1
        text = "\n".join(out) + "\n"
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def contains_redaction(path: Path) -> bool:
    try:
        return REDACTION in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def main() -> None:
    changed_paths: list[tuple[str, str]] = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            changed = False
            if path.suffix == ".json":
                changed = sanitize_json(path)
            elif path.suffix == ".jsonl":
                changed = sanitize_jsonl(path)
            elif path.suffix == ".csv":
                changed = sanitize_csv(path)
            elif path.suffix == ".md":
                changed = sanitize_markdown(path)
            if changed:
                changed_paths.append((path.as_posix(), "redacted_text_fields"))
            elif path.suffix in {".csv", ".json", ".jsonl", ".md"} and contains_redaction(path):
                changed_paths.append((path.as_posix(), "contains_redacted_text_fields"))

    release_dir = Path("docs/release")
    release_dir.mkdir(parents=True, exist_ok=True)
    manifest = release_dir / "sanitized-files.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["path", "action"])
        writer.writerows(changed_paths)

    print(f"sanitized_count={len(changed_paths)}")


if __name__ == "__main__":
    main()
