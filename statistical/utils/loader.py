from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from statistical.config import HT_DIR, HT_SYSTEM_NAME, MT_DIR, SOURCE_DIRS


LANG_SUFFIX_RE = re.compile(r"_(?:[a-z]{2})_(?:[a-z]{2})$")


@dataclass(frozen=True)
class TextRecord:
    book: str
    system: str
    path: Path
    pipeline: str
    model: str


def book_key(path: Path) -> str:
    stem = path.stem
    stem = LANG_SUFFIX_RE.sub("", stem)
    return stem


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def discover_source_books(source_dirs: tuple[Path, ...] = SOURCE_DIRS) -> set[str]:
    books: set[str] = set()
    for source_dir in source_dirs:
        if source_dir.exists():
            books.update(book_key(path) for path in source_dir.glob("*.txt"))
    return books


def discover_ht(ht_dir: Path = HT_DIR) -> dict[str, TextRecord]:
    records: dict[str, TextRecord] = {}
    if not ht_dir.exists():
        return records
    for path in sorted(ht_dir.glob("*.txt")):
        book = book_key(path)
        records[book] = TextRecord(book, HT_SYSTEM_NAME, path, "HT", "Human")
    return records


def display_pipeline(pipeline_dir_name: str) -> str:
    match = re.search(r"(\d+)$", pipeline_dir_name)
    return f"P{match.group(1)}" if match else pipeline_dir_name


def display_system(pipeline_dir_name: str, model: str) -> str:
    return f"[{display_pipeline(pipeline_dir_name)}] {model}"


def discover_mt(mt_dir: Path = MT_DIR) -> dict[str, dict[str, TextRecord]]:
    records: dict[str, dict[str, TextRecord]] = {}
    if not mt_dir.exists():
        return records
    for pipeline_dir in sorted(path for path in mt_dir.iterdir() if path.is_dir()):
        direct_files = sorted(pipeline_dir.glob("*.txt"))
        if direct_files:
            system = display_system(pipeline_dir.name, pipeline_dir.name)
            for path in direct_files:
                book = book_key(path)
                records.setdefault(system, {})[book] = TextRecord(
                    book=book,
                    system=system,
                    path=path,
                    pipeline=display_pipeline(pipeline_dir.name),
                    model=pipeline_dir.name,
                )

        for model_dir in sorted(path for path in pipeline_dir.iterdir() if path.is_dir()):
            system = display_system(pipeline_dir.name, model_dir.name)
            for path in sorted(model_dir.glob("*.txt")):
                book = book_key(path)
                records.setdefault(system, {})[book] = TextRecord(
                    book=book,
                    system=system,
                    path=path,
                    pipeline=display_pipeline(pipeline_dir.name),
                    model=model_dir.name,
                )
    return records


def load_corpus(require_source_match: bool = True) -> tuple[set[str], dict[str, dict[str, str]], dict[str, str]]:
    source_books = discover_source_books()
    ht_records = discover_ht()
    mt_records = discover_mt()

    if require_source_match and source_books:
        books = source_books & set(ht_records)
    else:
        books = set(ht_records)
    for system_records in mt_records.values():
        books |= set(system_records) & set(ht_records)
    if require_source_match and source_books:
        books &= source_books

    systems: dict[str, dict[str, str]] = {HT_SYSTEM_NAME: {}}
    for book in sorted(books):
        record = ht_records.get(book)
        if record:
            systems[HT_SYSTEM_NAME][book] = read_text(record.path)

    for system, records in sorted(mt_records.items()):
        matched = sorted(set(records) & set(systems[HT_SYSTEM_NAME]))
        if not matched:
            continue
        systems[system] = {book: read_text(records[book].path) for book in matched}

    pipelines = {HT_SYSTEM_NAME: "HT"}
    for system, records in mt_records.items():
        if system in systems and records:
            pipelines[system] = next(iter(records.values())).pipeline

    return set(systems[HT_SYSTEM_NAME]), systems, pipelines


def common_books_for_systems(systems: dict[str, dict[str, str]], selected: list[str]) -> list[str]:
    if not selected:
        return []
    common = set(systems[selected[0]])
    for system in selected[1:]:
        common &= set(systems.get(system, {}))
    return sorted(common)


def mt_systems(systems: dict[str, dict[str, str]]) -> list[str]:
    return sorted(system for system in systems if system != HT_SYSTEM_NAME)


def pipeline_texts(
    systems: dict[str, dict[str, str]],
    pipelines: dict[str, str],
    include_ht: bool = True,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    if include_ht and HT_SYSTEM_NAME in systems:
        grouped[HT_SYSTEM_NAME] = [systems[HT_SYSTEM_NAME][book] for book in sorted(systems[HT_SYSTEM_NAME])]
    for system in mt_systems(systems):
        pipeline = pipelines.get(system, "unknown")
        grouped.setdefault(pipeline, [])
        grouped[pipeline].extend(systems[system][book] for book in sorted(systems[system]))
    return grouped
