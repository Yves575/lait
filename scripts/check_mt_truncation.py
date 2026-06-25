"""Check machine-translated book files for likely truncation.

The script compares every TXT file under books/MT/<pipeline>/<model>/ against the
best available complete reference:

1. Human translation with the same book/source/target tuple.
2. Source book from books/eval or books/dev with the same book/source language.

The heuristics are intentionally conservative. A single weak signal usually marks
a file as SUSPICIOUS, while LIKELY_TRUNCATED requires a strong size deficit or a
combination of size and ending-quality signals.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_BOOKS_DIR = Path("books")
STATUSES = ("OK", "SUSPICIOUS", "LIKELY_TRUNCATED")

# Reference-size thresholds. HT comparisons are expected to be closer than source
# comparisons because source and target languages naturally differ in length.
HT_LIKELY_CHAR_RATIO = 0.65
HT_SUSPICIOUS_CHAR_RATIO = 0.78
HT_LIKELY_LINE_RATIO = 0.60
HT_SUSPICIOUS_LINE_RATIO = 0.75
HT_LIKELY_PARAGRAPH_RATIO = 0.60
HT_SUSPICIOUS_PARAGRAPH_RATIO = 0.75

SOURCE_LIKELY_CHAR_RATIO = 0.45
SOURCE_SUSPICIOUS_CHAR_RATIO = 0.58
SOURCE_LIKELY_LINE_RATIO = 0.45
SOURCE_SUSPICIOUS_LINE_RATIO = 0.58
SOURCE_LIKELY_PARAGRAPH_RATIO = 0.45
SOURCE_SUSPICIOUS_PARAGRAPH_RATIO = 0.58

VERY_SHORT_CHAR_COUNT = 1_000
TAIL_SAMPLE_CHARS = 500
ABRUPT_TAIL_WORD_LIMIT = 12
MIN_REFERENCE_CHARS_FOR_RATIO = 1_000

SENTENCE_END_RE = re.compile(r'[.!?。！？]["\')\]\}»”’]*\s*$')
WEAK_END_RE = re.compile(r'[,;:،؛、，：；\-–—/\\]\s*$')
LANGUAGE_CODE_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class BookKey:
    """A parsed book identity from <name>_<source-language>_<target-language>.txt."""

    name: str
    source_lang: str
    target_lang: str


@dataclass(frozen=True)
class BookFile:
    """A parsed book file and its location."""

    key: BookKey
    path: Path


@dataclass(frozen=True)
class TextMetrics:
    """Simple text measurements used by the truncation heuristics."""

    chars: int
    nonspace_chars: int
    lines: int
    nonempty_lines: int
    paragraphs: int
    words: int
    ends_with_newline: bool
    sentence_end: bool
    weak_end: bool
    tail_words: int
    trailing_fragment: str


@dataclass(frozen=True)
class ReferenceMatch:
    """The selected reference file for an MT file."""

    path: Path | None
    kind: str
    metrics: TextMetrics | None


@dataclass(frozen=True)
class MTResult:
    """Report row for a single MT file."""

    pipeline: str
    model: str
    filename: str
    mt_path: Path
    reference_path: Path | None
    reference_kind: str
    metrics: dict[str, str]
    status: str
    explanation: str


def parse_book_filename(path: Path) -> BookKey | None:
    """Parse <name>_<source-language>_<target-language>.txt.

    The rule is deterministic: the last two underscore-separated tokens are
    treated as language codes only if both look like language identifiers.
    """
    if path.suffix.lower() != ".txt":
        return None

    stem = path.stem
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None

    name, source_lang, target_lang = parts
    if not name or not LANGUAGE_CODE_RE.match(source_lang) or not LANGUAGE_CODE_RE.match(target_lang):
        return None

    return BookKey(name=name, source_lang=source_lang.lower(), target_lang=target_lang.lower())


def iter_book_files(root: Path) -> Iterable[BookFile]:
    """Yield parsed TXT book files below root in deterministic path order."""
    if not root.exists():
        return

    for path in sorted(root.rglob("*.txt")):
        key = parse_book_filename(path)
        if key is not None:
            yield BookFile(key=key, path=path)


def read_text(path: Path) -> str:
    """Read text using UTF-8, tolerating bad bytes so one file cannot stop a scan."""
    return path.read_text(encoding="utf-8", errors="replace")


def count_paragraphs(text: str) -> int:
    """Count non-empty paragraph blocks separated by one or more blank lines."""
    return len([block for block in re.split(r"\n\s*\n+", text.strip()) if block.strip()])


def trailing_fragment(text: str) -> str:
    """Return the final non-empty line, trimmed to a compact diagnostic fragment."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return re.sub(r"\s+", " ", stripped)[-120:]
    return ""


def compute_metrics(path: Path) -> TextMetrics:
    """Compute length and ending-quality signals for a text file."""
    text = read_text(path)
    stripped = text.rstrip()
    tail = stripped[-TAIL_SAMPLE_CHARS:]
    tail_words = len(re.findall(r"\S+", tail))

    return TextMetrics(
        chars=len(text),
        nonspace_chars=len(re.sub(r"\s+", "", text)),
        lines=len(text.splitlines()),
        nonempty_lines=sum(1 for line in text.splitlines() if line.strip()),
        paragraphs=count_paragraphs(text),
        words=len(re.findall(r"\S+", text)),
        ends_with_newline=text.endswith("\n"),
        sentence_end=bool(SENTENCE_END_RE.search(stripped)),
        weak_end=bool(WEAK_END_RE.search(stripped)),
        tail_words=tail_words,
        trailing_fragment=trailing_fragment(text),
    )


def ratio(numerator: int, denominator: int, min_denominator: int = 1) -> float | None:
    """Return numerator / denominator, or None if the denominator is too small."""
    if denominator < min_denominator:
        return None
    return numerator / denominator


def format_ratio(value: float | None) -> str:
    """Format a ratio for the report."""
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def build_ht_index(ht_root: Path) -> dict[BookKey, BookFile]:
    """Index HT files by the full book/source/target key.

    If duplicates exist, the lexicographically first path wins. This keeps the
    choice stable across runs and avoids depending on filesystem traversal order.
    """
    index: dict[BookKey, BookFile] = {}
    for book_file in iter_book_files(ht_root):
        index.setdefault(book_file.key, book_file)
    return index


def build_source_index(source_roots: Iterable[Path]) -> dict[tuple[str, str], BookFile]:
    """Index source files by (book name, source language).

    Source references usually have source_lang == target_lang. If both eval and
    dev contain a match, eval is preferred because roots are passed in that order;
    ties within a root are resolved lexicographically by path.
    """
    index: dict[tuple[str, str], BookFile] = {}
    for root in source_roots:
        for book_file in iter_book_files(root):
            source_key = (book_file.key.name, book_file.key.source_lang)
            index.setdefault(source_key, book_file)
    return index


def choose_reference(
    mt_key: BookKey,
    ht_index: dict[BookKey, BookFile],
    source_index: dict[tuple[str, str], BookFile],
) -> ReferenceMatch:
    """Choose the best deterministic reference for an MT book."""
    ht_match = ht_index.get(mt_key)
    if ht_match is not None:
        return ReferenceMatch(ht_match.path, "HT", compute_metrics(ht_match.path))

    source_match = source_index.get((mt_key.name, mt_key.source_lang))
    if source_match is not None:
        return ReferenceMatch(source_match.path, "SOURCE", compute_metrics(source_match.path))

    return ReferenceMatch(None, "NONE", None)


def threshold_set(reference_kind: str) -> dict[str, float]:
    """Return ratio thresholds appropriate for the selected reference kind."""
    if reference_kind == "HT":
        return {
            "likely_chars": HT_LIKELY_CHAR_RATIO,
            "suspicious_chars": HT_SUSPICIOUS_CHAR_RATIO,
            "likely_lines": HT_LIKELY_LINE_RATIO,
            "suspicious_lines": HT_SUSPICIOUS_LINE_RATIO,
            "likely_paragraphs": HT_LIKELY_PARAGRAPH_RATIO,
            "suspicious_paragraphs": HT_SUSPICIOUS_PARAGRAPH_RATIO,
        }

    return {
        "likely_chars": SOURCE_LIKELY_CHAR_RATIO,
        "suspicious_chars": SOURCE_SUSPICIOUS_CHAR_RATIO,
        "likely_lines": SOURCE_LIKELY_LINE_RATIO,
        "suspicious_lines": SOURCE_SUSPICIOUS_LINE_RATIO,
        "likely_paragraphs": SOURCE_LIKELY_PARAGRAPH_RATIO,
        "suspicious_paragraphs": SOURCE_SUSPICIOUS_PARAGRAPH_RATIO,
    }


def ending_signals(metrics: TextMetrics) -> list[str]:
    """Return human-readable ending issues found in an MT file."""
    signals: list[str] = []
    if metrics.chars == 0:
        signals.append("empty file")
    elif not metrics.sentence_end:
        signals.append("does not end with sentence punctuation")
    if metrics.weak_end:
        signals.append("ends with weak/incomplete punctuation")
    if 0 < metrics.tail_words <= ABRUPT_TAIL_WORD_LIMIT and not metrics.sentence_end:
        signals.append("very short final fragment")
    return signals


def classify(
    mt_metrics: TextMetrics,
    ref_match: ReferenceMatch,
) -> tuple[str, dict[str, str], str]:
    """Classify one MT file as OK, SUSPICIOUS, or LIKELY_TRUNCATED."""
    if ref_match.metrics is None:
        end_issues = ending_signals(mt_metrics)
        metrics = {
            "mt_chars": str(mt_metrics.chars),
            "mt_lines": str(mt_metrics.lines),
            "mt_paragraphs": str(mt_metrics.paragraphs),
            "char_ratio": "n/a",
            "line_ratio": "n/a",
            "paragraph_ratio": "n/a",
            "ends_with_newline": str(mt_metrics.ends_with_newline),
            "ending_signals": "; ".join(end_issues) or "none",
        }
        if mt_metrics.chars == 0:
            return "LIKELY_TRUNCATED", metrics, "no reference found and MT file is empty"
        if mt_metrics.chars < VERY_SHORT_CHAR_COUNT or len(end_issues) >= 2:
            return "SUSPICIOUS", metrics, "no reference found; weak standalone completeness signals"
        return "OK", metrics, "no reference found; standalone ending signals look acceptable"

    ref_metrics = ref_match.metrics
    thresholds = threshold_set(ref_match.kind)
    char_ratio = ratio(mt_metrics.chars, ref_metrics.chars, MIN_REFERENCE_CHARS_FOR_RATIO)
    line_ratio = ratio(mt_metrics.nonempty_lines, ref_metrics.nonempty_lines)
    paragraph_ratio = ratio(mt_metrics.paragraphs, ref_metrics.paragraphs)
    end_issues = ending_signals(mt_metrics)

    metrics = {
        "mt_chars": str(mt_metrics.chars),
        "ref_chars": str(ref_metrics.chars),
        "char_ratio": format_ratio(char_ratio),
        "mt_nonempty_lines": str(mt_metrics.nonempty_lines),
        "ref_nonempty_lines": str(ref_metrics.nonempty_lines),
        "line_ratio": format_ratio(line_ratio),
        "mt_paragraphs": str(mt_metrics.paragraphs),
        "ref_paragraphs": str(ref_metrics.paragraphs),
        "paragraph_ratio": format_ratio(paragraph_ratio),
        "ends_with_newline": str(mt_metrics.ends_with_newline),
        "ending_signals": "; ".join(end_issues) or "none",
    }

    likely_reasons: list[str] = []
    suspicious_reasons: list[str] = []

    if mt_metrics.chars == 0:
        likely_reasons.append("MT file is empty")
    if char_ratio is not None and char_ratio < thresholds["likely_chars"]:
        likely_reasons.append(f"character ratio {char_ratio:.3f} is very low")
    elif char_ratio is not None and char_ratio < thresholds["suspicious_chars"]:
        suspicious_reasons.append(f"character ratio {char_ratio:.3f} is low")

    if line_ratio is not None and line_ratio < thresholds["likely_lines"]:
        likely_reasons.append(f"line ratio {line_ratio:.3f} is very low")
    elif line_ratio is not None and line_ratio < thresholds["suspicious_lines"]:
        suspicious_reasons.append(f"line ratio {line_ratio:.3f} is low")

    if paragraph_ratio is not None and paragraph_ratio < thresholds["likely_paragraphs"]:
        likely_reasons.append(f"paragraph ratio {paragraph_ratio:.3f} is very low")
    elif paragraph_ratio is not None and paragraph_ratio < thresholds["suspicious_paragraphs"]:
        suspicious_reasons.append(f"paragraph ratio {paragraph_ratio:.3f} is low")

    if end_issues:
        suspicious_reasons.extend(end_issues)

    low_size_signal_count = len(likely_reasons) + sum(
        1 for reason in suspicious_reasons if "ratio" in reason
    )
    if likely_reasons and (end_issues or low_size_signal_count >= 2):
        return "LIKELY_TRUNCATED", metrics, "; ".join(likely_reasons + end_issues)
    if len(likely_reasons) >= 2:
        return "LIKELY_TRUNCATED", metrics, "; ".join(likely_reasons)
    if suspicious_reasons or likely_reasons:
        return "SUSPICIOUS", metrics, "; ".join(likely_reasons + suspicious_reasons)

    return "OK", metrics, "size ratios and ending signals look complete"


def iter_mt_files(mt_root: Path) -> Iterable[tuple[str, str, BookFile]]:
    """Yield MT files with their pipeline and model names."""
    if not mt_root.exists():
        return

    for path in sorted(mt_root.glob("*/*/*.txt")):
        key = parse_book_filename(path)
        if key is None:
            continue
        try:
            relative = path.relative_to(mt_root)
        except ValueError:
            continue
        if len(relative.parts) != 3:
            continue
        pipeline, model, _filename = relative.parts
        yield pipeline, model, BookFile(key=key, path=path)


def analyze_mt_file(
    pipeline: str,
    model: str,
    mt_file: BookFile,
    ht_index: dict[BookKey, BookFile],
    source_index: dict[tuple[str, str], BookFile],
) -> MTResult:
    """Analyze one machine-translated book file."""
    mt_metrics = compute_metrics(mt_file.path)
    ref_match = choose_reference(mt_file.key, ht_index, source_index)
    status, metrics, explanation = classify(mt_metrics, ref_match)

    return MTResult(
        pipeline=pipeline,
        model=model,
        filename=mt_file.path.name,
        mt_path=mt_file.path,
        reference_path=ref_match.path,
        reference_kind=ref_match.kind,
        metrics=metrics,
        status=status,
        explanation=explanation,
    )


def analyze_books(books_dir: Path) -> list[MTResult]:
    """Analyze every MT book below books_dir."""
    ht_index = build_ht_index(books_dir / "HT")
    source_index = build_source_index([books_dir / "eval", books_dir / "dev"])

    results: list[MTResult] = []
    for pipeline, model, mt_file in iter_mt_files(books_dir / "MT"):
        results.append(analyze_mt_file(pipeline, model, mt_file, ht_index, source_index))
    return results


def filter_results(results: list[MTResult], statuses: set[str]) -> list[MTResult]:
    """Return report rows whose status is enabled by the CLI filter."""
    return [result for result in results if result.status in statuses]


def print_text_report(results: list[MTResult], statuses: set[str]) -> None:
    """Print selected files and summary counts to stdout."""
    selected_results = filter_results(results, statuses)

    for result in selected_results:
        reference = str(result.reference_path) if result.reference_path else "NO_MATCH"
        metric_parts = [f"{key}={value}" for key, value in result.metrics.items()]
        print(f"Pipeline: {result.pipeline}")
        print(f"Model: {result.model}")
        print(f"File: {result.filename}")
        print(f"MT path: {result.mt_path}")
        print(f"Reference ({result.reference_kind}): {reference}")
        print(f"Metrics: {', '.join(metric_parts)}")
        print(f"Status: {result.status}")
        print(f"Explanation: {result.explanation}")
        print()

    if not selected_results:
        print(f"No files found for selected status filter: {', '.join(sorted(statuses))}.")
        print()

    print_summary(results)


def print_csv_report(results: list[MTResult], statuses: set[str]) -> None:
    """Print selected CSV rows and a final comment-line summary for all files."""
    selected_results = filter_results(results, statuses)
    metric_keys = sorted({key for result in selected_results for key in result.metrics})
    fieldnames = [
        "pipeline",
        "model",
        "filename",
        "mt_path",
        "reference_kind",
        "reference_path",
        *metric_keys,
        "status",
        "explanation",
    ]

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for result in selected_results:
        row = {
            "pipeline": result.pipeline,
            "model": result.model,
            "filename": result.filename,
            "mt_path": str(result.mt_path),
            "reference_kind": result.reference_kind,
            "reference_path": str(result.reference_path) if result.reference_path else "",
            "status": result.status,
            "explanation": result.explanation,
        }
        row.update(result.metrics)
        writer.writerow(row)

    counts = Counter(result.status for result in results)
    print(
        "# Summary: "
        + ", ".join(f"{status}={counts.get(status, 0)}" for status in STATUSES)
    )


def print_summary(results: list[MTResult]) -> None:
    """Print status counts."""
    counts = Counter(result.status for result in results)
    print("Summary:")
    for status in STATUSES:
        print(f"{status}: {counts.get(status, 0)}")
    print(f"TOTAL: {len(results)}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Check books/MT translations for likely truncation."
    )
    parser.add_argument(
        "--books-dir",
        type=Path,
        default=DEFAULT_BOOKS_DIR,
        help="Path to the books directory. Defaults to ./books.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "csv"),
        default="text",
        help="Report format. Defaults to text.",
    )
    parser.add_argument(
        "--status",
        choices=STATUSES,
        action="append",
        help=(
            "Status to print. Can be passed multiple times, for example "
            "--status SUSPICIOUS --status LIKELY_TRUNCATED. Defaults to all statuses."
        ),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    args = build_parser().parse_args()
    books_dir = args.books_dir

    if not books_dir.exists():
        print(f"Error: books directory does not exist: {books_dir}", file=sys.stderr)
        return 2

    results = analyze_books(books_dir)
    statuses = set(args.status or STATUSES)
    if args.format == "csv":
        print_csv_report(results, statuses)
    else:
        print_text_report(results, statuses)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
