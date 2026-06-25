#!/usr/bin/env python3
"""Flag suspicious paragraph alignments in Par3 alignment pickle files.

The alignment script stores one pickle per book with this shape:

    {
        "source_paras": [...],
        "gt_paras": [...],
        "translator_data": {
            "<pipeline>": {
                "translator_paras": [...],
                "sent_alignments": [...],
            },
            ...
        },
    }

Some locally generated files also contain empty placeholder entries in
``translator_data`` keyed by the original translation file path.  Those entries
are skipped here unless they contain a usable ``translator_paras`` list.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CSV_FIELDS = [
    "issue_type",
    "book",
    "pkl_path",
    "paragraph_id",
    "segment_index",
    "pipeline",
    "system_name",
    "reference_pipeline",
    "gt_length",
    "compared_length",
    "ratio",
    "bleu_score",
    "threshold",
    "low_bleu_repetitions",
    "total_bleu_comparisons",
    "bleu_scores",
    "text_preview",
    "reference_text_preview",
    "message",
    "low_bleu_references",
]


@dataclass
class BookAlignment:
    """Normalized view of one book's paragraph alignment data."""

    name: str
    path: Path
    gt_paras: Sequence[Any]
    pipelines: Dict[str, Sequence[Any]]


@dataclass
class ScanStats:
    books_scanned: int = 0
    paragraph_groups_scanned: int = 0
    suspicious_length_cases: int = 0
    suspicious_bleu_cases: int = 0
    warnings: int = 0


@dataclass
class PreparedText:
    """Reusable token/ngram representation for one paragraph."""

    text: str
    tokens: List[str]
    ngram_counts: Dict[int, Counter[Tuple[str, ...]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Par3 paragraph alignment pickle files for suspicious MT alignments."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to par3_dataset/ or to one split such as par3_dataset/dev/.",
    )
    parser.add_argument(
        "--length-ratio-threshold",
        type=float,
        default=3.0,
        help="Flag paragraph lengths whose larger/smaller ratio exceeds this value.",
    )
    parser.add_argument(
        "--min-length-check-chars",
        type=int,
        default=10,
        help=(
            "Skip the length-ratio check when both paragraphs are shorter than this many "
            "characters. This avoids noise from tiny headings or section numbers."
        ),
    )
    parser.add_argument(
        "--bleu-threshold",
        type=float,
        default=1.5,
        help="Flag pairwise MT BLEU scores below this value on a 0-100 scale.",
    )
    parser.add_argument(
        "--min-bleu-repetitions",
        type=int,
        default=None,
        help=(
            "Minimum number of low-BLEU references required to flag one pipeline. "
            "Defaults to a majority of available comparisons for the paragraph."
        ),
    )
    parser.add_argument(
        "--include-non-pipeline-translations",
        action="store_true",
        help="Also scan non-pipeline translator entries such as *_ht human translations.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for a detailed report file.",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv"],
        help="Report format. Currently only csv is supported.",
    )
    parser.add_argument(
        "--only-suspicious",
        action="store_true",
        help="Write only rows flagged by the current thresholds, matching the old report behavior.",
    )
    return parser.parse_args()


def find_pickle_files(dataset: Path) -> List[Path]:
    """Return likely alignment pickles below a dataset root or split directory."""
    if dataset.is_file() and dataset.suffix == ".pkl":
        return [dataset]

    if not dataset.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset}")
    if not dataset.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset}")

    # Supports both:
    #   par3_dataset/dev/book/book.pkl
    #   par3_dataset/book/book.pkl
    candidates = sorted(dataset.glob("*/*.pkl")) + sorted(dataset.glob("*/*/*.pkl"))
    seen = set()
    unique: List[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_alignment(path: Path, include_non_pipeline_translations: bool = False) -> Optional[BookAlignment]:
    """Load and normalize one alignment pickle, returning None for unsupported data."""
    with path.open("rb") as handle:
        data = pickle.load(handle)

    # Official releases may be one large dict keyed by book.  The local aligner
    # writes each book dict directly.  For this script we expect per-book files,
    # but this branch makes single-book wrapped pickles work too.
    if isinstance(data, Mapping) and "gt_paras" not in data and len(data) == 1:
        only_key, only_value = next(iter(data.items()))
        if isinstance(only_value, Mapping):
            data = only_value
            book_name = str(only_key)
        else:
            book_name = infer_book_name(path)
    else:
        book_name = infer_book_name(path)

    if not isinstance(data, Mapping):
        return None

    gt_paras = data.get("gt_paras")
    translator_data = data.get("translator_data")
    if not isinstance(gt_paras, Sequence) or isinstance(gt_paras, (str, bytes)):
        return None
    if not isinstance(translator_data, Mapping):
        return None

    pipelines: Dict[str, Sequence[Any]] = {}
    for raw_name, payload in translator_data.items():
        if not isinstance(payload, Mapping):
            continue
        paras = payload.get("translator_paras")
        if not isinstance(paras, Sequence) or isinstance(paras, (str, bytes)):
            continue
        if len(paras) == 0:
            continue
        pipeline_name = normalize_pipeline_name(raw_name)
        if not include_non_pipeline_translations and not is_mt_pipeline_name(pipeline_name):
            continue
        pipelines[pipeline_name] = paras

    return BookAlignment(name=book_name, path=path, gt_paras=gt_paras, pipelines=pipelines)


def infer_book_name(path: Path) -> str:
    return path.parent.name if path.parent.name else path.stem


def normalize_pipeline_name(name: Any) -> str:
    """Prefer stable file stems for path-like translator keys."""
    text = str(name)
    stem = Path(text).stem
    return stem or text


def is_mt_pipeline_name(name: str) -> bool:
    """Return True for generated MT pipeline outputs and False for HT entries."""
    lowered = name.lower()
    return "pipeline" in lowered and not lowered.endswith("_ht")


def clean_text(value: Any) -> str:
    """Convert valid paragraph values to text and treat missing values as empty."""
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, bytes):
        return " ".join(value.decode("utf-8", errors="replace").split())
    return ""


def text_at(paras: Sequence[Any], index: int) -> str:
    if index < 0 or index >= len(paras):
        return ""
    return clean_text(paras[index])


def preview(text: str, limit: int = 180) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1))


def sentence_bleu(candidate: str, reference: str, max_order: int = 4) -> float:
    """Compute a simple smoothed sentence BLEU score on the usual 0-100 scale."""
    return prepared_sentence_bleu(prepare_text(candidate), prepare_text(reference), max_order=max_order)


def prepare_text(text: str, max_order: int = 4) -> PreparedText:
    tokens = tokenize(text)
    return PreparedText(
        text=text,
        tokens=tokens,
        ngram_counts={order: ngrams(tokens, order) for order in range(1, min(max_order, len(tokens)) + 1)},
    )


def prepared_sentence_bleu(candidate: PreparedText, reference: PreparedText, max_order: int = 4) -> float:
    """Compute BLEU using pre-tokenized text to avoid repeated paragraph work."""
    cand_tokens = candidate.tokens
    ref_tokens = reference.tokens
    if not cand_tokens or not ref_tokens:
        return 0.0

    effective_order = min(max_order, len(cand_tokens))
    log_precisions = []
    for order in range(1, effective_order + 1):
        cand_ngrams = candidate.ngram_counts.get(order, Counter())
        ref_ngrams = reference.ngram_counts.get(order, Counter())
        total = sum(cand_ngrams.values())
        if total == 0:
            continue
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        # Small epsilon smoothing avoids making every short paragraph BLEU zero
        # only because one higher-order n-gram has no overlap.
        precision = overlap / total if overlap else 0.1 / total
        log_precisions.append(math.log(precision))

    if not log_precisions:
        return 0.0

    brevity_penalty = 1.0
    if len(cand_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1.0 - (len(ref_tokens) / len(cand_tokens)))

    return 100.0 * brevity_penalty * math.exp(sum(log_precisions) / len(log_precisions))


def make_issue(**kwargs: Any) -> Dict[str, Any]:
    if "segment_index" not in kwargs and "paragraph_id" in kwargs:
        kwargs["segment_index"] = kwargs["paragraph_id"]
    if "system_name" not in kwargs and "pipeline" in kwargs:
        kwargs["system_name"] = kwargs["pipeline"]
    row = {field: "" for field in CSV_FIELDS}
    row.update(kwargs)
    return row


def merge_related_issues(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Merge length and BLEU rows that point to the same paragraph pipeline."""
    merged: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    passthrough: List[Dict[str, Any]] = []

    for row in rows:
        issue_type = str(row.get("issue_type", ""))
        if issue_type not in {"length", "bleu"}:
            passthrough.append(make_issue(**dict(row)))
            continue

        key = (
            str(row.get("book", "")),
            str(row.get("pkl_path", "")),
            str(row.get("paragraph_id", "")),
            str(row.get("pipeline", "")),
        )
        if key not in merged:
            merged[key] = make_issue(**dict(row))
            continue

        existing = merged[key]
        existing_type = str(existing.get("issue_type", ""))
        if existing_type != issue_type:
            existing["issue_type"] = "both"

        for field in CSV_FIELDS:
            new_value = row.get(field, "")
            if not new_value:
                continue
            if not existing.get(field):
                existing[field] = new_value
                continue
            if field == "threshold" and str(existing[field]) != str(new_value):
                if existing_type == "length" and issue_type == "bleu":
                    existing[field] = f"length={existing[field]};bleu={new_value}"
                elif existing_type == "bleu" and issue_type == "length":
                    existing[field] = f"length={new_value};bleu={existing[field]}"
                continue
            if field == "message" and str(new_value) not in str(existing[field]):
                existing[field] = f"{existing[field]} {new_value}"

    return passthrough + list(merged.values())


def scan_book(
    book: BookAlignment,
    length_ratio_threshold: float,
    min_length_check_chars: int,
    bleu_threshold: float,
    min_bleu_repetitions: Optional[int],
    only_suspicious: bool = False,
) -> Tuple[ScanStats, List[Dict[str, Any]]]:
    stats = ScanStats(books_scanned=1)
    issues: List[Dict[str, Any]] = []

    paragraph_count = len(book.gt_paras)
    stats.paragraph_groups_scanned = paragraph_count

    if not book.pipelines:
        stats.warnings += 1
        issues.append(
            make_issue(
                issue_type="warning",
                book=book.name,
                pkl_path=str(book.path),
                message="No populated translator_data entries with translator_paras were found.",
            )
        )
        return stats, issues

    for para_idx in range(paragraph_count):
        paragraph_id = str(para_idx)
        gt_text = text_at(book.gt_paras, para_idx)
        gt_length = len(gt_text)
        pipeline_texts: Dict[str, str] = {}
        prepared_texts: Dict[str, PreparedText] = {}

        for pipeline, paras in book.pipelines.items():
            if para_idx >= len(paras):
                stats.warnings += 1
                issues.append(
                    make_issue(
                        issue_type="warning",
                        book=book.name,
                        pkl_path=str(book.path),
                        paragraph_id=paragraph_id,
                        pipeline=pipeline,
                        message="Pipeline is missing this paragraph index; skipped comparisons.",
                    )
                )
                continue

            compared_text = text_at(paras, para_idx)
            pipeline_texts[pipeline] = compared_text
            if compared_text:
                prepared_texts[pipeline] = prepare_text(compared_text)
            compared_length = len(compared_text)

            ratio = length_ratio(gt_length, compared_length)
            length_flagged = (
                ratio is not None
                and max(gt_length, compared_length) >= min_length_check_chars
                and ratio > length_ratio_threshold
            )
            if length_flagged:
                stats.suspicious_length_cases += 1

        stats.suspicious_bleu_cases += add_bleu_issues(
            issues=issues,
            book=book,
            paragraph_id=paragraph_id,
            pipeline_texts=pipeline_texts,
            prepared_texts=prepared_texts,
            gt_text=gt_text,
            gt_length=gt_length,
            length_ratio_threshold=length_ratio_threshold,
            min_length_check_chars=min_length_check_chars,
            bleu_threshold=bleu_threshold,
            min_bleu_repetitions=min_bleu_repetitions,
            only_suspicious=only_suspicious,
        )

    return stats, issues


def add_bleu_issues(
    issues: List[Dict[str, Any]],
    book: BookAlignment,
    paragraph_id: str,
    pipeline_texts: Mapping[str, str],
    prepared_texts: Mapping[str, PreparedText],
    gt_text: str,
    gt_length: int,
    length_ratio_threshold: float,
    min_length_check_chars: int,
    bleu_threshold: float,
    min_bleu_repetitions: Optional[int],
    only_suspicious: bool = False,
) -> int:
    """Add one diagnostic row per pipeline, with threshold flags as metadata."""
    pipelines = sorted(pipeline_texts)
    prepared_pipelines = sorted(prepared_texts)
    total_comparisons = max(0, len(prepared_pipelines) - 1)

    required_repetitions = min_bleu_repetitions
    if required_repetitions is None:
        required_repetitions = max(1, math.ceil(total_comparisons / 2))
    required_repetitions = max(1, required_repetitions)

    issue_count = 0
    for candidate_pipeline in pipelines:
        low_refs: List[Tuple[str, float]] = []
        bleu_scores: List[Tuple[str, float]] = []
        if candidate_pipeline in prepared_texts:
            for reference_pipeline in prepared_pipelines:
                if candidate_pipeline == reference_pipeline:
                    continue
                bleu = prepared_sentence_bleu(
                    prepared_texts[candidate_pipeline],
                    prepared_texts[reference_pipeline],
                )
                bleu_scores.append((reference_pipeline, bleu))
                if bleu < bleu_threshold:
                    low_refs.append((reference_pipeline, bleu))

        compared_text = pipeline_texts[candidate_pipeline]
        compared_length = len(compared_text)
        ratio = length_ratio(gt_length, compared_length)
        length_flagged = (
            ratio is not None
            and max(gt_length, compared_length) >= min_length_check_chars
            and ratio > length_ratio_threshold
        )
        bleu_flagged = total_comparisons > 0 and len(low_refs) >= required_repetitions

        if length_flagged and bleu_flagged:
            issue_type = "both"
        elif length_flagged:
            issue_type = "length"
        elif bleu_flagged:
            issue_type = "bleu"
        else:
            issue_type = ""

        if only_suspicious and not issue_type:
            continue

        if bleu_flagged:
            issue_count += 1

        low_refs.sort(key=lambda item: item[1])
        bleu_scores.sort(key=lambda item: item[1])
        lowest_reference = low_refs[0][0] if low_refs else (bleu_scores[0][0] if bleu_scores else "")
        lowest_bleu = low_refs[0][1] if low_refs else (bleu_scores[0][1] if bleu_scores else None)
        messages: List[str] = []
        if length_flagged:
            messages.append("Paragraph length ratio exceeds threshold.")
        if bleu_flagged:
            messages.append(
                "Pipeline has low BLEU against "
                f"{len(low_refs)}/{total_comparisons} other pipeline outputs."
            )

        issues.append(
            make_issue(
                issue_type=issue_type,
                book=book.name,
                pkl_path=str(book.path),
                paragraph_id=paragraph_id,
                pipeline=candidate_pipeline,
                reference_pipeline=lowest_reference or ("gt" if length_flagged else ""),
                gt_length=gt_length,
                compared_length=compared_length,
                ratio="" if ratio is None else f"{ratio:.3f}",
                bleu_score="" if lowest_bleu is None else f"{lowest_bleu:.3f}",
                threshold=f"length={length_ratio_threshold};bleu={bleu_threshold}",
                low_bleu_repetitions=len(low_refs),
                total_bleu_comparisons=total_comparisons,
                bleu_scores=";".join(f"{ref}:{score:.3f}" for ref, score in bleu_scores),
                text_preview=preview(compared_text),
                reference_text_preview=preview(
                    pipeline_texts[lowest_reference] if lowest_reference in pipeline_texts else gt_text
                ),
                message=" ".join(messages),
                low_bleu_references=";".join(ref for ref, _ in low_refs),
            )
        )

    return issue_count


def length_ratio(gt_length: int, compared_length: int, min_length_check_chars: int = 0) -> Optional[float]:
    """Return the larger/smaller length ratio, or None when both are empty."""
    larger = max(gt_length, compared_length)
    smaller = min(gt_length, compared_length)
    if larger < min_length_check_chars:
        return None
    if larger == 0:
        return None
    if smaller == 0:
        return math.inf
    return larger / smaller


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def format_count_with_percent(count: int, denominator: int) -> str:
    if denominator <= 0:
        return f"{count} (n/a)"
    return f"{count} ({(count / denominator) * 100:.2f}%)"


def issue_paragraph_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("book", "")),
        str(row.get("pkl_path", "")),
        str(row.get("paragraph_id", "")),
    )


def unique_paragraph_count(rows: Iterable[Mapping[str, Any]], issue_types: set) -> int:
    return len(
        {
            issue_paragraph_key(row)
            for row in rows
            if row.get("issue_type") in issue_types
        }
    )


def format_cases_with_unique_percent(cases: int, unique_paragraphs: int, denominator: int) -> str:
    if denominator <= 0:
        return f"{cases} cases; {unique_paragraphs} unique paragraphs (n/a)"
    percent = (unique_paragraphs / denominator) * 100
    return f"{cases} cases; {unique_paragraphs} unique paragraphs ({percent:.2f}%)"


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset)
    if args.length_ratio_threshold <= 0:
        print("--length-ratio-threshold must be greater than 0", file=sys.stderr)
        return 2
    if args.min_length_check_chars < 0:
        print("--min-length-check-chars must be non-negative", file=sys.stderr)
        return 2
    if args.bleu_threshold < 0:
        print("--bleu-threshold must be non-negative", file=sys.stderr)
        return 2
    if args.min_bleu_repetitions is not None and args.min_bleu_repetitions <= 0:
        print("--min-bleu-repetitions must be greater than 0", file=sys.stderr)
        return 2

    try:
        pickle_files = find_pickle_files(dataset)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    all_issues: List[Dict[str, Any]] = []
    totals = ScanStats()

    for path in pickle_files:
        try:
            book = load_alignment(
                path,
                include_non_pipeline_translations=args.include_non_pipeline_translations,
            )
        except Exception as exc:  # keep one bad pickle from aborting the scan
            totals.warnings += 1
            all_issues.append(
                make_issue(
                    issue_type="warning",
                    book=infer_book_name(path),
                    pkl_path=str(path),
                    message=f"Could not load pickle: {exc}",
                )
            )
            continue

        if book is None:
            totals.warnings += 1
            all_issues.append(
                make_issue(
                    issue_type="warning",
                    book=infer_book_name(path),
                    pkl_path=str(path),
                    message="Pickle does not match the expected alignment schema.",
                )
            )
            continue

        stats, issues = scan_book(
            book,
            length_ratio_threshold=args.length_ratio_threshold,
            min_length_check_chars=args.min_length_check_chars,
            bleu_threshold=args.bleu_threshold,
            min_bleu_repetitions=args.min_bleu_repetitions,
            only_suspicious=args.only_suspicious,
        )
        totals.books_scanned += stats.books_scanned
        totals.paragraph_groups_scanned += stats.paragraph_groups_scanned
        totals.suspicious_length_cases += stats.suspicious_length_cases
        totals.suspicious_bleu_cases += stats.suspicious_bleu_cases
        totals.warnings += stats.warnings
        all_issues.extend(issues)

    report_rows = merge_related_issues(all_issues)
    merged_issue_counts = Counter(
        row.get("issue_type") for row in report_rows if row.get("issue_type") in {"length", "bleu", "both"}
    )
    suspicious_total = sum(
        1 for row in report_rows if row.get("issue_type") in {"length", "bleu", "both"}
    )
    unique_length_paragraphs = unique_paragraph_count(report_rows, {"length", "both"})
    unique_bleu_paragraphs = unique_paragraph_count(report_rows, {"bleu", "both"})
    unique_both_paragraphs = unique_paragraph_count(report_rows, {"both"})
    unique_suspicious_paragraphs = unique_paragraph_count(report_rows, {"length", "bleu", "both"})

    print(f"Books scanned: {totals.books_scanned}")
    print(f"Paragraph groups scanned: {totals.paragraph_groups_scanned}")
    print(
        "Suspicious length cases: "
        f"{format_cases_with_unique_percent(totals.suspicious_length_cases, unique_length_paragraphs, totals.paragraph_groups_scanned)}"
    )
    print(
        "Suspicious BLEU cases: "
        f"{format_cases_with_unique_percent(totals.suspicious_bleu_cases, unique_bleu_paragraphs, totals.paragraph_groups_scanned)}"
    )
    print(
        "Rows flagged by both methods: "
        f"{format_cases_with_unique_percent(merged_issue_counts['both'], unique_both_paragraphs, totals.paragraph_groups_scanned)}"
    )
    print(
        "Total suspicious cases: "
        f"{format_cases_with_unique_percent(suspicious_total, unique_suspicious_paragraphs, totals.paragraph_groups_scanned)}"
    )
    if totals.warnings:
        print(f"Warnings: {totals.warnings}")

    if args.output:
        write_csv(Path(args.output), report_rows)
        print(f"Detailed report written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
