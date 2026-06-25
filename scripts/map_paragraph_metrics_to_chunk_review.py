"""Map paragraph-level metric scores onto chunk-review segments.

This reuses paragraph-level metric outputs from ``results_all_metrics`` for the
approximately 300-word chunks in ``results_chunk_review_eval``. Matching is done
on source text because both HT and MT systems share the same source chunks.

By default, the script writes augmented copies under root-level
``results_mapped_metrics`` and does not modify the original chunk-review
metric outputs.

Example:
  .venv/bin/python scripts/map_paragraph_metrics_to_chunk_review.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_ALL_RESULTS_DIR = Path("results_all_metrics")
DEFAULT_CHUNK_RESULTS_DIR = Path("results_chunk_review_eval")
DEFAULT_OUTPUT_DIR = Path("results_mapped_metrics")
BOOK_RESULTS_DIR_NAME = "books"
ANALYSIS_DIR_NAME = "analysis"
REQUESTED_METRICS = ("comet22", "cometkiwi", "metricx-qe", "metricx")
HIGHER_IS_BETTER_DEFAULTS = {
    "comet22": True,
    "cometkiwi": True,
    "metricx-qe": False,
    "metricx": False,
}
SYSTEM_KINDS = {"ht", "mt"}


@dataclass(frozen=True)
class MetricSource:
    split: str
    book: str
    system_kind: str
    path: Path


@dataclass
class ParagraphMatch:
    paragraph_index: int
    chunk_index: int
    score: float
    method: str


@dataclass
class SystemMapping:
    book: str
    system_kind: str
    source: MetricSource
    system_dir: Path
    chunk_path: Path
    paragraph_rows: list[dict[str, Any]]
    chunk_rows: list[dict[str, Any]]
    matches: list[ParagraphMatch]
    mapped_scores: dict[int, dict[str, float]]
    mapped_score_paragraph_indices: dict[int, dict[str, list[int]]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSON value is not an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category[0] in {"L", "N"}:
            chars.append(char)
    return "".join(chars)


def partial_containment_score(needle: str, haystack: str) -> tuple[float, str]:
    if not needle:
        return 1.0, "empty"
    if needle in haystack:
        return 1.0, "normalized_substring"
    if len(needle) <= 8:
        return 0.0, "too_short_for_fuzzy"

    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    longest = matcher.find_longest_match(0, len(needle), 0, len(haystack)).size
    longest_score = longest / len(needle)

    # Ratio catches mild edits distributed across the paragraph, while the
    # longest-match score catches paragraphs embedded in larger chunks.
    ratio_score = matcher.ratio()
    return max(longest_score, ratio_score), "fuzzy"


def best_chunk_for_paragraph(
    paragraph_text: str,
    normalized_chunks: list[str],
    start_chunk: int,
    lookahead: int,
) -> tuple[int, float, str]:
    normalized_paragraph = normalize_for_match(paragraph_text)
    if not normalized_chunks:
        return -1, 0.0, "no_chunks"

    end_chunk = min(len(normalized_chunks), start_chunk + lookahead + 1)
    candidates = range(start_chunk, end_chunk)
    best_index = start_chunk
    best_score = -1.0
    best_method = ""

    for chunk_index in candidates:
        score, method = partial_containment_score(
            normalized_paragraph, normalized_chunks[chunk_index]
        )
        if score > best_score:
            best_index = chunk_index
            best_score = score
            best_method = method
        if score == 1.0:
            break
    return best_index, best_score, best_method


def match_paragraphs_to_chunks(
    paragraph_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    lookahead: int,
) -> list[ParagraphMatch]:
    normalized_chunks = [normalize_for_match(str(row.get("source", ""))) for row in chunk_rows]
    current_chunk = 0
    matches: list[ParagraphMatch] = []

    for paragraph in paragraph_rows:
        paragraph_index = int(paragraph["segment_index"])
        chunk_index, score, method = best_chunk_for_paragraph(
            str(paragraph.get("source", "")),
            normalized_chunks,
            current_chunk,
            lookahead,
        )
        if chunk_index >= 0:
            current_chunk = chunk_index
        matches.append(
            ParagraphMatch(
                paragraph_index=paragraph_index,
                chunk_index=chunk_index,
                score=score,
                method=method,
            )
        )
    return matches


def system_kind_from_summary_row(row: dict[str, str]) -> str | None:
    if row["pipeline"] == "ht":
        return "ht"
    if row["pipeline"] == "pipeline3":
        return "mt"
    return None


def collect_metric_sources(all_results_dir: Path) -> dict[tuple[str, str], MetricSource]:
    summary_path = all_results_dir / "all_results.csv"
    sources: dict[tuple[str, str], MetricSource] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "eval":
                continue
            system_kind = system_kind_from_summary_row(row)
            if system_kind is None or row["metric"] not in REQUESTED_METRICS:
                continue
            source = MetricSource(
                split=row["split"],
                book=row["book"],
                system_kind=system_kind,
                path=Path(row["path"]),
            )
            key = (source.book, source.system_kind)
            previous = sources.get(key)
            if previous is not None and previous.path != source.path:
                raise ValueError(f"Multiple paths for {key}: {previous.path} and {source.path}")
            sources[key] = source
    return sources


def split_chunk_book_dir_name(name: str) -> tuple[str, str] | None:
    match = re.match(r"^(?P<book>.+)_(?P<lang>fr|ja|pl)_en$", name)
    if not match:
        return None
    return match.group("book"), match.group("lang")


def book_results_dir(results_dir: Path) -> Path:
    return results_dir / BOOK_RESULTS_DIR_NAME


def analysis_results_dir(results_dir: Path) -> Path:
    return results_dir / ANALYSIS_DIR_NAME


def chunk_system_dirs(chunk_results_dir: Path) -> list[tuple[str, str, Path]]:
    dirs: list[tuple[str, str, Path]] = []
    source_dir = book_results_dir(chunk_results_dir)
    if not source_dir.exists():
        source_dir = chunk_results_dir
    for book_dir in sorted(source_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        parsed = split_chunk_book_dir_name(book_dir.name)
        if parsed is None:
            continue
        book, _lang = parsed
        for system_dir in sorted(book_dir.iterdir()):
            if not system_dir.is_dir():
                continue
            system_kind = system_dir.name.rsplit("_", 1)[-1]
            if system_kind in SYSTEM_KINDS and (system_dir / "segment_scores.jsonl").exists():
                dirs.append((book, system_kind, system_dir))
    return dirs


def metric_means_by_chunk(
    paragraph_rows: list[dict[str, Any]],
    matches: list[ParagraphMatch],
) -> dict[int, dict[str, float]]:
    values = metric_values_by_chunk(paragraph_rows, matches)
    return metric_means_from_values(values)


def metric_values_by_chunk(
    paragraph_rows: list[dict[str, Any]],
    matches: list[ParagraphMatch],
) -> dict[int, dict[str, dict[int, float]]]:
    paragraph_by_index = {int(row["segment_index"]): row for row in paragraph_rows}
    values: dict[int, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))

    for match in matches:
        row = paragraph_by_index[match.paragraph_index]
        scores = row.get("scores", {})
        if not isinstance(scores, dict):
            continue
        for metric in REQUESTED_METRICS:
            score = scores.get(metric)
            if score is not None:
                values[match.chunk_index][metric][match.paragraph_index] = float(score)

    return {
        chunk_index: {metric: dict(metric_values) for metric, metric_values in metrics.items()}
        for chunk_index, metrics in values.items()
    }


def metric_means_from_values(
    values: dict[int, dict[str, dict[int, float]]],
) -> dict[int, dict[str, float]]:
    return {
        chunk_index: {
            metric: mean(metric_values.values()) for metric, metric_values in metrics.items()
        }
        for chunk_index, metrics in values.items()
    }


def shared_metrics(system_values: dict[str, dict[int, dict[str, dict[int, float]]]]) -> set[str]:
    metrics_by_system: dict[str, set[str]] = {}
    for system_kind, chunk_values in system_values.items():
        metrics_by_system[system_kind] = {
            metric for chunk_metrics in chunk_values.values() for metric in chunk_metrics
        }
    if not metrics_by_system:
        return set()
    return set.intersection(*metrics_by_system.values())


def paired_metric_means_by_chunk(
    system_values: dict[str, dict[int, dict[str, dict[int, float]]]],
) -> dict[str, dict[int, dict[str, float]]]:
    """Average shared HT/MT metrics over the same paragraph IDs per chunk.

    Metrics that exist for only one system, such as reference-based scores that
    are not computed for HT, keep their original per-system paragraph average.
    Metrics present on both systems use the intersection of paragraph IDs with a
    score on both sides for each chunk.
    """

    selected_indices = paired_metric_paragraph_indices_by_chunk(system_values)
    paired: dict[str, dict[int, dict[str, float]]] = {
        system_kind: {} for system_kind in system_values
    }

    for system_kind, chunk_values in system_values.items():
        for chunk_index, chunk_metrics in chunk_values.items():
            for metric, paragraph_values in chunk_metrics.items():
                selected_values = [
                    paragraph_values[paragraph_id]
                    for paragraph_id in selected_indices.get(system_kind, {})
                    .get(chunk_index, {})
                    .get(metric, [])
                ]
                if selected_values:
                    paired.setdefault(system_kind, {}).setdefault(chunk_index, {})[metric] = mean(
                        selected_values
                    )

    return paired


def paired_metric_paragraph_indices_by_chunk(
    system_values: dict[str, dict[int, dict[str, dict[int, float]]]],
) -> dict[str, dict[int, dict[str, list[int]]]]:
    common_metrics = shared_metrics(system_values)
    selected: dict[str, dict[int, dict[str, list[int]]]] = {
        system_kind: {} for system_kind in system_values
    }

    for system_kind, chunk_values in system_values.items():
        for chunk_index, chunk_metrics in chunk_values.items():
            for metric, paragraph_values in chunk_metrics.items():
                if metric in common_metrics:
                    paragraph_ids = [
                        set(
                            other_values.get(chunk_index, {})
                            .get(metric, {})
                            .keys()
                        )
                        for other_values in system_values.values()
                    ]
                    selected_ids = set.intersection(*paragraph_ids) if paragraph_ids else set()
                else:
                    selected_ids = set(paragraph_values)

                if selected_ids:
                    selected.setdefault(system_kind, {}).setdefault(chunk_index, {})[metric] = (
                        sorted(selected_ids)
                    )

    return selected


def augment_chunk_rows(
    chunk_rows: list[dict[str, Any]],
    mapped_scores: dict[int, dict[str, float]],
    score_prefix: str,
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for row in chunk_rows:
        chunk_index = int(row["segment_index"])
        new_row = dict(row)
        scores = dict(new_row.get("scores", {}))
        for metric, value in mapped_scores.get(chunk_index, {}).items():
            scores[f"{score_prefix}{metric}"] = value
        new_row["scores"] = scores
        augmented.append(new_row)
    return augmented


def output_system_dir(output_dir: Path, source_system_dir: Path) -> Path:
    relative = source_system_dir.relative_to(DEFAULT_CHUNK_RESULTS_DIR)
    return output_dir / relative


def write_system_scores(
    source_system_dir: Path,
    target_system_dir: Path,
    mapped_scores: dict[int, dict[str, float]],
    score_prefix: str,
    all_results_dir: Path,
) -> None:
    source_path = source_system_dir / "system_scores.json"
    if not source_path.exists():
        return
    data = json.loads(source_path.read_text(encoding="utf-8"))
    metrics = dict(data.get("metrics", {}))
    for metric in REQUESTED_METRICS:
        values = [
            chunk_metrics[metric]
            for chunk_metrics in mapped_scores.values()
            if metric in chunk_metrics
        ]
        if not values:
            continue
        mapped_name = f"{score_prefix}{metric}"
        metrics[mapped_name] = {
            "score": mean(values),
            "num_scored_segments": len(values),
            "higher_is_better": HIGHER_IS_BETTER_DEFAULTS[metric],
            "source": "paragraph_mean_from_results_all_metrics",
        }
    data["metrics"] = metrics
    data["paragraph_metric_mapping"] = {
        "source_results_dir": str(all_results_dir),
        "score_prefix": score_prefix,
    }
    target_path = target_system_dir / "system_scores.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "book",
        "system_kind",
        "paragraph_index",
        "chunk_index",
        "match_score",
        "match_method",
        "metrics_available",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "book",
        "system_kind",
        "source_path",
        "chunk_path",
        "num_paragraphs",
        "num_chunks",
        "num_chunks_with_scores",
        "min_match_score",
        "num_low_confidence_matches",
        "metrics_mapped",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metrics_available(scores: Any) -> list[str]:
    if not isinstance(scores, dict):
        return []
    return [metric for metric in REQUESTED_METRICS if metric in scores]


def build_chunk_alignment_rows(
    book: str,
    system_kind: str,
    source_path: Path,
    chunk_path: Path,
    paragraph_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    matches: list[ParagraphMatch],
    mapped_scores: dict[int, dict[str, float]],
    mapped_score_paragraph_indices: dict[int, dict[str, list[int]]],
) -> list[dict[str, Any]]:
    paragraph_by_index = {int(row["segment_index"]): row for row in paragraph_rows}
    matches_by_chunk: dict[int, list[ParagraphMatch]] = defaultdict(list)
    for match in matches:
        matches_by_chunk[match.chunk_index].append(match)

    rows: list[dict[str, Any]] = []
    for chunk in chunk_rows:
        chunk_index = int(chunk["segment_index"])
        chunk_matches = matches_by_chunk.get(chunk_index, [])
        mapped_paragraphs: list[dict[str, Any]] = []
        for match in chunk_matches:
            paragraph = paragraph_by_index[match.paragraph_index]
            scores = paragraph.get("scores", {})
            mapped_paragraphs.append(
                {
                    "paragraph_index": match.paragraph_index,
                    "match_score": match.score,
                    "match_method": match.method,
                    "metrics_available": metrics_available(scores),
                    "source": str(paragraph.get("source", "")),
                    "hypothesis": str(paragraph.get("hypothesis", "")),
                    "scores": {
                        metric: scores[metric]
                        for metric in REQUESTED_METRICS
                        if isinstance(scores, dict) and metric in scores
                    },
                }
            )

        rows.append(
            {
                "book": book,
                "system_kind": system_kind,
                "chunk_index": chunk_index,
                "source_path": str(source_path),
                "chunk_path": str(chunk_path),
                "chunk_source": str(chunk.get("source", "")),
                "chunk_hypothesis": str(chunk.get("hypothesis", "")),
                "paragraph_count": len(mapped_paragraphs),
                "paragraph_indices": [p["paragraph_index"] for p in mapped_paragraphs],
                "match_scores": [p["match_score"] for p in mapped_paragraphs],
                "match_methods": [p["match_method"] for p in mapped_paragraphs],
                "mapped_scores": mapped_scores.get(chunk_index, {}),
                "mapped_score_paragraph_indices": mapped_score_paragraph_indices.get(
                    chunk_index, {}
                ),
                "paragraphs": mapped_paragraphs,
            }
        )
    return rows


def write_chunk_alignment_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path.with_suffix(".jsonl"), rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "book",
        "system_kind",
        "chunk_index",
        "source_path",
        "chunk_path",
        "paragraph_count",
        "paragraph_indices",
        "match_scores",
        "match_methods",
        "mapped_scores",
        "mapped_score_paragraph_indices",
        "chunk_source",
        "paragraph_sources",
        "chunk_hypothesis",
        "paragraph_hypotheses",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            paragraphs = row["paragraphs"]
            writer.writerow(
                {
                    "book": row["book"],
                    "system_kind": row["system_kind"],
                    "chunk_index": row["chunk_index"],
                    "source_path": row["source_path"],
                    "chunk_path": row["chunk_path"],
                    "paragraph_count": row["paragraph_count"],
                    "paragraph_indices": "|".join(
                        str(index) for index in row["paragraph_indices"]
                    ),
                    "match_scores": "|".join(
                        f"{score:.6f}" for score in row["match_scores"]
                    ),
                    "match_methods": "|".join(row["match_methods"]),
                    "mapped_scores": json.dumps(
                        row["mapped_scores"], ensure_ascii=False, sort_keys=True
                    ),
                    "mapped_score_paragraph_indices": json.dumps(
                        row["mapped_score_paragraph_indices"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "chunk_source": row["chunk_source"],
                    "paragraph_sources": "\n\n--- PARAGRAPH ---\n\n".join(
                        paragraph["source"] for paragraph in paragraphs
                    ),
                    "chunk_hypothesis": row["chunk_hypothesis"],
                    "paragraph_hypotheses": "\n\n--- PARAGRAPH ---\n\n".join(
                        paragraph["hypothesis"] for paragraph in paragraphs
                    ),
                }
            )


def map_metrics(args: argparse.Namespace) -> None:
    metric_sources = collect_metric_sources(args.all_results_dir)
    mappings_by_book: dict[str, list[SystemMapping]] = defaultdict(list)
    audit_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    chunk_alignment_rows: list[dict[str, Any]] = []

    for book, system_kind, system_dir in chunk_system_dirs(args.chunk_results_dir):
        source = metric_sources.get((book, system_kind))
        if source is None:
            continue

        paragraph_rows = read_jsonl(source.path)
        chunk_path = system_dir / "segment_scores.jsonl"
        chunk_rows = read_jsonl(chunk_path)
        matches = match_paragraphs_to_chunks(paragraph_rows, chunk_rows, args.lookahead)
        mappings_by_book[book].append(
            SystemMapping(
                book=book,
                system_kind=system_kind,
                source=source,
                system_dir=system_dir,
                chunk_path=chunk_path,
                paragraph_rows=paragraph_rows,
                chunk_rows=chunk_rows,
                matches=matches,
                mapped_scores=metric_means_by_chunk(paragraph_rows, matches),
                mapped_score_paragraph_indices={},
            )
        )

    for book_mappings in mappings_by_book.values():
        values_by_system = {
            mapping.system_kind: metric_values_by_chunk(mapping.paragraph_rows, mapping.matches)
            for mapping in book_mappings
        }
        paired_scores = paired_metric_means_by_chunk(values_by_system)
        paired_indices = paired_metric_paragraph_indices_by_chunk(values_by_system)
        for mapping in book_mappings:
            mapping.mapped_scores = paired_scores.get(mapping.system_kind, {})
            mapping.mapped_score_paragraph_indices = paired_indices.get(mapping.system_kind, {})

    for book_mappings in mappings_by_book.values():
        for mapping in book_mappings:
            target_dir = args.output_dir / mapping.system_dir.relative_to(args.chunk_results_dir)
            augmented_rows = augment_chunk_rows(
                mapping.chunk_rows, mapping.mapped_scores, args.score_prefix
            )
            write_jsonl(target_dir / "segment_scores.jsonl", augmented_rows)
            write_system_scores(
                mapping.system_dir,
                target_dir,
                mapping.mapped_scores,
                args.score_prefix,
                args.all_results_dir,
            )

            for extra_name in ("skipped_segments.jsonl",):
                extra_source = mapping.system_dir / extra_name
                if extra_source.exists():
                    (target_dir / extra_name).write_text(
                        extra_source.read_text(encoding="utf-8"), encoding="utf-8"
                    )

            paragraph_by_index = {
                int(row["segment_index"]): row for row in mapping.paragraph_rows
            }
            for match in mapping.matches:
                scores = paragraph_by_index[match.paragraph_index].get("scores", {})
                audit_rows.append(
                    {
                        "book": mapping.book,
                        "system_kind": mapping.system_kind,
                        "paragraph_index": match.paragraph_index,
                        "chunk_index": match.chunk_index,
                        "match_score": f"{match.score:.6f}",
                        "match_method": match.method,
                        "metrics_available": "|".join(
                            metric for metric in REQUESTED_METRICS if metric in scores
                        ),
                    }
                )

            chunk_alignment_rows.extend(
                build_chunk_alignment_rows(
                    book=mapping.book,
                    system_kind=mapping.system_kind,
                    source_path=mapping.source.path,
                    chunk_path=mapping.chunk_path,
                    paragraph_rows=mapping.paragraph_rows,
                    chunk_rows=mapping.chunk_rows,
                    matches=mapping.matches,
                    mapped_scores=mapping.mapped_scores,
                    mapped_score_paragraph_indices=mapping.mapped_score_paragraph_indices,
                )
            )

            metrics_mapped = sorted(
                {
                    metric
                    for chunk_metrics in mapping.mapped_scores.values()
                    for metric in chunk_metrics
                }
            )
            low_confidence = [
                match for match in mapping.matches if match.score < args.min_match_score
            ]
            summary_rows.append(
                {
                    "book": mapping.book,
                    "system_kind": mapping.system_kind,
                    "source_path": str(mapping.source.path),
                    "chunk_path": str(mapping.chunk_path),
                    "num_paragraphs": len(mapping.paragraph_rows),
                    "num_chunks": len(mapping.chunk_rows),
                    "num_chunks_with_scores": len(mapping.mapped_scores),
                    "min_match_score": (
                        f"{min((m.score for m in mapping.matches), default=0.0):.6f}"
                    ),
                    "num_low_confidence_matches": len(low_confidence),
                    "metrics_mapped": "|".join(metrics_mapped),
                }
            )

    analysis_dir = analysis_results_dir(args.output_dir)
    write_audit_csv(analysis_dir / "paragraph_chunk_matches.csv", audit_rows)
    write_summary_csv(analysis_dir / "mapping_summary.csv", summary_rows)
    write_chunk_alignment_audit(
        analysis_dir / "chunk_paragraph_alignment_audit.csv",
        chunk_alignment_rows,
    )
    if args.agreement_audit_dir:
        write_chunk_alignment_audit(
            args.agreement_audit_dir / "chunk_paragraph_alignment_audit.csv",
            chunk_alignment_rows,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-results-dir", type=Path, default=DEFAULT_ALL_RESULTS_DIR)
    parser.add_argument("--chunk-results-dir", type=Path, default=DEFAULT_CHUNK_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--score-prefix",
        default="paragraph_mean_",
        help="Prefix for mapped score names added to each chunk scores object.",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=3,
        help="Number of future chunks considered for each paragraph during ordered matching.",
    )
    parser.add_argument(
        "--min-match-score",
        type=float,
        default=0.80,
        help="Audit threshold for low-confidence fuzzy matches.",
    )
    parser.add_argument(
        "--agreement-audit-dir",
        type=Path,
        default=DEFAULT_CHUNK_RESULTS_DIR / ANALYSIS_DIR_NAME / "metric_agreement_analysis",
        help=(
            "Optional extra directory that receives chunk_paragraph_alignment_audit.csv/jsonl "
            "for manual metric-agreement review."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    map_metrics(parse_args())
