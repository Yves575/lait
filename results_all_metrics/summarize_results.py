#!/usr/bin/env python3
"""Summarize LitMT metric results.

Run from anywhere:

    python results_all_metrics/summarize_results.py

By default this scans the directory containing this script, recomputes system
means from ``segment_scores.jsonl`` when available, writes ``all_results.csv``
and ``summary.json`` there, and prints a compact summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


TARGET_SYSTEMS = {
    "pipeline1_gemini",
    "pipeline1_gpt54_high",
    "pipeline2_gemini",
    "pipeline2_gpt54_high",
    "pipeline3",
}

CSV_FIELDS = [
    "split",
    "book",
    "system_folder",
    "pipeline",
    "model",
    "pipeline_model",
    "system_name",
    "metric",
    "score",
    "higher_is_better",
    "num_scored_segments",
    "num_filtered_segments",
    "num_available_segments",
    "path",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir,
        help="Directory to scan recursively for segment_scores.jsonl or system_scores.json files.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=script_dir / "all_results.csv",
        help="Path for the full aggregated CSV table.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=script_dir / "summary.json",
        help="Path for the JSON summary.",
    )
    parser.add_argument(
        "--alignment-csv",
        type=Path,
        default=None,
        help="Optional diagnostics CSV from par3/verify_alignment.py used for downstream filtering.",
    )
    parser.add_argument(
        "--length-ratio-threshold",
        type=float,
        default=None,
        help="When set with --alignment-csv, filter segments whose GT/MT length ratio is above this value.",
    )
    parser.add_argument(
        "--min-length-check-chars",
        type=int,
        default=10,
        help="Ignore length-ratio filtering when both compared strings are shorter than this many characters.",
    )
    parser.add_argument(
        "--bleu-threshold",
        type=float,
        default=None,
        help="When set with --alignment-csv, filter segments with enough pairwise BLEU scores below this value.",
    )
    parser.add_argument(
        "--min-bleu-repetitions",
        type=int,
        default=None,
        help="Minimum low-BLEU comparisons needed to filter a segment. Defaults to a majority.",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print the readable summary.")
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warn(f"skipping malformed JSON {path}: {exc}")
        return None
    except OSError as exc:
        warn(f"skipping unreadable file {path}: {exc}")
        return None
    if not isinstance(data, dict):
        warn(f"skipping {path}: top-level JSON is not an object")
        return None
    return data


def parse_location(path: Path, results_dir: Path) -> tuple[str, str, str]:
    try:
        parts = path.relative_to(results_dir).parts
    except ValueError:
        parts = path.parts

    split = parts[0] if parts and parts[0] in {"dev", "eval"} else "unknown"
    book = parts[1] if len(parts) > 1 and split != "unknown" else "unknown"
    system_folder = path.parent.name
    return split, book, system_folder


def parse_system_name(system_folder: str, book: str) -> tuple[str, str | None, str]:
    short_name = system_folder
    book_prefix = f"{book}_"
    if book != "unknown" and short_name.startswith(book_prefix):
        short_name = short_name[len(book_prefix) :]

    if short_name == "ht" or short_name.endswith("_ht"):
        return "ht", None, "ht"

    match = re.match(r"^(pipeline\d+)(?:_(.+))?$", short_name)
    if match:
        pipeline = match.group(1)
        model = match.group(2)
        pipeline_model = pipeline if model is None else f"{pipeline}_{model}"
        return pipeline, model, pipeline_model

    parts = short_name.split("_", 1)
    pipeline = parts[0] if parts and parts[0] else "unknown"
    model = parts[1] if len(parts) > 1 else None
    pipeline_model = pipeline if model is None else f"{pipeline}_{model}"
    return pipeline, model, pipeline_model


def as_float(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return score


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def metric_higher_is_better(metric: str) -> bool:
    return metric in {"comet22", "cometkiwi"}


def parse_bleu_scores(raw_scores: str) -> list[float]:
    scores: list[float] = []
    for item in raw_scores.split(";"):
        if not item:
            continue
        _, _, raw_score = item.rpartition(":")
        score = as_float(raw_score)
        if score is not None:
            scores.append(score)
    return scores


def load_alignment_diagnostics(csv_path: Path | None) -> dict[tuple[str, str, int], dict[str, str]]:
    if csv_path is None:
        return {}
    diagnostics: dict[tuple[str, str, int], dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            book = row.get("book", "")
            system = row.get("system_name") or row.get("pipeline") or ""
            raw_index = row.get("segment_index") or row.get("paragraph_id")
            if not book or not system or raw_index is None:
                warn(f"skipping alignment row {row_number}: missing book/system/segment")
                continue
            try:
                segment_index = int(raw_index)
            except ValueError:
                warn(f"skipping alignment row {row_number}: invalid segment index {raw_index!r}")
                continue
            diagnostics[(book, system, segment_index)] = row
    return diagnostics


def alignment_keys(book: str, system_folder: str, pipeline_model: str, segment_index: int) -> list[tuple[str, str, int]]:
    system_names = [system_folder]
    prefixed = f"{book}_{pipeline_model}"
    if prefixed not in system_names:
        system_names.append(prefixed)
    if pipeline_model not in system_names:
        system_names.append(pipeline_model)
    return [(book, system, segment_index) for system in system_names]


def find_alignment_diagnostic(
    diagnostics: dict[tuple[str, str, int], dict[str, str]],
    book: str,
    system_folder: str,
    pipeline_model: str,
    segment_index: int,
) -> dict[str, str] | None:
    for key in alignment_keys(book, system_folder, pipeline_model, segment_index):
        if key in diagnostics:
            return diagnostics[key]
    return None


def should_filter_segment(row: dict[str, str], args: argparse.Namespace) -> bool:
    if args.length_ratio_threshold is not None:
        ratio = as_float(row.get("ratio"))
        gt_length = as_int(row.get("gt_length"))
        compared_length = as_int(row.get("compared_length"))
        max_length = max(gt_length or 0, compared_length or 0)
        if ratio is not None and max_length >= args.min_length_check_chars and ratio > args.length_ratio_threshold:
            return True

    if args.bleu_threshold is not None:
        scores = parse_bleu_scores(row.get("bleu_scores", ""))
        if not scores:
            fallback = as_float(row.get("bleu_score"))
            if fallback is not None:
                scores = [fallback]
        low_count = sum(1 for score in scores if score < args.bleu_threshold)
        total = len(scores)
        if total:
            required = args.min_bleu_repetitions
            if required is None:
                required = max(1, math.ceil(total / 2))
            if low_count >= required:
                return True

    return False


def read_segment_score_rows(
    results_dir: Path,
    diagnostics: dict[tuple[str, str, int], dict[str, str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path in sorted(results_dir.rglob("segment_scores.jsonl")):
        split, book, system_folder = parse_location(path, results_dir)
        pipeline, model, pipeline_model = parse_system_name(system_folder, book)
        metric_scores: dict[str, list[float]] = defaultdict(list)
        filtered_counts: dict[str, int] = defaultdict(int)
        available_counts: dict[str, int] = defaultdict(int)

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            warn(f"skipping unreadable file {path}: {exc}")
            warnings.append(f"skipped {path}")
            continue

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                segment = json.loads(line)
            except json.JSONDecodeError as exc:
                warn(f"skipping malformed JSONL row {path}:{line_number}: {exc}")
                continue
            if not isinstance(segment, dict):
                continue
            segment_index = as_int(segment.get("segment_index"))
            scores = segment.get("scores")
            if segment_index is None or not isinstance(scores, dict):
                continue

            diagnostic = find_alignment_diagnostic(
                diagnostics,
                book,
                system_folder,
                pipeline_model,
                segment_index,
            )
            filtered = diagnostic is not None and should_filter_segment(diagnostic, args)
            for metric_name, raw_score in scores.items():
                score = as_float(raw_score)
                if score is None:
                    continue
                metric_name = str(metric_name)
                available_counts[metric_name] += 1
                if filtered:
                    filtered_counts[metric_name] += 1
                    continue
                metric_scores[metric_name].append(score)

        for metric_name, scores in sorted(metric_scores.items()):
            if not scores:
                continue
            rows.append(
                {
                    "split": split,
                    "book": book,
                    "system_folder": system_folder,
                    "pipeline": pipeline,
                    "model": model or "",
                    "pipeline_model": pipeline_model,
                    "system_name": pipeline_model,
                    "metric": metric_name,
                    "score": sum(scores) / len(scores),
                    "higher_is_better": metric_higher_is_better(metric_name),
                    "num_scored_segments": len(scores),
                    "num_filtered_segments": filtered_counts[metric_name],
                    "num_available_segments": available_counts[metric_name],
                    "path": str(path),
                }
            )

    return rows, warnings


def read_system_score_rows(results_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path in sorted(results_dir.rglob("system_scores.json")):
        data = load_json(path)
        if data is None:
            warnings.append(f"skipped {path}")
            continue

        metrics = data.get("metrics")
        if not isinstance(metrics, dict):
            warn(f"skipping {path}: missing or invalid metrics object")
            warnings.append(f"missing metrics: {path}")
            continue

        split, book, folder_system = parse_location(path, results_dir)
        system_folder = str(data.get("system") or folder_system)
        pipeline, model, pipeline_model = parse_system_name(system_folder, book)

        for metric_name, metric_data in sorted(metrics.items()):
            if not isinstance(metric_data, dict):
                warn(f"skipping {path} metric {metric_name}: metric value is not an object")
                continue
            score = as_float(metric_data.get("score"))
            if score is None:
                warn(f"skipping {path} metric {metric_name}: missing or invalid score")
                continue
            higher_is_better = bool(metric_data.get("higher_is_better"))
            rows.append(
                {
                    "split": split,
                    "book": book,
                    "system_folder": system_folder,
                    "pipeline": pipeline,
                    "model": model or "",
                    "pipeline_model": pipeline_model,
                    "system_name": pipeline_model,
                    "metric": str(metric_name),
                    "score": score,
                    "higher_is_better": higher_is_better,
                    "num_scored_segments": int(metric_data.get("num_scored_segments") or 0),
                    "num_filtered_segments": 0,
                    "num_available_segments": int(metric_data.get("num_scored_segments") or 0),
                    "path": str(path),
                }
            )

    return rows, warnings


def read_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    diagnostics = load_alignment_diagnostics(args.alignment_csv)
    use_segment_scores = bool(diagnostics) or any(args.results_dir.rglob("segment_scores.jsonl"))
    if use_segment_scores:
        return read_segment_score_rows(args.results_dir, diagnostics, args)
    return read_system_score_rows(args.results_dir)


def best_and_worst(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    higher_is_better = bool(rows[0]["higher_is_better"])
    best = max(rows, key=lambda row: row["score"]) if higher_is_better else min(rows, key=lambda row: row["score"])
    worst = min(rows, key=lambda row: row["score"]) if higher_is_better else max(rows, key=lambda row: row["score"])
    return {"best": public_row(best), "worst": public_row(worst)}


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in CSV_FIELDS}


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return dict(grouped)


def summarize_grouped(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for key_values, group in sorted(group_rows(rows, keys).items()):
        extremes = best_and_worst(group)
        if extremes is None:
            continue
        item = {key: value for key, value in zip(keys, key_values)}
        item.update(extremes)
        summary.append(item)
    return summary


def build_summary(rows: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    metric_extremes = {
        metric: best_and_worst(group)
        for (metric,), group in sorted(group_rows(rows, ("metric",)).items())
    }
    metric_extremes = {metric: value for metric, value in metric_extremes.items() if value is not None}

    requested = {
        target: [public_row(row) for row in rows if row["pipeline_model"] == target]
        for target in sorted(TARGET_SYSTEMS)
    }

    return {
        "num_rows": len(rows),
        "num_system_score_files": len({row["path"] for row in rows}),
        "metrics": sorted({row["metric"] for row in rows}),
        "metric_extremes": metric_extremes,
        "best_worst_system_per_metric": metric_extremes,
        "best_worst_system_per_pipeline": summarize_grouped(rows, ("metric", "pipeline_model")),
        "best_worst_system_per_book": summarize_grouped(rows, ("metric", "book")),
        "requested_systems": requested,
        "warnings": warnings,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_row(row: dict[str, Any]) -> str:
    direction = "higher" if row["higher_is_better"] else "lower"
    filtered = ""
    if row.get("num_filtered_segments"):
        filtered = (
            f", filtered={row['num_filtered_segments']}/"
            f"{row.get('num_available_segments', row['num_scored_segments'])}"
        )
    return (
        f"{row['score']:.6g} ({direction} is better), n={row['num_scored_segments']}{filtered} | "
        f"{row['split']} / {row['book']} / {row['system_folder']} | {row['path']}"
    )


def print_summary(summary: dict[str, Any], csv_path: Path, json_path: Path) -> None:
    print(f"Rows: {summary['num_rows']}")
    print(f"System score files: {summary['num_system_score_files']}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print()

    print("Best/Worst By Metric")
    for metric, extremes in summary["metric_extremes"].items():
        print(f"- {metric}")
        print(f"  best:  {format_row(extremes['best'])}")
        print(f"  worst: {format_row(extremes['worst'])}")
    print()

    print("Requested Systems")
    for target, rows in summary["requested_systems"].items():
        print(f"- {target}: {len(rows)} metric rows")
        by_metric = group_rows(rows, ("metric",))
        for (metric,), group in sorted(by_metric.items()):
            extremes = best_and_worst(group)
            if extremes:
                print(f"  {metric} best:  {format_row(extremes['best'])}")
                print(f"  {metric} worst: {format_row(extremes['worst'])}")


def main() -> None:
    args = parse_args()
    if args.min_length_check_chars < 0:
        raise SystemExit("--min-length-check-chars must be non-negative")
    if args.min_bleu_repetitions is not None and args.min_bleu_repetitions <= 0:
        raise SystemExit("--min-bleu-repetitions must be greater than 0")

    rows, warnings = read_rows(args)
    if not rows:
        raise SystemExit(f"No valid metric rows found under {args.results_dir}")

    summary = build_summary(rows, warnings)
    summary["alignment_filter"] = {
        "alignment_csv": str(args.alignment_csv) if args.alignment_csv else None,
        "length_ratio_threshold": args.length_ratio_threshold,
        "min_length_check_chars": args.min_length_check_chars,
        "bleu_threshold": args.bleu_threshold,
        "min_bleu_repetitions": args.min_bleu_repetitions,
    }
    write_csv(args.csv, rows)
    write_json(args.json, summary)
    if not args.quiet:
        print_summary(summary, args.csv, args.json)


if __name__ == "__main__":
    main()
