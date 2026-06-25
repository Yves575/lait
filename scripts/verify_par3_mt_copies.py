"""Verify copied MT files under par3_dataset against their books/MT sources.

The script checks:

1. The copied file is placed under the matching split/title folder.
2. The filename encodes a valid pipeline/model combination.
3. The file content matches the expected source text after removing blank lines,
   which is the same normalization used when the files were copied.

Examples:
  python3 scripts/verify_par3_mt_copies.py
  python3 scripts/verify_par3_mt_copies.py --split eval --show-ok
  python3 scripts/verify_par3_mt_copies.py --title fog
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET_DIR = Path("par3_dataset")
DEFAULT_MT_DIR = Path("books/MT")
SPLITS = {"dev", "eval"}
PIPELINES_WITH_MODELS = {"pipeline1", "pipeline2"}
PIPELINE_WITHOUT_MODEL = "pipeline3"
TARGET_PATTERN = re.compile(
    r"^(?P<title>.+)_(?P<pipeline>pipeline[1-3])(?:_(?P<model>.+))?\.txt$"
)


@dataclass(frozen=True)
class VerificationResult:
    status: str
    target: Path
    source: Path | None
    message: str


def read_normalized_lines(path: Path) -> list[str]:
    """Read a text file and drop blank lines, preserving remaining line text."""
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.strip():
                lines.append(line)
    return lines


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_target(target: Path, dataset_dir: Path) -> tuple[str, str, str, str | None] | None:
    """Parse a par3_dataset MT copy path into split/title/pipeline/model."""
    try:
        rel = target.relative_to(dataset_dir)
    except ValueError:
        return None

    if len(rel.parts) != 4:
        return None

    split, title, subdir, filename = rel.parts
    if split not in SPLITS or subdir != "trans_txts":
        return None

    match = TARGET_PATTERN.fullmatch(filename)
    if not match:
        return None

    file_title = match.group("title")
    pipeline = match.group("pipeline")
    model = match.group("model")
    return split, title, file_title, pipeline, model


def resolve_source(
    mt_dir: Path, split: str, title: str, pipeline: str, model: str | None
) -> tuple[Path | None, str | None]:
    """Resolve the expected source file and return an error message on failure."""
    if pipeline == PIPELINE_WITHOUT_MODEL:
        if model is not None:
            return None, "pipeline3 target should not include a model suffix"
        source_dir = mt_dir / pipeline
    elif pipeline in PIPELINES_WITH_MODELS:
        if not model:
            return None, f"{pipeline} target is missing a model suffix"
        source_dir = mt_dir / pipeline / model
        if model in {"gpt54", "gpt54_high"} and split == "eval":
            source_dir = source_dir / "eval"
    else:
        return None, f"unsupported pipeline: {pipeline}"

    if not source_dir.is_dir():
        return None, f"source folder does not exist: {relative(source_dir)}"

    matches = sorted(source_dir.glob(f"{title}_*.txt"))
    if not matches:
        return None, f"no source file found in {relative(source_dir)} for title {title!r}"
    if len(matches) > 1:
        paths = ", ".join(relative(path) for path in matches)
        return None, f"multiple source files found for title {title!r}: {paths}"
    return matches[0], None


def compare_files(source: Path, target: Path) -> str | None:
    """Return a mismatch explanation, or None if the normalized texts match."""
    source_lines = read_normalized_lines(source)
    target_lines = read_normalized_lines(target)

    if source_lines == target_lines:
        return None

    if len(source_lines) != len(target_lines):
        return (
            "normalized content differs: "
            f"{len(source_lines)} non-empty lines in source vs "
            f"{len(target_lines)} in target"
        )

    for index, (source_line, target_line) in enumerate(zip(source_lines, target_lines), start=1):
        if source_line != target_line:
            return (
                "normalized content differs at line "
                f"{index}: source={source_line[:80]!r} target={target_line[:80]!r}"
            )

    return "normalized content differs"


def verify_target(target: Path, dataset_dir: Path, mt_dir: Path) -> VerificationResult:
    parsed = parse_target(target, dataset_dir)
    if parsed is None:
        return VerificationResult(
            status="SKIP",
            target=target,
            source=None,
            message="path does not match par3_dataset/<split>/<title>/trans_txts/<title>_pipeline*.txt",
        )

    split, folder_title, file_title, pipeline, model = parsed
    if folder_title != file_title:
        return VerificationResult(
            status="BAD_NAME",
            target=target,
            source=None,
            message=f"folder title {folder_title!r} does not match filename title {file_title!r}",
        )

    source, error = resolve_source(mt_dir=mt_dir, split=split, title=folder_title, pipeline=pipeline, model=model)
    if error is not None:
        return VerificationResult(status="MISSING_SOURCE", target=target, source=None, message=error)

    mismatch = compare_files(source, target)
    if mismatch is not None:
        return VerificationResult(status="MISMATCH", target=target, source=source, message=mismatch)

    return VerificationResult(status="OK", target=target, source=source, message="content matches source")


def iter_targets(dataset_dir: Path, split: str | None, title: str | None) -> list[Path]:
    pattern = "*_pipeline*.txt"
    if split and title:
        roots = [dataset_dir / split / title / "trans_txts"]
    elif split:
        roots = [dataset_dir / split]
    elif title:
        roots = list(dataset_dir.glob(f"*/{title}/trans_txts"))
    else:
        roots = [dataset_dir]

    targets: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        targets.extend(sorted(root.rglob(pattern)))
    return sorted(set(targets))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that copied MT files under par3_dataset are in the correct "
            "folder and match the expected books/MT source after blank-line removal."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset root to scan. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--mt-dir",
        type=Path,
        default=DEFAULT_MT_DIR,
        help=f"books/MT root. Default: {DEFAULT_MT_DIR}",
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLITS),
        help="Only verify one dataset split.",
    )
    parser.add_argument(
        "--title",
        help="Only verify one title.",
    )
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="Print OK rows in addition to failures.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.dataset_dir.is_dir():
        print(f"Dataset folder does not exist: {args.dataset_dir}", file=sys.stderr)
        return 1
    if not args.mt_dir.is_dir():
        print(f"MT folder does not exist: {args.mt_dir}", file=sys.stderr)
        return 1

    targets = iter_targets(args.dataset_dir, args.split, args.title)
    if not targets:
        print("No matching copied MT files found.", file=sys.stderr)
        return 1

    results = [verify_target(target, args.dataset_dir, args.mt_dir) for target in targets]

    print("status\ttarget\tsource\tmessage")
    for result in results:
        if result.status == "OK" and not args.show_ok:
            continue
        source = "-" if result.source is None else relative(result.source)
        print(f"{result.status}\t{relative(result.target)}\t{source}\t{result.message}")

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    print()
    print(f"Verified files: {len(results):,}")
    for status in sorted(counts):
        print(f"{status}: {counts[status]:,}")

    failures = {"BAD_NAME", "MISSING_SOURCE", "MISMATCH"}
    return 1 if any(result.status in failures for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
