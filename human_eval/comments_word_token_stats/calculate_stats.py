"""Calculate word and token count stats for free-text comments in the human evaluation study.

Mirrors `book_stats/calculate_stats.py` but operates on the per-response text fields in
`human_eval/data/study-data-full.json` rather than book chunks.

Comment fields covered (one stats table per field):
  - single_q5         : responses.q5 from first + second single-reading questionnaires
  - single_q6         : responses.q6 from first + second single-reading questionnaires
  - comparison_q4     : responses.q4 from the comparison questionnaire
  - comparison_q7     : responses.q7 from the comparison questionnaire
  - chunk_justification : justification field from per-chunk annotations

Empty / missing responses are skipped (not counted as zero).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import tiktoken

# Match the participant exclusions used by `human_eval/export_study_data_csv.py` so that
# downstream analyses see the same population.
DEFAULT_EXCLUDED_PARTICIPANT_PREFIXES = ("humeval_p009", "humeval_p012")

COMMENT_FIELDS = (
    "single_q5",
    "single_q6",
    "comparison_q4",
    "comparison_q7",
    "chunk_justification",
)


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def count_words(text: str) -> int:
    """Count whitespace-delimited words, matching `export_study_data_csv.whitespace_word_count`."""
    return len(text.split())


def is_excluded(participant_id: object, excluded_prefixes: tuple[str, ...]) -> bool:
    if not excluded_prefixes:
        return False
    pid = str(participant_id or "")
    return any(pid.startswith(prefix) for prefix in excluded_prefixes)


def extract_comments(
    study_data: dict,
    excluded_prefixes: tuple[str, ...],
) -> dict[str, list[dict]]:
    """Return a mapping of comment-field name -> list of {participant_id, text} rows.

    Skips empty / whitespace-only responses.
    """
    buckets: dict[str, list[dict]] = {name: [] for name in COMMENT_FIELDS}

    for response in study_data.get("questionnaireResponses", []):
        if is_excluded(response.get("participant_id"), excluded_prefixes):
            continue

        q_type = response.get("questionnaire_type")
        answers = response.get("responses") or {}
        if not isinstance(answers, dict):
            continue

        if q_type in ("first", "second"):
            field_map = {"single_q5": "q5", "single_q6": "q6"}
        elif q_type == "comparison":
            field_map = {"comparison_q4": "q4", "comparison_q7": "q7"}
        else:
            continue

        for bucket_name, answer_key in field_map.items():
            text = str(answers.get(answer_key, "") or "").strip()
            if not text:
                continue
            buckets[bucket_name].append(
                {
                    "participant_id": response.get("participant_id"),
                    "assignment_id": response.get("assignment_id"),
                    "questionnaire_type": q_type,
                    "text": text,
                }
            )

    for annotation in study_data.get("chunkAnnotations", []):
        if is_excluded(annotation.get("participant_id"), excluded_prefixes):
            continue
        text = str(annotation.get("justification") or "").strip()
        if not text:
            continue
        buckets["chunk_justification"].append(
            {
                "participant_id": annotation.get("participant_id"),
                "assignment_id": annotation.get("assignment_id"),
                "book_id": annotation.get("book_id"),
                "chunk_id": annotation.get("chunk_id"),
                "text": text,
            }
        )

    return buckets


def build_counts_frame(rows: list[dict]) -> pd.DataFrame:
    """Add word_count and token_count columns to a list of comment rows."""
    if not rows:
        return pd.DataFrame(columns=["text", "word_count", "token_count"])

    df = pd.DataFrame(rows)
    df["word_count"] = df["text"].map(count_words)
    df["token_count"] = df["text"].map(count_tokens)
    return df


def calculate_stats(counts: pd.Series, metric_name: str) -> pd.DataFrame:
    """Return max, min, std, mean, median, and total for a count series."""
    if counts.empty:
        return pd.DataFrame(
            {"stat": ["max", "min", "std", "mean", "median", "sum"], metric_name: [pd.NA] * 6}
        )
    stats = counts.agg(["max", "min", "std", "mean", "median", "sum"])
    return stats.rename_axis("stat").reset_index(name=metric_name)


def combine_stats(word_stats: pd.DataFrame, token_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge per-metric stats frames into one (stat, word_count, token_count) frame."""
    return word_stats.merge(token_stats, on="stat", how="outer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate word- and token-count statistics for free-text comments in the "
            "human evaluation study."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("human_eval/data/study-data-full.json"),
        help="Path to study-data-full.json (default: human_eval/data/study-data-full.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("human_eval/comments_word_token_stats/output"),
        help=(
            "Directory to write CSV outputs into (default: "
            "human_eval/comments_word_token_stats/output)."
        ),
    )
    parser.add_argument(
        "--exclude-participant-prefix",
        action="append",
        default=None,
        help=(
            "Participant ID prefix to exclude. May be passed multiple times. "
            "Defaults match export_study_data_csv.py "
            f"({', '.join(DEFAULT_EXCLUDED_PARTICIPANT_PREFIXES)}). "
            "Pass --exclude-participant-prefix '' to disable exclusions."
        ),
    )
    parser.add_argument(
        "--write-per-response",
        action="store_true",
        help="Also write per-response word/token counts to <field>_per_response.csv.",
    )
    return parser.parse_args()


def resolve_excluded_prefixes(cli_value: list[str] | None) -> tuple[str, ...]:
    if cli_value is None:
        return DEFAULT_EXCLUDED_PARTICIPANT_PREFIXES
    return tuple(prefix for prefix in cli_value if prefix)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    excluded_prefixes = resolve_excluded_prefixes(args.exclude_participant_prefix)

    if not input_path.is_file():
        print(f"Error: {input_path} is not a file", file=sys.stderr)
        raise SystemExit(1)

    with input_path.open("r", encoding="utf-8") as fh:
        study_data = json.load(fh)

    buckets = extract_comments(study_data, excluded_prefixes)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for field in COMMENT_FIELDS:
        rows = buckets[field]
        counts_df = build_counts_frame(rows)

        word_stats = calculate_stats(counts_df["word_count"], "word_count")
        token_stats = calculate_stats(counts_df["token_count"], "token_count")
        combined = combine_stats(word_stats, token_stats)
        combined.insert(0, "field", field)
        combined.insert(1, "n_responses", len(counts_df))

        out_path = output_dir / f"{field}_stats.csv"
        combined.to_csv(out_path, index=False)
        print(f"Wrote {field} stats to {out_path} (n={len(counts_df)})")

        if args.write_per_response and not counts_df.empty:
            per_resp_path = output_dir / f"{field}_per_response.csv"
            counts_df.to_csv(per_resp_path, index=False)
            print(f"Wrote {field} per-response counts to {per_resp_path}")

        summary_rows.append(combined)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_path = output_dir / "all_comments_stats.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote combined summary to {summary_path}")


if __name__ == "__main__":
    main()
