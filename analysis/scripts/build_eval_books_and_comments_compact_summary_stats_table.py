"""Build a compact evaluation-data summary statistics LaTeX table.

This is a smaller companion to
``build_eval_books_and_comments_summary_stats_table.py``. It keeps only the
book-level token/word statistics and aggregates all participant free-text
comments into one group, with a ``# Comment/Book`` column analogous to the
``# Claim/Book`` column in the compact dataset-summary table.

Reads:
  - ``book_stats/human_translation_counts/book_{word,token}_count_stats.csv``
  - ``human_eval/comments_word_token_stats/output/all_comments_stats.csv``
  - ``human_eval/data/study-data-full.json`` (for comments-per-book counts)

Output is written to
``analysis/manuscript_tables/tex/eval_books_and_comments_compact_summary_stats.tex``
and also printed to stdout.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOK_STATS_ROOT = REPO_ROOT / "book_stats" / "human_translation_counts"
COMMENT_STATS_PATH = (
    REPO_ROOT / "human_eval" / "comments_word_token_stats" / "output" / "all_comments_stats.csv"
)
STUDY_DATA_PATH = REPO_ROOT / "human_eval" / "data" / "study-data-full.json"
HT_BOOKS_DIR = REPO_ROOT / "books" / "HT" / "eval"
OUTPUT_PATH = (
    REPO_ROOT
    / "analysis"
    / "manuscript_tables"
    / "tables"
    / "tex"
    / "eval_books_and_comments_compact_summary_stats.tex"
)

DEFAULT_EXCLUDED_PARTICIPANT_PREFIXES = ("humeval_p009", "humeval_p012")
COMMENT_FIELDS = (
    "single_q5",
    "single_q6",
    "comparison_q4",
    "comparison_q7",
    "chunk_justification",
)
STATS_ORDER = ("mean", "std", "max", "min")
STAT_LABELS = {
    "mean": r"\textsc{Mean}",
    "std": r"\textsc{St. Dev.}",
    "max": r"\textsc{Max}",
    "min": r"\textsc{Min}",
}

BOOKS_ICON = r"\includegraphics[height=1.1em]{figs/books.png}"
COMMENTS_ICON = r"\includegraphics[height=1em]{figs/Icons/comments.png}"


def read_simple_stats(path: Path, metric: str) -> dict[str, float]:
    """Read a stats CSV with rows ``stat,<metric>_count``."""
    column = f"{metric}_count"
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stat = row.get("stat", "")
            value = row.get(column, "")
            if stat and value not in ("", None):
                out[stat] = float(value)
    return out


def read_comment_field_stats(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Read all per-field comment stats from ``all_comments_stats.csv``."""
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            field = row["field"]
            stat = row["stat"]
            out[field][stat] = {
                "n": float(row["n_responses"]),
                "word": float(row["word_count"]),
                "token": float(row["token_count"]),
            }
    return out


def combine_comment_stats(
    field_stats: dict[str, dict[str, dict[str, float]]],
    metric: str,
) -> dict[str, float]:
    """Combine per-field summary stats into one all-comments summary.

    The combined standard deviation uses the sample-variance merge formula,
    matching the pandas ``Series.std`` convention used by the source script.
    """
    fields = [field for field in COMMENT_FIELDS if field in field_stats]
    total_n = int(sum(field_stats[field]["mean"]["n"] for field in fields))
    total_sum = sum(field_stats[field]["sum"][metric] for field in fields)
    mean = total_sum / total_n

    variance_numerator = 0.0
    for field in fields:
        n = int(field_stats[field]["mean"]["n"])
        field_mean = field_stats[field]["mean"][metric]
        field_std = field_stats[field]["std"][metric]
        variance_numerator += (n - 1) * (field_std**2)
        variance_numerator += n * ((field_mean - mean) ** 2)

    return {
        "mean": mean,
        "std": (variance_numerator / (total_n - 1)) ** 0.5 if total_n > 1 else 0.0,
        "max": max(field_stats[field]["max"][metric] for field in fields),
        "min": min(field_stats[field]["min"][metric] for field in fields),
        "n": float(total_n),
    }


def is_excluded(participant_id: object) -> bool:
    pid = str(participant_id or "")
    return any(pid.startswith(prefix) for prefix in DEFAULT_EXCLUDED_PARTICIPANT_PREFIXES)


def normalize_book_id(book_id: object) -> str:
    """Collapse known aliases so fixed and non-fixed Needle's Eye count together."""
    return str(book_id or "").replace("_FIXED_", "_")


def count_comments_by_book(study_data_path: Path) -> Counter[str]:
    """Count non-empty participant free-text comments per normalized book ID."""
    with study_data_path.open("r", encoding="utf-8") as fh:
        study_data = json.load(fh)

    assignment_to_book: dict[str, str] = {}
    for assignment in study_data.get("assignments", []):
        if is_excluded(assignment.get("participant_id")):
            continue
        assignment_to_book[str(assignment.get("id"))] = normalize_book_id(
            assignment.get("book_id")
        )

    counts: Counter[str] = Counter()
    for response in study_data.get("questionnaireResponses", []):
        if is_excluded(response.get("participant_id")):
            continue
        book_id = assignment_to_book.get(str(response.get("assignment_id")))
        if not book_id:
            continue

        q_type = response.get("questionnaire_type")
        answers = response.get("responses") or {}
        if q_type in ("first", "second"):
            keys = ("q5", "q6")
        elif q_type == "comparison":
            keys = ("q4", "q7")
        else:
            keys = ()

        for key in keys:
            if str(answers.get(key, "") or "").strip():
                counts[book_id] += 1

    for annotation in study_data.get("chunkAnnotations", []):
        if is_excluded(annotation.get("participant_id")):
            continue
        if not str(annotation.get("justification") or "").strip():
            continue
        counts[normalize_book_id(annotation.get("book_id"))] += 1

    return counts


def summary_stats(values: list[int]) -> dict[str, float]:
    if not values:
        raise SystemExit("Error: cannot summarize an empty value list")
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "max": float(max(values)),
        "min": float(min(values)),
    }


def fmt_intish(value: float) -> str:
    return f"{int(round(value)):,}"


def fmt_decimal(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def fmt_book(value: float, stat: str) -> str:
    if stat in {"mean", "std", "max", "min"}:
        return fmt_intish(value)
    return fmt_decimal(value)


def fmt_short(value: float, stat: str) -> str:
    if stat in {"max", "min"} or float(value).is_integer():
        return fmt_intish(value)
    return fmt_decimal(value)


def build_table() -> str:
    book_stats = {
        "token": read_simple_stats(BOOK_STATS_ROOT / "book_token_count_stats.csv", "token"),
        "word": read_simple_stats(BOOK_STATS_ROOT / "book_word_count_stats.csv", "word"),
    }
    n_books = len(list(HT_BOOKS_DIR.glob("*.jsonl")))

    comment_field_stats = read_comment_field_stats(COMMENT_STATS_PATH)
    comment_stats = {
        "token": combine_comment_stats(comment_field_stats, "token"),
        "word": combine_comment_stats(comment_field_stats, "word"),
    }
    n_comments = int(comment_stats["word"]["n"])
    comments_per_book = summary_stats(list(count_comments_by_book(STUDY_DATA_PATH).values()))

    body = []
    for stat in STATS_ORDER:
        cells = [
            STAT_LABELS[stat],
            fmt_book(book_stats["token"][stat], stat),
            fmt_book(book_stats["word"][stat], stat),
            fmt_short(comment_stats["token"][stat], stat),
            fmt_short(comment_stats["word"][stat], stat),
            fmt_short(comments_per_book[stat], stat),
        ]
        body.append("        " + " & ".join(cells) + r" \\")

    lines = [
        r"\begin{table}[t!]",
        r"    \centering",
        r"    \small",
        r"    \resizebox{0.48\textwidth}{!}{%",
        r"    \begin{tabular}{lcc|cc|c}",
        r"        \toprule",
        rf"         & \multicolumn{{2}}{{c}}{{\textbf{{Books}} {BOOKS_ICON}}} & \multicolumn{{3}}{{c}}{{\textbf{{Participant Comments}} {COMMENTS_ICON}}} \\",
        rf"         & \multicolumn{{2}}{{c}}{{(\textit{{n={n_books}}})}} & \multicolumn{{3}}{{c}}{{(\textit{{n={n_comments}}})}} \\",
        r"        \cmidrule(lr){2-3} \cmidrule(lr){4-6}",
        r"         & \textsc{Tokens} & \textsc{Words} & \textsc{Tokens} & \textsc{Words} & \textsc{\# Comment/Book}\\",
        r"        \midrule",
        *body,
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    }",
        r"    \caption{Summary statistics for evaluation books and participant evaluation comments.}",
        r"    \label{tab:eval_books_comments_compact_summary_stats}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    output = build_table() + "\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(output)
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
