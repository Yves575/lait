"""Build the unified evaluation-data summary statistics LaTeX table.

Reads:
  - ``book_stats/{human_translation_counts,machine_translation_counts,source_text_counts}/{book,chunk}_{word,token}_count_stats.csv``
  - ``human_eval/comments_word_token_stats/output/{chunk_justification,single_q5,
    single_q6,comparison_q4,comparison_q7}_stats.csv``
  - ``books/HT/eval/*.jsonl`` (to derive the chunks-per-book distribution).

Emits a single LaTeX ``table*`` with two stacked sub-tables (``Words`` on top,
``Tokens`` on bottom). For Books and Chunks, the ``Words`` sub-table reports
``HT`` and ``MT`` only; the ``Tokens`` sub-table also includes ``Source`` rows.
For ``# Chunks/Book`` and the Participant-Comments columns (which do not split
by translation version), the value spans all version rows in that sub-table via
``\\multirow``.

Output is written to
``analysis/manuscript_tables/tex/eval_books_and_comments_summary_stats.tex`` and also printed to
stdout.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOK_STATS_ROOT = REPO_ROOT / "book_stats"
COMMENT_STATS_ROOT = (
    REPO_ROOT / "human_eval" / "comments_word_token_stats" / "output"
)
HT_BOOKS_DIR = REPO_ROOT / "books" / "HT" / "eval"
OUTPUT_PATH = (
    REPO_ROOT / "analysis" / "manuscript_tables" / "tables" / "tex" / "eval_books_and_comments_summary_stats.tex"
)

VERSIONS = ("HT", "MT", "SRC")
WORD_VERSIONS = ("HT", "MT")
VERSION_STATS_DIRS = {
    "HT": "human_translation_counts",
    "MT": "machine_translation_counts",
    "SRC": "source_text_counts",
}
VERSION_ROW_LABELS = {
    "HT": r"\htr ~ \includegraphics[height=1em]{figs/Icons/human.png}",
    "MT": r"\mtr ~ \includegraphics[height=1em]{figs/Icons/robot.png}",
    "SRC": r"\srcr ~ \includegraphics[height=1em]{figs/Icons/foreign_book.png}",
}
METRICS = ("word", "token")
STATS_ORDER = ("mean", "median", "std", "max", "min", "sum")
STAT_LABELS = {
    "mean": r"\textsc{Mean}",
    "median": r"\textsc{Median}",
    "std": r"\textsc{St. Dev.}",
    "max": r"\textsc{Max}",
    "min": r"\textsc{Min}",
    "sum": r"\textsc{Total}",
}
SECTION_LABELS = {
    "word": r"\textbf{\textsc{Words}}",
    "token": r"\textbf{\textsc{Tokens}}",
}

COMMENT_FIELDS = (
    "single_q5",
    "single_q6",
    "comparison_q4",
    "comparison_q7",
    "chunk_justification",
)
COMMENT_HEADERS = {
    "chunk_justification": r"\textsc{Chunk Justif.}",
    "single_q5": r"\textsc{Single Q5}",
    "single_q6": r"\textsc{Single Q6}",
    "comparison_q4": r"\textsc{Compar. Q4}",
    "comparison_q7": r"\textsc{Compar. Q7}",
}

# Stats that always render as integers; others get one decimal place.
INT_STATS = {"max", "min", "sum"}


def read_book_chunk_stats(path: Path, metric: str) -> dict[str, float]:
    """Read a ``book_stats`` CSV with rows ``stat,<word|token>_count``."""
    column = f"{metric}_count"
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stat = row.get("stat", "")
            value_str = row.get(column, "")
            if not stat or not value_str:
                continue
            try:
                out[stat] = float(value_str)
            except ValueError:
                continue
    return out


def read_comment_stats(path: Path, metric: str) -> dict[str, float]:
    """Read a comment-stats CSV (one ``field`` per file) and return its stats."""
    column = f"{metric}_count"
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stat = row.get("stat", "")
            value_str = row.get(column, "")
            if not stat or value_str in (None, ""):
                continue
            try:
                out[stat] = float(value_str)
            except ValueError:
                continue
    return out


def compute_chunks_per_book_stats(books_dir: Path) -> tuple[dict[str, float], int]:
    """Compute per-book chunk-count distribution and book count from JSONL files."""
    counts: list[int] = []
    for path in sorted(books_dir.glob("*.jsonl")):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as fh:
            counts.append(sum(1 for line in fh if line.strip()))
    if not counts:
        raise SystemExit(f"Error: no JSONL files found in {books_dir}")
    stats = {
        "mean": statistics.mean(counts),
        "median": statistics.median(counts),
        "std": statistics.stdev(counts) if len(counts) > 1 else 0.0,
        "max": float(max(counts)),
        "min": float(min(counts)),
        "sum": float(sum(counts)),
    }
    return stats, len(counts)


def fmt(value: float, stat: str) -> str:
    """Format a number: ``,``-separated ints for max/min/sum, else 1 decimal."""
    if value is None:
        return "--"
    # Integer-valued stats that happen to be exact half-integers (e.g. medians
    # like 41.5) should still display the fractional part.
    if stat in INT_STATS and float(value).is_integer():
        return f"{int(round(value)):,}"
    if float(value).is_integer():
        return f"{int(round(value)):,}"
    # Default: one decimal place, with thousands separators.
    return f"{value:,.1f}"


def load_all_stats() -> dict:
    """Load every stats source needed for the table."""
    data: dict = {
        "books": {v: {m: {} for m in METRICS} for v in VERSIONS},
        "chunks": {v: {m: {} for m in METRICS} for v in VERSIONS},
        "comments": {f: {m: {} for m in METRICS} for f in COMMENT_FIELDS},
        "chunks_per_book": {},
    }

    for version in VERSIONS:
        for metric in METRICS:
            data["books"][version][metric] = read_book_chunk_stats(
                BOOK_STATS_ROOT / VERSION_STATS_DIRS[version] / f"book_{metric}_count_stats.csv",
                metric,
            )
            data["chunks"][version][metric] = read_book_chunk_stats(
                BOOK_STATS_ROOT / VERSION_STATS_DIRS[version] / f"chunk_{metric}_count_stats.csv",
                metric,
            )

    for field in COMMENT_FIELDS:
        path = COMMENT_STATS_ROOT / f"{field}_stats.csv"
        for metric in METRICS:
            data["comments"][field][metric] = read_comment_stats(path, metric)

    chunks_per_book_stats, n_books = compute_chunks_per_book_stats(HT_BOOKS_DIR)
    data["chunks_per_book"] = chunks_per_book_stats
    data["n_books"] = n_books
    return data


def comment_n(field: str, data: dict) -> int:
    """Return n_responses for a comment field (read from the CSV header)."""
    path = COMMENT_STATS_ROOT / f"{field}_stats.csv"
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row = next(reader, None)
        if row is None:
            return 0
        try:
            return int(float(row["n_responses"]))
        except (KeyError, ValueError):
            return 0


def build_data_row(
    stat: str,
    metric: str,
    version: str,
    data: dict,
) -> list[str]:
    """Build the per-version cells: [book, chunk] values."""
    book_val = data["books"][version][metric].get(stat)
    chunk_val = data["chunks"][version][metric].get(stat)
    return [fmt(book_val, stat), fmt(chunk_val, stat)]


def build_shared_cells(stat: str, metric: str, data: dict) -> list[str]:
    """Build the cells that don't split HT/MT: #/Bk + 5 comment columns."""
    chunks_per_book = fmt(data["chunks_per_book"].get(stat), stat)
    comment_cells = [
        fmt(data["comments"][field][metric].get(stat), stat)
        for field in COMMENT_FIELDS
    ]
    return [chunks_per_book] + comment_cells


BOOKS_ICON = r"\includegraphics[height=1.1em]{figs/books.png}"
CHUNKS_ICON = r"\chunkicon"
COMMENTS_ICON = r"\includegraphics[height=1em]{figs/Icons/comments.png}"


def build_section_rows(
    metric: str,
    data: dict,
    *,
    versions: tuple[str, ...] = VERSIONS,
) -> list[str]:
    """Build the body rows for one section (Words or Tokens)."""
    section_label = SECTION_LABELS[metric]
    n_versions = len(versions)
    n_body_rows = len(STATS_ORDER) * n_versions
    section_cell = (
        f"\\multirow{{{n_body_rows}}}{{*}}{{\\rotatebox{{90}}{{{section_label}}}}}"
    )

    lines: list[str] = []
    for stat_idx, stat in enumerate(STATS_ORDER):
        stat_label = STAT_LABELS[stat]
        shared = build_shared_cells(stat, metric, data)
        shared_cells_multirow = [f"\\multirow{{{n_versions}}}{{*}}{{{cell}}}" for cell in shared]
        empty_shared = ["" for _ in shared]

        for version_idx, version in enumerate(versions):
            book_val, chunk_val = build_data_row(stat, metric, version, data)
            cells = [
                section_cell if stat_idx == 0 and version_idx == 0 else "",
                f"\\multirow{{{n_versions}}}{{*}}{{{stat_label}}}" if version_idx == 0 else "",
                VERSION_ROW_LABELS[version],
                book_val,
                chunk_val,
                *(shared_cells_multirow if version_idx == 0 else empty_shared),
            ]
            lines.append("            " + " & ".join(cells) + r" \\")

        if stat_idx < len(STATS_ORDER) - 1:
            lines.append(r"            \addlinespace")
    return lines


def build_table(data: dict) -> str:
    """Build the full ``table*`` environment."""
    n_books = data["n_books"]
    n_chunks = int(data["chunks_per_book"]["sum"])

    # Header rows.
    n_responses = {field: comment_n(field, data) for field in COMMENT_FIELDS}

    header_section_row = (
        rf"            & & & \multicolumn{{1}}{{c}}{{\textbf{{Books}} {BOOKS_ICON}}}"
        rf" & \multicolumn{{1}}{{c}}{{\textbf{{Chunks}} {CHUNKS_ICON}}}"
        r" & "
        rf" & \multicolumn{{5}}{{c}}{{\textbf{{Participant Comments}} {COMMENTS_ICON}}} \\"
    )
    header_n_row = (
        rf"            & & & \multicolumn{{1}}{{c}}{{(\textit{{n={n_books}}})}}"
        rf" & \multicolumn{{1}}{{c}}{{(\textit{{n={n_chunks}}})}}"
        r" & "
        + " ".join(
            f"& \\multicolumn{{1}}{{c}}{{(\\textit{{n={n_responses[field]}}})}}"
            for field in COMMENT_FIELDS
        )
        + r" \\"
    )
    cmidrules = (
        r"            \cmidrule(lr){4-4}\cmidrule(lr){5-5}"
        r"\cmidrule(lr){6-6}\cmidrule(lr){7-11}"
    )
    comment_headers = " & ".join(COMMENT_HEADERS[f] for f in COMMENT_FIELDS)
    header_col_row = (
        r"            & & &     &     & \textsc{\#/Bk}"
        rf" & {comment_headers} \\"
    )

    body: list[str] = []
    body.extend(build_section_rows("word", data, versions=WORD_VERSIONS))
    body.append(r"            \midrule")
    body.extend(build_section_rows("token", data))

    column_spec = "lll cc c ccccc"
    caption = (
        r"Summary statistics for evaluation books, chunks, and participant "
        r"free-text comments, split into a \textbf{\textsc{Words}} sub-table "
        r"(whitespace-delimited word counts; we do not report \srcr{} word "
        r"counts because Japanese cannot be split on whitespaces) "
        r"and a \textbf{\textsc{Tokens}} "
        r"sub-table (\texttt{tiktoken} \texttt{o200k\_base} token counts). "
        r"Books and chunks are reported separately for human-translated "
        r"(\htr) and machine-translated (\mtr) versions in the "
        r"\textbf{\textsc{Words}} sub-table; the \textbf{\textsc{Tokens}} "
        r"sub-table also includes source-language (\srcr) versions. "
        r"The \textsc{\#/Bk} (chunks per book) column does not depend on the "
        r"metric and is identical across both sub-tables (and across "
        r"\htr ~/~ \mtr ~/~ \srcr{}, since chunks are aligned across versions). "
        r"\textsc{Participant Comments} include: "
        r"\textit{Single Q5/Q6} are free-text responses from the single-reading questionnaires; "
        r"\textit{Compar.\ Q4/Q7} are free-text responses from the comparison questionnaire; "
        r"\textit{Chunk Justif.}\ are per-chunk preference justifications."
    )

    lines = [
        r"\begin{table*}[t!]",
        r"    \centering",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{5pt}",
        r"    \resizebox{\textwidth}{!}{%",
        f"        \\begin{{tabular}}{{{column_spec}}}",
        r"            \toprule",
        header_section_row,
        header_n_row,
        cmidrules,
        header_col_row,
        r"            \midrule",
        *body,
        r"            \bottomrule",
        r"        \end{tabular}",
        r"    }",
        f"    \\caption{{{caption}}}",
        r"    \label{tab:basic_summary_stats_dataset}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def main() -> None:
    data = load_all_stats()
    output = build_table(data) + "\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(output)
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
