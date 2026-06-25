#!/usr/bin/env python3
"""Generate LaTeX annotation-scheme table from coding_Q1_Q2_Q3_Q5.csv."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT = Path("human_eval/annotations/coding_Q1_Q2_Q3_Q5.csv")
DEFAULT_OUTPUT = Path("analysis/manuscript_tables/tex/annotation_scheme_Q1_Q2_Q3_Q5.tex")

CATEGORY_RE = re.compile(r"^([A-D])\.\s+(.+)$")
CODE_RE = re.compile(r"^([A-D]\d+[a-z]?)\.\s*(.+)$")

# Abbreviations that should be followed by \@ before a space (line-break control).
ABBREV_FOLLOW = ("e.g.", "i.e.", "etc.", "vs.", "cf.")

# Extra vertical space after each data row (in addition to \\arraystretch).
ROW_END = r"\\[5pt]"
ARRAY_STRETCH = "1.25"


@dataclass
class CodeRow:
    code: str
    aspect: str
    definition: str
    positive: str
    negative: str


@dataclass
class Category:
    letter: str
    title: str
    codes: list[CodeRow]


def latex_escape(text: str) -> str:
    """Escape text for LaTeX tabular cells (not math mode)."""
    if not text:
        return ""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\^{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    # Straight double quotes -> LaTeX quotes
    text = re.sub(r'"([^"]*)"', r"``\1''", text)
    # Remaining straight quotes (odd count)
    text = text.replace('"', "''")
    # Unicode dashes / ellipsis
    text = text.replace("\u2014", "---")
    text = text.replace("\u2013", "--")
    text = text.replace("\u2026", r"\ldots{}")
    text = text.replace("→", r"$\rightarrow$")
    text = text.replace("…", r"\ldots{}")
    # clichéd etc.
    text = text.replace("clichéd", r"clich\'{e}d")
    text = text.replace("cliché", r"clich\'{e}")
    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def add_abbrev_line_breaks(text: str) -> str:
    for abbrev in ABBREV_FOLLOW:
        text = text.replace(abbrev + " ", abbrev + r"\ ")
    return text


def polarity(positive: str, negative: str) -> str:
    pos = positive.strip().lower()
    neg = negative.strip().lower()
    has_pos = pos == "positive"
    has_neg = neg == "negative"
    if has_pos and has_neg:
        return r"$+/-$"
    if has_neg or pos == "negative":
        return r"$-$"
    if has_pos:
        return r"$+$"
    return ""


def parse_csv(path: Path) -> tuple[list[Category], list[str]]:
    categories: list[Category] = []
    notes: list[str] = []
    current: Category | None = None

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {path}")

        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue

            first = row[0].strip()
            if not first:
                continue

            if first.upper().startswith("NOTE"):
                continue

            if first.startswith("For Q"):
                notes.append(first)
                continue

            examples = row[1].strip() if len(row) > 1 else ""
            definition = row[2].strip() if len(row) > 2 else ""
            positive = row[3].strip() if len(row) > 3 else ""
            negative = row[4].strip() if len(row) > 4 else ""

            cat_match = CATEGORY_RE.match(first)
            if cat_match and not CODE_RE.match(first):
                if current is not None:
                    categories.append(current)
                letter, title = cat_match.group(1), cat_match.group(2).strip()
                current = Category(letter=letter, title=title, codes=[])
                continue

            code_match = CODE_RE.match(first)
            if code_match is None:
                continue
            if current is None:
                raise ValueError(f"Code row before any category: {first!r}")

            code, aspect = code_match.group(1), code_match.group(2).strip()
            if not definition and examples:
                definition = examples
            current.codes.append(
                CodeRow(
                    code=code,
                    aspect=aspect,
                    definition=definition,
                    positive=positive,
                    negative=negative,
                )
            )

    if current is not None:
        categories.append(current)

    return categories, notes


def build_caption(notes: list[str], csv_name: str) -> str:
    base = (
        f"Annotation scheme for participant comments on Single Q5, Q6, Compar. Q4, Q7, and Chunk Justif. "
    )
    if not notes:
        return base
    note_text = "; ".join(n.strip().rstrip(".") for n in notes) + "."
    return f"{base} {latex_escape(note_text)}"


def render_category_block(cat: Category, *, cat_col_width: str = "0.13") -> list[str]:
    if not cat.codes:
        return []

    lines: list[str] = []
    cat_cell = (
        f"\\parbox[t]{{{cat_col_width}\\textwidth}}"
        f"{{\\codelabel{{{cat.letter}}}.\\\\{latex_escape(cat.title)}}}" # \codelabel should be defined in main latex file
    )

    for i, code in enumerate(cat.codes):
        definition = add_abbrev_line_breaks(latex_escape(code.definition))
        pol = polarity(code.positive, code.negative)
        aspect = latex_escape(code.aspect)
        category_col = cat_cell if i == 0 else ""

        lines.append(
            f"        {category_col} & \\codelabel{{{code.code}}} " # \codelabel should be defined in main latex file
            f"& {aspect} &\n"
            f"        {definition} &\n"
            f"        {pol} {ROW_END}"
        )
    return lines


TABLE_HEADER = (
    r"        \toprule"
    "\n"
    r"        \textbf{Category} & \textbf{Code} & \textbf{Aspect} & "
    r"\textbf{Definition} & \textbf{Pol.} \\"
    "\n"
    r"        \midrule"
)
TABLE_CONTINUED_HEAD = (
    r"        \multicolumn{5}{c}{\textbf{Table \thetable{} -- continued from previous page}} \\"
    "\n"
    + TABLE_HEADER
)
TABLE_CONTINUED_FOOT = (
    r"        \midrule"
    "\n"
    r"        \multicolumn{5}{r}{\textit{Continued on next page}} \\"
    "\n"
    r"        \midrule"
)
TABLE_LAST_FOOT = (
    r"        \bottomrule"
    "\n"
    r"        \noalign{\vskip 0.75em}"
    "\n"
    r"        \caption{CPTN_PLACEHOLDER}"
    "\n"
    r"        \label{tab:annotation-scheme-q1-q2-q3-q5} \\"
)


def render_tex(
    categories: list[Category],
    notes: list[str],
    *,
    csv_name: str,
) -> str:
    # caption = build_caption(notes, csv_name)
    caption = (
        "Annotation scheme for participant comments on Single Q5, Q6, Compar. Q4, and Chunk Justif. "
        "Only positive codes are applied to comments for Single Q5, and only negative codes are applied for Single Q6. "
        "For Compar. Q4 and Chunk Justif., positive codes are applied to match the comments' positive aspects for the preferred translation, "
        "and negative codes applied to match the negative aspects of the non-preferred translation."
    )
    body_lines: list[str] = []

    for i, cat in enumerate(categories):
        if i > 0:
            body_lines.append("        \\midrule")
        body_lines.extend(render_category_block(cat))

    body = "\n".join(body_lines)

    last_foot = TABLE_LAST_FOOT.replace("CPTN_PLACEHOLDER", caption)

    return (
        "% Auto-generated by build_annotation_scheme_table.py; do not edit by hand.\n"
        "% Requires: booktabs, longtable, array\n"
        "%\n"
        "% \\input{analysis/manuscript_tables/tex/annotation_scheme_Q1_Q2_Q3_Q5.tex}\n"
        "%\n"
        "% Reference examples: \\ref{A}, \\ref{B3}, \\ref{D4a}\n"
        "\n"
        "{\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        f"\\renewcommand{{\\arraystretch}}{{{ARRAY_STRETCH}}}\n"
        "\\onecolumn"
        "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}"
        "p{0.13\\textwidth} >{\\raggedright\\arraybackslash}"
        "p{0.06\\textwidth} >{\\raggedright\\arraybackslash}"
        "p{0.17\\textwidth} >{\\raggedright\\arraybackslash}"
        "p{0.56\\textwidth} c@{}}\n"
        f"{TABLE_HEADER}\n"
        "\\endfirsthead\n"
        f"{TABLE_CONTINUED_HEAD}\n"
        "\\endhead\n"
        f"{TABLE_CONTINUED_FOOT}\n"
        "\\endfoot\n"
        f"{last_foot}\n"
        "\\endlastfoot\n"
        f"{body}\n"
        "\\end{longtable}\n"
        "\\twocolumn"
        "}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX annotation scheme table from coding CSV."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input coding CSV (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .tex file (default: {DEFAULT_OUTPUT.name})",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1

    categories, notes = parse_csv(args.input)
    tex = render_tex(categories, notes, csv_name=args.input.name)
    args.output.write_text(tex, encoding="utf-8")
    n_codes = sum(len(c.codes) for c in categories)
    print(f"Wrote {args.output} ({len(categories)} categories, {n_codes} codes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
