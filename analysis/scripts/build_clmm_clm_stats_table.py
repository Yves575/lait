#!/usr/bin/env python3
"""Build the isolated reading questionnaire CLM/CLMM stats LaTeX table.

Reads ``human_eval/analysis_outputs/part_1/single_reading/ordinal/q*_*_{clmm,clm}.txt``
and writes a compact ``table*`` with type (MT vs HT) fixed-effect estimates.
A single monospaced banner above the column headers lists the ``clmm`` and ``clm``
R calls (with ``q*`` placeholders). Question-to-``q*`` mapping is given in the caption.

Model choice per row matches the paper table (CLM fallback for q2/q3 when the
CLMM is singular; see ``human_eval/generate_analysis_summaries.R``).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from clmm_table_config import (  # noqa: E402
    CLM_FALLBACK_ON_SINGULAR,
    QUESTION_STEMS,
    TABLE_ROWS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "analysis" / "manuscript_tables" / "tables" / "tex" / "clmm_clm_stats_results.tex"
)
DEFAULT_INPUT_DIR = REPO_ROOT / "human_eval" / "analysis_outputs" / "part_1" / "single_reading" / "ordinal"

SINGULAR_FIT_RE = re.compile(r"SINGULAR FIT", re.IGNORECASE)
N_TABLE_COLS = 8

CLMM_R_CALL = "clmm(q* ~ type + order + (1 | reader) + (1 | book), data = model_data, Hess = TRUE)"
CLM_R_CALL = "clm(q* ~ type + order, data = model_data, Hess = TRUE)"


@dataclass(frozen=True)
class TypeEffect:
    estimate: float
    std_error: float
    z_value: float
    p_value: float
    ci_lower: float
    ci_upper: float

    @property
    def odds_ratio(self) -> float:
        return math.exp(self.estimate)

    @property
    def or_ci_lower(self) -> float:
        return math.exp(self.ci_lower)

    @property
    def or_ci_upper(self) -> float:
        return math.exp(self.ci_upper)


def parse_type_effect(path: Path) -> TypeEffect:
    if not path.is_file():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="utf-8").splitlines()

    coef_idx = next(i for i, line in enumerate(lines) if line.strip() == "Coefficients:")
    coef_lines = lines[coef_idx + 1 :]
    dash_idx = next((i for i, line in enumerate(coef_lines) if line.strip() == "---"), len(coef_lines))
    coef_lines = coef_lines[:dash_idx]

    type_row = next((ln for ln in coef_lines if ln.startswith("typeMT")), None)
    if type_row is None:
        raise ValueError(f"No typeMT coefficient in {path}")

    parts = type_row.split()
    if len(parts) < 5:
        raise ValueError(f"Malformed typeMT row in {path}: {type_row!r}")

    p_raw = re.sub(r"[^0-9.eE+-]", "", parts[4])
    estimate = float(parts[1])
    std_error = float(parts[2])
    z_value = float(parts[3])
    p_value = float(p_raw)

    ci_idx = next((i for i, line in enumerate(lines) if line.strip() == "Confidence intervals"), None)
    if ci_idx is None:
        raise ValueError(f"No Confidence intervals section in {path}")

    ci_row = next((ln for ln in lines[ci_idx + 1 :] if ln.startswith("typeMT")), None)
    if ci_row is None:
        raise ValueError(f"No typeMT confidence interval in {path}")

    ci_parts = ci_row.split()
    ci_lower = float(ci_parts[1])
    ci_upper = float(ci_parts[2])

    return TypeEffect(
        estimate=estimate,
        std_error=std_error,
        z_value=z_value,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def clmm_is_singular(path: Path) -> bool:
    if not path.is_file():
        return False
    return bool(SINGULAR_FIT_RE.search(path.read_text(encoding="utf-8")))


def latex_escape_tt(text: str) -> str:
    """Escape text for use inside \\texttt{...}."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("{", r"\{"),
        ("}", r"\}"),
        (">", r"\textgreater{}"),
        ("|", r"\textbar{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return re.sub(r"\s*~\s*", r" \\textasciitilde~ ", text)


def render_formula_banner() -> list[str]:
    """Shared CLMM/CLM R-call banner rows above the column header."""
    lines = [CLMM_R_CALL, CLM_R_CALL]
    return [
        f"        \\multicolumn{{{N_TABLE_COLS}}}{{@{{}}c@{{}}}}{{\\texttt{{{latex_escape_tt(line)}}}}} \\\\"
        for line in lines
    ]


def build_caption() -> str:
    return (
        r"Isolated reading questionnaire ratings statistical model results. "
        r"Questions are \texttt{q1} (acceptability), \texttt{q2} (smoothness), "
        r"\texttt{q3} (immersion), \texttt{q4} (would continue reading). "
        r"Rows use \texttt{clmm} for acceptability and would continue reading, "
        r"and \texttt{clm} for smoothness and immersion (singular CLMM fits for the latter). "
        r"OR (odds ratio) and 95\% CI (confidence interval) are for the MT vs HT fixed effect "
        r"(\texttt{typeMT}). Full model outputs are displayed in \S\ref{sec:stat-test-results}."
    )


def resolve_model_path(input_dir: Path, question_id: str, model_tag: str) -> Path:
    """Return the analysis output file for a table row."""
    stem = QUESTION_STEMS[question_id]
    clmm_path = input_dir / f"{stem}_clmm.txt"
    clm_path = input_dir / f"{stem}_clm.txt"

    if model_tag == "CLM":
        return clm_path
    if model_tag == "CLMM":
        if question_id in CLM_FALLBACK_ON_SINGULAR and clmm_is_singular(clmm_path):
            raise ValueError(
                f"{question_id}: CLMM is singular; table expects CLMM but analysis uses CLM "
                f"({clm_path.name})"
            )
        return clmm_path
    raise ValueError(f"Unknown model tag: {model_tag}")


def fmt_coef(x: float) -> str:
    return f"{x:.2f}"


def fmt_p(p: float) -> str:
    if p < 0.0001:
        return "< 0.0001"
    text = f"{round(p, 4):.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        return text
    return text


def fmt_or(x: float) -> str:
    return f"{x:.2f}"


def fmt_ci(low: float, high: float) -> str:
    return f"[{low:.2f}, {high:.2f}]"


def render_table_row(question: str, model: str, effect: TypeEffect) -> str:
    cells = [
        question,
        model,
        fmt_coef(effect.estimate),
        fmt_coef(effect.std_error),
        fmt_coef(effect.z_value),
        fmt_p(effect.p_value),
        fmt_or(effect.odds_ratio),
        fmt_ci(effect.or_ci_lower, effect.or_ci_upper),
    ]
    return "        " + " & ".join(cells) + r" \\"


def render_table(input_dir: Path) -> str:
    body_rows: list[str] = []
    sources: list[str] = []

    for question, question_id, model_tag in TABLE_ROWS:
        path = resolve_model_path(input_dir, question_id, model_tag)
        effect = parse_type_effect(path)
        body_rows.append(render_table_row(question, model_tag, effect))
        sources.append(path.name)

    source_comment = ", ".join(sources)
    lines = [
        "% Auto-generated by build_clmm_clm_stats_table.py; do not edit by hand.",
        f"% Sources: {source_comment}",
        "% Requires: booktabs",
        "%",
        f"% \\input{{analysis/manuscript_tables/tex/clmm_clm_stats_results.tex}}",
        "",
        r"\begin{table*}[h!]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{6pt}",
        r"    \begin{tabular}{llllllll}",
        r"        \toprule",
        *render_formula_banner(),
        r"        \midrule",
        r"        Question & Model & $\beta$ & SE & $z$ & $p$ & OR (MT vs HT) & 95\% CI \\",
        r"        \midrule",
        *body_rows,
        r"        \bottomrule",
        r"    \end{tabular}",
        f"    \\caption{{{build_caption()}}}",
        r"    \label{tab:clmm_clm_stats_results}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build LaTeX table of CLM/CLMM type (MT vs HT) fixed effects."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory with q*_*_{clmm,clm}.txt ordinal model outputs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .tex path",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir if args.input_dir.is_absolute() else REPO_ROOT / args.input_dir
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    try:
        tex = render_table(input_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")
    print(f"Wrote {output_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
