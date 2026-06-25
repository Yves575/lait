#!/usr/bin/env python3
"""Convert human-eval CLM/CLMM text outputs into manuscript LaTeX tables.

Reads R ordinal-model summary files under
``human_eval/analysis_outputs/part_1/single_reading/ordinal/`` and writes
``longtable`` snippets under ``analysis/manuscript_tables/tex/`` (default:
four primary appendix tables ``single_q*_clm*.tex``).

Everything from the ``Model summary`` line through the end of the file is rendered.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from clmm_table_config import (  # noqa: E402
    DEFAULT_ORDINAL_INPUT_DIR,
    HTR_MACRO,
    MAIN_STATS_TABLE_LABEL,
    MTR_MACRO,
    PRIMARY_APPENDIX_TABLES,
    QUESTION_CAPTION_META,
    QuestionCaptionMeta,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_GLOB = f"{DEFAULT_ORDINAL_INPUT_DIR}/q*_*_clm*.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "manuscript_tables" / "tables" / "tex"

MODEL_SUMMARY_MARKER = "Model summary"
KEY_VALUE_RE = re.compile(r"^(formula|data)\s*:\s*(.*)$", re.IGNORECASE)
CI_HEADER_RE = re.compile(r"([\d.]+\s*%)\s+([\d.]+\s*%)")
SIGNIF_STAR_RE = re.compile(r"^(\*\*\*|\*\*|\*)$")
RESPONSE_LEVEL_RE = re.compile(r"^q\d+\s*=\s*\d+:\s*$", re.IGNORECASE)
Q_FROM_STEM_RE = re.compile(r"(?:^single_)?(q\d+).*(clmm|clm)$", re.IGNORECASE)


@dataclass
class ProseBlock:
    kind: Literal["prose"] = "prose"
    start_line: int = 0
    lines: list[str] = field(default_factory=list)


@dataclass
class KeyValueBlock:
    kind: Literal["kv"] = "kv"
    start_line: int = 0
    key: str = ""
    value: str = ""


@dataclass
class TableBlock:
    kind: Literal["table"] = "table"
    title: str | None = None
    title_line: int | None = None
    table_start_line: int = 0
    rows: list[list[str]] = field(default_factory=list)
    trailing: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class OrdinalModelReport:
    question_id: str
    question_label: str
    question_title: str
    model_kind: Literal["clm", "clmm"]
    source_lines: list[str]
    blocks: list[ProseBlock | KeyValueBlock | TableBlock]
    source_name: str = ""
    output_stem: str = ""


ContentBlock = ProseBlock | KeyValueBlock | TableBlock


VSKIP_2MM = r"            \noalign{\vskip 2mm}\\"


def paragraph_gap_before(source_lines: list[str], line_index: int) -> bool:
    """True when a blank source line precedes ``line_index`` (paragraph break)."""
    if line_index <= 0:
        return False
    return not source_lines[line_index - 1].strip()


def emit_row(out: list[str], source_lines: list[str], line_index: int, row: str) -> None:
    if paragraph_gap_before(source_lines, line_index):
        out.append(VSKIP_2MM)
    out.append(row)


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
    text = re.sub(r"\s*~\s*", r" \\textasciitilde~ ", text)
    return text


def latex_escape_caption(text: str) -> str:
    """Escape plain text for use inside \\caption{...}."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("_", r"\_"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def split_ws_row(line: str) -> list[str]:
    return line.split()


def is_prose_only_line(line: str) -> bool:
    stripped = line.strip()
    if RESPONSE_LEVEL_RE.match(stripped):
        return True
    if stripped.startswith("Results are averaged over the levels of:"):
        return True
    if stripped.startswith("Confidence level used:"):
        return True
    if stripped.startswith("Cumulative Link Mixed Model"):
        return True
    return False


def is_table_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if is_prose_only_line(stripped):
        return False
    if stripped == "---":
        return False
    lowered = stripped.lower()
    if "estimate" in lowered and "error" in lowered:
        return True
    if lowered.startswith("groups "):
        return True
    if lowered.startswith("link ") or re.match(r"^logit\b", lowered):
        return True
    if lowered.startswith("type ") and ("mean.class" in lowered or "prob" in lowered):
        return True
    if CI_HEADER_RE.search(stripped):
        return True
    parts = split_ws_row(stripped)
    if len(parts) >= 2 and (
        re.search(r"^-?\d", parts[-1])
        or re.match(r"^\d+\|\d+", parts[0])
        or parts[0] in {"book", "reader", "typeMT", "orderMT-first", "HT", "MT"}
    ):
        return True
    return False


def looks_like_table_header(line: str) -> bool:
    lowered = line.lower()
    if "estimate" in lowered or lowered.startswith("groups"):
        return True
    if lowered.startswith("link ") or ("threshold" in lowered and "nobs" in lowered):
        return True
    if lowered.startswith("type ") and ("mean.class" in lowered or "prob" in lowered):
        return True
    if CI_HEADER_RE.search(line):
        return True
    return False


def merge_trailing_signif(parts: list[str]) -> list[str]:
    if len(parts) >= 2 and SIGNIF_STAR_RE.match(parts[-1]):
        parts = parts[:-2] + [f"{parts[-2]} {parts[-1]}"]
    return parts


def normalize_table_rows(raw_lines: list[str], title: str | None) -> list[list[str]]:
    """Normalize R summary tables whose columns use inconsistent spacing."""
    lines = [ln for ln in raw_lines if ln.strip()]
    if not lines:
        return []

    first = lines[0]
    title_text = title or ""

    if "max.grad" in first and "cond.H" in first:
        return [split_ws_row(ln) for ln in lines]

    if first.startswith("Groups "):
        return [split_ws_row(ln) for ln in lines]

    if "Estimate" in first and "Std." in first and "Pr(" in first:
        rows: list[list[str]] = [["", "Estimate", "Std. Error", "z value", "Pr(>|z|)"]]
        for ln in lines[1:]:
            rows.append(merge_trailing_signif(split_ws_row(ln)))
        return rows

    if title_text.startswith("Threshold coefficients"):
        rows = [["", "Estimate", "Std. Error", "z value"]]
        for ln in lines[1:]:
            rows.append(split_ws_row(ln))
        return rows

    if CI_HEADER_RE.search(first):
        match = CI_HEADER_RE.search(first)
        assert match is not None
        rows: list[list[str]] = [["", match.group(1).strip(), match.group(2).strip()]]
        for ln in lines[1:]:
            rows.append(split_ws_row(ln))
        return rows

    return [split_ws_row(ln) for ln in lines]


def is_section_title(line: str) -> bool:
    stripped = line.strip()
    if stripped == MODEL_SUMMARY_MARKER:
        return True
    if stripped in {
        "Random effects:",
        "Coefficients:",
        "Threshold coefficients:",
        "Confidence intervals",
        "Model warnings",
    }:
        return True
    if stripped.startswith("Estimated marginal means"):
        return True
    return False


def is_non_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped == "---":
        return True
    if stripped.startswith("Signif. codes:"):
        return True
    if stripped.startswith("Number of groups:"):
        return True
    if KEY_VALUE_RE.match(stripped):
        return True
    if is_section_title(stripped):
        return True
    return False


def question_meta_from_path(path: Path) -> tuple[str, str, str, Literal["clm", "clmm"]]:
    stem_match = Q_FROM_STEM_RE.search(path.stem)
    if stem_match:
        question_id = stem_match.group(1).lower()
        model_kind: Literal["clm", "clmm"] = stem_match.group(2).lower()  # type: ignore[assignment]
    else:
        question_id = "q?"
        model_kind = "clmm"
    meta = QUESTION_CAPTION_META.get(question_id)
    if meta:
        return question_id, meta.title, meta.slug, model_kind
    return question_id, question_id.upper(), question_id, model_kind


def parse_nobs(blocks: list[ContentBlock]) -> int | None:
    """Extract observation count from the model-fit summary row, if present."""
    for block in blocks:
        if not isinstance(block, TableBlock) or not block.rows:
            continue
        header = [cell.lower() for cell in block.rows[0]]
        if "nobs" not in header:
            continue
        nobs_idx = header.index("nobs")
        for row in block.rows[1:]:
            if len(row) > nobs_idx:
                try:
                    return int(float(row[nobs_idx]))
                except ValueError:
                    continue
    return None


def is_primary_appendix_table(question_id: str, model_kind: Literal["clm", "clmm"]) -> bool:
    meta = QUESTION_CAPTION_META.get(question_id)
    return meta is not None and meta.primary_model == model_kind


def build_caption(report: OrdinalModelReport) -> tuple[str, str]:
    """Return (short, long) captions for EMNLP appendix longtables."""
    meta = QUESTION_CAPTION_META.get(report.question_id)
    model_tag = report.model_kind.upper()

    if meta is None or not is_primary_appendix_table(report.question_id, report.model_kind):
        short = f"{report.question_title} {model_tag} results"
        long = f"{report.question_title} {model_tag} results."
        return short, long

    nobs = parse_nobs(report.blocks)
    n_text = f"$N={nobs}$" if nobs is not None else "$N=60$"
    return _build_primary_caption(meta, model_tag, n_text, report)


def _formula_latex(response: str, *, mixed: bool) -> str:
    base = (
        f"\\texttt{{{response}}} $\\sim$ \\texttt{{type}} + \\texttt{{order}}"
    )
    if mixed:
        return f"{base} + \\texttt{{(1|reader)}} + \\texttt{{(1|book)}}"
    return base


def _type_contrast_phrase() -> str:
    return (
        f"translation \\texttt{{type}} ({HTR_MACRO} vs.\\ {MTR_MACRO}, "
        f"with {HTR_MACRO} as reference)"
    )


def _order_contrast_phrase() -> str:
    return (
        f"reading \\texttt{{order}} ({HTR_MACRO}-first vs.\\ {MTR_MACRO}-first)"
    )


def _build_primary_caption(
    meta: QuestionCaptionMeta,
    model_tag: str,
    n_text: str,
    report: OrdinalModelReport,
) -> tuple[str, str]:
    short = f"{meta.title} ({model_tag})"

    scale = f"1 = {meta.scale_low}, 5 = {meta.scale_high}"
    response = report.question_id
    response_tt = f"\\texttt{{{response}}}"
    formula = _formula_latex(response, mixed=report.model_kind == "clmm")

    if report.model_kind == "clmm":
        model_family = "ordinal cumulative link mixed model"
        formula_tail = (
            f"and random intercepts for each \\texttt{{reader}} and \\texttt{{book}} "
            "(because each participant rates two excerpts and each book appears twice)."
        )
    else:
        model_family = "ordinal cumulative link model"
        formula_tail = (
            "without random effects because the CLMM fit was singular."
        )

    formula_meaning = (
        f"In this formula, the {response_tt} Likert rating is modeled as a function of "
        f"{_type_contrast_phrase()}, {_order_contrast_phrase()}, {formula_tail} "
    )

    model_note = (
        f"Full \\texttt{{summary()}} from an {model_family} "
        f"(\\textbf{{{model_tag}}}; {formula}). {formula_meaning}"
    )

    long = (
        f"\\textbf{{{latex_escape_caption(meta.title)}}}, Part~1 isolated reading "
        f"({scale}; {n_text}). "
        f"{model_note}"
        f"A negative \\texttt{{typeMT}} coefficient means {MTR_MACRO} was rated lower "
        f"than {HTR_MACRO}. "
        "Summary OR (odds ratio) and 95\\% CI (confidence interval) displayed in "
        f"\\autoref{{{MAIN_STATS_TABLE_LABEL}}}. "
        f"{meta.result_blurb}"
    )
    return short, long


def lines_after_model_summary(lines: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        if line.strip() == MODEL_SUMMARY_MARKER:
            return lines[i:]
    raise ValueError(f'Line "{MODEL_SUMMARY_MARKER}" not found')


def parse_table_run(lines: list[str], start: int) -> tuple[TableBlock | None, int]:
    """Parse consecutive table rows starting at index ``start``."""
    if start >= len(lines):
        return None, start

    title: str | None = None
    title_line: int | None = None
    table_start_line = start
    i = start
    first = lines[i].strip()

    if is_section_title(first) and first != MODEL_SUMMARY_MARKER:
        title = first
        title_line = i
        i += 1
        table_start_line = i

    if i >= len(lines):
        return TableBlock(title=title, title_line=title_line, table_start_line=table_start_line), i

    table_rows: list[str] = []
    trailing: list[tuple[int, str]] = []

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            break
        if is_section_title(stripped) and not table_rows:
            break
        if is_section_title(stripped) and table_rows:
            break

        kv_match = KEY_VALUE_RE.match(stripped)
        if kv_match:
            break

        if stripped == "---" or stripped.startswith("Signif. codes:"):
            break

        if stripped.startswith("Number of groups:"):
            trailing.append((i, stripped))
            i += 1
            break

        if is_prose_only_line(stripped):
            break

        if is_non_table_line(stripped) and not is_table_row(stripped):
            break

        if not table_rows and looks_like_table_header(stripped):
            table_start_line = i
            table_rows.append(stripped)
            i += 1
            continue

        if table_rows or is_table_row(stripped):
            if not table_rows:
                table_start_line = i
            table_rows.append(stripped)
            i += 1
            continue
        break

    if not table_rows:
        return None, start

    normalized = normalize_table_rows(table_rows, title)
    return TableBlock(
        title=title,
        title_line=title_line,
        table_start_line=table_start_line,
        rows=normalized,
        trailing=trailing,
    ), i


def parse_post_model_summary(lines: list[str]) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    i = 0
    n = len(lines)

    while i < n:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break

        stripped = lines[i].strip()

        if stripped == MODEL_SUMMARY_MARKER:
            blocks.append(ProseBlock(start_line=i, lines=[stripped]))
            i += 1
            continue

        kv_match = KEY_VALUE_RE.match(stripped)
        if kv_match:
            blocks.append(
                KeyValueBlock(
                    start_line=i,
                    key=kv_match.group(1),
                    value=kv_match.group(2).strip(),
                )
            )
            i += 1
            continue

        if stripped == "---" or stripped.startswith("Signif. codes:"):
            blocks.append(ProseBlock(start_line=i, lines=[stripped]))
            i += 1
            continue

        if is_section_title(stripped):
            if i + 1 < n and (
                is_table_row(lines[i + 1]) or looks_like_table_header(lines[i + 1])
            ):
                table_block, next_i = parse_table_run(lines, i)
                if table_block is not None:
                    blocks.append(table_block)
                    i = next_i
                    continue
            blocks.append(ProseBlock(start_line=i, lines=[stripped]))
            i += 1
            continue

        if is_table_row(stripped) or looks_like_table_header(stripped):
            table_block, next_i = parse_table_run(lines, i)
            if table_block is not None:
                blocks.append(table_block)
                i = next_i
                continue

        prose_start = i
        prose: list[str] = []
        while i < n:
            line = lines[i]
            s = line.strip()
            if not s:
                break
            if (
                is_section_title(s)
                or KEY_VALUE_RE.match(s)
                or s == "---"
                or s.startswith("Signif. codes:")
                or (is_table_row(s) and prose)
            ):
                break
            if is_table_row(s) and not prose:
                break
            prose.append(s)
            i += 1
        if prose:
            blocks.append(ProseBlock(start_line=prose_start, lines=prose))
        else:
            i += 1

    return blocks


def parse_ordinal_model(path: Path, *, output_stem: str = "") -> OrdinalModelReport:
    lines = path.read_text(encoding="utf-8").splitlines()
    question_id, question_title, slug, model_kind = question_meta_from_path(path)
    post_summary = lines_after_model_summary(lines)
    blocks = parse_post_model_summary(post_summary)
    if not blocks:
        raise ValueError(f"No content found after {MODEL_SUMMARY_MARKER!r} in {path}")
    stem = output_stem or path.stem
    return OrdinalModelReport(
        question_id=question_id,
        question_label=slug,
        question_title=question_title,
        model_kind=model_kind,
        source_lines=post_summary,
        blocks=blocks,
        source_name=path.name,
        output_stem=stem,
    )


def render_table_block(block: TableBlock, source_lines: list[str]) -> list[str]:
    out: list[str] = []
    if block.title and block.title_line is not None:
        title_tt = latex_escape_tt(block.title)
        emit_row(
            out,
            source_lines,
            block.title_line,
            f"            & \\texttt{{{title_tt}}}\\\\",
        )

    if not block.rows:
        return out

    ncol = max(len(row) for row in block.rows)
    col_spec = "l" * ncol

    def format_cell(cell: str) -> str:
        if not cell:
            return ""
        return rf"\texttt{{{latex_escape_tt(cell)}}}"

    header_cells = [format_cell(h) for h in block.rows[0]]
    header_row = " & ".join(header_cells)
    data_rows = block.rows[1:]

    body_rows: list[str] = []
    for row in data_rows:
        padded = row + [""] * (ncol - len(row))
        cells = [format_cell(cell) for cell in padded]
        body_rows.append(" & ".join(cells))

    table_lines = [
        f"            & \\begin{{tabular}}{{{col_spec}}}",
        f"                {header_row} \\\\",
        *[f"                {row} \\\\" for row in body_rows],
        r"            \end{tabular} \\",
    ]
    table_row = "\n".join(table_lines)
    emit_row(out, source_lines, block.table_start_line, table_row)

    for line_index, text in block.trailing:
        emit_row(
            out,
            source_lines,
            line_index,
            f"            & \\texttt{{{latex_escape_tt(text)}}}\\\\",
        )

    return out


def render_blocks(blocks: list[ContentBlock], source_lines: list[str]) -> list[str]:
    out: list[str] = []

    for block in blocks:
        if isinstance(block, ProseBlock):
            for offset, line in enumerate(block.lines):
                line_index = block.start_line + offset
                emit_row(
                    out,
                    source_lines,
                    line_index,
                    f"            & \\texttt{{{latex_escape_tt(line)}}}\\\\",
                )

        elif isinstance(block, KeyValueBlock):
            text = f"{block.key}: {block.value}"
            emit_row(
                out,
                source_lines,
                block.start_line,
                f"            & \\texttt{{{latex_escape_tt(text)}}}\\\\",
            )

        elif isinstance(block, TableBlock):
            out.extend(render_table_block(block, source_lines))

    return out


def longtable_title_row(title: str) -> str:
    return rf"        & \multicolumn{{1}}{{@{{}}l}}{{\bf {title}}} \\"


def render_tex(report: OrdinalModelReport) -> str:
    model_tag = report.model_kind.upper()
    title = f"{report.question_title} {model_tag} Model Summary"
    short_caption, long_caption = build_caption(report)
    label = f"tab:{report.question_id}_{report.question_label}_{report.model_kind}"
    tex_stem = report.output_stem or Path(report.source_name).stem

    table_header = (
        r"        \toprule" + "\n" + longtable_title_row(title) + "\n" + r"        \midrule"
    )
    continued_head = (
        r"        \multicolumn{2}{@{}l}{\textbf{Table \thetable{} -- continued from previous page}} \\"
        + "\n"
        + table_header
    )
    continued_foot = (
        r"        \midrule"
        "\n"
        r"        \multicolumn{2}{@{}r}{\textit{Continued on next page}} \\"
    )
    last_foot = (
        r"        \bottomrule"
        "\n"
        r"        \noalign{\vskip 2mm}"
        "\n"
        f"        \\caption[{short_caption}]{{{long_caption}}}"
        "\n"
        f"        \\label{{{label}}} \\\\"
    )

    body_lines = [
        r"\onecolumn",
        r"{\tiny",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \begin{longtable}{@{}l >{\raggedright\arraybackslash}p{\linewidth}@{}}",
        table_header,
        r"    \endfirsthead",
        continued_head,
        r"    \endhead",
        continued_foot,
        r"    \endfoot",
        last_foot,
        r"    \endlastfoot",
        *render_blocks(report.blocks, report.source_lines),
        r"    \end{longtable}",
        r"}",
        r"\twocolumn",
        "",
    ]

    header = (
        "% Auto-generated by build_clmm_latex_table.py; do not edit by hand.\n"
        f"% Source: {report.source_name}\n"
        "% Requires: booktabs, longtable, array; hyperref (or cleveref) for \\autoref\n"
        "%\n"
        f"% \\input{{analysis/manuscript_tables/tex/{tex_stem}.tex}}\n"
        "\n"
    )
    return header + "\n".join(body_lines)


def resolve_inputs(pattern: str) -> list[Path]:
    paths = sorted(REPO_ROOT.glob(pattern))
    return [p for p in paths if p.is_file()]


def resolve_primary_inputs(ordinal_dir: Path) -> list[tuple[Path, str]]:
    """Return (input path, output stem) for the four primary appendix tables."""
    resolved: list[tuple[Path, str]] = []
    for _qid, _model, source_name, output_stem in PRIMARY_APPENDIX_TABLES:
        path = ordinal_dir / source_name
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved.append((path, output_stem))
    return resolved


def default_output_path(input_path: Path, output_dir: Path, output_stem: str = "") -> Path:
    stem = output_stem or input_path.stem
    return output_dir / f"{stem}.tex"


def output_stem_for_input(path: Path) -> str:
    for _qid, _model, source_name, output_stem in PRIMARY_APPENDIX_TABLES:
        if path.name == source_name or path.stem == output_stem:
            return output_stem
    return path.stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert CLM/CLMM text output into a manuscript LaTeX table."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        action="append",
        help=(
            "CLM/CLMM .txt file (repeatable). Default: four primary tables from "
            f"{DEFAULT_ORDINAL_INPUT_DIR}/"
        ),
    )
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help=f"Glob for inputs when --input is omitted (default: {DEFAULT_INPUT_GLOB})",
    )
    parser.add_argument(
        "--primary-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process only the four primary appendix tables (default: true)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .tex path (only valid with a single --input)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for batch mode (default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--ordinal-dir",
        type=Path,
        default=None,
        help=f"Directory with q*_*_clm*.txt files (default: {DEFAULT_ORDINAL_INPUT_DIR})",
    )
    args = parser.parse_args(argv)

    ordinal_dir = args.ordinal_dir
    if ordinal_dir is None:
        ordinal_dir = REPO_ROOT / DEFAULT_ORDINAL_INPUT_DIR
    elif not ordinal_dir.is_absolute():
        ordinal_dir = REPO_ROOT / ordinal_dir

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        inputs = [p if p.is_absolute() else REPO_ROOT / p for p in args.input]
        input_jobs = [(p, output_stem_for_input(p)) for p in inputs]
    elif args.primary_only:
        try:
            input_jobs = resolve_primary_inputs(ordinal_dir)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        paths = resolve_inputs(args.input_glob)
        if not paths:
            print("error: no CLM/CLMM input files found", file=sys.stderr)
            return 1
        input_jobs = [(p, p.stem) for p in paths]

    if not input_jobs:
        print("error: no CLM/CLMM input files found", file=sys.stderr)
        return 1

    if args.output is not None and len(input_jobs) != 1:
        print("error: --output requires exactly one --input", file=sys.stderr)
        return 1

    for input_path, out_stem in input_jobs:
        if not input_path.is_file():
            print(f"error: input not found: {input_path}", file=sys.stderr)
            return 1

        report = parse_ordinal_model(input_path, output_stem=out_stem)
        tex = render_tex(report)

        if args.output is not None:
            out_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        else:
            out_path = default_output_path(input_path, output_dir, out_stem)

        out_path.write_text(tex, encoding="utf-8")
        print(
            f"Wrote {out_path.relative_to(REPO_ROOT)} "
            f"({report.question_id} {report.model_kind}, {len(report.blocks)} blocks)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
