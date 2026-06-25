"""Shared config for isolated-reading CLM/CLMM LaTeX table builders.

Keep in sync with ``build_clmm_clm_stats_table.py`` and ``build_clmm_latex_table.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelKind = Literal["clm", "clmm"]
ModelTag = Literal["CLMM", "CLM"]

DEFAULT_ORDINAL_INPUT_DIR = (
    "human_eval/analysis_outputs/part_1/single_reading/ordinal"
)

# Basenames under the ordinal output directory (see human_eval/analysis_output_paths.R).
QUESTION_STEMS: dict[str, str] = {
    "q1": "q1_acceptability",
    "q2": "q2_smoothness",
    "q3": "q3_immersion",
    "q4": "q4_continue_reading",
}

# Paper row order: display name, question id, model tag shown in main stats table.
TABLE_ROWS: list[tuple[str, str, ModelTag]] = [
    ("Acceptability", "q1", "CLMM"),
    ("Smoothness", "q2", "CLM"),
    ("Immersion", "q3", "CLM"),
    ("Would Continue Reading", "q4", "CLMM"),
]

CLM_FALLBACK_ON_SINGULAR = frozenset({"q2", "q3"})

MAIN_STATS_TABLE_LABEL = "tab:clmm_clm_stats_results"


@dataclass(frozen=True)
class QuestionCaptionMeta:
    title: str
    slug: str
    scale_low: str
    scale_high: str
    primary_model: ModelKind
    result_blurb: str


# LaTeX macros for human vs machine translation (defined in the paper preamble).
HTR_MACRO = r"\htr{}"
MTR_MACRO = r"\mtr{}"

QUESTION_CAPTION_META: dict[str, QuestionCaptionMeta] = {
    "q1": QuestionCaptionMeta(
        title="Acceptability",
        slug="acceptability",
        scale_low="unacceptable",
        scale_high="acceptable",
        primary_model="clmm",
        result_blurb=(
            f"{HTR_MACRO} rated more acceptable than {MTR_MACRO} ($p < 0.01$)."
        ),
    ),
    "q2": QuestionCaptionMeta(
        title="Smoothness",
        slug="smoothness",
        scale_low="unsmooth",
        scale_high="smooth",
        primary_model="clm",
        result_blurb=f"{HTR_MACRO} rated smoother than {MTR_MACRO} ($p < 0.01$).",
    ),
    "q3": QuestionCaptionMeta(
        title="Immersion",
        slug="immersion",
        scale_low="interfered",
        scale_high="supported immersion",
        primary_model="clm",
        result_blurb=(
            f"{HTR_MACRO} rated higher on immersion than {MTR_MACRO}; "
            r"marginal at $\alpha = 0.05$ in the main table."
        ),
    ),
    "q4": QuestionCaptionMeta(
        title="Continue reading",
        slug="continue_reading",
        scale_low="would not continue",
        scale_high="would continue",
        primary_model="clmm",
        result_blurb=(
            f"No reliable difference between {HTR_MACRO} and {MTR_MACRO} ($p > 0.05$)."
        ),
    ),
}

# Primary appendix longtables: (question_id, model_kind, R output basename, .tex output stem).
PRIMARY_APPENDIX_TABLES: list[tuple[str, ModelKind, str, str]] = [
    ("q1", "clmm", "q1_acceptability_clmm.txt", "single_q1_clmm"),
    ("q2", "clm", "q2_smoothness_clm.txt", "single_q2_clm"),
    ("q3", "clm", "q3_immersion_clm.txt", "single_q3_clm"),
    ("q4", "clmm", "q4_continue_reading_clmm.txt", "single_q4_clmm"),
]
