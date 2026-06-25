#!/usr/bin/env python3
"""Analyze qualitative category profiles for undetected MT cases.

This script builds per-response A/B/C/D category-count tables from the human-eval
qualitative coding exports, then compares focal cases where MT was treated as
human against relevant baselines.

Run from the project root:
    .venv/bin/python analysis/human_eval/analyze_mt_undetectability_category_intervals.py

Outputs are written to:
    analysis/human_eval/mt_undetectability_category_intervals/
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .label_collapse import collapse_label_code
except ImportError:  # pragma: no cover - direct script execution.
    from label_collapse import collapse_label_code

try:
    from scipy.stats import mannwhitneyu
except ImportError:  # pragma: no cover - scipy is expected in this project environment.
    mannwhitneyu = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PART1 = REPO_ROOT / "human_eval" / "data" / "part1-study-data-full.csv"
DEFAULT_Q1 = REPO_ROOT / "analysis" / "human_eval" / "q1_pos_isolated.csv"
DEFAULT_Q2 = REPO_ROOT / "analysis" / "human_eval" / "q2_neg_isolated.csv"
DEFAULT_Q3 = REPO_ROOT / "analysis" / "human_eval" / "q3_comparison.csv"
DEFAULT_Q4 = REPO_ROOT / "analysis" / "human_eval" / "q4_ai.csv"
DEFAULT_OUT = REPO_ROOT / "analysis" / "human_eval" / "mt_undetectability_category_intervals"

CATEGORY_PREFIXES = ("A", "B", "C", "D")
CATEGORY_COLUMNS = {
    "A": "A. Language-level features",
    "B": "B. Narrative-level features",
    "C": "C. Reader experience",
    "D": "D. Meta-translation",
}
CATEGORY_LABELS = {
    "A": "A: Language",
    "B": "B: Narrative",
    "C": "C: Reader exp.",
    "D": "D: Meta",
}
ABCD_SUBGROUPS = {
    "SG_A_grammar": {
        "label": "A1: Grammar/tense",
        "codes": {"A1", "A1b"},
    },
    "SG_A_word_choice": {
        "label": "A: Word choice",
        "codes": {"A2", "A3", "A4", "A5", "A6"},
    },
    "SG_A_cultural": {
        "label": "A: Cultural",
        "codes": {"A7"},
    },
    "SG_A_sentence": {
        "label": "A: Sentence structure",
        "codes": {"A8"},
    },
    "SG_A_consistency": {
        "label": "A9: Consistency",
        "codes": {"A9"},
    },
    "SG_A_surface": {
        "label": "A: Surface consistency",
        "codes": {"A10", "A11", "A12"},
    },
    "SG_B_dialogue": {
        "label": "B: Dialogue",
        "codes": {"B1"},
    },
    "SG_B_character": {
        "label": "B2: Character voice/portrayal",
        "codes": {"B2", "B3"},
    },
    "SG_B_imagery": {
        "label": "B: Imagery/figurative",
        "codes": {"B4", "B5"},
    },
    "SG_B_emotion": {
        "label": "B: Emotion",
        "codes": {"B6"},
    },
    "SG_B_narrative_flow": {
        "label": "B: Narrative flow",
        "codes": {"B6", "B7", "B8"},
    },
    "SG_C_comprehension": {
        "label": "C: Comprehension",
        "codes": {"C1"},
    },
    "SG_C_smoothness": {
        "label": "C2: Smoothness/cadence",
        "codes": {"C2", "A9"},
    },
    "SG_C_engagement": {
        "label": "C3: Engagement/hook",
        "codes": {"C3", "C4"},
    },
    "SG_C_humanness": {
        "label": "C: Humanness",
        "codes": {"C4"},
    },
    "SG_C_enjoyment": {
        "label": "C: Enjoyment",
        "codes": {"C5"},
    },
    "SG_D_translation_relation": {
        "label": "D: Translation relation",
        "codes": {"D1", "D2", "D3"},
    },
    "SG_D_ai_mt_verdict": {
        "label": "D4: AI/MT verdict/tell",
        "codes": {"D4a", "D4b"},
    },
}
Q4_M_LABELS = {
    "M1": "M1: Literalism",
    "M2": "M2: Word choice",
    "M3": "M3: Sentence flow",
    "M4": "M4: Grammar",
    "M5": "M5: Formatting",
    "M6": "M6: Tone",
    "M7": "M7: Register",
    "M8": "M8: Wordiness",
    "M9": "M9: Voice/emotion",
    "M10": "M10: Meaning errors",
    "M11": "M11: Creative success",
    "M12": "M12: Comprehension",
    "M13": "M13: Folk theory",
}
ABCD_COUNT_COLUMNS = ["A_count", "B_count", "C_count", "D_count"]
ABCD_SUBGROUP_COUNT_COLUMNS = [f"{code}_count" for code in ABCD_SUBGROUPS]
Q4_M_COUNT_COLUMNS = [f"{code}_count" for code in Q4_M_LABELS]
COUNT_COLUMNS = (
    ABCD_COUNT_COLUMNS
    + ["total_abcd_count"]
    + ABCD_SUBGROUP_COUNT_COLUMNS
    + Q4_M_COUNT_COLUMNS
    + ["total_m_count"]
)
PLOT_COUNT_COLUMNS = ABCD_COUNT_COLUMNS
SUBGROUP_PLOT_COUNT_COLUMNS = ABCD_SUBGROUP_COUNT_COLUMNS
Q4_PLOT_COUNT_COLUMNS = Q4_M_COUNT_COLUMNS
CODE_TO_SUBGROUP = {
    code: subgroup for subgroup, spec in ABCD_SUBGROUPS.items() for code in spec["codes"]
}

HT_BLUE = "#0072B2"
MT_ORANGE = "#D55E00"
NEUTRAL_GRAY = "#9A9A9A"
LIGHT_GRAY = "#D8D8D8"
GROUP_COLORS = {
    "isolated_undetected_mt": "#E69F00",
    "isolated_detected_mt": "#D55E00",
    "isolated_ht_recognized": "#0072B2",
    "isolated_ht_misclassified": "#56B4E9",
    "comparison_undetected_preferred_mt": "#E69F00",
    "comparison_detected_preferred_mt": "#D55E00",
    "comparison_normal_preferred_ht": "#0072B2",
    "comparison_ht_preferred_ht_judged_ai": "#56B4E9",
    "preferred_mt": "#D55E00",
    "preferred_ht": "#0072B2",
    "thought_ai_ht": "#56B4E9",
    "thought_ai_mt": "#D55E00",
}


@dataclass(frozen=True)
class ScenarioComparison:
    scope: str
    comparison: str
    focal_group_column: str
    focal_group_value: str
    baseline_group_column: str
    baseline_group_value: str
    focal_label: str
    baseline_label: str
    plot_stem: str | None = None


REQUIRED_COMPARISONS = [
    ScenarioComparison(
        scope="q1_pos_isolated",
        comparison="undetected_mt_vs_detected_mt",
        focal_group_column="isolated_detection_group",
        focal_group_value="isolated_undetected_mt",
        baseline_group_column="isolated_detection_group",
        baseline_group_value="isolated_detected_mt",
        focal_label="Undetected MT",
        baseline_label="Detected MT",
        plot_stem="q1_pos_detected_mt_vs_undetected_mt",
    ),
    ScenarioComparison(
        scope="q2_neg_isolated",
        comparison="undetected_mt_vs_detected_mt",
        focal_group_column="isolated_detection_group",
        focal_group_value="isolated_undetected_mt",
        baseline_group_column="isolated_detection_group",
        baseline_group_value="isolated_detected_mt",
        focal_label="Undetected MT",
        baseline_label="Detected MT",
        plot_stem="q2_neg_detected_mt_vs_undetected_mt",
    ),
    ScenarioComparison(
        scope="q3_pos",
        comparison="preferred_mt_thought_ht_vs_preferred_ht_thought_mt",
        focal_group_column="comparison_detection_group",
        focal_group_value="comparison_undetected_preferred_mt",
        baseline_group_column="comparison_detection_group",
        baseline_group_value="comparison_normal_preferred_ht",
        focal_label="Preferred MT; judged HT AI-like",
        baseline_label="Preferred HT; judged MT AI-like",
        plot_stem="q3_pos_preferred_mt_thought_ht_vs_preferred_ht_thought_mt",
    ),
    ScenarioComparison(
        scope="q3_neg",
        comparison="preferred_mt_thought_ht_vs_preferred_ht_thought_mt",
        focal_group_column="comparison_detection_group",
        focal_group_value="comparison_undetected_preferred_mt",
        baseline_group_column="comparison_detection_group",
        baseline_group_value="comparison_normal_preferred_ht",
        focal_label="Preferred MT; judged HT AI-like",
        baseline_label="Preferred HT; judged MT AI-like",
        plot_stem="q3_neg_preferred_mt_thought_ht_vs_preferred_ht_thought_mt",
    ),
    ScenarioComparison(
        scope="q4_ai_telling",
        comparison="preferred_mt_thought_ht_vs_thought_mt",
        focal_group_column="comparison_detection_group",
        focal_group_value="comparison_undetected_preferred_mt",
        baseline_group_column="thought_ai_group",
        baseline_group_value="thought_ai_mt",
        focal_label="Preferred MT; judged HT AI-like",
        baseline_label="Judged MT AI-like",
        plot_stem="q4_ai_telling_preferred_mt_thought_ht_vs_thought_mt",
    ),
    ScenarioComparison(
        scope="q4_human_telling",
        comparison="preferred_mt_thought_ht_vs_thought_mt",
        focal_group_column="comparison_detection_group",
        focal_group_value="comparison_undetected_preferred_mt",
        baseline_group_column="thought_ai_group",
        baseline_group_value="thought_ai_mt",
        focal_label="Preferred MT; judged HT AI-like",
        baseline_label="Judged MT AI-like",
        plot_stem="q4_human_telling_preferred_mt_thought_ht_vs_thought_mt",
    ),
]

EXPLORATORY_COMPARISONS = [
    ScenarioComparison(
        scope="q1_pos_isolated",
        comparison="undetected_mt_vs_isolated_ht_recognized",
        focal_group_column="isolated_detection_group",
        focal_group_value="isolated_undetected_mt",
        baseline_group_column="isolated_detection_group",
        baseline_group_value="isolated_ht_recognized",
        focal_label="Undetected MT",
        baseline_label="Recognized HT",
    ),
    ScenarioComparison(
        scope="q2_neg_isolated",
        comparison="undetected_mt_vs_isolated_ht_recognized",
        focal_group_column="isolated_detection_group",
        focal_group_value="isolated_undetected_mt",
        baseline_group_column="isolated_detection_group",
        baseline_group_value="isolated_ht_recognized",
        focal_label="Undetected MT",
        baseline_label="Recognized HT",
    ),
    ScenarioComparison(
        scope="q3_pos",
        comparison="preferred_mt_vs_preferred_ht",
        focal_group_column="preferred_group",
        focal_group_value="preferred_mt",
        baseline_group_column="preferred_group",
        baseline_group_value="preferred_ht",
        focal_label="Preferred MT",
        baseline_label="Preferred HT",
    ),
    ScenarioComparison(
        scope="q3_neg",
        comparison="preferred_mt_vs_preferred_ht",
        focal_group_column="preferred_group",
        focal_group_value="preferred_mt",
        baseline_group_column="preferred_group",
        baseline_group_value="preferred_ht",
        focal_label="Preferred MT",
        baseline_label="Preferred HT",
    ),
    ScenarioComparison(
        scope="q3_pos",
        comparison="thought_ht_vs_thought_mt",
        focal_group_column="thought_ai_group",
        focal_group_value="thought_ai_ht",
        baseline_group_column="thought_ai_group",
        baseline_group_value="thought_ai_mt",
        focal_label="Judged HT AI-like",
        baseline_label="Judged MT AI-like",
    ),
    ScenarioComparison(
        scope="q3_neg",
        comparison="thought_ht_vs_thought_mt",
        focal_group_column="thought_ai_group",
        focal_group_value="thought_ai_ht",
        baseline_group_column="thought_ai_group",
        baseline_group_value="thought_ai_mt",
        focal_label="Judged HT AI-like",
        baseline_label="Judged MT AI-like",
    ),
]

ALL_COMPARISONS = REQUIRED_COMPARISONS + EXPLORATORY_COMPARISONS


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def done_mask(df: pd.DataFrame) -> pd.Series:
    done_col = "done" if "done" in df.columns else "DONE" if "DONE" in df.columns else None
    if done_col is None:
        return pd.Series(True, index=df.index)
    values = df[done_col].astype(str).str.strip().str.upper()
    return values.eq("TRUE")


def canonical_book_id(book_id: object) -> str:
    return str(book_id or "").strip().replace(
        "polish_eval_FIXED_needle_s_eye", "polish_eval_needle_s_eye"
    )


def infer_source_lang(book_id: object) -> str:
    text = canonical_book_id(book_id).lower()
    if text.startswith("french_"):
        return "French"
    if text.startswith("japanese_"):
        return "Japanese"
    if text.startswith("polish_"):
        return "Polish"
    return ""


def participant_base(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    match = re.search(r"p0*(\d+)", text)
    if not match:
        return text
    return f"p{int(match.group(1)):03d}"


def assignment_number(row: pd.Series) -> str:
    raw = str(row.get("ID", "") or "").strip()
    if raw.isdigit():
        return f"{int(raw):02d}"
    return ""


def normalize_choice(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if "NO DIFF" in text:
        return "NO DIFF"
    match = re.search(r"\((HT|MT)\)", text)
    if match:
        return match.group(1)
    if text in {"HT", "MT"}:
        return text
    if text.startswith("T") and " HT" in text:
        return "HT"
    if text.startswith("T") and " MT" in text:
        return "MT"
    return ""


def split_labels(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan"}:
        return []
    labels: list[str] = []
    for raw in text.split(","):
        label = re.sub(r"\s+", " ", raw).strip()
        if label and label.lower() not in {"none", "nan"}:
            labels.append(label)
    return labels


def label_code(label: str) -> str:
    match = re.match(r"^([A-Z]\d+[a-z]?)\.", label)
    return match.group(1) if match else ""


def collapsed_category_prefix(label: str) -> str:
    return collapse_label_code(label)[:1]


def count_labels_in_columns(row: pd.Series, columns_by_prefix: dict[str, str]) -> dict[str, int]:
    counts = {prefix: 0 for prefix in CATEGORY_PREFIXES}
    for column in columns_by_prefix.values():
        if column in row.index:
            for label in split_labels(row.get(column, "")):
                prefix = collapsed_category_prefix(label)
                if prefix in counts:
                    counts[prefix] += 1
    return counts


def count_subgroups_in_columns(row: pd.Series, columns_by_prefix: dict[str, str]) -> dict[str, int]:
    counts = {subgroup: 0 for subgroup in ABCD_SUBGROUPS}
    for column in columns_by_prefix.values():
        if column not in row.index:
            continue
        for label in split_labels(row.get(column, "")):
            subgroup = CODE_TO_SUBGROUP.get(collapse_label_code(label))
            if subgroup:
                counts[subgroup] += 1
    return counts


def count_labels_by_prefix(value: object) -> tuple[dict[str, int], dict[str, int]]:
    counts = {prefix: 0 for prefix in CATEGORY_PREFIXES}
    m_counts = {code: 0 for code in Q4_M_LABELS}
    for label in split_labels(value):
        prefix = collapsed_category_prefix(label)
        if not prefix:
            continue
        if prefix in counts:
            counts[prefix] += 1
            continue
        m_match = re.match(r"^(M\d+)\.", label)
        if m_match and m_match.group(1) in m_counts:
            m_counts[m_match.group(1)] += 1
    return counts, m_counts


def source_lang_lookup(part1: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    lookup: dict[tuple[str, str, str], str] = {}
    for _, row in part1.iterrows():
        base = participant_base(row.get("user_id", ""))
        suffix = str(row.get("user_id", "")).split("_")[-1]
        book_id = canonical_book_id(row.get("book_id", ""))
        if base and suffix and book_id:
            lookup[(base, suffix.zfill(2), book_id)] = str(row.get("source_lang", ""))
    return lookup


def part1_row_lookup(part1: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    lookup: dict[tuple[str, str, str], pd.Series] = {}
    for _, row in part1.iterrows():
        base = participant_base(row.get("user_id", ""))
        suffix = str(row.get("user_id", "")).split("_")[-1]
        book_id = canonical_book_id(row.get("book_id", ""))
        if base and suffix and book_id:
            lookup[(base, suffix.zfill(2), book_id)] = row
    return lookup


def add_common_fields(
    row: pd.Series,
    source_file: str,
    scope: str,
    polarity_or_scope: str,
    counts: dict[str, int],
    subgroup_counts: dict[str, int] | None,
    m_counts: dict[str, int] | None,
    part1_lookup: dict[tuple[str, str, str], pd.Series],
    source_lookup: dict[tuple[str, str, str], str],
    comment: str,
) -> dict[str, object]:
    book_id = canonical_book_id(row.get("book_id", ""))
    base = participant_base(row.get("user", ""))
    suffix = assignment_number(row)
    part1_row = part1_lookup.get((base, suffix, book_id))
    source_lang = source_lookup.get((base, suffix, book_id), "") or infer_source_lang(book_id)
    current_trans = normalize_choice(row.get("current_trans", ""))
    preferred_continue = normalize_choice(row.get("preferred_continue", ""))
    thought_ai = normalize_choice(row.get("thought_ai", ""))

    if scope in {"q1_pos_isolated", "q2_neg_isolated"} and part1_row is not None:
        display_label = str(row.get("display_label", "")).strip().upper()
        if display_label == "T1":
            thought_ai = normalize_choice(part1_row.get("first_q7_decipher", thought_ai))
        elif display_label == "T2":
            thought_ai = normalize_choice(part1_row.get("second_q7_decipher", thought_ai))
    elif scope in {"q3_pos", "q3_neg"} and part1_row is not None:
        thought_ai = normalize_choice(part1_row.get("comparison_q5_decipher", thought_ai))

    record: dict[str, object] = {
        "source_file": source_file,
        "scope": scope,
        "polarity_or_scope": polarity_or_scope,
        "user": row.get("user", ""),
        "participant_id": (
            part1_row.get("participant_id", "") if part1_row is not None else ""
        ),
        "assignment_id": row.get("assignment_id", ""),
        "book_id": book_id,
        "source_lang": source_lang,
        "order": row.get("order", ""),
        "current_trans": current_trans,
        "preferred_continue": preferred_continue,
        "thought_ai": thought_ai,
        "A_count": counts["A"],
        "B_count": counts["B"],
        "C_count": counts["C"],
        "D_count": counts["D"],
        "total_abcd_count": sum(counts.values()),
        "total_m_count": sum((m_counts or {}).values()),
        "comment": comment,
    }
    for code in Q4_M_LABELS:
        record[f"{code}_count"] = (m_counts or {}).get(code, 0)
    for subgroup in ABCD_SUBGROUPS:
        record[f"{subgroup}_count"] = (subgroup_counts or {}).get(subgroup, 0)
    record.update(build_scenario_fields(record))
    return record


def build_scenario_fields(record: dict[str, object]) -> dict[str, str]:
    scope = str(record.get("scope", ""))
    current_trans = str(record.get("current_trans", ""))
    preferred = str(record.get("preferred_continue", ""))
    thought_ai = str(record.get("thought_ai", ""))

    isolated_detection_group = ""
    if scope in {"q1_pos_isolated", "q2_neg_isolated"}:
        if current_trans == "MT" and thought_ai == "HT":
            isolated_detection_group = "isolated_undetected_mt"
        elif current_trans == "MT" and thought_ai == "MT":
            isolated_detection_group = "isolated_detected_mt"
        elif current_trans == "HT" and thought_ai == "HT":
            isolated_detection_group = "isolated_ht_recognized"
        elif current_trans == "HT" and thought_ai == "MT":
            isolated_detection_group = "isolated_ht_misclassified"

    comparison_detection_group = ""
    if scope in {"q3_pos", "q3_neg", "q4_ai_telling", "q4_human_telling"}:
        if preferred == "MT" and thought_ai == "HT":
            comparison_detection_group = "comparison_undetected_preferred_mt"
        elif preferred == "HT" and thought_ai == "MT":
            comparison_detection_group = "comparison_normal_preferred_ht"
        elif preferred == "MT" and thought_ai == "MT":
            comparison_detection_group = "comparison_detected_preferred_mt"
        elif preferred == "HT" and thought_ai == "HT":
            comparison_detection_group = "comparison_ht_preferred_ht_judged_ai"

    preferred_group = f"preferred_{preferred.lower()}" if preferred in {"HT", "MT"} else ""
    thought_ai_group = f"thought_ai_{thought_ai.lower()}" if thought_ai in {"HT", "MT"} else ""

    groups = [
        "all_rows",
        isolated_detection_group,
        comparison_detection_group,
        preferred_group,
        thought_ai_group,
    ]
    return {
        "isolated_detection_group": isolated_detection_group,
        "comparison_detection_group": comparison_detection_group,
        "preferred_group": preferred_group,
        "thought_ai_group": thought_ai_group,
        "scenario_groups": "|".join(group for group in groups if group),
    }


def build_tidy_table(
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    q4: pd.DataFrame,
    part1: pd.DataFrame,
) -> pd.DataFrame:
    part1_lookup = part1_row_lookup(part1)
    source_lookup = source_lang_lookup(part1)
    rows: list[dict[str, object]] = []

    for source_file, scope, polarity, df in [
        ("q1_pos_isolated.csv", "q1_pos_isolated", "POS", q1),
        ("q2_neg_isolated.csv", "q2_neg_isolated", "NEG", q2),
    ]:
        for _, row in df[done_mask(df)].iterrows():
            counts = count_labels_in_columns(row, CATEGORY_COLUMNS)
            subgroup_counts = count_subgroups_in_columns(row, CATEGORY_COLUMNS)
            rows.append(
                add_common_fields(
                    row=row,
                    source_file=source_file,
                    scope=scope,
                    polarity_or_scope=polarity,
                    counts=counts,
                    subgroup_counts=subgroup_counts,
                    m_counts=None,
                    part1_lookup=part1_lookup,
                    source_lookup=source_lookup,
                    comment=str(row.get("comment", "")),
                )
            )

    q3_columns = {
        "POS": {prefix: f"POS\n{column}" for prefix, column in CATEGORY_COLUMNS.items()},
        "NEG": {prefix: f"NEG\n{column}" for prefix, column in CATEGORY_COLUMNS.items()},
    }
    for _, row in q3[done_mask(q3)].iterrows():
        for polarity, scope in [("POS", "q3_pos"), ("NEG", "q3_neg")]:
            counts = count_labels_in_columns(row, q3_columns[polarity])
            subgroup_counts = count_subgroups_in_columns(row, q3_columns[polarity])
            rows.append(
                add_common_fields(
                    row=row,
                    source_file="q3_comparison.csv",
                    scope=scope,
                    polarity_or_scope=polarity,
                    counts=counts,
                    subgroup_counts=subgroup_counts,
                    m_counts=None,
                    part1_lookup=part1_lookup,
                    source_lookup=source_lookup,
                    comment=str(row.get("comment", "")),
                )
            )

    for _, row in q4[done_mask(q4)].iterrows():
        for column, scope, label in [
            ("AI-telling", "q4_ai_telling", "AI-telling cues"),
            ("Human-telling", "q4_human_telling", "Human-telling cues"),
        ]:
            counts, m_counts = count_labels_by_prefix(row.get(column, ""))
            rows.append(
                add_common_fields(
                    row=row,
                    source_file="q4_ai.csv",
                    scope=scope,
                    polarity_or_scope=label,
                    counts=counts,
                    subgroup_counts=None,
                    m_counts=m_counts,
                    part1_lookup=part1_lookup,
                    source_lookup=source_lookup,
                    comment=str(row.get("comment", "")),
                )
            )

    tidy = pd.DataFrame(rows)
    if tidy.empty:
        return tidy
    sort_cols = ["scope", "book_id", "user", "polarity_or_scope"]
    return tidy.sort_values(sort_cols).reset_index(drop=True)


def is_q4_scope(scope: str) -> bool:
    return scope in {"q4_ai_telling", "q4_human_telling"}


def plot_count_columns_for_scope(scope: str) -> list[str]:
    return Q4_PLOT_COUNT_COLUMNS if is_q4_scope(scope) else PLOT_COUNT_COLUMNS


def stat_count_columns_for_scope(scope: str) -> list[str]:
    if is_q4_scope(scope):
        return Q4_M_COUNT_COLUMNS + ["total_m_count"]
    return ABCD_COUNT_COLUMNS + ["total_abcd_count"]


def label_for_count_column(column: str) -> str:
    if column.endswith("_count"):
        key = column.removesuffix("_count")
        return (
            CATEGORY_LABELS.get(key)
            or Q4_M_LABELS.get(key)
            or ABCD_SUBGROUPS.get(key, {}).get("label")
            or key
        )
    if column == "total_abcd_count":
        return "Total A-D"
    if column == "total_m_count":
        return "Total M"
    return column


def long_counts(
    df: pd.DataFrame, count_columns: Iterable[str] = PLOT_COUNT_COLUMNS
) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=[column for column in df.columns if column not in COUNT_COLUMNS],
        value_vars=list(count_columns),
        var_name="category_count",
        value_name="count",
    )
    long_df["category"] = long_df["category_count"].str.replace("_count", "", regex=False)
    long_df["category_label"] = long_df["category_count"].map(label_for_count_column)
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce").fillna(0)
    return long_df


def summarize_group(df: pd.DataFrame, scope: str, group_type: str, group_value: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    subset = df[df["scope"] == scope].copy()
    if group_type != "all_rows":
        subset = subset[subset[group_type] == group_value]
    for category in stat_count_columns_for_scope(scope):
        values = pd.to_numeric(subset[category], errors="coerce").dropna()
        rows.append(
            {
                "scope": scope,
                "group_type": group_type,
                "group": group_value,
                "category": category,
                "n": int(values.shape[0]),
                "mean": values.mean() if not values.empty else np.nan,
                "median": values.median() if not values.empty else np.nan,
                "q1": values.quantile(0.25) if not values.empty else np.nan,
                "q3": values.quantile(0.75) if not values.empty else np.nan,
                "min": values.min() if not values.empty else np.nan,
                "max": values.max() if not values.empty else np.nan,
                "p05": values.quantile(0.05) if not values.empty else np.nan,
                "p95": values.quantile(0.95) if not values.empty else np.nan,
            }
        )
    return rows


def build_interval_summary(tidy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in sorted(tidy["scope"].unique()):
        scope_df = tidy[tidy["scope"] == scope]
        rows.extend(summarize_group(tidy, scope, "all_rows", "all_rows"))
        for group_type in [
            "isolated_detection_group",
            "comparison_detection_group",
            "preferred_group",
            "thought_ai_group",
        ]:
            for group_value in sorted(value for value in scope_df[group_type].unique() if value):
                rows.extend(summarize_group(tidy, scope, group_type, group_value))
    return pd.DataFrame(rows)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    greater = 0
    less = 0
    for value in a:
        greater += int(np.sum(value > b))
        less += int(np.sum(value < b))
    return (greater - less) / (len(a) * len(b))


def mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0 or mannwhitneyu is None:
        return np.nan
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0
    return float(mannwhitneyu(a, b, alternative="two-sided", method="auto").pvalue)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    valid = [(idx, value) for idx, value in enumerate(p_values) if pd.notna(value)]
    adjusted = [np.nan for _ in p_values]
    if not valid:
        return adjusted
    sorted_valid = sorted(valid, key=lambda item: item[1])
    m = len(sorted_valid)
    running_min = 1.0
    for reverse_rank, (idx, value) in enumerate(reversed(sorted_valid), start=1):
        rank = m - reverse_rank + 1
        running_min = min(running_min, value * m / rank)
        adjusted[idx] = min(running_min, 1.0)
    return adjusted


def build_test_and_delta_tables(tidy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for comparison in ALL_COMPARISONS:
        scope_df = tidy[tidy["scope"] == comparison.scope]
        focal = scope_df[scope_df[comparison.focal_group_column] == comparison.focal_group_value]
        baseline = scope_df[scope_df[comparison.baseline_group_column] == comparison.baseline_group_value]
        for category in stat_count_columns_for_scope(comparison.scope):
            focal_values = pd.to_numeric(focal[category], errors="coerce").dropna().to_numpy(float)
            baseline_values = pd.to_numeric(baseline[category], errors="coerce").dropna().to_numpy(float)
            mean_diff = (
                np.mean(focal_values) - np.mean(baseline_values)
                if len(focal_values) and len(baseline_values)
                else np.nan
            )
            median_diff = (
                np.median(focal_values) - np.median(baseline_values)
                if len(focal_values) and len(baseline_values)
                else np.nan
            )
            p_value = mann_whitney_p(focal_values, baseline_values)
            cliff = cliffs_delta(focal_values, baseline_values)
            common = {
                "scope": comparison.scope,
                "comparison": comparison.comparison,
                "category": category,
                "focal_group_type": comparison.focal_group_column,
                "focal_group": comparison.focal_group_value,
                "baseline_group_type": comparison.baseline_group_column,
                "baseline_group": comparison.baseline_group_value,
                "focal_n": int(len(focal_values)),
                "baseline_n": int(len(baseline_values)),
                "focal_mean": np.mean(focal_values) if len(focal_values) else np.nan,
                "baseline_mean": np.mean(baseline_values) if len(baseline_values) else np.nan,
                "mean_diff_focal_minus_baseline": mean_diff,
                "focal_median": np.median(focal_values) if len(focal_values) else np.nan,
                "baseline_median": np.median(baseline_values) if len(baseline_values) else np.nan,
                "median_diff_focal_minus_baseline": median_diff,
                "cliffs_delta": cliff,
            }
            test_rows.append(
                {
                    **common,
                    "test": "Mann-Whitney U" if mannwhitneyu is not None else "not_available",
                    "p_value": p_value,
                }
            )
            delta_rows.append(common)

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests["bh_q_within_scope"] = np.nan
        for scope, index in tests.groupby("scope").groups.items():
            tests.loc[index, "bh_q_within_scope"] = benjamini_hochberg(
                tests.loc[index, "p_value"].tolist()
            )
    return tests, pd.DataFrame(delta_rows)


def comparison_subset(tidy: pd.DataFrame, comparison: ScenarioComparison) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_df = tidy[tidy["scope"] == comparison.scope]
    focal = scope_df[scope_df[comparison.focal_group_column] == comparison.focal_group_value].copy()
    baseline = scope_df[
        scope_df[comparison.baseline_group_column] == comparison.baseline_group_value
    ].copy()
    focal["plot_group"] = comparison.focal_label
    baseline["plot_group"] = comparison.baseline_label
    return focal, baseline


def plot_required_comparison(tidy: pd.DataFrame, comparison: ScenarioComparison, out_dir: Path) -> None:
    if comparison.plot_stem is None:
        return
    focal, baseline = comparison_subset(tidy, comparison)
    plot_columns = plot_count_columns_for_scope(comparison.scope)
    plot_df = long_counts(pd.concat([baseline, focal], ignore_index=True), plot_columns)
    if plot_df.empty:
        return
    baseline_long = plot_df[plot_df["plot_group"] == comparison.baseline_label]
    focal_long = plot_df[plot_df["plot_group"] == comparison.focal_label]
    fig_width = 12.5 if is_q4_scope(comparison.scope) else 7.2
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    if not baseline_long.empty:
        sns.boxplot(
            data=baseline_long,
            x="category_label",
            y="count",
            color=LIGHT_GRAY,
            width=0.55,
            fliersize=0,
            linewidth=1.0,
            ax=ax,
        )
        sns.stripplot(
            data=baseline_long,
            x="category_label",
            y="count",
            color=NEUTRAL_GRAY,
            size=5.0,
            jitter=0.18,
            alpha=0.70,
            ax=ax,
        )
    if not focal_long.empty:
        sns.stripplot(
            data=focal_long,
            x="category_label",
            y="count",
            color=MT_ORANGE,
            edgecolor="white",
            linewidth=0.8,
            size=8,
            jitter=0.12,
            alpha=0.95,
            ax=ax,
        )
    max_count = int(max(plot_df["count"].max(), 1))
    ax.set_ylim(-0.2, max_count + 0.8)
    ax.set_yticks(range(0, max_count + 1))
    ax.set_xlabel("")
    ylabel = (
        "Per-response origin-cue label count"
        if is_q4_scope(comparison.scope)
        else "Per-response category-label count"
    )
    ax.set_ylabel(ylabel)
    if is_q4_scope(comparison.scope):
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    ax.set_title(f"{comparison.scope}: {comparison.focal_label} vs baseline")
    ax.text(
        0.01,
        0.98,
        f"Baseline: {comparison.baseline_label} (n={baseline.shape[0]}); "
        f"focal: {comparison.focal_label} (n={focal.shape[0]})",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9, "pad": 4},
    )
    save_figure(fig, out_dir, comparison.plot_stem)


def readable_group_label(value: str) -> str:
    labels = {
        "isolated_undetected_mt": "MT guessed HT",
        "isolated_detected_mt": "MT guessed MT",
        "isolated_ht_recognized": "HT guessed HT",
        "isolated_ht_misclassified": "HT guessed MT",
        "comparison_undetected_preferred_mt": "Pref MT; HT AI-like",
        "comparison_normal_preferred_ht": "Pref HT; MT AI-like",
        "comparison_detected_preferred_mt": "Pref MT; MT AI-like",
        "comparison_ht_preferred_ht_judged_ai": "Pref HT; HT AI-like",
        "preferred_mt": "Preferred MT",
        "preferred_ht": "Preferred HT",
        "thought_ai_ht": "Judged HT AI-like",
        "thought_ai_mt": "Judged MT AI-like",
    }
    return labels.get(value, value.replace("_", " "))


def plot_exploratory_groups(
    tidy: pd.DataFrame,
    scope: str,
    group_column: str,
    group_order: list[str],
    stem: str,
    title: str,
    out_dir: Path,
) -> None:
    subset = tidy[(tidy["scope"] == scope) & (tidy[group_column].isin(group_order))].copy()
    if subset.empty:
        return
    subset["plot_group"] = pd.Categorical(
        subset[group_column].map(readable_group_label),
        categories=[readable_group_label(group) for group in group_order],
        ordered=True,
    )
    plot_df = long_counts(subset)
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    palette = {
        readable_group_label(group): GROUP_COLORS.get(group, NEUTRAL_GRAY)
        for group in group_order
    }
    sns.boxplot(
        data=plot_df,
        x="category_label",
        y="count",
        hue="plot_group",
        palette=palette,
        width=0.68,
        fliersize=0,
        linewidth=0.9,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="category_label",
        y="count",
        hue="plot_group",
        dodge=True,
        palette=palette,
        size=5.2,
        jitter=0.12,
        alpha=0.72,
        legend=False,
        ax=ax,
    )
    max_count = int(max(plot_df["count"].max(), 1))
    ax.set_ylim(-0.2, max_count + 0.8)
    ax.set_yticks(range(0, max_count + 1))
    ax.set_xlabel("")
    ax.set_ylabel("Per-response category-label count")
    ax.set_title(title)
    counts = subset.groupby("plot_group", observed=True).size()
    legend = ax.legend(title="", loc="upper right", frameon=True)
    for text in legend.get_texts():
        label = text.get_text()
        if label in counts.index:
            text.set_text(f"{label} (n={counts[label]})")
    save_figure(fig, out_dir, stem)


def plot_exploratory_subgroups(
    tidy: pd.DataFrame,
    scope: str,
    group_column: str,
    group_order: list[str],
    stem: str,
    title: str,
    out_dir: Path,
) -> None:
    subset = tidy[(tidy["scope"] == scope) & (tidy[group_column].isin(group_order))].copy()
    if subset.empty:
        return
    subgroup_columns = [
        column for column in SUBGROUP_PLOT_COUNT_COLUMNS if pd.to_numeric(subset[column]).sum() > 0
    ]
    if not subgroup_columns:
        return
    subset["plot_group"] = pd.Categorical(
        subset[group_column].map(readable_group_label),
        categories=[readable_group_label(group) for group in group_order],
        ordered=True,
    )
    plot_df = long_counts(subset, subgroup_columns)
    fig, ax = plt.subplots(figsize=(16, 6.4))
    palette = {
        readable_group_label(group): GROUP_COLORS.get(group, NEUTRAL_GRAY)
        for group in group_order
    }
    sns.boxplot(
        data=plot_df,
        x="category_label",
        y="count",
        hue="plot_group",
        palette=palette,
        width=0.68,
        fliersize=0,
        linewidth=0.9,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="category_label",
        y="count",
        hue="plot_group",
        dodge=True,
        palette=palette,
        size=4.6,
        jitter=0.12,
        alpha=0.72,
        legend=False,
        ax=ax,
    )
    max_count = int(max(plot_df["count"].max(), 1))
    ax.set_ylim(-0.2, max_count + 0.8)
    ax.set_yticks(range(0, max_count + 1))
    ax.set_xlabel("")
    ax.set_ylabel("Per-response subgroup-label count")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    counts = subset.groupby("plot_group", observed=True).size()
    legend = ax.legend(title="", loc="upper right", frameon=True)
    for text in legend.get_texts():
        label = text.get_text()
        if label in counts.index:
            text.set_text(f"{label} (n={counts[label]})")
    save_figure(fig, out_dir, stem)


def make_plots(tidy: pd.DataFrame, out_dir: Path) -> None:
    for comparison in REQUIRED_COMPARISONS:
        plot_required_comparison(tidy, comparison, out_dir)

    isolated_order = [
        "isolated_undetected_mt",
        "isolated_detected_mt",
        "isolated_ht_recognized",
        "isolated_ht_misclassified",
    ]
    plot_exploratory_groups(
        tidy,
        scope="q1_pos_isolated",
        group_column="isolated_detection_group",
        group_order=isolated_order,
        stem="exploratory_q1_pos_isolated_all_detection_groups",
        title="Q1 positive comments: isolated detection groups",
        out_dir=out_dir,
    )
    plot_exploratory_subgroups(
        tidy,
        scope="q1_pos_isolated",
        group_column="isolated_detection_group",
        group_order=isolated_order,
        stem="exploratory_subgroups_q1_pos_isolated_all_detection_groups",
        title="Q1 positive comments: isolated detection groups by code subgroup",
        out_dir=out_dir,
    )
    plot_exploratory_groups(
        tidy,
        scope="q2_neg_isolated",
        group_column="isolated_detection_group",
        group_order=isolated_order,
        stem="exploratory_q2_neg_isolated_all_detection_groups",
        title="Q2 negative comments: isolated detection groups",
        out_dir=out_dir,
    )
    plot_exploratory_subgroups(
        tidy,
        scope="q2_neg_isolated",
        group_column="isolated_detection_group",
        group_order=isolated_order,
        stem="exploratory_subgroups_q2_neg_isolated_all_detection_groups",
        title="Q2 negative comments: isolated detection groups by code subgroup",
        out_dir=out_dir,
    )
    comparison_order = [
        "comparison_undetected_preferred_mt",
        "comparison_detected_preferred_mt",
        "comparison_normal_preferred_ht",
        "comparison_ht_preferred_ht_judged_ai",
    ]
    plot_exploratory_groups(
        tidy,
        scope="q3_pos",
        group_column="comparison_detection_group",
        group_order=comparison_order,
        stem="exploratory_q3_pos_all_comparison_groups",
        title="Q3 POS rationale counts: preference and AI-attribution groups",
        out_dir=out_dir,
    )
    plot_exploratory_subgroups(
        tidy,
        scope="q3_pos",
        group_column="comparison_detection_group",
        group_order=comparison_order,
        stem="exploratory_subgroups_q3_pos_all_comparison_groups",
        title="Q3 POS rationale counts: preference and AI-attribution groups by code subgroup",
        out_dir=out_dir,
    )
    plot_exploratory_groups(
        tidy,
        scope="q3_neg",
        group_column="comparison_detection_group",
        group_order=comparison_order,
        stem="exploratory_q3_neg_all_comparison_groups",
        title="Q3 NEG rationale counts: preference and AI-attribution groups",
        out_dir=out_dir,
    )
    plot_exploratory_subgroups(
        tidy,
        scope="q3_neg",
        group_column="comparison_detection_group",
        group_order=comparison_order,
        stem="exploratory_subgroups_q3_neg_all_comparison_groups",
        title="Q3 NEG rationale counts: preference and AI-attribution groups by code subgroup",
        out_dir=out_dir,
    )


def format_number(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    numeric = float(value)
    if math.isclose(numeric, round(numeric)):
        return str(int(round(numeric)))
    return f"{numeric:.{digits}f}"


def strongest_deltas(tests: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    if tests.empty:
        return tests
    required_keys = {(comparison.scope, comparison.comparison) for comparison in REQUIRED_COMPARISONS}
    subset = tests[
        tests.apply(lambda row: (row["scope"], row["comparison"]) in required_keys, axis=1)
    ].copy()
    subset = subset[
        subset.apply(
            lambda row: row["category"] in plot_count_columns_for_scope(str(row["scope"])), axis=1
        )
    ]
    subset["abs_mean_diff"] = subset["mean_diff_focal_minus_baseline"].abs()
    return subset.sort_values(["abs_mean_diff", "focal_n"], ascending=[False, False]).head(limit)


def is_primary_plot_category(row: pd.Series) -> bool:
    return str(row["category"]) in plot_count_columns_for_scope(str(row["scope"]))


def write_summary(
    tidy: pd.DataFrame,
    intervals: pd.DataFrame,
    tests: pd.DataFrame,
    out_dir: Path,
) -> None:
    scope_counts = tidy.groupby("scope").size().to_dict()
    scenario_counts = {
        "isolated_detection_group": tidy["isolated_detection_group"].value_counts().to_dict(),
        "comparison_detection_group": tidy["comparison_detection_group"].value_counts().to_dict(),
    }
    strongest = strongest_deltas(tests)
    significant = tests[
        tests["bh_q_within_scope"].notna()
        & (tests["bh_q_within_scope"] <= 0.05)
        & tests.apply(is_primary_plot_category, axis=1)
    ]

    lines = [
        "# MT Undetectability Category-Interval Analysis",
        "",
        "## What Was Counted",
        "",
        "Each Q1-Q3 coded open-ended response was reduced to counts of broad "
        "qualitative-code families: A = language-level features, B = narrative-level "
        "features, C = reader experience, and D = meta-translation. Q4 responses use "
        "their own M-style origin-cue codebook. Counts are per response, not global "
        "label totals. For example, two comma-separated A labels in one response "
        "contribute `A_count = 2` for that response.",
        "",
        "Q3 comparison rationales were split into `q3_pos` and `q3_neg`: POS labels "
        "describe why the preferred translation was good, and NEG labels describe why "
        "the rejected translation was bad. Q4 AI-origin rationales were split into "
        "`q4_ai_telling` and `q4_human_telling`; because the Q4 export uses M-style "
        "origin-cue labels, Q4 plots and tests use `M1`-`M13` counts instead of "
        "A/B/C/D counts.",
        "",
        "The intervals in the summary table and plots are empirical distribution "
        "summaries: median, Q1, Q3, min/max, and 5th/95th percentiles. They are not "
        "confidence intervals.",
        "",
        "## Scenario Definitions",
        "",
        "- Isolated undetected MT: actual `current_trans == MT`, guessed `thought_ai == HT`.",
        "- Isolated detected MT: actual `current_trans == MT`, guessed `thought_ai == MT`.",
        "- Isolated HT baselines: `HT` recognized as `HT` or misclassified as `MT`.",
        "- Comparison undetected/preferred MT: `preferred_continue == MT` and `thought_ai == HT`.",
        "- Comparison normal/preferred HT: `preferred_continue == HT` and `thought_ai == MT`.",
        "- Additional exploratory groups include preference-only and AI-attribution-only splits.",
        "",
        "## Generated Files",
        "",
        "- `category_counts_by_response.csv`: tidy per-response category counts.",
        "- `category_interval_summary.csv`: empirical interval summaries by scope and group "
        "(A/B/C/D for Q1-Q3; M1-M13 for Q4).",
        "- `scenario_category_deltas.csv`: focal-minus-baseline differences and Cliff's delta.",
        "- `category_scenario_tests.csv`: Mann-Whitney U tests with BH adjustment within scope.",
        "- `*.png` and `*.pdf`: required and exploratory interval/point plots.",
        "",
        "## Scope Sizes",
        "",
    ]
    for scope, count in sorted(scope_counts.items()):
        lines.append(f"- `{scope}`: {count} response rows")
    lines.extend(["", "## Scenario Counts", ""])
    for group_column, counts in scenario_counts.items():
        lines.append(f"`{group_column}`:")
        for group, count in counts.items():
            if group:
                lines.append(f"- `{group}`: {count}")
        lines.append("")

    lines.extend(
        [
            "## Strongest Required-Comparison Deltas",
            "",
            "These are descriptive focal-minus-baseline differences sorted by absolute mean "
            "difference. Small sample sizes, especially for `comparison_undetected_preferred_mt`, "
            "make these hypothesis-generating rather than confirmatory.",
            "",
        ]
    )
    if strongest.empty:
        lines.append("No required-comparison deltas were available.")
    else:
        lines.append(
            "| Scope | Comparison | Category | Focal n | Baseline n | Mean delta | "
            "Median delta | Cliff's delta | p | BH q |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in strongest.iterrows():
            lines.append(
                f"| `{row['scope']}` | `{row['comparison']}` | `{row['category']}` | "
                f"{int(row['focal_n'])} | {int(row['baseline_n'])} | "
                f"{format_number(row['mean_diff_focal_minus_baseline'])} | "
                f"{format_number(row['median_diff_focal_minus_baseline'])} | "
                f"{format_number(row['cliffs_delta'])} | "
                f"{format_number(row['p_value'], 3)} | "
                f"{format_number(row['bh_q_within_scope'], 3)} |"
            )

    lines.extend(["", "## BH-Adjusted Results", ""])
    if significant.empty:
        lines.append(
            "No primary category-count comparisons survive Benjamini-Hochberg adjustment "
            "at q <= 0.05 within scope."
        )
    else:
        lines.append("The following primary category tests have BH q <= 0.05 within scope:")
        for _, row in significant.iterrows():
            lines.append(
                f"- `{row['scope']}` / `{row['comparison']}` / `{row['category']}`: "
                f"mean delta {format_number(row['mean_diff_focal_minus_baseline'])}, "
                f"BH q {format_number(row['bh_q_within_scope'], 3)}"
            )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- The primary purpose is visualization and hypothesis generation, not definitive "
            "inference.",
            "- Some focal groups are very small; point overlays should be read before p-values.",
        "- Q1/Q2 origin guesses are recovered from Part 1 because the coding exports leave "
        "`thought_ai` blank for isolated-reading rows.",
        "- Q4 is not directly comparable to Q1-Q3 category families because it uses a "
        "separate M-style AI/human-origin cue codebook.",
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, default=DEFAULT_PART1)
    parser.add_argument("--q1", type=Path, default=DEFAULT_Q1)
    parser.add_argument("--q2", type=Path, default=DEFAULT_Q2)
    parser.add_argument("--q3", type=Path, default=DEFAULT_Q3)
    parser.add_argument("--q4", type=Path, default=DEFAULT_Q4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    part1 = read_csv(args.part1)
    q1 = read_csv(args.q1)
    q2 = read_csv(args.q2)
    q3 = read_csv(args.q3)
    q4 = read_csv(args.q4)

    tidy = build_tidy_table(q1=q1, q2=q2, q3=q3, q4=q4, part1=part1)
    intervals = build_interval_summary(tidy)
    tests, deltas = build_test_and_delta_tables(tidy)

    tidy.to_csv(out_dir / "category_counts_by_response.csv", index=False, encoding="utf-8")
    intervals.to_csv(out_dir / "category_interval_summary.csv", index=False, encoding="utf-8")
    deltas.to_csv(out_dir / "scenario_category_deltas.csv", index=False, encoding="utf-8")
    tests.to_csv(out_dir / "category_scenario_tests.csv", index=False, encoding="utf-8")

    make_plots(tidy, out_dir)
    write_summary(tidy=tidy, intervals=intervals, tests=tests, out_dir=out_dir)

    print(f"Wrote {tidy.shape[0]} tidy response rows to {out_dir}")


if __name__ == "__main__":
    main()
