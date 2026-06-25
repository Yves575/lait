#!/usr/bin/env python3
"""Analyze cases where readers preferred MT but judged HT as more AI-like.

Run from the project root:
    python3 analysis/human_eval/analyze_mt_preference.py

Outputs are written to:
    analysis/human_eval/mt_preference_analysis/
"""

from __future__ import annotations

import argparse
import csv
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from .label_collapse import collapse_label, collapse_label_code
except ImportError:  # pragma: no cover - direct script execution.
    from label_collapse import collapse_label, collapse_label_code

try:
    from scipy.stats import fisher_exact
except ImportError:  # pragma: no cover - scipy is in project requirements.
    fisher_exact = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PART1 = REPO_ROOT / "human_eval" / "data" / "part1-study-data-full.csv"
DEFAULT_PART2 = REPO_ROOT / "human_eval" / "data" / "part2-study-data-full.csv"
DEFAULT_Q3 = REPO_ROOT / "analysis" / "human_eval" / "q3_comparison.csv"
DEFAULT_Q4 = REPO_ROOT / "analysis" / "human_eval" / "q4_ai.csv"
DEFAULT_Q5 = REPO_ROOT / "analysis" / "human_eval" / "q5_chunks.csv"
DEFAULT_OUT = REPO_ROOT / "analysis" / "human_eval" / "mt_preference_analysis"

TRANSLATION_COLORS = {"HT": "#0072B2", "MT": "#D55E00", "NO DIFF": "#7A7A7A"}
ROLE_COLORS = {
    "MT positive": "#1B9E77",
    "HT negative": "#D95F02",
    "AI cue": "#7570B3",
    "Human cue": "#66A61E",
    "MT preference": "#D55E00",
    "HT preference": "#0072B2",
}
SCENARIO_COLORS = {
    "Relative MT advantage": "#4C78A8",
    "MT strength": "#1B9E77",
    "HT weakness": "#D95F02",
    "Weak/uncoded evidence": "#7A7A7A",
}

Q_LABELS = {
    "q1": "Acceptability",
    "q2": "Smoothness",
    "q3": "Immersion",
    "q4": "Continue",
}

FAMILY_NAMES = {
    "A. Language-level features": "Language",
    "B. Narrative-level features": "Narrative",
    "C. Reader experience": "Reader experience",
    "D. Meta-translation": "Meta-translation",
    "M. AI/human-origin cues": "AI/human cues",
}
CODE_FAMILY_NAMES = {
    "A": "Language",
    "B": "Narrative",
    "C": "Reader experience",
    "D": "Meta-translation",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
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


def split_labels(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    labels = []
    for raw in text.split(","):
        label = re.sub(r"\s+", " ", raw).strip()
        if label and label.lower() not in {"none", "nan"}:
            labels.append(collapse_label(label))
    return labels


def short_label(label: str) -> str:
    label = re.sub(r"^[A-Z][0-9a-z]*\.\s*", "", label)
    return label


def wrap_label(label: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def canonical_book_id(book_id: str) -> str:
    return book_id.replace("polish_eval_FIXED_needle_s_eye", "polish_eval_needle_s_eye")


def q3_pos_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("POS\n")]


def q3_neg_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("NEG\n")]


def family_from_column(column: str) -> str:
    family = column.split("\n")[-1].strip()
    return FAMILY_NAMES.get(family, family)


def family_from_label_or_column(label: str, column: str) -> str:
    collapsed_family = collapse_label_code(label)[:1]
    if collapsed_family in {"A", "B", "C", "D"}:
        return CODE_FAMILY_NAMES[collapsed_family]
    return family_from_column(column)


def count_labels_by_columns(row: pd.Series, columns: list[str]) -> tuple[Counter[str], Counter[str]]:
    label_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for column in columns:
        labels = split_labels(row.get(column, ""))
        if not labels:
            continue
        for label in labels:
            family = family_from_label_or_column(label, column)
            label_counts[label] += 1
            family_counts[family] += 1
    return label_counts, family_counts


def count_ai_labels(row: pd.Series, column: str) -> Counter[str]:
    return Counter(split_labels(row.get(column, "")))


def q3_strength(raw_value: str) -> str:
    raw = str(raw_value).strip()
    if raw in {"1", "4"}:
        return "clear"
    if raw in {"2", "3"}:
        return "slight"
    return ""


def classify_scenario(pos_count: int, neg_count: int) -> str:
    if pos_count >= 1 and neg_count >= 1:
        return "Relative MT advantage"
    if pos_count >= 2 and neg_count == 0:
        return "MT strength"
    if neg_count >= 2 and pos_count == 0:
        return "HT weakness"
    return "Weak/uncoded evidence"


def build_mt_case_rows(
    part1: pd.DataFrame, part2: pd.DataFrame, q3: pd.DataFrame, q4: pd.DataFrame
) -> list[dict[str, object]]:
    q3_index = {
        row["assignment_id"]: row
        for _, row in q3.iterrows()
        if row.get("assignment_id") and row.get("done", row.get("DONE", "")).upper() != "FALSE"
    }
    q4_index = {
        row["assignment_id"]: row
        for _, row in q4.iterrows()
        if row.get("assignment_id") and row.get("done", row.get("DONE", "")).upper() != "FALSE"
    }
    part2_counts_by_case: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    part2_difficulty_by_case: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for _, row in part2.iterrows():
        user_id = row.get("user_id", "")
        book_id = canonical_book_id(row.get("book_id", ""))
        preferred = row.get("preferred_translation", "")
        difficulty = row.get("difficulty", "")
        if user_id and book_id and preferred in {"HT", "MT"}:
            key = (user_id, book_id)
            part2_counts_by_case[key][preferred] += 1
            if difficulty:
                part2_difficulty_by_case[key][f"{preferred}_{difficulty}"] += 1

    pos_cols = q3_pos_columns(q3)
    neg_cols = q3_neg_columns(q3)
    rows: list[dict[str, object]] = []
    for _, row in part1.iterrows():
        if row.get("comparison_q3_decipher") != "MT":
            continue
        assignment_id = row.get("participant_id", "")
        # The part1 CSV does not expose assignment_id, so join through q3 by participant/book.
        q3_row = match_coding_row(q3_index.values(), row)
        q4_row = match_coding_row(q4_index.values(), row)
        pos_labels, pos_families = (
            count_labels_by_columns(q3_row, pos_cols) if q3_row is not None else (Counter(), Counter())
        )
        neg_labels, neg_families = (
            count_labels_by_columns(q3_row, neg_cols) if q3_row is not None else (Counter(), Counter())
        )
        ai_labels = count_ai_labels(q4_row, "AI-telling") if q4_row is not None else Counter()
        human_labels = count_ai_labels(q4_row, "Human-telling") if q4_row is not None else Counter()
        q3_assignment = q3_row.get("assignment_id", "") if q3_row is not None else ""
        case_key = (row.get("user_id", ""), canonical_book_id(row.get("book_id", "")))
        chunk_counts = part2_counts_by_case.get(case_key, Counter())
        difficulty_counts = part2_difficulty_by_case.get(case_key, Counter())
        rows.append(
            {
                "user_id": row.get("user_id", ""),
                "participant_id": row.get("participant_id", ""),
                "assignment_id": q3_assignment,
                "book_id": canonical_book_id(row.get("book_id", "")),
                "source_lang": row.get("source_lang", ""),
                "order": f"{row.get('first_version', '')}-first",
                "comparison_q3_raw": row.get("comparison_q3", ""),
                "comparison_q3_strength": q3_strength(row.get("comparison_q3", "")),
                "preferred_continue": row.get("comparison_q3_decipher", ""),
                "thought_ai": row.get("comparison_q5_decipher", ""),
                "ai_confidence": row.get("comparison_q6", ""),
                "mt_positive_label_count": sum(pos_labels.values()),
                "ht_negative_label_count": sum(neg_labels.values()),
                "ai_telling_label_count": sum(ai_labels.values()),
                "human_telling_label_count": sum(human_labels.values()),
                "scenario": classify_scenario(sum(pos_labels.values()), sum(neg_labels.values())),
                "top_mt_positive_labels": "; ".join(label for label, _ in pos_labels.most_common(4)),
                "top_ht_negative_labels": "; ".join(label for label, _ in neg_labels.most_common(4)),
                "top_ai_telling_labels": "; ".join(label for label, _ in ai_labels.most_common(4)),
                "top_human_telling_labels": "; ".join(label for label, _ in human_labels.most_common(4)),
                "mt_positive_families": "; ".join(
                    f"{family}:{count}" for family, count in pos_families.most_common()
                ),
                "ht_negative_families": "; ".join(
                    f"{family}:{count}" for family, count in neg_families.most_common()
                ),
                "chunk_pref_ht": chunk_counts.get("HT", 0),
                "chunk_pref_mt": chunk_counts.get("MT", 0),
                "chunk_pref_mt_rate": (
                    chunk_counts.get("MT", 0) / sum(chunk_counts.values()) if sum(chunk_counts.values()) else 0
                ),
                "chunk_pref_mt_significant": difficulty_counts.get("MT_significantly_better", 0),
                "chunk_pref_mt_better": difficulty_counts.get("MT_better", 0),
                "chunk_pref_mt_similar": difficulty_counts.get("MT_similar_quality", 0),
                "q3_comment": q3_row.get("comment", "") if q3_row is not None else "",
                "q4_ai_comment": q4_row.get("comment", "") if q4_row is not None else "",
            }
        )
    return rows


def match_coding_row(rows: object, part1_row: pd.Series) -> pd.Series | None:
    participant = str(part1_row.get("user_id", "")).split("_")[0]
    book_id = canonical_book_id(str(part1_row.get("book_id", "")))
    candidates = []
    for row in rows:
        row_book = canonical_book_id(str(row.get("book_id", "")))
        row_user = str(row.get("user", ""))
        row_id = str(row.get("ID", ""))
        part_number = re.sub(r"^p0*", "", participant)
        user_match = row_user.startswith(participant) or row_user == participant
        id_match = row_id == part_number
        if row_book == book_id and (user_match or id_match):
            candidates.append(row)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[0]
    return None


def label_records(df: pd.DataFrame, rows: list[dict[str, object]], kind: str) -> list[dict[str, object]]:
    assignment_ids = {row["assignment_id"] for row in rows if row.get("assignment_id")}
    if kind == "q3":
        pos_cols = q3_pos_columns(df)
        neg_cols = q3_neg_columns(df)
        records = []
        for _, row in df.iterrows():
            if row.get("assignment_id") not in assignment_ids:
                continue
            for role, columns in (("MT positive", pos_cols), ("HT negative", neg_cols)):
                for column in columns:
                    for label in split_labels(row.get(column, "")):
                        records.append(
                            {
                                "assignment_id": row.get("assignment_id", ""),
                                "role": role,
                                "family": family_from_label_or_column(label, column),
                                "label": label,
                            }
                        )
        return records
    if kind == "q4":
        records = []
        for _, row in df.iterrows():
            if row.get("assignment_id") not in assignment_ids:
                continue
            for role, column in (("AI cue", "AI-telling"), ("Human cue", "Human-telling")):
                for label in split_labels(row.get(column, "")):
                    records.append(
                        {
                            "assignment_id": row.get("assignment_id", ""),
                            "role": role,
                            "family": "AI/human cues",
                            "label": label,
                        }
                    )
        return records
    if kind == "q5":
        pos_cols = q3_pos_columns(df)
        neg_cols = q3_neg_columns(df)
        records = []
        for _, row in df.iterrows():
            if row.get("assignment_id") not in assignment_ids:
                continue
            chunk_preferred = row.get("chunk_preferred", "")
            if chunk_preferred not in {"HT", "MT"}:
                continue
            rejected = "HT" if chunk_preferred == "MT" else "MT"
            for polarity, columns in (("POS", pos_cols), ("NEG", neg_cols)):
                for column in columns:
                    for label in split_labels(row.get(column, "")):
                        records.append(
                            {
                                "assignment_id": row.get("assignment_id", ""),
                                "book_id": canonical_book_id(row.get("book_id", "")),
                                "user": row.get("user", ""),
                                "chunk_id": row.get("chunk_id", ""),
                                "chunk_preferred": chunk_preferred,
                                "translation_role": "preferred" if polarity == "POS" else "rejected",
                                "translation": chunk_preferred if polarity == "POS" else rejected,
                                "polarity": polarity,
                                "family": family_from_label_or_column(label, column),
                                "label": label,
                            }
                        )
        return records
    raise ValueError(kind)


def all_q3_preference_label_records(q3: pd.DataFrame) -> list[dict[str, object]]:
    pos_cols = q3_pos_columns(q3)
    neg_cols = q3_neg_columns(q3)
    records = []
    for _, row in q3.iterrows():
        preferred = row.get("preferred_continue", "")
        if preferred not in {"HT", "MT"}:
            continue
        if row.get("done", row.get("DONE", "")).upper() == "FALSE":
            continue
        rejected = "HT" if preferred == "MT" else "MT"
        for polarity, columns in (("POS", pos_cols), ("NEG", neg_cols)):
            for column in columns:
                for label in split_labels(row.get(column, "")):
                    records.append(
                        {
                            "assignment_id": row.get("assignment_id", ""),
                            "unit_id": row.get("assignment_id", ""),
                            "book_id": canonical_book_id(row.get("book_id", "")),
                            "user": row.get("user", ""),
                            "preferred_continue": preferred,
                            "translation_role": "preferred" if polarity == "POS" else "rejected",
                            "translation": preferred if polarity == "POS" else rejected,
                            "polarity": polarity,
                            "family": family_from_label_or_column(label, column),
                            "label": label,
                        }
                    )
    return records


def all_q5_chunk_label_records(q5: pd.DataFrame) -> list[dict[str, object]]:
    pos_cols = q3_pos_columns(q5)
    neg_cols = q3_neg_columns(q5)
    records = []
    for _, row in q5.iterrows():
        preferred = row.get("chunk_preferred", "")
        if preferred not in {"HT", "MT"}:
            continue
        rejected = "HT" if preferred == "MT" else "MT"
        for polarity, columns in (("POS", pos_cols), ("NEG", neg_cols)):
            for column in columns:
                for label in split_labels(row.get(column, "")):
                    records.append(
                        {
                            "assignment_id": row.get("assignment_id", ""),
                            "unit_id": "|".join(
                                [
                                    row.get("assignment_id", ""),
                                    canonical_book_id(row.get("book_id", "")),
                                    row.get("chunk_id", ""),
                                ]
                            ),
                            "book_id": canonical_book_id(row.get("book_id", "")),
                            "user": row.get("user", ""),
                            "chunk_id": row.get("chunk_id", ""),
                            "chunk_preferred": preferred,
                            "translation_role": "preferred" if polarity == "POS" else "rejected",
                            "translation": preferred if polarity == "POS" else rejected,
                            "polarity": polarity,
                            "family": family_from_label_or_column(label, column),
                            "label": label,
                        }
                    )
    return records


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    indexed = [
        (idx, value)
        for idx, value in enumerate(p_values)
        if value is not None and pd.notna(value)
    ]
    if not indexed:
        return [None for _ in p_values]
    ranked = sorted(indexed, key=lambda item: item[1], reverse=True)
    total = len(ranked)
    adjusted: dict[int, float] = {}
    running_min = 1.0
    for reverse_rank, (idx, value) in enumerate(ranked, start=1):
        rank = total - reverse_rank + 1
        running_min = min(running_min, value * total / rank)
        adjusted[idx] = min(running_min, 1.0)
    return [adjusted.get(idx) for idx in range(len(p_values))]


def fisher_p(a_present: int, a_total: int, b_present: int, b_total: int) -> float | None:
    if fisher_exact is None:
        return None
    table = [
        [a_present, max(a_total - a_present, 0)],
        [b_present, max(b_total - b_present, 0)],
    ]
    return float(fisher_exact(table, alternative="two-sided").pvalue)


def label_lift_table(
    records: pd.DataFrame,
    unit_column: str,
    group_column: str,
    group_a: str,
    group_b: str,
    label_scope: str,
) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    rows = []
    unit_groups = records[[unit_column, group_column]].drop_duplicates()
    totals = {
        group: int(unit_groups[unit_groups[group_column] == group][unit_column].nunique())
        for group in (group_a, group_b)
    }
    for (polarity, family, label), subset in records.groupby(["polarity", "family", "label"]):
        present = {
            group: int(
                subset[subset[group_column] == group][unit_column].drop_duplicates().shape[0]
            )
            for group in (group_a, group_b)
        }
        rate_a = present[group_a] / totals[group_a] if totals[group_a] else 0.0
        rate_b = present[group_b] / totals[group_b] if totals[group_b] else 0.0
        rows.append(
            {
                "scope": label_scope,
                "polarity": polarity,
                "family": family,
                "label": label,
                f"{group_a}_cases_with_label": present[group_a],
                f"{group_b}_cases_with_label": present[group_b],
                f"{group_a}_total_cases": totals[group_a],
                f"{group_b}_total_cases": totals[group_b],
                f"{group_a}_rate": rate_a,
                f"{group_b}_rate": rate_b,
                "rate_delta_mt_minus_ht": rate_a - rate_b,
                "rate_ratio_mt_over_ht": rate_a / rate_b if rate_b else "",
                "fisher_p": fisher_p(present[group_a], totals[group_a], present[group_b], totals[group_b]),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["bh_q"] = benjamini_hochberg(result["fisher_p"].tolist())
    return result.sort_values(
        ["polarity", "rate_delta_mt_minus_ht", f"{group_a}_cases_with_label"],
        ascending=[True, False, False],
    )


def family_rate_table(
    records: pd.DataFrame,
    unit_column: str,
    group_column: str,
    group_a: str,
    group_b: str,
    label_scope: str,
) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    family_presence = records[[unit_column, group_column, "polarity", "family"]].drop_duplicates()
    rows = []
    unit_groups = records[[unit_column, group_column]].drop_duplicates()
    totals = {
        group: int(unit_groups[unit_groups[group_column] == group][unit_column].nunique())
        for group in (group_a, group_b)
    }
    for (polarity, family), subset in family_presence.groupby(["polarity", "family"]):
        present = {
            group: int(subset[subset[group_column] == group][unit_column].nunique())
            for group in (group_a, group_b)
        }
        rows.append(
            {
                "scope": label_scope,
                "polarity": polarity,
                "family": family,
                f"{group_a}_cases_with_family": present[group_a],
                f"{group_b}_cases_with_family": present[group_b],
                f"{group_a}_total_cases": totals[group_a],
                f"{group_b}_total_cases": totals[group_b],
                f"{group_a}_rate": present[group_a] / totals[group_a] if totals[group_a] else 0,
                f"{group_b}_rate": present[group_b] / totals[group_b] if totals[group_b] else 0,
            }
        )
    return pd.DataFrame(rows)


def plot_preference_ai_crosstab(part1: pd.DataFrame, out_dir: Path) -> None:
    ct = pd.crosstab(part1["comparison_q3_decipher"], part1["comparison_q5_decipher"])
    ct = ct.reindex(index=["HT", "MT"], columns=["HT", "MT"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", cbar=False, linewidths=1, ax=ax)
    ax.set_title("Continue-reading preference vs AI attribution")
    ax.set_xlabel("Translation judged more likely AI-translated")
    ax.set_ylabel("Translation preferred for continued reading")
    save_figure(fig, out_dir, "01_preference_vs_ai_guess_heatmap")


def plot_scenarios(cases: pd.DataFrame, out_dir: Path) -> None:
    counts = cases["scenario"].value_counts().reindex(SCENARIO_COLORS.keys(), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(counts.index, counts.values, color=[SCENARIO_COLORS[key] for key in counts.index])
    ax.bar_label(bars, padding=4)
    ax.set_title("Why MT won in whole-excerpt comparison Q3")
    ax.set_xlabel("MT-preference cases")
    ax.set_ylabel("")
    ax.invert_yaxis()
    save_figure(fig, out_dir, "02_mt_preference_scenario_counts")


def plot_q3_label_families(records: pd.DataFrame, out_dir: Path) -> None:
    if records.empty:
        return
    summary = records.groupby(["role", "family"]).size().reset_index(name="n")
    pivot = summary.pivot(index="family", columns="role", values="n").fillna(0)
    pivot = pivot.reindex(columns=["MT positive", "HT negative"], fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    y = range(len(pivot))
    ax.barh(y, pivot["MT positive"], color=ROLE_COLORS["MT positive"], label="MT positive")
    ax.barh(
        y,
        -pivot["HT negative"],
        color=ROLE_COLORS["HT negative"],
        label="HT negative",
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(list(y), pivot.index)
    limit = max(pivot.max().max(), 1) + 1
    ax.set_xlim(-limit, limit)
    ax.set_title("Q3 explanation label families in MT-preference cases")
    ax.set_xlabel("Label count; HT-negative left, MT-positive right")
    ax.legend(loc="lower right")
    save_figure(fig, out_dir, "03_q3_mt_positive_vs_ht_negative_families")


def plot_top_q3_labels(records: pd.DataFrame, out_dir: Path) -> None:
    if records.empty:
        return
    for role, stem in (("MT positive", "04_top_mt_positive_q3_labels"), ("HT negative", "05_top_ht_negative_q3_labels")):
        subset = records[records["role"] == role]
        counts = subset["label"].value_counts().head(12).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(
            [wrap_label(short_label(label), 42) for label in counts.index],
            counts.values,
            color=ROLE_COLORS[role],
        )
        ax.bar_label(bars, padding=3)
        ax.set_title(f"Top Q3 labels: {role}")
        ax.set_xlabel("Label count")
        ax.set_ylabel("")
        save_figure(fig, out_dir, stem)


def plot_q4_ai_cues(records: pd.DataFrame, out_dir: Path) -> None:
    if records.empty:
        return
    for role, stem in (("AI cue", "06_top_false_ai_cues_for_ht"), ("Human cue", "07_top_human_cues_for_mt")):
        subset = records[records["role"] == role]
        counts = subset["label"].value_counts().head(12).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(
            [wrap_label(short_label(label), 42) for label in counts.index],
            counts.values,
            color=ROLE_COLORS[role],
        )
        ax.bar_label(bars, padding=3)
        title = "Cues that made HT look AI-like" if role == "AI cue" else "Cues that made MT look human-like"
        ax.set_title(title)
        ax.set_xlabel("Label count among MT-preference false-AI cases")
        ax.set_ylabel("")
        save_figure(fig, out_dir, stem)


def plot_chunk_preferences(cases: pd.DataFrame, out_dir: Path) -> None:
    cases = cases.sort_values(["source_lang", "book_id", "user_id"])
    labels = [
        f"{row.user_id}\n{row.book_id.replace('_eval_', ': ').replace('_', ' ')}"
        for row in cases.itertuples()
    ]
    ht = cases["chunk_pref_ht"].astype(float)
    mt = cases["chunk_pref_mt"].astype(float)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    y = range(len(cases))
    ax.barh(y, ht, color=TRANSLATION_COLORS["HT"], label="HT chunks preferred")
    ax.barh(y, mt, left=ht, color=TRANSLATION_COLORS["MT"], label="MT chunks preferred")
    ax.set_yticks(list(y), labels, fontsize=7.5)
    ax.set_title("Chunk-level preferences inside whole-excerpt MT-preference assignments")
    ax.set_xlabel("Chunk judgments")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    save_figure(fig, out_dir, "08_chunk_preferences_for_mt_preference_cases")


def plot_q5_chunk_labels(records: pd.DataFrame, out_dir: Path) -> None:
    if records.empty:
        return
    mt_chunks = records[records["chunk_preferred"] == "MT"]
    if mt_chunks.empty:
        return
    family_summary = (
        mt_chunks.groupby(["polarity", "family"]).size().reset_index(name="n")
    )
    pivot = family_summary.pivot(index="family", columns="polarity", values="n").fillna(0)
    pivot = pivot.reindex(columns=["POS", "NEG"], fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    y = range(len(pivot))
    ax.barh(y, pivot["POS"], color=ROLE_COLORS["MT positive"], label="POS for preferred MT")
    ax.barh(y, -pivot["NEG"], color=ROLE_COLORS["HT negative"], label="NEG for rejected HT")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(list(y), pivot.index)
    limit = max(pivot.max().max(), 1) + 1
    ax.set_xlim(-limit, limit)
    ax.set_title("Chunk-level Q5 labels when MT-preferring readers chose MT chunks")
    ax.set_xlabel("Label count; rejected-HT NEG left, preferred-MT POS right")
    ax.legend(loc="lower right")
    save_figure(fig, out_dir, "11_q5_mt_chunk_positive_vs_ht_negative_families")

    for polarity, stem, title, color in (
        ("POS", "12_top_q5_pos_labels_for_mt_chunks", "Top chunk POS labels for preferred MT", ROLE_COLORS["MT positive"]),
        ("NEG", "13_top_q5_neg_labels_for_rejected_ht_chunks", "Top chunk NEG labels for rejected HT", ROLE_COLORS["HT negative"]),
    ):
        subset = mt_chunks[mt_chunks["polarity"] == polarity]
        counts = subset["label"].value_counts().head(12).sort_values()
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh([wrap_label(short_label(label), 42) for label in counts.index], counts.values, color=color)
        ax.bar_label(bars, padding=3)
        ax.set_title(title)
        ax.set_xlabel("Label count")
        ax.set_ylabel("")
        save_figure(fig, out_dir, stem)


def plot_family_norms(family_rates: pd.DataFrame, out_dir: Path, stem: str, title: str) -> None:
    if family_rates.empty:
        return
    plot_df = family_rates.copy()
    plot_df["family_label"] = plot_df["polarity"] + ": " + plot_df["family"]
    plot_df = plot_df.sort_values(["polarity", "family"])
    y = range(len(plot_df))
    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.45 * len(plot_df) + 1.6)))
    ax.scatter(
        100 * plot_df["HT_rate"],
        y,
        color=ROLE_COLORS["HT preference"],
        s=70,
        label="HT-preference cases",
        zorder=3,
    )
    ax.scatter(
        100 * plot_df["MT_rate"],
        y,
        color=ROLE_COLORS["MT preference"],
        s=70,
        label="MT-preference cases",
        zorder=3,
    )
    for idx, row in enumerate(plot_df.itertuples()):
        ax.plot(
            [100 * row.HT_rate, 100 * row.MT_rate],
            [idx, idx],
            color="#B0B0B0",
            linewidth=1.8,
            zorder=1,
        )
    ax.set_yticks(list(y), plot_df["family_label"])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Share of cases with at least one label in this family (%)")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.legend(loc="lower right")
    save_figure(fig, out_dir, stem)


def plot_label_lift(
    lift: pd.DataFrame,
    out_dir: Path,
    polarity: str,
    stem: str,
    title: str,
    min_presence: int = 2,
) -> None:
    if lift.empty:
        return
    plot_df = lift[lift["polarity"] == polarity].copy()
    if plot_df.empty:
        return
    plot_df["max_presence"] = plot_df[["MT_cases_with_label", "HT_cases_with_label"]].max(axis=1)
    plot_df = plot_df[plot_df["max_presence"] >= min_presence]
    if plot_df.empty:
        return
    plot_df = plot_df.reindex(
        plot_df["rate_delta_mt_minus_ht"].abs().sort_values(ascending=False).head(14).index
    ).sort_values("rate_delta_mt_minus_ht")
    colors = [
        ROLE_COLORS["MT preference"] if value > 0 else ROLE_COLORS["HT preference"]
        for value in plot_df["rate_delta_mt_minus_ht"]
    ]
    labels = []
    for row in plot_df.itertuples():
        sig = ""
        if pd.notna(row.bh_q) and row.bh_q < 0.1:
            sig = " *"
        labels.append(f"{wrap_label(short_label(row.label), 40)}{sig}")
    fig, ax = plt.subplots(figsize=(10, max(5.5, 0.5 * len(plot_df) + 1.5)))
    bars = ax.barh(labels, 100 * plot_df["rate_delta_mt_minus_ht"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.bar_label(bars, labels=[f"{value:+.0f} pp" for value in 100 * plot_df["rate_delta_mt_minus_ht"]], padding=3)
    ax.set_title(title)
    ax.set_xlabel("Rate difference, MT-preference cases minus HT-preference cases")
    ax.set_ylabel("")
    ax.text(
        0.99,
        0.02,
        "* BH q < 0.10",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    save_figure(fig, out_dir, stem)


def plot_isolated_rating_deltas(part1: pd.DataFrame, cases: pd.DataFrame, out_dir: Path) -> None:
    case_keys = set(zip(cases["user_id"], cases["book_id"]))
    rows = []
    for _, row in part1.iterrows():
        key = (row["user_id"], canonical_book_id(row["book_id"]))
        if key not in case_keys:
            continue
        for q in Q_LABELS:
            values = {}
            for prefix in ("first", "second"):
                version = row[f"{prefix}_version"]
                values[version] = float(row[f"{prefix}_{q}"])
            if {"HT", "MT"} <= set(values):
                rows.append({"question": Q_LABELS[q], "HT": values["HT"], "MT": values["MT"]})
    df = pd.DataFrame(rows)
    means = df.groupby("question")[["HT", "MT"]].mean().reindex(Q_LABELS.values())
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = range(len(means))
    for idx, (_, vals) in enumerate(means.iterrows()):
        ax.plot([vals["HT"], vals["MT"]], [idx, idx], color="#999999", linewidth=2)
        ax.scatter(vals["HT"], idx, color=TRANSLATION_COLORS["HT"], s=90, label="HT" if idx == 0 else "")
        ax.scatter(vals["MT"], idx, color=TRANSLATION_COLORS["MT"], s=90, label="MT" if idx == 0 else "")
        ax.text(vals["HT"] - 0.04, idx + 0.12, f"{vals['HT']:.2f}", ha="right", va="center", fontsize=9)
        ax.text(vals["MT"] + 0.04, idx + 0.12, f"{vals['MT']:.2f}", ha="left", va="center", fontsize=9)
    ax.set_yticks(list(y), means.index)
    ax.set_xlim(1, 5)
    ax.set_title("Single-reading ratings for assignments that later preferred MT")
    ax.set_xlabel("Mean isolated rating, 1 worst to 5 best")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    save_figure(fig, out_dir, "09_isolated_rating_means_for_mt_preference_cases")


def plot_book_summary(part1: pd.DataFrame, part2: pd.DataFrame, cases: pd.DataFrame, out_dir: Path) -> None:
    part1 = part1.copy()
    part1["book_id"] = part1["book_id"].map(canonical_book_id)
    part2 = part2.copy()
    part2["book_id"] = part2["book_id"].map(canonical_book_id)
    rows = []
    for book_id, book_rows in part1.groupby("book_id"):
        chunk_rows = part2[part2["book_id"].map(canonical_book_id) == book_id]
        case_count = int((book_rows["comparison_q3_decipher"] == "MT").sum())
        false_ai_count = int(
            (
                (book_rows["comparison_q3_decipher"] == "MT")
                & (book_rows["comparison_q5_decipher"] == "HT")
            ).sum()
        )
        total_chunks = len(chunk_rows)
        rows.append(
            {
                "book_id": book_id,
                "comparison_mt_preference_count": case_count,
                "mt_preference_false_ai_count": false_ai_count,
                "chunk_mt_rate": (
                    (chunk_rows["preferred_translation"] == "MT").sum() / total_chunks
                    if total_chunks
                    else 0
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["comparison_mt_preference_count", "chunk_mt_rate"], ascending=[False, False]
    )
    summary.to_csv(out_dir / "book_level_mt_favorability.csv", index=False)
    plot_df = summary[summary["comparison_mt_preference_count"] > 0].copy()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(plot_df))
    ax1.bar(
        x,
        plot_df["comparison_mt_preference_count"],
        color=TRANSLATION_COLORS["MT"],
        alpha=0.85,
        label="Whole-excerpt MT preferences",
    )
    ax1.bar(
        x,
        plot_df["mt_preference_false_ai_count"],
        color="#8C2D04",
        alpha=0.55,
        label="Also judged HT more AI-like",
    )
    ax1.set_ylabel("Comparison-case count")
    ax1.set_xticks(x, [wrap_label(book.replace("_eval_", ": ").replace("_", " "), 22) for book in plot_df["book_id"]], rotation=45, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, 100 * plot_df["chunk_mt_rate"], color="#333333", marker="o", linewidth=2, label="Chunk MT preference rate")
    ax2.set_ylabel("Chunk-level MT preference rate (%)")
    ax2.set_ylim(0, 100)
    ax1.set_title("Books where readers preferred MT in whole-excerpt comparison")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    save_figure(fig, out_dir, "10_book_level_mt_favorability")


def write_markdown_summary(
    out_dir: Path,
    cases: pd.DataFrame,
    crosstab: pd.DataFrame,
    q3_lift: pd.DataFrame,
) -> None:
    scenario_counts = cases["scenario"].value_counts()
    lines = [
        "# MT Preference And AI Undetectability Analysis",
        "",
        "This analysis focuses on whole-excerpt comparison cases where participants preferred MT",
        "for continued reading and judged HT as more likely AI-translated.",
        "",
        "## Core Counts",
        "",
        f"- Part 1 comparison rows: {int(crosstab.to_numpy().sum())}",
        f"- MT-preference cases: {len(cases)}",
        f"- MT-preference cases that judged HT more AI-like: {int((cases['thought_ai'] == 'HT').sum())}",
        "",
        "Preference by AI-attribution crosstab:",
        "",
        "| Preferred to continue | Thought HT AI | Thought MT AI |",
        "| --- | ---: | ---: |",
    ]
    for preferred in crosstab.index:
        lines.append(
            f"| {preferred} | {int(crosstab.loc[preferred, 'HT'])} | "
            f"{int(crosstab.loc[preferred, 'MT'])} |"
        )
    lines.extend(
        [
            "",
            "## Scenario Counts",
            "",
            "| Scenario | Cases |",
            "| --- | ---: |",
        ]
    )
    for scenario, count in scenario_counts.items():
        lines.append(f"| {scenario} | {int(count)} |")
    if not q3_lift.empty:
        lines.extend(
            [
                "",
                "## Strongest Q3 Label Differences Against HT-Preference Norm",
                "",
                "Positive values mean the label appears in a larger share of MT-preference cases than",
                "HT-preference cases. Fisher p-values and BH q-values are exploratory because n is small.",
                "",
                "| Polarity | Label | MT rate | HT rate | Delta | Fisher p | BH q |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        top = q3_lift.copy()
        top["abs_delta"] = top["rate_delta_mt_minus_ht"].abs()
        top = top.sort_values(["abs_delta", "MT_cases_with_label"], ascending=[False, False]).head(8)
        for row in top.itertuples():
            p_value = "" if pd.isna(row.fisher_p) else f"{row.fisher_p:.3f}"
            q_value = "" if pd.isna(row.bh_q) else f"{row.bh_q:.3f}"
            lines.append(
                f"| {row.polarity} | {short_label(row.label)} | {100 * row.MT_rate:.0f}% | "
                f"{100 * row.HT_rate:.0f}% | {100 * row.rate_delta_mt_minus_ht:+.0f} pp | "
                f"{p_value} | {q_value} |"
            )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `mt_preference_cases.csv`: one row per whole-excerpt MT-preference case.",
            "- `q3_mt_preference_label_records.csv`: POS/NEG labels from comparison preference explanations.",
            "- `q4_false_ai_label_records.csv`: AI/human-origin cue labels for the same cases.",
            "- `q5_chunk_label_records_for_mt_preference_cases.csv`: chunk-level POS/NEG labels for the same assignments.",
            "- `q3_preference_norm_label_lift.csv`: Q3 MT-preference vs HT-preference label rates and tests.",
            "- `q3_preference_norm_family_rates.csv`: Q3 family-level baseline rates by preference group.",
            "- `q5_chunk_preference_norm_label_lift.csv`: Q5 chunk-level MT-preferred vs HT-preferred label rates and tests.",
            "- `q5_chunk_preference_norm_family_rates.csv`: Q5 family-level baseline rates by chunk preference.",
            "- `book_level_mt_favorability.csv`: book-level comparison and chunk preference summary.",
            "- `*.png` / `*.pdf`: plots for the analysis.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, default=DEFAULT_PART1)
    parser.add_argument("--part2", type=Path, default=DEFAULT_PART2)
    parser.add_argument("--q3", type=Path, default=DEFAULT_Q3)
    parser.add_argument("--q4", type=Path, default=DEFAULT_Q4)
    parser.add_argument("--q5", type=Path, default=DEFAULT_Q5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_plot_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    part1 = read_csv(args.part1)
    part1["book_id"] = part1["book_id"].map(canonical_book_id)
    part2 = read_csv(args.part2)
    part2["book_id"] = part2["book_id"].map(canonical_book_id)
    q3 = read_csv(args.q3)
    q4 = read_csv(args.q4)
    q5 = read_csv(args.q5)

    case_rows = build_mt_case_rows(part1, part2, q3, q4)
    fieldnames = list(case_rows[0].keys()) if case_rows else []
    write_csv(args.out_dir / "mt_preference_cases.csv", case_rows, fieldnames)
    cases = pd.DataFrame(case_rows)

    q3_records = pd.DataFrame(label_records(q3, case_rows, "q3"))
    q4_records = pd.DataFrame(label_records(q4, case_rows, "q4"))
    q5_records = pd.DataFrame(label_records(q5, case_rows, "q5"))
    all_q3_records = pd.DataFrame(all_q3_preference_label_records(q3))
    all_q5_records = pd.DataFrame(all_q5_chunk_label_records(q5))
    q3_records.to_csv(args.out_dir / "q3_mt_preference_label_records.csv", index=False)
    q4_records.to_csv(args.out_dir / "q4_false_ai_label_records.csv", index=False)
    q5_records.to_csv(args.out_dir / "q5_chunk_label_records_for_mt_preference_cases.csv", index=False)
    all_q3_records.to_csv(args.out_dir / "q3_all_preference_label_records.csv", index=False)
    all_q5_records.to_csv(args.out_dir / "q5_all_chunk_preference_label_records.csv", index=False)

    q3_lift = label_lift_table(
        all_q3_records,
        unit_column="unit_id",
        group_column="preferred_continue",
        group_a="MT",
        group_b="HT",
        label_scope="Q3 whole-excerpt preference explanations",
    )
    q3_family_rates = family_rate_table(
        all_q3_records,
        unit_column="unit_id",
        group_column="preferred_continue",
        group_a="MT",
        group_b="HT",
        label_scope="Q3 whole-excerpt preference explanations",
    )
    q5_lift = label_lift_table(
        all_q5_records,
        unit_column="unit_id",
        group_column="chunk_preferred",
        group_a="MT",
        group_b="HT",
        label_scope="Q5 chunk preference justifications",
    )
    q5_family_rates = family_rate_table(
        all_q5_records,
        unit_column="unit_id",
        group_column="chunk_preferred",
        group_a="MT",
        group_b="HT",
        label_scope="Q5 chunk preference justifications",
    )
    q3_lift.to_csv(args.out_dir / "q3_preference_norm_label_lift.csv", index=False)
    q3_family_rates.to_csv(args.out_dir / "q3_preference_norm_family_rates.csv", index=False)
    q5_lift.to_csv(args.out_dir / "q5_chunk_preference_norm_label_lift.csv", index=False)
    q5_family_rates.to_csv(args.out_dir / "q5_chunk_preference_norm_family_rates.csv", index=False)

    crosstab = pd.crosstab(part1["comparison_q3_decipher"], part1["comparison_q5_decipher"])
    crosstab = crosstab.reindex(index=["HT", "MT"], columns=["HT", "MT"], fill_value=0)
    crosstab.to_csv(args.out_dir / "preference_vs_ai_guess_crosstab.csv")

    plot_preference_ai_crosstab(part1, args.out_dir)
    plot_scenarios(cases, args.out_dir)
    plot_q3_label_families(q3_records, args.out_dir)
    plot_top_q3_labels(q3_records, args.out_dir)
    plot_q4_ai_cues(q4_records, args.out_dir)
    plot_chunk_preferences(cases, args.out_dir)
    plot_q5_chunk_labels(q5_records, args.out_dir)
    plot_family_norms(
        q3_family_rates,
        args.out_dir,
        "14_q3_family_rates_mt_preference_vs_ht_preference",
        "Q3 family rates: MT-preference cases against HT-preference norm",
    )
    plot_label_lift(
        q3_lift,
        args.out_dir,
        "POS",
        "15_q3_pos_label_lift_mt_preference_vs_ht_preference",
        "Q3 POS label lift: preferred MT cases vs preferred HT cases",
    )
    plot_label_lift(
        q3_lift,
        args.out_dir,
        "NEG",
        "16_q3_neg_label_lift_mt_preference_vs_ht_preference",
        "Q3 NEG label lift: rejected HT in MT cases vs rejected MT in HT cases",
    )
    plot_family_norms(
        q5_family_rates,
        args.out_dir,
        "17_q5_family_rates_mt_chunks_vs_ht_chunks",
        "Q5 family rates: MT-preferred chunks against HT-preferred chunk norm",
    )
    plot_label_lift(
        q5_lift,
        args.out_dir,
        "POS",
        "18_q5_pos_label_lift_mt_chunks_vs_ht_chunks",
        "Q5 POS label lift: MT-preferred chunks vs HT-preferred chunks",
        min_presence=8,
    )
    plot_label_lift(
        q5_lift,
        args.out_dir,
        "NEG",
        "19_q5_neg_label_lift_mt_chunks_vs_ht_chunks",
        "Q5 NEG label lift: rejected HT in MT chunks vs rejected MT in HT chunks",
        min_presence=8,
    )
    plot_isolated_rating_deltas(part1, cases, args.out_dir)
    plot_book_summary(part1, part2, cases, args.out_dir)
    write_markdown_summary(args.out_dir, cases, crosstab, q3_lift)

    print(f"Wrote MT preference analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
