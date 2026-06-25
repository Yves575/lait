"""Analyze chunk-review automatic metrics against human HT/MT preferences.

The script joins MetricX-QE and LitTransProQA chunk scores from
``results_chunk_review_eval/books`` with Part 2 human-evaluation preferences from
``human_eval/data/part2-study-data-full.csv``. It writes tidy CSV tables,
Kendall tau summaries, and plots under ``results_chunk_review_eval/analysis``.

Example:
  .venv/bin/python scripts/analyze_chunk_review_metrics.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer


DEFAULT_RESULTS_DIR = Path("results_chunk_review_eval")
DEFAULT_HUMAN_PATH = Path("human_eval/data/part2-study-data-full.csv")
BOOK_RESULTS_DIR_NAME = "books"
ANALYSIS_DIR_NAME = "analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / ANALYSIS_DIR_NAME / "humeval_metric_analysis"

LANG_PREFIXES = {
    "french_eval_": ("fr", "French"),
    "japanese_eval_": ("ja", "Japanese"),
    "polish_eval_": ("pl", "Polish"),
}

LANG_LABELS = {"fr": "French", "ja": "Japanese", "pl": "Polish"}
SYSTEM_LABELS = {"ht": "HT", "mt": "MT"}
PREFERENCE_ORDER = ["HT", "MT", "tie"]
PREFERENCE_COLORS = {"HT": "#4C78A8", "MT": "#F58518", "tie": "#9CA3AF"}
METRIC_LABELS = {
    "litransproqa": "LitTransProQA",
    "metricx-qe": "MetricX-QE",
    "paragraph_mean_comet22": "COMET-22 (paragraph mean)",
    "paragraph_mean_cometkiwi": "COMETKiwi (paragraph mean)",
    "paragraph_mean_metricx-qe": "MetricX-QE (paragraph mean)",
    "paragraph_mean_metricx": "MetricX (paragraph mean)",
}


def ci95(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))


ERROR_KW = {"ecolor": "black", "elinewidth": 1.4, "capsize": 5, "capthick": 1.4}


def annotate_bars(ax: plt.Axes, fmt: str = ".3g", fontsize: int = 8) -> None:
    for container in ax.containers:
        if not isinstance(container, BarContainer):
            continue
        ax.bar_label(container, fmt=f"%{fmt}", padding=2, fontsize=fontsize)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: JSON value is not an object")
            rows.append(record)
    return rows


def source_lang_from_pkl(pkl: str) -> str:
    for suffix, lang in (("_fr_en", "fr"), ("_ja_en", "ja"), ("_pl_en", "pl")):
        if pkl.endswith(suffix):
            return lang
    return ""


def slug_from_pkl(pkl: str) -> str:
    lang = source_lang_from_pkl(pkl)
    if lang:
        return pkl[: -len(f"_{lang}_en")]
    return pkl


def canonical_human_slug(book_id: str) -> tuple[str, str]:
    slug = book_id
    lang = ""
    for prefix, (lang_code, _lang_label) in LANG_PREFIXES.items():
        if slug.startswith(prefix):
            slug = slug[len(prefix) :]
            lang = lang_code
            break
    if slug.startswith("FIXED_"):
        slug = slug[len("FIXED_") :]
    return slug, lang


def collect_metric_pairs(
    results_dir: Path,
    tie_epsilon: float,
) -> pd.DataFrame:
    metric_info: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    book_results_dir = results_dir / BOOK_RESULTS_DIR_NAME
    if not book_results_dir.exists():
        book_results_dir = results_dir

    for system_score_path in sorted(book_results_dir.glob("*/*/system_scores.json")):
        system_dir = system_score_path.parent
        segment_path = system_dir / "segment_scores.jsonl"
        if not segment_path.exists():
            continue

        system_data = json.loads(system_score_path.read_text(encoding="utf-8"))
        pkl = str(system_data["pkl"])
        system_name = str(system_data["system"])
        system_kind = system_name.rsplit("_", 1)[-1]
        if system_kind not in SYSTEM_LABELS:
            continue

        for metric, details in system_data.get("metrics", {}).items():
            metric_info[metric] = {
                "higher_is_better": bool(details.get("higher_is_better", True)),
            }

        for segment in read_jsonl(segment_path):
            scores = segment.get("scores", {})
            if not isinstance(scores, dict):
                continue
            for metric, score in scores.items():
                if score is None:
                    continue
                rows.append(
                    {
                        "pkl": pkl,
                        "book_slug": slug_from_pkl(pkl),
                        "source_lang": source_lang_from_pkl(pkl),
                        "source_lang_label": LANG_LABELS.get(source_lang_from_pkl(pkl), ""),
                        "chunk_id": int(segment["segment_index"]),
                        "system": SYSTEM_LABELS[system_kind],
                        "metric": metric,
                        "score": float(score),
                    }
                )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError(f"No segment scores found under {results_dir}")

    pair_df = (
        long_df.pivot_table(
            index=["pkl", "book_slug", "source_lang", "source_lang_label", "chunk_id", "metric"],
            columns="system",
            values="score",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pair_df = pair_df.rename(columns={"HT": "ht_score", "MT": "mt_score"})
    pair_df = pair_df.dropna(subset=["ht_score", "mt_score"]).copy()

    pair_df["higher_is_better"] = pair_df["metric"].map(
        lambda metric: bool(metric_info.get(metric, {}).get("higher_is_better", True))
    )
    pair_df["raw_delta_ht_minus_mt"] = pair_df["ht_score"] - pair_df["mt_score"]
    pair_df["preference_delta"] = pair_df.apply(
        lambda row: row["ht_score"] - row["mt_score"]
        if row["higher_is_better"]
        else row["mt_score"] - row["ht_score"],
        axis=1,
    )
    pair_df["metric_preference"] = pair_df["preference_delta"].map(
        lambda value: "tie"
        if abs(value) <= tie_epsilon
        else ("HT" if value > 0 else "MT")
    )
    pair_df["metric_preference_num"] = pair_df["metric_preference"].map(
        {"HT": 1, "MT": -1, "tie": 0}
    )
    pair_df["metric_label"] = pair_df["metric"].map(lambda value: METRIC_LABELS.get(value, value))
    return pair_df.sort_values(["pkl", "chunk_id", "metric"]).reset_index(drop=True)


def collect_human_preferences(human_path: Path) -> pd.DataFrame:
    human_df = pd.read_csv(human_path)
    human_df = human_df.rename(columns={"preferred_translation": "human_preference"})
    human_df["human_preference"] = human_df["human_preference"].fillna("")
    human_df = human_df[human_df["human_preference"].isin(["HT", "MT"])].copy()

    slugs_langs = human_df["book_id"].map(canonical_human_slug)
    human_df["book_slug"] = slugs_langs.map(lambda item: item[0])
    human_df["source_lang"] = slugs_langs.map(lambda item: item[1])
    human_df["pkl"] = human_df["book_slug"] + "_" + human_df["source_lang"] + "_en"
    human_df["chunk_id"] = human_df["chunk_id"].astype(int)
    human_df["difficulty"] = human_df["difficulty"].fillna("")
    human_df["human_preference_num"] = human_df["human_preference"].map({"HT": 1, "MT": -1})
    human_df["human_preference_hard_as_tie"] = human_df.apply(
        lambda row: "tie" if row["difficulty"] == "similar_quality" else row["human_preference"],
        axis=1,
    )
    human_df["human_preference_hard_as_tie_num"] = human_df[
        "human_preference_hard_as_tie"
    ].map({"HT": 1, "MT": -1, "tie": 0})
    agreement = (
        human_df.groupby(["pkl", "book_slug", "source_lang", "chunk_id"])
        .agg(
            human_chunk_n_judgments=("human_preference", "size"),
            human_chunk_n_unique_preferences=("human_preference", "nunique"),
            human_chunk_preferences=("human_preference", lambda values: ",".join(sorted(values))),
        )
        .reset_index()
    )
    agreement["human_chunk_raters_agree"] = (
        (agreement["human_chunk_n_judgments"] >= 2)
        & (agreement["human_chunk_n_unique_preferences"] == 1)
    )
    agreement["human_chunk_agreed_preference"] = agreement.apply(
        lambda row: row["human_chunk_preferences"].split(",")[0]
        if row["human_chunk_raters_agree"]
        else "",
        axis=1,
    )
    human_df = human_df.merge(
        agreement,
        on=["pkl", "book_slug", "source_lang", "chunk_id"],
        how="left",
    )
    return human_df[
        [
            "participant_id",
            "book_id",
            "pkl",
            "book_slug",
            "source_lang",
            "chunk_id",
            "human_preference",
            "human_preference_num",
            "human_preference_hard_as_tie",
            "human_preference_hard_as_tie_num",
            "difficulty",
            "human_chunk_n_judgments",
            "human_chunk_n_unique_preferences",
            "human_chunk_preferences",
            "human_chunk_raters_agree",
            "human_chunk_agreed_preference",
            "justification",
        ]
    ].copy()


def build_participant_table(metric_pairs: pd.DataFrame, human_df: pd.DataFrame) -> pd.DataFrame:
    joined = metric_pairs.merge(
        human_df,
        on=["pkl", "book_slug", "source_lang", "chunk_id"],
        how="left",
    )
    joined["human_agrees_with_metric"] = joined.apply(
        lambda row: bool(row["metric_preference"] == row["human_preference"])
        if row.get("human_preference") in {"HT", "MT"} and row["metric_preference"] in {"HT", "MT"}
        else pd.NA,
        axis=1,
    )
    joined["hard_as_tie_agrees_with_metric"] = joined.apply(
        lambda row: bool(row["metric_preference"] == row["human_preference_hard_as_tie"])
        if row.get("human_preference_hard_as_tie") in {"HT", "MT", "tie"}
        else pd.NA,
        axis=1,
    )
    return joined.sort_values(["pkl", "chunk_id", "participant_id", "metric"]).reset_index(
        drop=True
    )


def build_majority_table(metric_pairs: pd.DataFrame, human_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        human_df.groupby(["pkl", "book_slug", "source_lang", "chunk_id", "human_preference"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for choice in ["HT", "MT"]:
        if choice not in counts:
            counts[choice] = 0
    counts = counts.rename(columns={"HT": "human_ht_count", "MT": "human_mt_count"})
    counts["human_total_count"] = counts["human_ht_count"] + counts["human_mt_count"]
    counts["human_majority_preference"] = counts.apply(
        lambda row: "HT"
        if row["human_ht_count"] > row["human_mt_count"]
        else ("MT" if row["human_mt_count"] > row["human_ht_count"] else "tie"),
        axis=1,
    )
    counts["human_majority_num"] = counts["human_majority_preference"].map(
        {"HT": 1, "MT": -1, "tie": 0}
    )
    joined = metric_pairs.merge(
        counts,
        on=["pkl", "book_slug", "source_lang", "chunk_id"],
        how="left",
    )
    joined["majority_agrees_with_metric"] = joined.apply(
        lambda row: bool(row["metric_preference"] == row["human_majority_preference"])
        if row.get("human_majority_preference") in {"HT", "MT", "tie"}
        else pd.NA,
        axis=1,
    )
    return joined.sort_values(["pkl", "chunk_id", "metric"]).reset_index(drop=True)


def kendall_summary(
    df: pd.DataFrame,
    human_col: str,
    metric_col: str,
    group_cols: list[str],
    label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.dropna(subset=[human_col, metric_col]).groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        clean = group[[human_col, metric_col]].dropna()
        total_pairs = len(clean)
        concordant_pairs = int(
            (
                clean[human_col].isin(["HT", "MT"])
                & clean[metric_col].isin(["HT", "MT"])
                & (clean[human_col] == clean[metric_col])
            ).sum()
        )
        discordant_pairs = int(
            (
                clean[human_col].isin(["HT", "MT"])
                & clean[metric_col].isin(["HT", "MT"])
                & (clean[human_col] != clean[metric_col])
            ).sum()
        )
        tie_pairs = total_pairs - concordant_pairs - discordant_pairs
        tau = (
            (concordant_pairs - discordant_pairs) / total_pairs
            if total_pairs
            else math.nan
        )
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "variant": label,
                "total_pairs": total_pairs,
                "concordant_pairs": concordant_pairs,
                "discordant_pairs": discordant_pairs,
                "tie_pairs": tie_pairs,
                "kendall_tau": tau,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_agreement(participant_table: pd.DataFrame, majority_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    participant_valid = participant_table.dropna(subset=["human_preference_num"]).copy()
    majority_valid = majority_table.dropna(subset=["human_majority_num"]).copy()
    agreed_chunk_valid = (
        participant_valid[participant_valid["human_chunk_raters_agree"] == True]
        .drop_duplicates(["pkl", "chunk_id", "metric"])
        .copy()
    )

    litrans_participant = participant_valid[participant_valid["metric"] == "litransproqa"].copy()
    metricx_participant = participant_valid[participant_valid["metric"] == "metricx-qe"].copy()
    litrans_agreed = agreed_chunk_valid[agreed_chunk_valid["metric"] == "litransproqa"].copy()
    metricx_agreed = agreed_chunk_valid[agreed_chunk_valid["metric"] == "metricx-qe"].copy()

    correlations = pd.concat(
        [
            kendall_summary(
                participant_valid,
                "human_preference",
                "metric_preference",
                ["metric"],
                "participant_strict",
            ),
            kendall_summary(
                litrans_participant,
                "human_preference_hard_as_tie",
                "metric_preference",
                ["metric"],
                "litransproqa_similar_quality_as_tie",
            ),
            kendall_summary(
                majority_valid,
                "human_majority_preference",
                "metric_preference",
                ["metric"],
                "chunk_majority",
            ),
            kendall_summary(
                agreed_chunk_valid,
                "human_preference",
                "metric_preference",
                ["metric"],
                "human_agreed_chunks_strict",
            ),
            kendall_summary(
                litrans_agreed,
                "human_preference_hard_as_tie",
                "metric_preference",
                ["metric"],
                "human_agreed_chunks_litransproqa_similar_quality_as_tie",
            ),
            kendall_summary(
                participant_valid,
                "human_preference",
                "metric_preference",
                ["metric", "source_lang"],
                "participant_strict_by_language",
            ),
            kendall_summary(
                litrans_participant,
                "human_preference_hard_as_tie",
                "metric_preference",
                ["metric", "source_lang"],
                "litransproqa_similar_quality_as_tie_by_language",
            ),
            kendall_summary(
                majority_valid,
                "human_majority_preference",
                "metric_preference",
                ["metric", "source_lang"],
                "chunk_majority_by_language",
            ),
            kendall_summary(
                agreed_chunk_valid,
                "human_preference",
                "metric_preference",
                ["metric", "source_lang"],
                "human_agreed_chunks_strict_by_language",
            ),
        ],
        ignore_index=True,
    )

    by_participant = kendall_summary(
        participant_valid,
        "human_preference",
        "metric_preference",
        ["metric", "participant_id"],
        "participant_strict",
    )

    agreement = (
        participant_valid.groupby(["metric", "metric_preference", "human_preference"])
        .size()
        .reset_index(name="n")
    )
    hard_agreement = (
        participant_valid.groupby(
            ["metric", "metric_preference", "human_preference_hard_as_tie"]
        )
        .size()
        .reset_index(name="n")
    )
    accuracy_rows: list[dict[str, Any]] = []
    for metric, group in participant_valid.groupby("metric"):
        no_tie = group[group["metric_preference"].isin(["HT", "MT"])]
        accuracy_rows.append(
            {
                "metric": metric,
                "variant": "participant_strict_no_metric_ties",
                "n": len(no_tie),
                "agreement_rate": float(no_tie["human_agrees_with_metric"].mean())
                if len(no_tie)
                else math.nan,
                "metric_ties": int((group["metric_preference"] == "tie").sum()),
                "human_similar_quality": int((group["difficulty"] == "similar_quality").sum()),
                "human_agreed_chunks": int((group["human_chunk_raters_agree"] == True).sum()),
            }
        )
        agreed = group[(group["human_chunk_raters_agree"] == True) & group["metric_preference"].isin(["HT", "MT"])]
        accuracy_rows.append(
            {
                "metric": metric,
                "variant": "human_agreed_chunks_strict_no_metric_ties",
                "n": len(agreed),
                "agreement_rate": float(agreed["human_agrees_with_metric"].mean())
                if len(agreed)
                else math.nan,
                "metric_ties": int((group["metric_preference"] == "tie").sum()),
                "human_similar_quality": int((group["difficulty"] == "similar_quality").sum()),
                "human_agreed_chunks": int((group["human_chunk_raters_agree"] == True).sum()),
            }
        )
        if metric == "litransproqa":
            hard = group[group["human_preference_hard_as_tie"].isin(["HT", "MT", "tie"])]
            accuracy_rows.append(
                {
                    "metric": metric,
                    "variant": "litransproqa_similar_quality_as_tie",
                    "n": len(hard),
                    "agreement_rate": float(hard["hard_as_tie_agrees_with_metric"].mean())
                    if len(hard)
                    else math.nan,
                    "metric_ties": int((group["metric_preference"] == "tie").sum()),
                    "human_similar_quality": int(
                        (group["difficulty"] == "similar_quality").sum()
                    ),
                    "human_agreed_chunks": int((group["human_chunk_raters_agree"] == True).sum()),
                }
            )
    accuracy = pd.DataFrame(accuracy_rows)

    return {
        "correlations": correlations,
        "correlations_by_participant": by_participant,
        "agreement_counts": agreement,
        "agreement_counts_similar_quality_as_tie": hard_agreement,
        "agreement_rates": accuracy,
    }


def paired_permutation_pvalue(
    deltas: np.ndarray,
    rng: np.random.Generator,
    num_resamples: int,
) -> float:
    """One-sided paired permutation p-value for mean(delta) >= observed mean."""
    clean = deltas[np.isfinite(deltas)]
    if len(clean) == 0:
        return math.nan
    if np.allclose(clean, 0.0):
        return 0.5

    observed = float(np.mean(clean))
    signs = rng.choice([-1.0, 1.0], size=(num_resamples, len(clean)))
    null_means = np.mean(signs * clean, axis=1)
    return float((np.count_nonzero(null_means >= observed) + 1) / (num_resamples + 1))


def build_spa_tables(
    metric_pairs: pd.DataFrame,
    human_df: pd.DataFrame,
    num_resamples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute an HT-vs-MT adaptation of WMT soft pairwise accuracy.

    WMT SPA is system-level and averages over all pairs of MT systems. Here there is only
    one pair, HT vs MT, so each book contributes one soft pairwise term per metric:
    ``1 - abs(human_p_value - metric_p_value)``.
    """
    rng = np.random.default_rng(random_seed)

    metric_rows: list[dict[str, Any]] = []
    for (pkl, metric), group in metric_pairs.groupby(["pkl", "metric"]):
        row0 = group.iloc[0]
        oriented = group["ht_score"] - group["mt_score"]
        if not bool(row0["higher_is_better"]):
            oriented = -oriented
        p_value = paired_permutation_pvalue(
            oriented.to_numpy(dtype=float),
            rng=rng,
            num_resamples=num_resamples,
        )
        metric_rows.append(
            {
                "pkl": pkl,
                "metric": metric,
                "metric_label": row0["metric_label"],
                "metric_p_value_ht_better": p_value,
                "metric_observed_mean_delta": float(oriented.mean()),
                "metric_n_segments": int(len(oriented)),
            }
        )
    metric_pvalues = pd.DataFrame(metric_rows)

    human_rows: list[dict[str, Any]] = []
    for pkl, group in human_df.groupby("pkl"):
        row0 = group.iloc[0]
        strict = group["human_preference_num"].to_numpy(dtype=float)
        hard_as_tie = group["human_preference_hard_as_tie_num"].to_numpy(dtype=float)
        for variant, values in [
            ("participant_strict", strict),
            ("participant_similar_quality_as_tie", hard_as_tie),
        ]:
            p_value = paired_permutation_pvalue(values, rng=rng, num_resamples=num_resamples)
            human_rows.append(
                {
                    "pkl": pkl,
                    "book_slug": row0["book_slug"],
                    "source_lang": row0["source_lang"],
                    "source_lang_label": LANG_LABELS.get(row0["source_lang"], ""),
                    "human_variant": variant,
                    "human_p_value_ht_better": p_value,
                    "human_observed_mean_delta": float(np.nanmean(values)),
                    "human_n_judgments": int(np.isfinite(values).sum()),
                }
            )
    human_pvalues = pd.DataFrame(human_rows)

    spa_by_book = metric_pvalues.merge(human_pvalues, on="pkl", how="inner")
    spa_by_book["spa"] = 1 - (
        spa_by_book["human_p_value_ht_better"] - spa_by_book["metric_p_value_ht_better"]
    ).abs()
    spa_by_book["hard_pairwise_agreement"] = (
        (spa_by_book["human_p_value_ht_better"] < 0.5)
        == (spa_by_book["metric_p_value_ht_better"] < 0.5)
    )

    group_cols = ["metric", "metric_label", "human_variant"]
    spa_summary = (
        spa_by_book.groupby(group_cols)
        .agg(
            n_books=("pkl", "nunique"),
            mean_spa=("spa", "mean"),
            median_spa=("spa", "median"),
            mean_hard_pairwise_agreement=("hard_pairwise_agreement", "mean"),
        )
        .reset_index()
    )
    by_lang = (
        spa_by_book.groupby(group_cols + ["source_lang", "source_lang_label"])
        .agg(
            n_books=("pkl", "nunique"),
            mean_spa=("spa", "mean"),
            median_spa=("spa", "median"),
            mean_hard_pairwise_agreement=("hard_pairwise_agreement", "mean"),
        )
        .reset_index()
    )
    spa_summary = pd.concat([spa_summary, by_lang], ignore_index=True)
    return spa_by_book.sort_values(["metric", "human_variant", "pkl"]), spa_summary


def write_csvs(
    output_dir: Path,
    metric_pairs: pd.DataFrame,
    human_df: pd.DataFrame,
    participant_table: pd.DataFrame,
    majority_table: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    spa_by_book: pd.DataFrame,
    spa_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_pairs.to_csv(output_dir / "metric_pair_table.csv", index=False)
    human_df.to_csv(output_dir / "human_preferences_normalized.csv", index=False)
    participant_table.to_csv(output_dir / "human_metric_participant_table.csv", index=False)
    majority_table.to_csv(output_dir / "human_metric_chunk_majority_table.csv", index=False)

    score_summary = (
        metric_pairs.groupby(["pkl", "book_slug", "source_lang", "metric", "metric_label"])
        .agg(
            n_chunks=("chunk_id", "nunique"),
            mean_ht_score=("ht_score", "mean"),
            mean_mt_score=("mt_score", "mean"),
            sd_ht_score=("ht_score", "std"),
            sd_mt_score=("mt_score", "std"),
            mean_preference_delta=("preference_delta", "mean"),
            median_preference_delta=("preference_delta", "median"),
        )
        .reset_index()
    )
    score_summary.to_csv(output_dir / "book_metric_score_summary.csv", index=False)

    language_score_summary = (
        metric_pairs.groupby(["source_lang", "source_lang_label", "metric", "metric_label"])
        .agg(
            n_chunks=("chunk_id", "count"),
            mean_ht_score=("ht_score", "mean"),
            mean_mt_score=("mt_score", "mean"),
            sd_ht_score=("ht_score", "std"),
            sd_mt_score=("mt_score", "std"),
            mean_preference_delta=("preference_delta", "mean"),
            median_preference_delta=("preference_delta", "median"),
            ci95_preference_delta=("preference_delta", ci95),
        )
        .reset_index()
    )
    language_score_summary.to_csv(output_dir / "language_metric_score_summary.csv", index=False)

    overall_score_summary = (
        metric_pairs.groupby(["metric", "metric_label"])
        .agg(
            n_chunks=("chunk_id", "count"),
            mean_ht_score=("ht_score", "mean"),
            mean_mt_score=("mt_score", "mean"),
            sd_ht_score=("ht_score", "std"),
            sd_mt_score=("mt_score", "std"),
            mean_preference_delta=("preference_delta", "mean"),
            median_preference_delta=("preference_delta", "median"),
            ci95_preference_delta=("preference_delta", ci95),
        )
        .reset_index()
    )
    overall_score_summary.to_csv(output_dir / "overall_metric_score_summary.csv", index=False)

    preference_summary = (
        metric_pairs.groupby(["pkl", "book_slug", "source_lang", "metric", "metric_preference"])
        .size()
        .reset_index(name="n")
    )
    preference_summary.to_csv(output_dir / "book_metric_preference_counts.csv", index=False)
    spa_by_book.to_csv(output_dir / "spa_by_book.csv", index=False)
    spa_summary.to_csv(output_dir / "spa_summary.csv", index=False)

    for name, df in summaries.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def save_plot(fig: plt.Figure, plots_dir: Path, stem: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(plots_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_mean_scores(metric_pairs: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    df = metric_pairs.melt(
        id_vars=["metric", "metric_label"],
        value_vars=["ht_score", "mt_score"],
        var_name="system",
        value_name="score",
    )
    df["system"] = df["system"].map({"ht_score": "HT", "mt_score": "MT"})
    summary = (
        df.groupby(["metric", "metric_label", "system"], as_index=False)
        .agg(mean_score=("score", "mean"), ci95_score=("score", ci95))
    )
    metrics = list(summary["metric"].drop_duplicates())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.4), squeeze=False)
    for ax, metric in zip(axes.flat, metrics, strict=False):
        group = summary[summary["metric"] == metric].set_index("system").reindex(["HT", "MT"])
        ax.bar(
            group.index,
            group["mean_score"],
            yerr=group["ci95_score"],
            color=[PREFERENCE_COLORS["HT"], PREFERENCE_COLORS["MT"]],
            error_kw=ERROR_KW,
        )
        ax.set_title(str(group["metric_label"].iloc[0]))
        ax.set_xlabel("")
        ax.set_ylabel("Mean segment score")
        annotate_bars(ax, ".3g")
    save_plot(fig, plots_dir, "metric_mean_scores_ht_vs_mt")


def plot_preference_counts(metric_pairs: pd.DataFrame, plots_dir: Path) -> None:
    counts = (
        metric_pairs.groupby(["metric_label", "metric_preference"])
        .size()
        .reset_index(name="n")
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sns.barplot(
        data=counts,
        x="metric_label",
        y="n",
        hue="metric_preference",
        hue_order=PREFERENCE_ORDER,
        palette=PREFERENCE_COLORS,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Chunks")
    ax.set_title("Metric preference counts")
    annotate_bars(ax, ".0f")
    ax.legend(title="Metric prefers")
    save_plot(fig, plots_dir, "metric_preference_counts")


def plot_preference_counts_by_language(metric_pairs: pd.DataFrame, plots_dir: Path) -> None:
    counts = (
        metric_pairs.groupby(["metric_label", "source_lang_label", "metric_preference"])
        .size()
        .reset_index(name="n")
    )
    g = sns.catplot(
        data=counts,
        x="source_lang_label",
        y="n",
        hue="metric_preference",
        hue_order=PREFERENCE_ORDER,
        col="metric_label",
        kind="bar",
        palette=PREFERENCE_COLORS,
        height=4.2,
        aspect=1.05,
        sharey=False,
    )
    g.set_axis_labels("", "Chunks")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        annotate_bars(ax, ".0f")
    save_plot(g.fig, plots_dir, "metric_preference_counts_by_language")


def plot_book_deltas(metric_pairs: pd.DataFrame, plots_dir: Path) -> None:
    book_delta = (
        metric_pairs.groupby(["metric", "metric_label", "pkl"], as_index=False)
        .agg(
            mean_preference_delta=("preference_delta", "mean"),
            ci95_preference_delta=("preference_delta", ci95),
        )
        .sort_values(["metric", "mean_preference_delta"])
    )
    for metric, group in book_delta.groupby("metric"):
        fig, ax = plt.subplots(figsize=(8.0, max(4.2, 0.36 * len(group))))
        colors = [
            PREFERENCE_COLORS["HT"] if value > 0 else PREFERENCE_COLORS["MT"]
            for value in group["mean_preference_delta"]
        ]
        ax.barh(
            group["pkl"],
            group["mean_preference_delta"],
            xerr=group["ci95_preference_delta"],
            color=colors,
            error_kw=ERROR_KW,
        )
        ax.axvline(0, color="#374151", linewidth=1)
        for y, value in enumerate(group["mean_preference_delta"]):
            offset = 0.01 * max(1.0, float(group["mean_preference_delta"].abs().max()))
            ax.text(
                value + (offset if value >= 0 else -offset),
                y,
                f"{value:.3g}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )
        ax.set_xlabel("Mean preference delta (positive favors HT, negative favors MT)")
        ax.set_ylabel("")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} mean HT-vs-MT delta by book")
        save_plot(fig, plots_dir, f"book_preference_delta_{metric}")


def plot_book_preference_counts(metric_pairs: pd.DataFrame, plots_dir: Path) -> None:
    counts = (
        metric_pairs.groupby(["metric", "pkl", "metric_preference"])
        .size()
        .reset_index(name="n")
    )
    for metric, group in counts.groupby("metric"):
        pivot = (
            group.pivot_table(index="pkl", columns="metric_preference", values="n", fill_value=0)
            .reindex(columns=PREFERENCE_ORDER, fill_value=0)
            .sort_values(["HT", "MT"], ascending=[True, False])
        )
        fig, ax = plt.subplots(figsize=(8.0, max(4.2, 0.36 * len(pivot))))
        left = pd.Series(0, index=pivot.index, dtype=float)
        for preference in PREFERENCE_ORDER:
            ax.barh(
                pivot.index,
                pivot[preference],
                left=left,
                label=preference,
                color=PREFERENCE_COLORS[preference],
            )
            for y, (start, width) in enumerate(zip(left, pivot[preference], strict=False)):
                if width > 0:
                    ax.text(
                        start + width / 2,
                        y,
                        f"{int(width)}",
                        va="center",
                        ha="center",
                        fontsize=7,
                        color="white" if preference != "tie" else "#111827",
                    )
            left = left + pivot[preference]
        ax.set_xlabel("Chunks")
        ax.set_ylabel("")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} preference counts by book")
        ax.legend(title="Metric prefers")
        save_plot(fig, plots_dir, f"book_preference_counts_{metric}")


def plot_book_mean_scores(metric_pairs: pd.DataFrame, plots_dir: Path) -> None:
    df = metric_pairs.melt(
        id_vars=["metric", "metric_label", "pkl"],
        value_vars=["ht_score", "mt_score"],
        var_name="system",
        value_name="score",
    )
    df["system"] = df["system"].map({"ht_score": "HT", "mt_score": "MT"})
    summary = (
        df.groupby(["metric", "metric_label", "pkl", "system"], as_index=False)
        .agg(score=("score", "mean"), ci95_score=("score", ci95))
    )
    for metric, group in summary.groupby("metric"):
        books = list(group["pkl"].drop_duplicates())
        y = np.arange(len(books))
        height = 0.36
        fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.42 * len(books))))
        for offset, system in [(-height / 2, "HT"), (height / 2, "MT")]:
            sub = group[group["system"] == system].set_index("pkl").reindex(books)
            bars = ax.barh(
                y + offset,
                sub["score"],
                xerr=sub["ci95_score"],
                height=height,
                color=PREFERENCE_COLORS[system],
                label=system,
                error_kw=ERROR_KW,
            )
            ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(books)
        ax.set_xlabel("Mean segment score")
        ax.set_ylabel("")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} mean score by book")
        ax.legend(title="")
        save_plot(fig, plots_dir, f"book_mean_scores_{metric}")


def plot_language_mean_scores(metric_pairs: pd.DataFrame, plots_dir: Path) -> None:
    df = metric_pairs.melt(
        id_vars=["metric", "metric_label", "source_lang_label"],
        value_vars=["ht_score", "mt_score"],
        var_name="system",
        value_name="score",
    )
    df["system"] = df["system"].map({"ht_score": "HT", "mt_score": "MT"})
    summary = (
        df.groupby(["metric", "metric_label", "source_lang_label", "system"], as_index=False)
        .agg(mean_score=("score", "mean"), ci95_score=("score", ci95))
    )
    metrics = list(summary["metric"].drop_duplicates())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.6), squeeze=False)
    for ax, metric in zip(axes.flat, metrics, strict=False):
        group = summary[summary["metric"] == metric]
        langs = list(group["source_lang_label"].drop_duplicates())
        x = np.arange(len(langs))
        width = 0.36
        for offset, system in [(-width / 2, "HT"), (width / 2, "MT")]:
            sub = group[group["system"] == system].set_index("source_lang_label").reindex(langs)
            bars = ax.bar(
                x + offset,
                sub["mean_score"],
                yerr=sub["ci95_score"],
                width=width,
                color=PREFERENCE_COLORS[system],
                label=system,
                error_kw=ERROR_KW,
            )
            ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(langs)
        ax.set_title(str(group["metric_label"].iloc[0]))
        ax.set_xlabel("")
        ax.set_ylabel("Mean segment score")
        ax.legend(title="")
    save_plot(fig, plots_dir, "language_mean_scores_ht_vs_mt")


def plot_mean_preference_delta(metric_pairs: pd.DataFrame, plots_dir: Path) -> None:
    overall = (
        metric_pairs.groupby(["metric", "metric_label"], as_index=False)
        .agg(
            mean_preference_delta=("preference_delta", "mean"),
            ci95_preference_delta=("preference_delta", ci95),
        )
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [
        PREFERENCE_COLORS["HT"] if value > 0 else PREFERENCE_COLORS["MT"]
        for value in overall["mean_preference_delta"]
    ]
    ax.bar(
        overall["metric_label"],
        overall["mean_preference_delta"],
        yerr=overall["ci95_preference_delta"],
        color=colors,
        error_kw=ERROR_KW,
    )
    ax.axhline(0, color="#374151", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Mean preference delta")
    ax.set_title("Overall mean HT-vs-MT delta with 95% CI")
    for idx, value in enumerate(overall["mean_preference_delta"]):
        ax.text(
            idx,
            value,
            f"{value:.3g}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    save_plot(fig, plots_dir, "overall_mean_preference_delta")

    by_lang = (
        metric_pairs.groupby(["metric", "metric_label", "source_lang_label"], as_index=False)
        .agg(
            mean_preference_delta=("preference_delta", "mean"),
            ci95_preference_delta=("preference_delta", ci95),
        )
    )
    for metric, group in by_lang.groupby("metric"):
        group = group.sort_values("source_lang_label")
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        colors = [
            PREFERENCE_COLORS["HT"] if value > 0 else PREFERENCE_COLORS["MT"]
            for value in group["mean_preference_delta"]
        ]
        ax.bar(
            group["source_lang_label"],
            group["mean_preference_delta"],
            yerr=group["ci95_preference_delta"],
            color=colors,
            error_kw=ERROR_KW,
        )
        ax.axhline(0, color="#374151", linewidth=1)
        ax.set_xlabel("")
        ax.set_ylabel("Mean preference delta")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} mean delta by language with 95% CI")
        for idx, value in enumerate(group["mean_preference_delta"]):
            ax.text(
                idx,
                value,
                f"{value:.3g}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
        save_plot(fig, plots_dir, f"language_mean_preference_delta_{metric}")


def plot_agreement_rates(agreement_rates: pd.DataFrame, plots_dir: Path) -> None:
    plot_df = agreement_rates.copy()
    plot_df["metric_label"] = plot_df["metric"].map(lambda value: METRIC_LABELS.get(value, value))
    plot_df["variant_label"] = plot_df["variant"].map(
        {
            "participant_strict_no_metric_ties": "strict, no metric ties",
            "human_agreed_chunks_strict_no_metric_ties": "agreed human chunks",
            "litransproqa_similar_quality_as_tie": "LitTransProQA: similar_quality as tie",
        }
    )
    plot_df = plot_df.dropna(subset=["variant_label"]).copy()
    plot_df["ci95_agreement"] = plot_df.apply(
        lambda row: 1.96
        * math.sqrt(row["agreement_rate"] * (1 - row["agreement_rate"]) / row["n"])
        if row["n"] and not pd.isna(row["agreement_rate"])
        else 0.0,
        axis=1,
    )
    metrics = list(plot_df["metric_label"].drop_duplicates())
    variants = list(plot_df["variant_label"].drop_duplicates())
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for idx, variant in enumerate(variants):
        sub = plot_df[plot_df["variant_label"] == variant].set_index("metric_label").reindex(metrics)
        bars = ax.bar(
            x + (idx - (len(variants) - 1) / 2) * width,
            sub["agreement_rate"],
            yerr=sub["ci95_agreement"],
            width=width,
            label=variant,
            color=["#4C78A8", "#72B7B2", "#54A24B", "#F58518"][idx % 4],
            error_kw=ERROR_KW,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Agreement rate")
    ax.set_title("Metric agreement with human chunk preferences")
    ax.legend(title="")
    save_plot(fig, plots_dir, "human_metric_agreement_rates")


def plot_kendall(correlations: pd.DataFrame, plots_dir: Path) -> None:
    plot_df = correlations[
        correlations["variant"].isin(
            [
                "participant_strict",
                "litransproqa_similar_quality_as_tie",
                "human_agreed_chunks_strict",
                "human_agreed_chunks_litransproqa_similar_quality_as_tie",
            ]
        )
    ].copy()
    plot_df["metric_label"] = plot_df["metric"].map(lambda value: METRIC_LABELS.get(value, value))
    plot_df["variant_label"] = plot_df["variant"].map(
        {
            "participant_strict": "participant strict",
            "litransproqa_similar_quality_as_tie": "LitTransProQA similar_quality as tie",
            "human_agreed_chunks_strict": "agreed chunks strict",
            "human_agreed_chunks_litransproqa_similar_quality_as_tie": (
                "agreed chunks LitTransProQA tie policy"
            ),
        }
    )
    plot_df = plot_df.dropna(subset=["variant_label"]).copy()
    metrics = list(plot_df["metric_label"].drop_duplicates())
    variants = list(plot_df["variant_label"].drop_duplicates())
    x = np.arange(len(metrics))
    width = min(0.18, 0.8 / max(1, len(variants)))
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#B279A2", "#E45756"]
    for idx, variant in enumerate(variants):
        sub = plot_df[plot_df["variant_label"] == variant].set_index("metric_label").reindex(metrics)
        bars = ax.bar(
            x + (idx - (len(variants) - 1) / 2) * width,
            sub["kendall_tau"],
            width=width,
            label=variant,
            color=colors[idx % len(colors)],
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.axhline(0, color="#374151", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Pairwise Kendall tau")
    ax.set_title("Pairwise Kendall agreement with human preferences")
    ax.legend(title="")
    save_plot(fig, plots_dir, "kendall_tau_summary")


def plot_spa_summary(spa_summary: pd.DataFrame, plots_dir: Path) -> None:
    plot_df = spa_summary[spa_summary["source_lang"].isna()].copy()
    plot_df["variant_label"] = plot_df["human_variant"].map(
        {
            "participant_strict": "participant strict",
            "participant_similar_quality_as_tie": "similar_quality as tie",
        }
    )
    plot_df["ci95_spa"] = plot_df.apply(
        lambda row: 1.96
        * math.sqrt(row["mean_spa"] * (1 - row["mean_spa"]) / row["n_books"])
        if row["n_books"] and not pd.isna(row["mean_spa"])
        else 0.0,
        axis=1,
    )
    metrics = list(plot_df["metric_label"].drop_duplicates())
    variants = list(plot_df["variant_label"].drop_duplicates())
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for idx, variant in enumerate(variants):
        sub = plot_df[plot_df["variant_label"] == variant].set_index("metric_label").reindex(metrics)
        bars = ax.bar(
            x + (idx - (len(variants) - 1) / 2) * width,
            sub["mean_spa"],
            yerr=sub["ci95_spa"],
            width=width,
            label=variant,
            color=["#4C78A8", "#72B7B2"][idx % 2],
            error_kw=ERROR_KW,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Mean adapted SPA")
    ax.set_title("Adapted SPA for HT-vs-MT system pair")
    ax.legend(title="")
    save_plot(fig, plots_dir, "spa_summary")


def plot_human_vs_metric_heatmaps(participant_table: pd.DataFrame, plots_dir: Path) -> None:
    valid = participant_table.dropna(subset=["human_preference"]).copy()
    for metric, group in valid.groupby("metric"):
        matrix = pd.crosstab(group["human_preference"], group["metric_preference"])
        matrix = matrix.reindex(index=["HT", "MT"], columns=PREFERENCE_ORDER, fill_value=0)
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Metric preference")
        ax.set_ylabel("Human preference")
        ax.set_title(f"Human vs {METRIC_LABELS.get(metric, metric)} preference")
        save_plot(fig, plots_dir, f"human_vs_metric_heatmap_{metric}")


def write_readme(output_dir: Path, tie_epsilon: float) -> None:
    readme = f"""# Human-Eval Metric Analysis

Generated by `scripts/analyze_chunk_review_metrics.py`.

## Main Tables

- `metric_pair_table.csv`: one row per book/chunk/metric with HT score, MT score, oriented score delta, and metric preference.
- `human_preferences_normalized.csv`: Part 2 human preferences with normalized book IDs that match `results_chunk_review_eval`.
- `human_metric_participant_table.csv`: participant-level join between human preferences and metric preferences.
- `human_metric_chunk_majority_table.csv`: chunk-level human majority preference joined to metric preferences.
- `book_metric_score_summary.csv`: per-book HT/MT score means, standard deviations, and oriented deltas.
- `book_metric_preference_counts.csv`: per-book counts of chunks where each metric prefers HT, MT, or tie.
- `language_metric_score_summary.csv`: per-language HT/MT score means, standard deviations, and oriented deltas.
- `overall_metric_score_summary.csv`: corpus-level HT/MT score means, standard deviations, and oriented deltas.
- `correlations.csv`: pairwise Kendall summaries overall and by source language.
- `correlations_by_participant.csv`: pairwise Kendall summaries per participant.
- `agreement_rates.csv`: simple metric-human pairwise agreement rates.
- `spa_by_book.csv`: adapted HT-vs-MT soft pairwise accuracy inputs and scores per book.
- `spa_summary.csv`: adapted SPA averages overall and by source language.

## Preference Direction

`preference_delta` is oriented so positive values favor HT and negative values favor MT.
The direction comes from each metric's `higher_is_better` flag in `system_scores.json`;
for example, LitTransProQA and COMETKiwi are higher-is-better, while MetricX-QE and
MetricX are lower-is-better.

Metric ties use absolute `preference_delta <= {tie_epsilon}`. There is no extra
close-tie threshold for MetricX-family metrics; Kendall results use hard HT-vs-MT
preferences.
The main strict Kendall variant compares every participant-level HT/MT choice against the
automatic metric preference for the same book/chunk.

Additional Kendall variants:

- `litransproqa_similar_quality_as_tie`: only for LitTransProQA. Human rows with
  `difficulty == "similar_quality"` are encoded as tie because LitTransProQA has many
  exact score ties.
- `human_agreed_chunks_*`: only keeps book/chunk items where at least two human raters
  chose the same `preferred_translation` (`HT,HT` or `MT,MT`). The duplicate rater rows
  are collapsed to one row per agreed book/chunk/metric before Kendall tau is computed.

Kendall tau is computed directly as an HT-vs-MT pairwise statistic, not by numeric
rank correlation. For each chunk-level HT--MT pair:

- concordant: human and metric both prefer HT, or both prefer MT
- discordant: one prefers HT and the other prefers MT
- tie: either side is encoded as tie

The reported `kendall_tau` is `(concordant_pairs - discordant_pairs) / total_pairs`.

## Adapted SPA

WMT SPA is a system-level meta-metric over all system pairs. This dataset has only one
system pair, HT vs MT, so `spa_by_book.csv` computes one SPA contribution per book:
`1 - abs(human_p_value_ht_better - metric_p_value_ht_better)`. The p-values come from
paired sign-flip permutation tests over chunk-level HT-minus-MT deltas. This is useful
as a WMT-inspired diagnostic, but it is not identical to a full WMT submission with many
MT systems.

## Plots

PNG and PDF versions are in `plots/`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def make_plots(
    output_dir: Path,
    metric_pairs: pd.DataFrame,
    participant_table: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    spa_summary: pd.DataFrame,
) -> None:
    plots_dir = output_dir / "plots"
    plot_metric_mean_scores(metric_pairs, plots_dir)
    plot_preference_counts(metric_pairs, plots_dir)
    plot_preference_counts_by_language(metric_pairs, plots_dir)
    plot_book_deltas(metric_pairs, plots_dir)
    plot_book_preference_counts(metric_pairs, plots_dir)
    plot_book_mean_scores(metric_pairs, plots_dir)
    plot_language_mean_scores(metric_pairs, plots_dir)
    plot_mean_preference_delta(metric_pairs, plots_dir)
    plot_agreement_rates(summaries["agreement_rates"], plots_dir)
    plot_kendall(summaries["correlations"], plots_dir)
    plot_spa_summary(spa_summary, plots_dir)
    plot_human_vs_metric_heatmaps(participant_table, plots_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join chunk-review metric scores with human preferences and create plots."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--human-path", type=Path, default=DEFAULT_HUMAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tie-epsilon",
        type=float,
        default=1e-9,
        help="Absolute oriented-delta threshold for declaring metric ties.",
    )
    parser.add_argument(
        "--spa-resamples",
        type=int,
        default=10000,
        help="Number of paired sign-flip permutation samples for adapted SPA p-values.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=13,
        help="Random seed for adapted SPA permutation tests.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metric_pairs = collect_metric_pairs(
        args.results_dir,
        tie_epsilon=args.tie_epsilon,
    )
    human_df = collect_human_preferences(args.human_path)
    participant_table = build_participant_table(metric_pairs, human_df)
    majority_table = build_majority_table(metric_pairs, human_df)
    summaries = summarize_agreement(participant_table, majority_table)
    spa_by_book, spa_summary = build_spa_tables(
        metric_pairs,
        human_df,
        num_resamples=args.spa_resamples,
        random_seed=args.random_seed,
    )

    write_csvs(
        args.output_dir,
        metric_pairs,
        human_df,
        participant_table,
        majority_table,
        summaries,
        spa_by_book,
        spa_summary,
    )
    make_plots(args.output_dir, metric_pairs, participant_table, summaries, spa_summary)
    write_readme(
        args.output_dir,
        tie_epsilon=args.tie_epsilon,
    )

    matched = participant_table["participant_id"].notna().sum()
    print(f"Wrote analysis to {args.output_dir}")
    print(f"Metric pair rows: {len(metric_pairs):,}")
    print(f"Participant-level joined rows: {matched:,}")
    print(f"Plots: {args.output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
