"""Analyze agreement between LitTransProQA and MetricX-QE chunk-review results.

This script reads book-level metric outputs from ``results_chunk_review_eval/books``.
It does not use human-evaluation answers. It compares the two automatic metrics on
the same HT-vs-MT chunk pairs and writes tables plus plots under
``results_chunk_review_eval/analysis``.

Example:
  .venv/bin/python scripts/analyze_chunk_review_metric_agreement.py
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
from scipy.stats import kendalltau


DEFAULT_RESULTS_DIR = Path("results_chunk_review_eval")
BOOK_RESULTS_DIR_NAME = "books"
ANALYSIS_DIR_NAME = "analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / ANALYSIS_DIR_NAME / "metric_agreement_analysis"

LANG_LABELS = {"fr": "French", "ja": "Japanese", "pl": "Polish"}
SYSTEM_LABELS = {"ht": "HT", "mt": "MT"}
METRIC_LABELS = {"litransproqa": "LitTransProQA", "metricx-qe": "MetricX-QE"}
PREFERENCE_ORDER = ["HT", "MT", "tie"]
PREFERENCE_COLORS = {"HT": "#4C78A8", "MT": "#F58518", "tie": "#9CA3AF"}
ERROR_KW = {"ecolor": "black", "elinewidth": 1.4, "capsize": 5, "capthick": 1.4}


def ci95(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))


def annotate_bars(ax: plt.Axes, fmt: str = ".3g", fontsize: int = 8) -> None:
    for container in ax.containers:
        if isinstance(container, BarContainer):
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


def collect_metric_pairs(results_dir: Path, tie_epsilon: float) -> pd.DataFrame:
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
        lambda value: "tie" if abs(value) <= tie_epsilon else ("HT" if value > 0 else "MT")
    )
    pair_df["metric_preference_num"] = pair_df["metric_preference"].map(
        {"HT": 1, "MT": -1, "tie": 0}
    )
    pair_df["metric_label"] = pair_df["metric"].map(lambda value: METRIC_LABELS.get(value, value))
    return pair_df.sort_values(["pkl", "chunk_id", "metric"]).reset_index(drop=True)


def build_metric_agreement_table(metric_pairs: pd.DataFrame) -> pd.DataFrame:
    wide = metric_pairs.pivot_table(
        index=["pkl", "book_slug", "source_lang", "source_lang_label", "chunk_id"],
        columns="metric",
        values=[
            "ht_score",
            "mt_score",
            "preference_delta",
            "metric_preference_num",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{value}" for value, metric in wide.columns]
    wide = wide.reset_index()

    pref = metric_pairs.pivot_table(
        index=["pkl", "book_slug", "source_lang", "source_lang_label", "chunk_id"],
        columns="metric",
        values="metric_preference",
        aggfunc="first",
    ).reset_index()
    pref = pref.rename(
        columns={
            "litransproqa": "litransproqa_preference",
            "metricx-qe": "metricx_qe_preference",
        }
    )

    joined = wide.merge(
        pref,
        on=["pkl", "book_slug", "source_lang", "source_lang_label", "chunk_id"],
        how="inner",
    )
    joined = joined.rename(
        columns={
            "litransproqa_ht_score": "litransproqa_ht_score",
            "litransproqa_mt_score": "litransproqa_mt_score",
            "litransproqa_preference_delta": "litransproqa_preference_delta",
            "litransproqa_metric_preference_num": "litransproqa_preference_num",
            "metricx-qe_ht_score": "metricx_qe_ht_score",
            "metricx-qe_mt_score": "metricx_qe_mt_score",
            "metricx-qe_preference_delta": "metricx_qe_preference_delta",
            "metricx-qe_metric_preference_num": "metricx_qe_preference_num",
        }
    )
    joined = joined.dropna(
        subset=[
            "litransproqa_preference",
            "metricx_qe_preference",
            "litransproqa_preference_num",
            "metricx_qe_preference_num",
        ]
    ).copy()
    joined["metrics_agree"] = (
        joined["litransproqa_preference"] == joined["metricx_qe_preference"]
    )
    joined["both_tie"] = (
        (joined["litransproqa_preference"] == "tie")
        & (joined["metricx_qe_preference"] == "tie")
    )
    joined["both_non_tie"] = (
        joined["litransproqa_preference"].isin(["HT", "MT"])
        & joined["metricx_qe_preference"].isin(["HT", "MT"])
    )
    joined["non_tie_metrics_agree"] = joined["metrics_agree"] & joined["both_non_tie"]
    return joined.sort_values(["pkl", "chunk_id"]).reset_index(drop=True)


def metric_agreement_summary(agreement_table: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = (
        [((), agreement_table)]
        if not group_cols
        else agreement_table.groupby(group_cols, dropna=False)
    )
    for keys, group in grouped:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)
        non_tie = group[group["both_non_tie"]]
        tau = math.nan
        p_value = math.nan
        if (
            len(group) >= 2
            and group["litransproqa_preference_num"].nunique() > 1
            and group["metricx_qe_preference_num"].nunique() > 1
        ):
            result = kendalltau(
                group["litransproqa_preference_num"],
                group["metricx_qe_preference_num"],
                nan_policy="omit",
            )
            tau = float(result.statistic) if result.statistic is not None else math.nan
            p_value = float(result.pvalue) if result.pvalue is not None else math.nan

        row = dict(zip(group_cols, keys, strict=False)) if group_cols else {}
        row.update(
            {
                "n_chunks": len(group),
                "agree_count": int(group["metrics_agree"].sum()),
                "agreement_rate": float(group["metrics_agree"].mean()) if len(group) else math.nan,
                "both_tie_count": int(group["both_tie"].sum()),
                "both_non_tie_chunks": len(non_tie),
                "non_tie_agree_count": int(non_tie["metrics_agree"].sum()),
                "non_tie_agreement_rate": float(non_tie["metrics_agree"].mean())
                if len(non_tie)
                else math.nan,
                "kendall_tau": tau,
                "kendall_p_value": p_value,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables(output_dir: Path, metric_pairs: pd.DataFrame, agreement_table: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_pairs.to_csv(output_dir / "metric_pair_table.csv", index=False)
    agreement_table.to_csv(output_dir / "metric_agreement_by_chunk.csv", index=False)

    overall_scores = (
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
    overall_scores.to_csv(output_dir / "overall_metric_score_summary.csv", index=False)

    language_scores = (
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
    language_scores.to_csv(output_dir / "language_metric_score_summary.csv", index=False)

    book_scores = (
        metric_pairs.groupby(["pkl", "book_slug", "source_lang", "metric", "metric_label"])
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
    book_scores.to_csv(output_dir / "book_metric_score_summary.csv", index=False)

    preference_counts = (
        metric_pairs.groupby(["metric", "metric_label", "metric_preference"])
        .size()
        .reset_index(name="n")
    )
    preference_counts.to_csv(output_dir / "metric_preference_counts.csv", index=False)

    language_preference_counts = (
        metric_pairs.groupby(
            ["source_lang", "source_lang_label", "metric", "metric_label", "metric_preference"]
        )
        .size()
        .reset_index(name="n")
    )
    language_preference_counts.to_csv(
        output_dir / "language_metric_preference_counts.csv", index=False
    )

    metric_agreement_summary(agreement_table, []).to_csv(
        output_dir / "metric_agreement_summary.csv", index=False
    )
    metric_agreement_summary(agreement_table, ["source_lang", "source_lang_label"]).to_csv(
        output_dir / "language_metric_agreement_summary.csv", index=False
    )
    metric_agreement_summary(agreement_table, ["pkl", "book_slug", "source_lang"]).to_csv(
        output_dir / "book_metric_agreement_summary.csv", index=False
    )

    cross_tab = pd.crosstab(
        agreement_table["litransproqa_preference"],
        agreement_table["metricx_qe_preference"],
    ).reindex(index=PREFERENCE_ORDER, columns=PREFERENCE_ORDER, fill_value=0)
    cross_tab.to_csv(output_dir / "metric_preference_crosstab.csv")


def save_plot(fig: plt.Figure, plots_dir: Path, stem: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(plots_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_mean_scores(metric_pairs: pd.DataFrame, plots_dir: Path) -> None:
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
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5), squeeze=False)
    for ax, metric in zip(axes.flat, metrics, strict=False):
        group = summary[summary["metric"] == metric].set_index("system").reindex(["HT", "MT"])
        bars = ax.bar(
            group.index,
            group["mean_score"],
            yerr=group["ci95_score"],
            color=[PREFERENCE_COLORS["HT"], PREFERENCE_COLORS["MT"]],
            error_kw=ERROR_KW,
        )
        ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=8)
        ax.set_title(str(group["metric_label"].iloc[0]))
        ax.set_ylabel("Mean segment score")
    save_plot(fig, plots_dir, "metric_mean_scores_ht_vs_mt")


def plot_metric_preference_counts(metric_pairs: pd.DataFrame, plots_dir: Path) -> None:
    counts = (
        metric_pairs.groupby(["metric_label", "metric_preference"]).size().reset_index(name="n")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.barplot(
        data=counts,
        x="metric_label",
        y="n",
        hue="metric_preference",
        hue_order=PREFERENCE_ORDER,
        palette=PREFERENCE_COLORS,
        ax=ax,
    )
    annotate_bars(ax, ".0f")
    ax.set_xlabel("")
    ax.set_ylabel("Chunks")
    ax.set_title("Automatic metric HT/MT preference counts")
    ax.legend(title="Metric prefers")
    save_plot(fig, plots_dir, "metric_preference_counts")


def plot_metric_agreement(agreement_table: pd.DataFrame, plots_dir: Path) -> None:
    overall = metric_agreement_summary(agreement_table, [])
    by_lang = metric_agreement_summary(agreement_table, ["source_lang_label"])

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar(["All chunks"], overall["agreement_rate"], color="#4C78A8")
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Agreement rate")
    ax.set_title("LitTransProQA and MetricX-QE preference agreement")
    save_plot(fig, plots_dir, "metric_agreement_rate_overall")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    by_lang = by_lang.sort_values("source_lang_label")
    bars = ax.bar(by_lang["source_lang_label"], by_lang["agreement_rate"], color="#4C78A8")
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Agreement rate")
    ax.set_title("Metric preference agreement by source language")
    save_plot(fig, plots_dir, "metric_agreement_rate_by_language")

    book = metric_agreement_summary(agreement_table, ["pkl"]).sort_values("agreement_rate")
    fig, ax = plt.subplots(figsize=(8.2, max(4.4, 0.36 * len(book))))
    bars = ax.barh(book["pkl"], book["agreement_rate"], color="#4C78A8")
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Agreement rate")
    ax.set_ylabel("")
    ax.set_title("Metric preference agreement by book")
    save_plot(fig, plots_dir, "metric_agreement_rate_by_book")


def plot_metric_preference_crosstab(agreement_table: pd.DataFrame, plots_dir: Path) -> None:
    matrix = pd.crosstab(
        agreement_table["litransproqa_preference"],
        agreement_table["metricx_qe_preference"],
    ).reindex(index=PREFERENCE_ORDER, columns=PREFERENCE_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("MetricX-QE preference")
    ax.set_ylabel("LitTransProQA preference")
    ax.set_title("Automatic metric preference cross-tab")
    save_plot(fig, plots_dir, "metric_preference_crosstab")


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
    bars = ax.bar(
        overall["metric_label"],
        overall["mean_preference_delta"],
        yerr=overall["ci95_preference_delta"],
        color=colors,
        error_kw=ERROR_KW,
    )
    ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=8)
    ax.axhline(0, color="#374151", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Mean preference delta")
    ax.set_title("Overall mean HT-vs-MT delta with 95% CI")
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
        bars = ax.bar(
            group["source_lang_label"],
            group["mean_preference_delta"],
            yerr=group["ci95_preference_delta"],
            color=colors,
            error_kw=ERROR_KW,
        )
        ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=8)
        ax.axhline(0, color="#374151", linewidth=1)
        ax.set_xlabel("")
        ax.set_ylabel("Mean preference delta")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} mean delta by language")
        save_plot(fig, plots_dir, f"language_mean_preference_delta_{metric}")


def write_readme(output_dir: Path, tie_epsilon: float) -> None:
    readme = f"""# Automatic Metric Agreement Analysis

Generated by `scripts/analyze_chunk_review_metric_agreement.py`.

This analysis uses only files inside `results_chunk_review_eval`. It compares
LitTransProQA and MetricX-QE on the same HT-vs-MT chunk pairs; it does not use
human-evaluation answers.

## Main Tables

- `metric_pair_table.csv`: one row per book/chunk/metric with HT score, MT score, oriented delta, and the metric's HT/MT/tie preference.
- `metric_agreement_by_chunk.csv`: one row per book/chunk with both metrics side by side and whether their preferences agree.
- `metric_agreement_summary.csv`: overall agreement rate and Kendall tau between metric preferences.
- `language_metric_agreement_summary.csv`: agreement rate by source language.
- `book_metric_agreement_summary.csv`: agreement rate by book.
- `metric_preference_crosstab.csv`: LitTransProQA preference by MetricX-QE preference.
- `overall_metric_score_summary.csv`, `language_metric_score_summary.csv`, and `book_metric_score_summary.csv`: HT/MT score means and oriented deltas.

## Preference Direction

`preference_delta` is oriented so positive values favor HT and negative values favor MT.
For LitTransProQA, higher is better. For MetricX-QE, lower is better.

Metric ties use absolute `preference_delta <= {tie_epsilon}`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def make_plots(
    output_dir: Path,
    metric_pairs: pd.DataFrame,
    agreement_table: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    plots_dir = output_dir / "plots"
    plot_metric_mean_scores(metric_pairs, plots_dir)
    plot_metric_preference_counts(metric_pairs, plots_dir)
    plot_metric_agreement(agreement_table, plots_dir)
    plot_metric_preference_crosstab(agreement_table, plots_dir)
    plot_mean_preference_delta(metric_pairs, plots_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare LitTransProQA and MetricX-QE chunk-review results."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tie-epsilon",
        type=float,
        default=1e-9,
        help="Absolute oriented-delta threshold for declaring metric ties.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metric_pairs = collect_metric_pairs(args.results_dir, tie_epsilon=args.tie_epsilon)
    agreement_table = build_metric_agreement_table(metric_pairs)
    write_tables(args.output_dir, metric_pairs, agreement_table)
    make_plots(args.output_dir, metric_pairs, agreement_table)
    write_readme(args.output_dir, tie_epsilon=args.tie_epsilon)

    summary = metric_agreement_summary(agreement_table, [])
    agreement_rate = summary["agreement_rate"].iloc[0]
    print(f"Wrote metric-only analysis to {args.output_dir}")
    print(f"Metric pair rows: {len(metric_pairs):,}")
    print(f"Comparable chunks: {len(agreement_table):,}")
    print(f"LitTransProQA/MetricX-QE agreement rate: {agreement_rate:.3f}")
    print(f"Plots: {args.output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
