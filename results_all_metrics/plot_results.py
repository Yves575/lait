#!/usr/bin/env python3
"""Plot LitMT pipeline evaluation results.

Run from anywhere:

    python results_all_metrics/plot_results.py

The script reads ``all_results.csv`` when it exists, which lets it plot the
filtered means produced by ``summarize_results.py``. Use ``--rebuild`` to ignore
the CSV and regenerate unfiltered rows from ``system_scores.json`` files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ModuleNotFoundError as exc:
    missing = exc.name or "plotting dependency"
    raise SystemExit(
        f"Missing required Python package: {missing}\n"
        "Install plotting dependencies with:\n"
        "  python -m pip install pandas matplotlib seaborn"
    ) from exc


TARGET_SYSTEMS = [
    "pipeline1_gemini",
    "pipeline1_gpt54_high",
    "pipeline2_gemini",
    "pipeline2_gpt54_high",
    "pipeline3",
]

COMPARISON_SYSTEMS = TARGET_SYSTEMS + ["ht"]

TARGET_SYSTEM_COLORS = {
    "pipeline1_gemini": "#8ecae6",
    "pipeline2_gemini": "#2176ae",
    "pipeline1_gpt54_high": "#f4a261",
    "pipeline2_gpt54_high": "#c65d21",
    "pipeline3": "#5b8e3e",
    "ht": "#6f6f6f",
}

CSV_COLUMNS = [
    "split",
    "book",
    "pipeline",
    "model",
    "pipeline_model",
    "system_name",
    "system_folder",
    "metric",
    "score",
    "higher_is_better",
    "num_scored_segments",
    "num_filtered_segments",
    "num_available_segments",
    "path",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir,
        help="Directory containing dev/ and eval/ result folders.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=script_dir / "plots",
        help="Directory where PNG plots will be saved.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=script_dir / "all_results.csv",
        help="Path for the aggregated CSV output.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving them.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore --csv input and rebuild unfiltered rows from system_scores.json files.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warn(f"Skipping malformed JSON {path}: {exc}")
        return None
    except OSError as exc:
        warn(f"Skipping unreadable file {path}: {exc}")
        return None

    if not isinstance(data, dict):
        warn(f"Skipping {path}: top-level JSON is not an object")
        return None
    return data


def parse_location(path: Path, results_dir: Path) -> tuple[str, str, str]:
    try:
        parts = path.relative_to(results_dir).parts
    except ValueError:
        parts = path.parts

    split = parts[0] if len(parts) > 0 and parts[0] in {"dev", "eval"} else "unknown"
    book = parts[1] if len(parts) > 1 and split != "unknown" else "unknown"
    system_folder = path.parent.name
    return split, book, system_folder


def parse_system(system_folder: str, book: str) -> tuple[str, str, str, str]:
    system_name = system_folder
    prefix = f"{book}_"
    if book != "unknown" and system_name.startswith(prefix):
        system_name = system_name[len(prefix) :]

    if system_name == "ht" or system_name.endswith("_ht"):
        return "ht", "", "ht", "ht"

    match = re.match(r"^(pipeline\d+)(?:_(.+))?$", system_name)
    if not match:
        return "unknown", "", system_name, system_name

    pipeline = match.group(1)
    model = match.group(2) or ""
    pipeline_model = pipeline if not model else f"{pipeline}_{model}"
    return pipeline, model, pipeline_model, system_name


def as_float(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return score


def normalize_results_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "system_name" not in df.columns:
        df["system_name"] = df.get("pipeline_model", df.get("system_folder", ""))
    if "num_filtered_segments" not in df.columns:
        df["num_filtered_segments"] = 0
    if "num_available_segments" not in df.columns:
        df["num_available_segments"] = df.get("num_scored_segments", 0)

    if "higher_is_better" in df.columns:
        df["higher_is_better"] = df["higher_is_better"].map(
            lambda value: str(value).lower() in {"true", "1", "yes"}
        )
    for column in ("score", "num_scored_segments", "num_filtered_segments", "num_available_segments"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_csv_results(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except OSError as exc:
        warn(f"Could not read {csv_path}: {exc}")
        return pd.DataFrame(columns=CSV_COLUMNS)
    missing = {"split", "book", "pipeline_model", "metric", "score", "higher_is_better"} - set(df.columns)
    if missing:
        warn(f"Ignoring {csv_path}: missing required columns {', '.join(sorted(missing))}")
        return pd.DataFrame(columns=CSV_COLUMNS)
    return normalize_results_df(df)


def load_system_score_results(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    score_files = sorted(
        path
        for split in ("dev", "eval")
        for path in (results_dir / split).glob("*/*/*/system_scores.json")
    )

    for path in score_files:
        data = read_json(path)
        if data is None:
            continue

        metrics = data.get("metrics")
        if not isinstance(metrics, dict):
            warn(f"Skipping {path}: missing or invalid metrics object")
            continue

        split, book, folder_system = parse_location(path, results_dir)
        system_folder = str(data.get("system") or folder_system)
        pipeline, model, pipeline_model, system_name = parse_system(system_folder, book)
        if pipeline == "unknown":
            continue

        for metric_name, metric_data in sorted(metrics.items()):
            if not isinstance(metric_data, dict):
                warn(f"Skipping {path} metric {metric_name}: metric value is not an object")
                continue

            score = as_float(metric_data.get("score"))
            if score is None:
                warn(f"Skipping {path} metric {metric_name}: missing or invalid score")
                continue

            rows.append(
                {
                    "split": split,
                    "book": book,
                    "pipeline": pipeline,
                    "model": model,
                    "pipeline_model": pipeline_model,
                    "system_name": system_name,
                    "system_folder": system_folder,
                    "metric": str(metric_name),
                    "score": score,
                    "higher_is_better": bool(metric_data.get("higher_is_better")),
                    "num_scored_segments": int(metric_data.get("num_scored_segments") or 0),
                    "num_filtered_segments": 0,
                    "num_available_segments": int(metric_data.get("num_scored_segments") or 0),
                    "path": str(path),
                }
            )

    return normalize_results_df(pd.DataFrame(rows, columns=CSV_COLUMNS))


def metric_direction(metric_df: pd.DataFrame) -> str:
    higher = bool(metric_df["higher_is_better"].mode().iloc[0])
    return "higher is better" if higher else "lower is better"


def format_score_label(value: float) -> str:
    if pd.isna(value):
        return ""
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def add_axis_headroom(ax: plt.Axes, *, top_fraction: float = 0.12, right_fraction: float = 0.06) -> None:
    if top_fraction:
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        if span:
            ax.set_ylim(ymin, ymax + span * top_fraction)
    if right_fraction:
        xmin, xmax = ax.get_xlim()
        span = xmax - xmin
        if span:
            ax.set_xlim(xmin, xmax + span * right_fraction)


def annotate_bars(
    ax: plt.Axes,
    *,
    orientation: str = "vertical",
    fontsize: int = 7,
    rotation: int = 0,
    padding: int = 8,
) -> None:
    """Add compact numeric labels beyond bar CI whiskers."""
    if orientation == "horizontal":
        xmin, xmax = ax.get_xlim()
        span = xmax - xmin
        offset = span * (padding / 650)
        for patch in ax.patches:
            width = patch.get_width()
            if not math.isfinite(width) or width == 0:
                continue
            y = patch.get_y() + patch.get_height() / 2
            x_right = width
            tol = max(patch.get_height(), 0.02)
            for line in ax.lines:
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) == 0:
                    continue
                near = [float(xv) for xv, yv in zip(xdata, ydata) if abs(float(yv) - y) <= tol]
                if near:
                    x_right = max(x_right, max(near))
            ax.text(
                x_right + offset,
                y,
                format_score_label(float(width)),
                ha="left",
                va="center",
                fontsize=fontsize,
                clip_on=False,
            )
        return

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    offset = span * (padding / 650)
    for patch in ax.patches:
        height = patch.get_height()
        if not math.isfinite(height) or height == 0:
            continue
        x = patch.get_x() + patch.get_width() / 2
        y_top = height
        tol = max(patch.get_width(), 0.02)
        for line in ax.lines:
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) == 0:
                continue
            near = [float(yv) for xv, yv in zip(xdata, ydata) if abs(float(xv) - x) <= tol]
            if near:
                y_top = max(y_top, max(near))
        ax.text(
            x,
            y_top + offset,
            format_score_label(float(height)),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
            clip_on=False,
        )


def annotate_points(ax: plt.Axes, plot_df: pd.DataFrame, x_col: str, y_col: str) -> None:
    for _, row in plot_df.iterrows():
        ax.annotate(
            format_score_label(float(row[y_col])),
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def add_mean_labels(ax: plt.Axes, means: pd.Series, order: list[str]) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.02
    for x_pos, key in enumerate(order):
        if key not in means.index:
            continue
        mean = float(means.loc[key])
        ax.text(
            x_pos,
            mean + offset,
            format_score_label(mean),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def order_by_mean(df: pd.DataFrame, column: str) -> list[str]:
    means = df.groupby(column, observed=True)["score"].mean().sort_values()
    return list(means.index)


def save_current_plot(path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_score_distributions(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    for metric, metric_df in df.groupby("metric", sort=True):
        plt.figure(figsize=(12, 6))
        order = order_by_mean(metric_df, "pipeline_model")
        ax = sns.boxplot(
            data=metric_df,
            x="pipeline_model",
            y="score",
            order=order,
            color="#8fb9d9",
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "#f6c85f",
                "markeredgecolor": "#5a4a00",
                "markersize": 4,
            },
        )
        sns.stripplot(
            data=metric_df,
            x="pipeline_model",
            y="score",
            order=order,
            color="#22313f",
            alpha=0.35,
            size=2.5,
            jitter=0.25,
        )
        means = metric_df.groupby("pipeline_model", observed=True)["score"].mean()
        add_mean_labels(ax, means, order)
        plt.title(f"{metric}: Score Distribution by Pipeline/Model ({metric_direction(metric_df)})")
        plt.xlabel("Pipeline / model")
        plt.ylabel("Score")
        plt.xticks(rotation=45, ha="right")
        save_current_plot(plots_dir / f"distribution_{safe_filename(metric)}.png", show)


def plot_average_per_pipeline(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    rankable_df = df[df["pipeline"] != "ht"]
    for metric, metric_df in rankable_df.groupby("metric", sort=True):
        higher = bool(metric_df["higher_is_better"].mode().iloc[0])
        order = (
            metric_df.groupby("pipeline", observed=True)["score"]
            .mean()
            .sort_values(ascending=not higher)
            .index.tolist()
        )

        plt.figure(figsize=(8, 5))
        ax = sns.barplot(
            data=metric_df,
            x="pipeline",
            y="score",
            hue="pipeline",
            order=order,
            hue_order=order,
            palette="Set2",
            legend=False,
            errorbar=("ci", 95),
            capsize=0.12,
            err_kws={"linewidth": 1.2},
        )
        add_axis_headroom(ax, top_fraction=0.18)
        annotate_bars(ax, fontsize=8, padding=12)
        plt.title(f"{metric}: Average Score per Pipeline ({metric_direction(metric_df)})")
        plt.xlabel("Pipeline")
        plt.ylabel("Mean score with 95% CI")
        save_current_plot(plots_dir / f"average_pipeline_{safe_filename(metric)}.png", show)


def plot_scores_per_book(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    for metric, metric_df in df.groupby("metric", sort=True):
        books = sorted(metric_df["book"].unique())
        pipelines = sorted(metric_df["pipeline"].unique())

        plt.figure(figsize=(max(12, 0.45 * len(books)), 6.5))
        ax = sns.barplot(
            data=metric_df,
            x="book",
            y="score",
            hue="pipeline",
            order=books,
            hue_order=pipelines,
            palette="Set2",
            errorbar=("ci", 95),
            capsize=0.05,
            err_kws={"linewidth": 0.9},
        )
        add_axis_headroom(ax, top_fraction=0.2)
        annotate_bars(ax, fontsize=5, rotation=90, padding=11)
        plt.title(f"{metric}: Average Scores per Book ({metric_direction(metric_df)})")
        plt.xlabel("Book")
        plt.ylabel("Mean score with 95% CI")
        plt.xticks(rotation=45, ha="right")
        plt.legend(
            title="Pipeline",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=min(len(pipelines), 4),
        )
        save_current_plot(plots_dir / f"scores_per_book_{safe_filename(metric)}.png", show)


def plot_eval_selected_scores_per_book(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    selected = df[(df["split"] == "eval") & (df["pipeline_model"].isin(COMPARISON_SYSTEMS))].copy()
    if selected.empty:
        warn("No eval rows found for selected pipeline/model book plots")
        return

    hue_order = [target for target in COMPARISON_SYSTEMS if target in set(selected["pipeline_model"])]
    for metric, metric_df in selected.groupby("metric", sort=True):
        books = sorted(metric_df["book"].unique())

        plt.figure(figsize=(max(12, 0.48 * len(books)), 6.5))
        ax = sns.barplot(
            data=metric_df,
            x="book",
            y="score",
            hue="pipeline_model",
            order=books,
            hue_order=hue_order,
            palette=TARGET_SYSTEM_COLORS,
            errorbar=("ci", 95),
            capsize=0.05,
            err_kws={"linewidth": 0.9},
        )
        add_axis_headroom(ax, top_fraction=0.2)
        annotate_bars(ax, fontsize=5, rotation=90, padding=11)
        plt.title(f"Eval Books - {metric}: Selected Pipeline/Model Scores with HT ({metric_direction(metric_df)})")
        plt.xlabel("Eval book")
        plt.ylabel("Mean score with 95% CI")
        plt.xticks(rotation=45, ha="right")
        plt.legend(
            title="Pipeline / model",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=min(len(hue_order), 3),
        )
        save_current_plot(plots_dir / f"eval_scores_per_book_selected_{safe_filename(metric)}.png", show)


def directional_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for metric, metric_df in df.groupby("metric", sort=True):
        metric_df = metric_df.copy()
        sign = 1 if bool(metric_df["higher_is_better"].mode().iloc[0]) else -1
        directional_score = sign * metric_df["score"]
        std = directional_score.std(ddof=0)
        if std == 0 or pd.isna(std):
            metric_df["directional_z_score"] = 0.0
        else:
            metric_df["directional_z_score"] = (directional_score - directional_score.mean()) / std
        pieces.append(metric_df)
    return pd.concat(pieces, ignore_index=True) if pieces else df.assign(directional_z_score=[])


def plot_selected_comparisons(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    selected = df[df["pipeline_model"].isin(COMPARISON_SYSTEMS)].copy()
    if selected.empty:
        warn("No selected pipeline/model systems found for comparison plot")
        return

    selected_z = directional_z_scores(selected)
    summary = (
        selected_z.groupby(["pipeline_model", "metric"], as_index=False, observed=True)["directional_z_score"]
        .mean()
        .sort_values(["pipeline_model", "metric"])
    )

    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=summary,
        x="metric",
        y="directional_z_score",
        hue="pipeline_model",
        marker="o",
        hue_order=[target for target in COMPARISON_SYSTEMS if target in set(summary["pipeline_model"])],
        palette=TARGET_SYSTEM_COLORS,
    )
    annotate_points(plt.gca(), summary, "metric", "directional_z_score")
    plt.axhline(0, color="#4c4c4c", linewidth=0.8, linestyle="--")
    plt.title("Selected Pipeline/Model Comparison Across Metrics with HT")
    plt.xlabel("Metric")
    plt.ylabel("Mean directional z-score (higher means better)")
    plt.legend(title="Pipeline / model", loc="best")
    save_current_plot(plots_dir / "selected_pipeline_model_comparison.png", show)

    for metric, metric_df in selected.groupby("metric", sort=True):
        plt.figure(figsize=(9, 5))
        order = [target for target in COMPARISON_SYSTEMS if target in set(metric_df["pipeline_model"])]
        ax = sns.barplot(
            data=metric_df,
            x="pipeline_model",
            y="score",
            hue="pipeline_model",
            order=order,
            hue_order=order,
            palette=TARGET_SYSTEM_COLORS,
            legend=False,
            errorbar=("ci", 95),
            capsize=0.12,
            err_kws={"linewidth": 1.2},
        )
        add_axis_headroom(ax, top_fraction=0.18)
        annotate_bars(ax, fontsize=8, padding=12)
        direction = metric_direction(df[df["metric"] == metric])
        plt.title(f"{metric}: Selected Pipeline/Model Mean Scores with HT ({direction})")
        plt.xlabel("Pipeline / model")
        plt.ylabel("Mean score with 95% CI")
        plt.xticks(rotation=35, ha="right")
        save_current_plot(plots_dir / f"selected_{safe_filename(metric)}.png", show)


def plot_best_worst_systems(df: pd.DataFrame, plots_dir: Path, show: bool, n: int = 10) -> None:
    rankable_df = df[df["pipeline_model"] != "ht"]
    for metric, metric_df in rankable_df.groupby("metric", sort=True):
        higher = bool(metric_df["higher_is_better"].mode().iloc[0])
        system_summary = (
            metric_df.groupby("pipeline_model", as_index=False, observed=True)["score"]
            .mean()
            .sort_values("score", ascending=not higher)
        )
        best = system_summary.head(n).assign(group="best")
        worst = system_summary.tail(n).assign(group="worst")
        plot_df = pd.concat([best, worst], ignore_index=True).drop_duplicates("pipeline_model")
        plot_df = plot_df.sort_values("score", ascending=not higher)
        order = list(plot_df["pipeline_model"])
        metric_plot_df = metric_df.merge(plot_df[["pipeline_model", "group"]], on="pipeline_model", how="inner")

        plt.figure(figsize=(9, max(5, 0.35 * len(plot_df))))
        ax = sns.barplot(
            data=metric_plot_df,
            y="pipeline_model",
            x="score",
            hue="group",
            order=order,
            hue_order=["best", "worst"],
            dodge=False,
            palette={"best": "#2e7d32", "worst": "#c62828"},
            errorbar=("ci", 95),
            capsize=0.12,
            err_kws={"linewidth": 1.1},
        )
        add_axis_headroom(ax, top_fraction=0, right_fraction=0.1)
        annotate_bars(ax, orientation="horizontal", fontsize=7, padding=14)
        plt.title(f"{metric}: Best and Worst Pipeline/Model Systems ({metric_direction(metric_df)})")
        plt.xlabel("Mean score with 95% CI")
        plt.ylabel("Pipeline / model")
        plt.legend(title="", loc="upper right", bbox_to_anchor=(1.0, -0.05), ncol=2)
        save_current_plot(plots_dir / f"best_worst_{safe_filename(metric)}.png", show)


def plot_metric_correlations(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    direction_by_metric = {
        metric: bool(metric_df["higher_is_better"].mode().iloc[0])
        for metric, metric_df in df.groupby("metric", sort=True)
    }
    pivot = (
        df.pivot_table(
            index=["split", "book", "pipeline_model", "system_name"],
            columns="metric",
            values="score",
            aggfunc="mean",
        )
        .dropna(axis=1, how="all")
        .reset_index()
    )
    metrics = [col for col in pivot.columns if col not in {"split", "book", "pipeline_model", "system_name"}]
    if len(metrics) < 2:
        warn("Not enough metrics for correlation plots")
        return

    directional_pivot = pivot.copy()
    for metric in metrics:
        if not direction_by_metric.get(metric, True):
            directional_pivot[metric] = -directional_pivot[metric]

    corr = directional_pivot[metrics].corr()
    plt.figure(figsize=(1.2 * len(metrics) + 4, 1.0 * len(metrics) + 3))
    sns.heatmap(corr, annot=True, cmap="vlag", vmin=-1, vmax=1, center=0, square=True)
    plt.title("Correlation Between Metrics (direction-normalized: higher is better)")
    save_current_plot(plots_dir / "metric_correlation_heatmap.png", show)

    for x_metric, y_metric in combinations(metrics, 2):
        pair_df = pivot[[x_metric, y_metric]].dropna()
        if len(pair_df) < 3:
            continue

        plt.figure(figsize=(6.5, 5.5))
        sns.regplot(
            data=pair_df,
            x=x_metric,
            y=y_metric,
            scatter_kws={"alpha": 0.5, "s": 24},
            line_kws={"color": "#c44e52"},
        )
        corr_value = pair_df[x_metric].corr(pair_df[y_metric])
        plt.title(f"{x_metric} vs {y_metric} (r={corr_value:.2f})")
        plt.xlabel(x_metric)
        plt.ylabel(y_metric)
        save_current_plot(
            plots_dir / f"correlation_{safe_filename(x_metric)}_vs_{safe_filename(y_metric)}.png",
            show,
        )


def plot_all(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_score_distributions(df, plots_dir, show)
    plot_average_per_pipeline(df, plots_dir, show)
    plot_scores_per_book(df, plots_dir, show)
    plot_eval_selected_scores_per_book(df, plots_dir, show)
    plot_selected_comparisons(df, plots_dir, show)
    plot_best_worst_systems(df, plots_dir, show)
    plot_metric_correlations(df, plots_dir, show)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    plots_dir = args.plots_dir.resolve()
    csv_path = args.csv.resolve()

    if csv_path.exists() and not args.rebuild:
        df = load_csv_results(csv_path)
        source = "summarized CSV"
        should_write_csv = False
    else:
        df = load_system_score_results(results_dir)
        source = "system score files"
        should_write_csv = True
    if df.empty:
        raise SystemExit(f"No valid pipeline results found under {results_dir}")

    if should_write_csv:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    plot_all(df, plots_dir, args.show)

    print(f"Loaded {len(df):,} metric rows from {df['path'].nunique():,} result paths in {source}.")
    if should_write_csv:
        print(f"Saved aggregated CSV to {csv_path}")
    else:
        print(f"Used existing summarized CSV at {csv_path}")
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
