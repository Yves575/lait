from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from statistical.config import HT_SYSTEM_NAME, PLOT_DIRS, TABLE_DIRS, ensure_output_dirs
from statistical.utils.loader import load_corpus, mt_systems
from statistical.utils.plotting import set_style
from statistical.utils.text_processing import mtld, normalized_words, sliding_windows, type_token_ratio


WINDOW_SIZE = 500
EVAL_PIPELINE_SYSTEMS = {
    "P1": ["[P1] gemini", "[P1] gpt54_high"],
    "P2": ["[P2] gemini", "[P2] gpt54_high"],
}


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _window_rows(system: str, pipeline: str, book: str, text: str) -> list[dict[str, object]]:
    tokens = normalized_words(text)
    rows = []
    for index, window in enumerate(sliding_windows(tokens, WINDOW_SIZE), start=1):
        rows.append({
            "system": system,
            "pipeline": pipeline,
            "book": book,
            "window_index": index,
            "token_count": len(window),
            "ttr": type_token_ratio(window),
            "mtld": mtld(window),
        })
    return rows


def _plot_metric(windows: pd.DataFrame, pipeline: str, metric: str) -> None:
    set_style()
    subset = windows[windows["pipeline"].isin(["HT", pipeline])]
    summary = (
        subset.groupby(["pipeline", "window_index"], as_index=False)[metric]
        .mean()
        .sort_values("window_index")
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"HT": "#111111", pipeline: "#c0392b"}
    for label, group in summary.groupby("pipeline"):
        ax.plot(group["window_index"], group[metric], label=HT_SYSTEM_NAME if label == "HT" else label,
                color=colors.get(label), linewidth=1.8)
    ax.set_title(f"{metric.upper()} over Text Windows: {pipeline} vs HT")
    ax.set_xlabel(f"Ordered {WINDOW_SIZE}-token window")
    ax.set_ylabel(metric.upper())
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(
        PLOT_DIRS["lexical_diversity"] / f"{_safe_name(pipeline)}_{metric}_windows.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_metric_for_systems(
    windows: pd.DataFrame,
    pipeline: str,
    selected_systems: list[str],
    metric: str,
    suffix: str,
) -> None:
    set_style()
    ht = windows[windows["pipeline"] == "HT"].copy()
    selected = windows[windows["system"].isin(selected_systems)].copy()
    if selected.empty:
        return

    ht["plot_group"] = "HT"
    selected["plot_group"] = pipeline
    subset = pd.concat([ht, selected], ignore_index=True)
    summary = (
        subset.groupby(["plot_group", "window_index"], as_index=False)[metric]
        .mean()
        .sort_values("window_index")
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"HT": "#111111", pipeline: "#c0392b"}
    for label, group in summary.groupby("plot_group"):
        ax.plot(
            group["window_index"],
            group[metric],
            label=HT_SYSTEM_NAME if label == "HT" else f"{label} selected models",
            color=colors.get(label),
            linewidth=1.8,
        )
    ax.set_title(f"{metric.upper()} over Text Windows: {pipeline} selected models vs HT")
    ax.set_xlabel(f"Ordered {WINDOW_SIZE}-token window")
    ax.set_ylabel(metric.upper())
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(
        PLOT_DIRS["lexical_diversity"] / f"{_safe_name(pipeline)}_{metric}_windows{suffix}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    _, systems, pipelines = load_corpus()
    rows = []
    system_order = [HT_SYSTEM_NAME, *mt_systems(systems)]
    for system in system_order:
        pipeline = pipelines.get(system, "HT" if system == HT_SYSTEM_NAME else "unknown")
        for book, text in systems[system].items():
            rows.extend(_window_rows(system, pipeline, book, text))

    windows = pd.DataFrame(rows)
    summary = (
        windows.groupby(["system", "pipeline"], as_index=False)
        .agg(
            mean_ttr=("ttr", "mean"),
            median_ttr=("ttr", "median"),
            mean_mtld=("mtld", "mean"),
            median_mtld=("mtld", "median"),
            windows=("window_index", "count"),
            mean_window_tokens=("token_count", "mean"),
        )
        .sort_values(["pipeline", "system"])
    )

    windows.to_csv(TABLE_DIRS["lexical_diversity"] / "lexical_diversity_windows.csv", index=False)
    summary.to_csv(TABLE_DIRS["lexical_diversity"] / "lexical_diversity_summary.csv", index=False)

    eval_systems = [system for selected in EVAL_PIPELINE_SYSTEMS.values() for system in selected]
    eval_windows = windows[(windows["pipeline"] == "HT") | (windows["system"].isin(eval_systems))].copy()
    eval_summary = summary[(summary["pipeline"] == "HT") | (summary["system"].isin(eval_systems))].copy()
    eval_windows.to_csv(TABLE_DIRS["lexical_diversity"] / "lexical_diversity_windows_eval.csv", index=False)
    eval_summary.to_csv(TABLE_DIRS["lexical_diversity"] / "lexical_diversity_summary_eval.csv", index=False)

    for pipeline in sorted(value for value in windows["pipeline"].unique() if value != "HT"):
        _plot_metric(windows, pipeline, "ttr")
        _plot_metric(windows, pipeline, "mtld")
    for pipeline, selected_systems in EVAL_PIPELINE_SYSTEMS.items():
        available_systems = [system for system in selected_systems if system in set(windows["system"])]
        _plot_metric_for_systems(windows, pipeline, available_systems, "ttr", "_eval")
        _plot_metric_for_systems(windows, pipeline, available_systems, "mtld", "_eval")
    return windows, summary


if __name__ == "__main__":
    run()
