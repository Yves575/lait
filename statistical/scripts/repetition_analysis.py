from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from statistical.config import HT_SYSTEM_NAME, PLOT_DIRS, TABLE_DIRS, ensure_output_dirs
from statistical.utils.loader import load_corpus, mt_systems
from statistical.utils.plotting import set_style
from statistical.utils.text_processing import chunk_text, normalized_words, repetition_rate


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _rows_for_text(system: str, pipeline: str, book: str, text: str) -> list[dict[str, object]]:
    rows = []
    for segment_index, segment in enumerate(chunk_text(text), start=1):
        tokens = normalized_words(segment)
        for n in (1, 2, 3):
            rows.append({
                "system": system,
                "pipeline": pipeline,
                "book": book,
                "segment_index": segment_index,
                "n": n,
                "token_count": len(tokens),
                "repetition_rate": repetition_rate(tokens, n),
            })
    return rows


def _plot_pipeline(data: pd.DataFrame, pipeline: str) -> None:
    set_style()
    subset = data[data["pipeline"].isin(["HT", pipeline])].copy()
    subset["group"] = subset["pipeline"].map({"HT": HT_SYSTEM_NAME}).fillna(subset["pipeline"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, n in zip(axes, (1, 2, 3), strict=True):
        sns.boxplot(
            data=subset[subset["n"] == n],
            x="group",
            y="repetition_rate",
            hue="group",
            ax=ax,
            palette={HT_SYSTEM_NAME: "#f4b000", pipeline: "#c0392b"},
            showfliers=False,
            legend=False,
        )
        ax.set_title(f"{n}-gram repetition")
        ax.set_xlabel("")
        ax.set_ylabel("Repetition rate" if n == 1 else "")
    fig.suptitle(f"Repetition Rate Distribution: {pipeline} vs HT", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIRS["repetition"] / f"{_safe_name(pipeline)}_repetition_rates.png", bbox_inches="tight")
    plt.close(fig)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    _, systems, pipelines = load_corpus()
    rows = []
    for system in [HT_SYSTEM_NAME, *mt_systems(systems)]:
        pipeline = pipelines.get(system, "HT" if system == HT_SYSTEM_NAME else "unknown")
        for book, text in systems[system].items():
            rows.extend(_rows_for_text(system, pipeline, book, text))

    data = pd.DataFrame(rows)
    summary = (
        data.groupby(["system", "pipeline", "n"], as_index=False)
        .agg(
            mean_repetition_rate=("repetition_rate", "mean"),
            median_repetition_rate=("repetition_rate", "median"),
            p95_repetition_rate=("repetition_rate", lambda values: values.quantile(0.95)),
            segments=("segment_index", "count"),
        )
        .sort_values(["pipeline", "system", "n"])
    )
    data.to_csv(TABLE_DIRS["repetition"] / "repetition_segment_rates.csv", index=False)
    summary.to_csv(TABLE_DIRS["repetition"] / "repetition_summary.csv", index=False)

    for pipeline in sorted(value for value in data["pipeline"].unique() if value != "HT"):
        _plot_pipeline(data, pipeline)
    return data, summary


if __name__ == "__main__":
    run()
