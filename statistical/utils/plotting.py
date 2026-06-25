from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from statistical.config import DPI


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def heatmap(
    data: pd.DataFrame,
    output_path: Path,
    title: str,
    cmap: str = "YlOrRd",
    focus_off_diagonal: bool = True,
    contrast_percentiles: tuple[float, float] = (1.0, 99.0),
    fmt: str = ".3f",
) -> None:
    set_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = data.to_numpy(dtype=float)
    if focus_off_diagonal and values.shape[0] == values.shape[1] and values.shape[0] > 1:
        mask = ~np.eye(values.shape[0], dtype=bool)
        off_diag = values[mask]
        vmin = float(np.nanpercentile(off_diag, contrast_percentiles[0]))
        vmax = float(np.nanpercentile(off_diag, contrast_percentiles[1]))
        if vmin == vmax:
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    else:
        vmin = float(np.nanpercentile(values, contrast_percentiles[0]))
        vmax = float(np.nanpercentile(values, contrast_percentiles[1]))
    if vmin == vmax:
        vmin, vmax = 0.0, 1.0

    width = max(8, min(18, 0.52 * len(data.columns) + 4))
    height = max(6, min(16, 0.42 * len(data.index) + 3))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=fmt,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Cosine similarity"},
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def system_map_plot(data: pd.DataFrame, output_path: Path) -> None:
    set_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=data,
        x="mean_similarity_to_ht",
        y="mean_similarity_to_mt",
        hue="pipeline",
        s=95,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )
    for _, row in data.iterrows():
        ax.annotate(row["system"], (row["mean_similarity_to_ht"], row["mean_similarity_to_mt"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_title("2D System Similarity Map")
    ax.set_xlabel("Mean similarity to Human Translation")
    ax.set_ylabel("Mean similarity to other MT systems")
    ax.legend(title="Pipeline", loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def grouped_barh(data: pd.DataFrame, output_path: Path) -> None:
    set_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long = data.melt(id_vars=["system", "pipeline"], var_name="metric", value_name="mape")
    fig_height = max(6, 0.42 * data["system"].nunique() + 2)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    sns.barplot(data=long, y="system", x="mape", hue="metric", ax=ax)
    ax.set_title("Structural Difference vs Human Translation")
    ax.set_xlabel("Mean absolute percentage difference")
    ax.set_ylabel("")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
