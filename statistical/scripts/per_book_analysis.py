from __future__ import annotations

import pandas as pd

from statistical.config import PLOT_DIRS, TABLES_DIR, ensure_output_dirs
from statistical.utils.loader import load_corpus, mt_systems
from statistical.utils.plotting import heatmap
from statistical.utils.similarity import tfidf_ht_per_book


def run() -> pd.DataFrame:
    ensure_output_dirs()
    _, systems, _ = load_corpus()
    matrix = tfidf_ht_per_book(systems, mt_systems(systems))
    matrix.to_csv(TABLES_DIR / "per_book_similarity.csv")
    heatmap(
        matrix,
        PLOT_DIRS["heatmaps"] / "per_book_similarity.png",
        "Per-Book HT vs MT Similarity",
        cmap="YlOrRd",
        focus_off_diagonal=False,
        contrast_percentiles=(1.0, 99.0),
        fmt=".2f",
    )
    return matrix


if __name__ == "__main__":
    run()
