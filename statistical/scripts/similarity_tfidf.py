from __future__ import annotations

import pandas as pd

from statistical.config import HT_SYSTEM_NAME, PLOT_DIRS, TABLES_DIR, ensure_output_dirs
from statistical.utils.loader import load_corpus
from statistical.utils.plotting import heatmap
from statistical.utils.similarity import tfidf_system_matrix


def run() -> pd.DataFrame:
    ensure_output_dirs()
    _, systems, _ = load_corpus()
    system_names = [HT_SYSTEM_NAME, *sorted(system for system in systems if system != HT_SYSTEM_NAME)]
    matrix = tfidf_system_matrix(systems, system_names)
    matrix.to_csv(TABLES_DIR / "ht_mt_similarity_tfidf.csv")
    heatmap(
        matrix,
        PLOT_DIRS["heatmaps"] / "ht_mt_similarity_tfidf.png",
        "HT/MT Similarity Heatmap (TF-IDF)",
        cmap="YlOrRd",
        contrast_percentiles=(1.0, 99.0),
    )
    return matrix


if __name__ == "__main__":
    run()
