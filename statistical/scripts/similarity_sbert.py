from __future__ import annotations

import pandas as pd

from statistical.config import HT_SYSTEM_NAME, PLOT_DIRS, SBERT_MODEL_NAME, TABLES_DIR, ensure_output_dirs
from statistical.utils.loader import load_corpus
from statistical.utils.plotting import heatmap
from statistical.utils.similarity import sbert_system_matrix


def run(model_name: str = SBERT_MODEL_NAME) -> pd.DataFrame:
    ensure_output_dirs()
    _, systems, _ = load_corpus()
    system_names = [HT_SYSTEM_NAME, *sorted(system for system in systems if system != HT_SYSTEM_NAME)]
    matrix = sbert_system_matrix(systems, system_names, model_name)
    matrix.to_csv(TABLES_DIR / "system_similarity_sbert.csv")
    heatmap(
        matrix,
        PLOT_DIRS["heatmaps"] / "system_similarity_sbert.png",
        "Global System Similarity Heatmap (SBERT)",
        cmap="YlOrRd",
        contrast_percentiles=(1.0, 99.0),
    )
    return matrix


if __name__ == "__main__":
    run()
