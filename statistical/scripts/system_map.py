from __future__ import annotations

import pandas as pd

from statistical.config import HT_SYSTEM_NAME, PLOT_DIRS, TABLES_DIR, ensure_output_dirs
from statistical.scripts.similarity_tfidf import run as run_tfidf
from statistical.utils.loader import load_corpus, mt_systems
from statistical.utils.plotting import system_map_plot


def build_system_map(similarity_matrix: pd.DataFrame, pipelines: dict[str, str]) -> pd.DataFrame:
    rows = []
    mt_names = [system for system in similarity_matrix.index if system != HT_SYSTEM_NAME]
    for system in mt_names:
        other_mt = [other for other in mt_names if other != system]
        rows.append({
            "system": system,
            "pipeline": pipelines.get(system, "unknown"),
            "mean_similarity_to_ht": float(similarity_matrix.loc[system, HT_SYSTEM_NAME]),
            "mean_similarity_to_mt": float(similarity_matrix.loc[system, other_mt].mean()) if other_mt else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["pipeline", "system"])


def run(similarity_matrix: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_output_dirs()
    _, systems, pipelines = load_corpus()
    if similarity_matrix is None:
        table_path = TABLES_DIR / "ht_mt_similarity_tfidf.csv"
        similarity_matrix = pd.read_csv(table_path, index_col=0) if table_path.exists() else run_tfidf()
    data = build_system_map(similarity_matrix.loc[[HT_SYSTEM_NAME, *mt_systems(systems)]], pipelines)
    data.to_csv(TABLES_DIR / "system_2d_map.csv", index=False)
    system_map_plot(data, PLOT_DIRS["system_maps"] / "system_2d_map.png")
    return data


if __name__ == "__main__":
    run()

