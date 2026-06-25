from __future__ import annotations

import pandas as pd

from statistical.config import HT_SYSTEM_NAME, PLOT_DIRS, TABLES_DIR, ensure_output_dirs
from statistical.utils.loader import load_corpus, mt_systems
from statistical.utils.plotting import grouped_barh
from statistical.utils.text_processing import mean_absolute_percentage_difference, text_stats


def run() -> pd.DataFrame:
    ensure_output_dirs()
    _, systems, pipelines = load_corpus()
    rows = []
    ht_stats = {book: text_stats(text) for book, text in systems[HT_SYSTEM_NAME].items()}

    for system in mt_systems(systems):
        common_books = sorted(set(systems[system]) & set(ht_stats))
        system_stats = {book: text_stats(systems[system][book]) for book in common_books}
        rows.append({
            "system": system,
            "pipeline": pipelines.get(system, "unknown"),
            "words": mean_absolute_percentage_difference(
                (system_stats[book].words, ht_stats[book].words) for book in common_books
            ),
            "sentences": mean_absolute_percentage_difference(
                (system_stats[book].sentences, ht_stats[book].sentences) for book in common_books
            ),
            "paragraphs": mean_absolute_percentage_difference(
                (system_stats[book].paragraphs, ht_stats[book].paragraphs) for book in common_books
            ),
        })

    data = pd.DataFrame(rows).sort_values(["pipeline", "system"])
    data.to_csv(TABLES_DIR / "structure_difference.csv", index=False)
    grouped_barh(data, PLOT_DIRS["structure"] / "structure_difference.png")
    return data


if __name__ == "__main__":
    run()

