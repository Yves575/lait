from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BOOKS_DIR = ROOT_DIR / "books"
STATISTICAL_DIR = ROOT_DIR / "statistical"
OUTPUTS_DIR = STATISTICAL_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TABLES_DIR = OUTPUTS_DIR / "tables"
CACHE_DIR = OUTPUTS_DIR / "cache"

SOURCE_DIRS = (BOOKS_DIR / "dev", BOOKS_DIR / "eval")
HT_DIR = BOOKS_DIR / "HT"
MT_DIR = BOOKS_DIR / "MT"

HT_SYSTEM_NAME = "Human (HT)"
SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SBERT_BATCH_SIZE = 16
CHUNK_WORDS = 220
MIN_CHUNK_WORDS = 40
RANDOM_STATE = 42
DPI = 300

PLOT_DIRS = {
    "heatmaps": PLOTS_DIR / "heatmaps",
    "system_maps": PLOTS_DIR / "system_maps",
    "structure": PLOTS_DIR / "structure",
    "ngrams": PLOTS_DIR / "ngrams",
    "token_distribution": PLOTS_DIR / "token_distribution",
    "lexical_diversity": PLOTS_DIR / "lexical_diversity",
    "repetition": PLOTS_DIR / "repetition",
}

TABLE_DIRS = {
    "ngrams": TABLES_DIR / "ngrams",
    "token_distribution": TABLES_DIR / "token_distribution",
    "lexical_diversity": TABLES_DIR / "lexical_diversity",
    "repetition": TABLES_DIR / "repetition",
}


def ensure_output_dirs() -> None:
    for path in [OUTPUTS_DIR, TABLES_DIR, CACHE_DIR, *PLOT_DIRS.values(), *TABLE_DIRS.values()]:
        path.mkdir(parents=True, exist_ok=True)
