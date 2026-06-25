from __future__ import annotations

import argparse

from statistical.config import ensure_output_dirs
from statistical.scripts.lexical_diversity import run as run_lexical_diversity
from statistical.scripts.ngram_analysis import run as run_ngrams
from statistical.scripts.per_book_analysis import run as run_per_book
from statistical.scripts.repetition_analysis import run as run_repetition
from statistical.scripts.similarity_sbert import run as run_sbert
from statistical.scripts.similarity_tfidf import run as run_tfidf
from statistical.scripts.structure_analysis import run as run_structure
from statistical.scripts.system_map import run as run_system_map
from statistical.scripts.token_distribution import run as run_token_distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HT/MT statistical analysis plots and tables.")
    parser.add_argument(
        "--skip-sbert",
        action="store_true",
        help="Skip SBERT heatmap generation. Useful when sentence-transformers/model downloads are unavailable.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    tfidf_matrix = run_tfidf()
    run_per_book()
    run_system_map(tfidf_matrix)
    run_structure()
    run_ngrams()
    run_token_distribution()
    run_lexical_diversity()
    run_repetition()
    if not args.skip_sbert:
        run_sbert()


if __name__ == "__main__":
    main()
