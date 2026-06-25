from __future__ import annotations

from statistical.config import HT_SYSTEM_NAME, SBERT_MODEL_NAME, ensure_output_dirs
from statistical.utils.loader import load_corpus
from statistical.utils.similarity import cached_embedding


def run(model_name: str = SBERT_MODEL_NAME) -> int:
    from sentence_transformers import SentenceTransformer

    ensure_output_dirs()
    _, systems, _ = load_corpus()
    model = SentenceTransformer(model_name)
    count = 0
    for system in [HT_SYSTEM_NAME, *sorted(system for system in systems if system != HT_SYSTEM_NAME)]:
        for text in systems[system].values():
            cached_embedding(model, model_name, text)
            count += 1
    return count


if __name__ == "__main__":
    total = run()
    print(f"Cached embeddings for {total} system/book texts.")

