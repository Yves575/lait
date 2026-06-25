from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from statistical.config import CACHE_DIR, HT_SYSTEM_NAME
from statistical.utils.loader import common_books_for_systems
from statistical.utils.text_processing import chunk_text, normalize_text


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def average_tfidf_similarity(texts: list[str]) -> np.ndarray:
    normalized = [normalize_text(text) for text in texts]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(normalized)
    return cosine_similarity(matrix)


def tfidf_system_matrix(systems: dict[str, dict[str, str]], system_names: list[str]) -> pd.DataFrame:
    totals = pd.DataFrame(0.0, index=system_names, columns=system_names)
    counts = pd.DataFrame(0, index=system_names, columns=system_names)
    books = common_books_for_systems(systems, system_names)

    for book in books:
        texts = [systems[system][book] for system in system_names]
        sim = average_tfidf_similarity(texts)
        totals += sim
        counts += 1

    result = totals.divide(counts.where(counts != 0)).fillna(0.0)
    for system in system_names:
        result.loc[system, system] = 1.0
    return result


def tfidf_ht_per_book(systems: dict[str, dict[str, str]], mt_system_names: list[str]) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    ht_books = systems[HT_SYSTEM_NAME]
    for book in sorted(ht_books):
        available = [system for system in mt_system_names if book in systems.get(system, {})]
        if not available:
            continue
        names = [HT_SYSTEM_NAME, *available]
        sim = average_tfidf_similarity([systems[name][book] for name in names])
        rows[book] = {system: float(sim[0, idx + 1]) for idx, system in enumerate(available)}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _cache_key(model_name: str, text: str) -> str:
    digest = hashlib.sha256(f"{model_name}\0{text}".encode("utf-8", errors="replace")).hexdigest()
    return digest


def cached_embedding(model, model_name: str, text: str, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model_name, text)
    path = cache_dir / f"{key}.npy"
    if path.exists():
        return np.load(path)
    chunks = chunk_text(text)
    if not chunks:
        embedding = np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
    else:
        chunk_embeddings = model.encode(chunks, batch_size=16, show_progress_bar=False, normalize_embeddings=True)
        embedding = np.asarray(chunk_embeddings, dtype=np.float32).mean(axis=0)
        norm = np.linalg.norm(embedding)
        if norm:
            embedding = embedding / norm
    np.save(path, embedding)
    return embedding


def sbert_system_matrix(
    systems: dict[str, dict[str, str]],
    system_names: list[str],
    model_name: str,
) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    totals = pd.DataFrame(0.0, index=system_names, columns=system_names)
    counts = pd.DataFrame(0, index=system_names, columns=system_names)
    books = common_books_for_systems(systems, system_names)

    for book in books:
        embeddings = {
            system: cached_embedding(model, model_name, systems[system][book])
            for system in system_names
        }
        for left in system_names:
            for right in system_names:
                totals.loc[left, right] += cosine(embeddings[left], embeddings[right])
                counts.loc[left, right] += 1

    result = totals.divide(counts.where(counts != 0)).fillna(0.0)
    for system in system_names:
        result.loc[system, system] = 1.0
    return result

