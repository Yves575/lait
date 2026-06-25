#!/usr/bin/env python3
"""Surface and lexical analysis for MT guessed as HT vs MT guessed as MT.

Run from the project root:
    .venv/bin/python analysis/human_eval/analyze_mt_detection_surface_features.py

Outputs are written to:
    analysis/human_eval/mt_detection_surface_features/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statistical.utils.text_processing import (
    mtld,
    normalized_words,
    paragraphs,
    repetition_rate,
    sentences,
    type_token_ratio,
    words,
)

try:
    from scipy.stats import mannwhitneyu, spearmanr
except ImportError:  # pragma: no cover - scipy is expected in the project environment.
    mannwhitneyu = None
    spearmanr = None


DEFAULT_PART1 = REPO_ROOT / "human_eval" / "data" / "part1-study-data-full.csv"
DEFAULT_QUAL_COUNTS = (
    REPO_ROOT
    / "analysis"
    / "human_eval"
    / "mt_undetectability_category_intervals"
    / "category_counts_by_response.csv"
)
DEFAULT_OUT = REPO_ROOT / "analysis" / "human_eval" / "mt_detection_surface_features"
DEFAULT_HT_DIR = REPO_ROOT / "books" / "HT" / "eval"

SOURCE_LANG_CODES = {
    "french": "fr",
    "japanese": "ja",
    "polish": "pl",
}

TEXT_FEATURE_COLUMNS = [
    "token_count",
    "type_count",
    "ttr",
    "mtld",
    "hapax_rate",
    "sentence_count",
    "mean_sentence_words",
    "median_sentence_words",
    "std_sentence_words",
    "p90_sentence_words",
    "max_sentence_words",
    "long_sentence_share",
    "paragraph_count",
    "mean_paragraph_words",
    "repeated_unigram_rate",
    "repeated_bigram_rate",
    "repeated_trigram_rate",
    "em_dash_rate",
    "en_dash_rate",
    "hyphen_rate",
    "dash_rate",
    "ellipsis_rate",
    "semicolon_rate",
    "colon_rate",
    "parenthesis_rate",
    "quote_rate",
    "slash_rate",
    "punctuation_density",
    "quote_style_count",
    "dash_style_count",
    "mixed_quote_style",
    "mixed_dash_style",
    "slash_n_artifact_count",
    "single_newline_rate",
    "repeated_blank_line_count",
    "surface_artifact_index",
]

PAIRED_FEATURE_BASES = [
    "mtld",
    "ttr",
    "median_sentence_words",
    "std_sentence_words",
    "em_dash_rate",
    "en_dash_rate",
    "dash_rate",
    "dash_style_count",
    "quote_rate",
    "slash_rate",
    "surface_artifact_index",
    "repeated_bigram_rate",
]
PAIRED_FEATURE_COLUMNS = [
    feature
    for base in PAIRED_FEATURE_BASES
    for feature in (
        f"mt_minus_ht_{base}",
        f"abs_mt_minus_ht_{base}",
        f"mt_div_ht_{base}",
    )
]

CHUNK_ALIGNED_FEATURE_COLUMNS = [
    "aligned_chunk_count",
    "mean_abs_chunk_delta_em_dash_rate",
    "mean_signed_chunk_delta_em_dash_rate",
    "share_chunks_mt_more_em_dash",
    "mean_abs_chunk_delta_en_dash_rate",
    "mean_signed_chunk_delta_en_dash_rate",
    "share_chunks_mt_more_en_dash",
    "mean_abs_chunk_delta_hyphen_rate",
    "mean_signed_chunk_delta_hyphen_rate",
    "share_chunks_mt_more_hyphen",
    "mean_abs_chunk_delta_dash_rate",
    "mean_signed_chunk_delta_dash_rate",
    "share_chunks_mt_more_dash",
    "mean_abs_chunk_delta_quote_rate",
    "mean_signed_chunk_delta_quote_rate",
    "share_chunks_mt_more_quote",
    "mean_abs_chunk_delta_surface_artifact_index",
    "mean_signed_chunk_delta_surface_artifact_index",
    "share_chunks_mt_more_surface_artifact",
    "mean_abs_chunk_delta_mtld",
    "mean_abs_chunk_delta_ttr",
    "mean_abs_chunk_delta_median_sentence_words",
    "share_chunks_dash_style_mismatch",
    "share_chunks_quote_style_mismatch",
]

PRIMARY_FEATURES = [
    "mtld",
    "ttr",
    "median_sentence_words",
    "std_sentence_words",
    "em_dash_rate",
    "surface_artifact_index",
    "mt_minus_ht_mtld",
    "abs_mt_minus_ht_mtld",
    "mt_minus_ht_em_dash_rate",
    "abs_mt_minus_ht_em_dash_rate",
    "mt_minus_ht_dash_rate",
    "abs_mt_minus_ht_dash_rate",
    "mt_minus_ht_surface_artifact_index",
    "abs_mt_minus_ht_surface_artifact_index",
    "mean_abs_chunk_delta_em_dash_rate",
    "share_chunks_mt_more_em_dash",
    "mean_abs_chunk_delta_quote_rate",
    "mean_abs_chunk_delta_surface_artifact_index",
    "share_chunks_dash_style_mismatch",
    "q1_SG_A_word_choice_count",
    "q1_SG_A_surface_count",
]

QUAL_FEATURES = [
    "SG_A_word_choice_count",
    "SG_A_surface_count",
    "SG_A_sentence_count",
    "SG_C_smoothness_count",
    "SG_D_ai_mt_verdict_count",
]

QUAL_SCOPES = {
    "q1_pos_isolated": "q1",
    "q2_neg_isolated": "q2",
}

PLOT_LABELS = {
    "mtld": "MTLD",
    "ttr": "TTR",
    "median_sentence_words": "Median sentence words",
    "std_sentence_words": "Sentence-length SD",
    "em_dash_rate": "Em dash / 1k words",
    "dash_rate": "All dashes / 1k words",
    "surface_artifact_index": "Surface artifact index",
    "mt_minus_ht_mtld": "MT minus HT MTLD",
    "abs_mt_minus_ht_mtld": "|MT minus HT| MTLD",
    "mt_minus_ht_ttr": "MT minus HT TTR",
    "abs_mt_minus_ht_ttr": "|MT minus HT| TTR",
    "mt_minus_ht_median_sentence_words": "MT minus HT median sentence words",
    "abs_mt_minus_ht_median_sentence_words": "|MT minus HT| median sentence words",
    "mt_minus_ht_std_sentence_words": "MT minus HT sentence-length SD",
    "abs_mt_minus_ht_std_sentence_words": "|MT minus HT| sentence-length SD",
    "mt_minus_ht_em_dash_rate": "MT minus HT em dash / 1k words",
    "abs_mt_minus_ht_em_dash_rate": "|MT minus HT| em dash / 1k words",
    "mt_minus_ht_dash_rate": "MT minus HT all dashes / 1k words",
    "abs_mt_minus_ht_dash_rate": "|MT minus HT| all dashes / 1k words",
    "mt_minus_ht_surface_artifact_index": "MT minus HT surface artifact index",
    "abs_mt_minus_ht_surface_artifact_index": "|MT minus HT| surface artifact index",
    "mean_abs_chunk_delta_em_dash_rate": "Mean |chunk MT-HT| em dash",
    "mean_signed_chunk_delta_em_dash_rate": "Mean chunk MT-HT em dash",
    "share_chunks_mt_more_em_dash": "Share chunks MT > HT em dash",
    "mean_abs_chunk_delta_dash_rate": "Mean |chunk MT-HT| all dashes",
    "mean_signed_chunk_delta_dash_rate": "Mean chunk MT-HT all dashes",
    "share_chunks_mt_more_dash": "Share chunks MT > HT all dashes",
    "mean_abs_chunk_delta_quote_rate": "Mean |chunk MT-HT| quotes",
    "mean_abs_chunk_delta_surface_artifact_index": "Mean |chunk MT-HT| surface artifacts",
    "share_chunks_dash_style_mismatch": "Share chunks dash-style mismatch",
    "q1_SG_A_word_choice_count": "Q1 word-choice labels",
    "q1_SG_A_surface_count": "Q1 surface labels",
    "q1_SG_C_smoothness_count": "Q1 smoothness/cadence labels",
    "q2_SG_C_smoothness_count": "Q2 smoothness/cadence labels",
    "q1_SG_D_ai_mt_verdict_count": "Q1 AI/MT verdict/tell labels",
    "q2_SG_D_ai_mt_verdict_count": "Q2 AI/MT verdict/tell labels",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def canonical_book_id(book_id: object) -> str:
    return str(book_id or "").strip().replace(
        "polish_eval_FIXED_needle_s_eye", "polish_eval_needle_s_eye"
    )


def book_id_to_mt_stem(book_id: str) -> str:
    canonical = canonical_book_id(book_id)
    for prefix, lang_code in SOURCE_LANG_CODES.items():
        marker = f"{prefix}_eval_"
        if canonical.startswith(marker):
            title = canonical.removeprefix(marker)
            return f"{title}_{lang_code}_en"
    raise ValueError(f"Cannot infer MT filename stem from book_id: {book_id}")


def resolve_mt_text_path(book_id: str, repo_root: Path = REPO_ROOT) -> Path:
    stem = book_id_to_mt_stem(book_id)
    candidates = [
        repo_root / "books" / "MT_chunks" / f"{stem}.jsonl",
        repo_root / "books" / "MT" / "pipeline3" / "eval" / f"{stem}.txt",
        repo_root / "books" / "MT" / "pipeline3" / f"{stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No MT text found for {book_id}. Tried:\n{joined}")


def resolve_ht_text_path(book_id: str, ht_dir: Path = DEFAULT_HT_DIR) -> Path:
    stem = book_id_to_mt_stem(book_id)
    candidate = ht_dir / f"{stem}.jsonl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No HT eval text found for {book_id}: {candidate}")


def read_translation_text(path: Path) -> str:
    return "\n\n".join(text for _, text in read_translation_chunks(path)).strip()


def read_translation_chunks(path: Path) -> list[tuple[int, str]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append((int(obj.get("chunk_id", len(rows) + 1)), str(obj.get("text", ""))))
        return sorted(rows, key=lambda item: item[0])
    return [(1, path.read_text(encoding="utf-8").strip())]


def safe_rate(count: float, denominator: float) -> float:
    return count / denominator if denominator else 0.0


def sentence_lengths(text: str) -> list[int]:
    return [len(words(sentence)) for sentence in sentences(text) if words(sentence)]


def paragraph_lengths(text: str) -> list[int]:
    return [len(words(paragraph)) for paragraph in paragraphs(text) if words(paragraph)]


def count_single_newlines(text: str) -> int:
    return len(re.findall(r"(?<!\n)\n(?!\n)", text))


def count_ellipsis(text: str) -> int:
    return text.count("...") + text.count("…")


def count_quote_styles(text: str) -> int:
    styles = [
        bool(re.search(r'["]', text)),
        bool(re.search(r"[“”]", text)),
        bool(re.search(r"[‘’]", text)),
        bool(re.search(r"[«»]", text)),
    ]
    return sum(styles)


def count_dash_styles(text: str) -> int:
    styles = [bool(re.search(pattern, text)) for pattern in ["—", "–", r"-"]]
    return sum(styles)


def count_punctuation(text: str) -> int:
    extra = "—–…“”‘’«»"
    return sum(1 for char in text if char in string.punctuation or char in extra)


def extract_text_features(text: str) -> dict[str, float]:
    norm_tokens = normalized_words(text)
    raw_words = words(text)
    token_count = len(norm_tokens)
    type_counts = Counter(norm_tokens)
    sent_lengths = sentence_lengths(text)
    para_lengths = paragraph_lengths(text)
    per_1k = token_count / 1000 if token_count else 0.0

    slash_n_artifacts = len(re.findall(r"(?:/n+/?|\\n+)", text, flags=re.IGNORECASE))
    single_newlines = count_single_newlines(text)
    repeated_blank_lines = len(re.findall(r"\n\s*\n+", text))
    quote_style_count = count_quote_styles(text)
    dash_style_count = count_dash_styles(text)
    surface_artifact_index = (
        slash_n_artifacts
        + int(quote_style_count > 1)
        + int(dash_style_count > 1)
        + repeated_blank_lines
        + safe_rate(single_newlines, per_1k)
    )

    return {
        "token_count": float(token_count),
        "word_count": float(len(raw_words)),
        "type_count": float(len(type_counts)),
        "ttr": type_token_ratio(norm_tokens),
        "mtld": mtld(norm_tokens),
        "hapax_rate": safe_rate(sum(1 for count in type_counts.values() if count == 1), token_count),
        "sentence_count": float(len(sent_lengths)),
        "mean_sentence_words": float(np.mean(sent_lengths)) if sent_lengths else 0.0,
        "median_sentence_words": float(np.median(sent_lengths)) if sent_lengths else 0.0,
        "std_sentence_words": float(np.std(sent_lengths)) if sent_lengths else 0.0,
        "p90_sentence_words": float(np.percentile(sent_lengths, 90)) if sent_lengths else 0.0,
        "max_sentence_words": float(max(sent_lengths)) if sent_lengths else 0.0,
        "long_sentence_share": safe_rate(sum(length >= 30 for length in sent_lengths), len(sent_lengths)),
        "paragraph_count": float(len(para_lengths)),
        "mean_paragraph_words": float(np.mean(para_lengths)) if para_lengths else 0.0,
        "repeated_unigram_rate": repetition_rate(norm_tokens, 1),
        "repeated_bigram_rate": repetition_rate(norm_tokens, 2),
        "repeated_trigram_rate": repetition_rate(norm_tokens, 3),
        "em_dash_rate": safe_rate(text.count("—"), per_1k),
        "en_dash_rate": safe_rate(text.count("–"), per_1k),
        "hyphen_rate": safe_rate(text.count("-"), per_1k),
        "dash_rate": safe_rate(text.count("—") + text.count("–") + text.count("-"), per_1k),
        "ellipsis_rate": safe_rate(count_ellipsis(text), per_1k),
        "semicolon_rate": safe_rate(text.count(";"), per_1k),
        "colon_rate": safe_rate(text.count(":"), per_1k),
        "parenthesis_rate": safe_rate(text.count("(") + text.count(")"), per_1k),
        "quote_rate": safe_rate(sum(text.count(char) for char in "\"“”‘’«»"), per_1k),
        "slash_rate": safe_rate(text.count("/") + text.count("\\"), per_1k),
        "punctuation_density": safe_rate(count_punctuation(text), len(text)),
        "quote_style_count": float(quote_style_count),
        "dash_style_count": float(dash_style_count),
        "mixed_quote_style": float(quote_style_count > 1),
        "mixed_dash_style": float(dash_style_count > 1),
        "slash_n_artifact_count": float(slash_n_artifacts),
        "single_newline_rate": safe_rate(single_newlines, per_1k),
        "repeated_blank_line_count": float(repeated_blank_lines),
        "surface_artifact_index": float(surface_artifact_index),
    }


def prefixed_features(prefix: str, features: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in features.items()}


def paired_feature_deltas(
    mt_features: dict[str, float], ht_features: dict[str, float]
) -> dict[str, float]:
    paired: dict[str, float] = {}
    for feature in PAIRED_FEATURE_BASES:
        mt_value = float(mt_features.get(feature, np.nan))
        ht_value = float(ht_features.get(feature, np.nan))
        paired[f"mt_minus_ht_{feature}"] = mt_value - ht_value
        paired[f"abs_mt_minus_ht_{feature}"] = abs(mt_value - ht_value)
        paired[f"mt_div_ht_{feature}"] = mt_value / ht_value if ht_value else np.nan
    return paired


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def share_positive(values: list[float]) -> float:
    return sum(value > 0 for value in values) / len(values) if values else np.nan


def aligned_chunk_feature_deltas(
    mt_chunks: list[tuple[int, str]], ht_chunks: list[tuple[int, str]]
) -> dict[str, float]:
    mt_by_id = {chunk_id: text for chunk_id, text in mt_chunks}
    ht_by_id = {chunk_id: text for chunk_id, text in ht_chunks}
    shared_ids = sorted(set(mt_by_id) & set(ht_by_id))
    mt_features = {chunk_id: extract_text_features(mt_by_id[chunk_id]) for chunk_id in shared_ids}
    ht_features = {chunk_id: extract_text_features(ht_by_id[chunk_id]) for chunk_id in shared_ids}

    def deltas(feature: str) -> list[float]:
        return [
            float(mt_features[chunk_id].get(feature, np.nan))
            - float(ht_features[chunk_id].get(feature, np.nan))
            for chunk_id in shared_ids
        ]

    em_dash = deltas("em_dash_rate")
    en_dash = deltas("en_dash_rate")
    hyphen = deltas("hyphen_rate")
    dash = deltas("dash_rate")
    quote = deltas("quote_rate")
    surface = deltas("surface_artifact_index")
    mtld_deltas = deltas("mtld")
    ttr_deltas = deltas("ttr")
    median_sentence_deltas = deltas("median_sentence_words")
    dash_style_mismatches = [
        float(mt_features[chunk_id]["dash_style_count"] != ht_features[chunk_id]["dash_style_count"])
        for chunk_id in shared_ids
    ]
    quote_style_mismatches = [
        float(mt_features[chunk_id]["quote_style_count"] != ht_features[chunk_id]["quote_style_count"])
        for chunk_id in shared_ids
    ]
    return {
        "aligned_chunk_count": float(len(shared_ids)),
        "mean_abs_chunk_delta_em_dash_rate": mean_or_nan([abs(value) for value in em_dash]),
        "mean_signed_chunk_delta_em_dash_rate": mean_or_nan(em_dash),
        "share_chunks_mt_more_em_dash": share_positive(em_dash),
        "mean_abs_chunk_delta_en_dash_rate": mean_or_nan([abs(value) for value in en_dash]),
        "mean_signed_chunk_delta_en_dash_rate": mean_or_nan(en_dash),
        "share_chunks_mt_more_en_dash": share_positive(en_dash),
        "mean_abs_chunk_delta_hyphen_rate": mean_or_nan([abs(value) for value in hyphen]),
        "mean_signed_chunk_delta_hyphen_rate": mean_or_nan(hyphen),
        "share_chunks_mt_more_hyphen": share_positive(hyphen),
        "mean_abs_chunk_delta_dash_rate": mean_or_nan([abs(value) for value in dash]),
        "mean_signed_chunk_delta_dash_rate": mean_or_nan(dash),
        "share_chunks_mt_more_dash": share_positive(dash),
        "mean_abs_chunk_delta_quote_rate": mean_or_nan([abs(value) for value in quote]),
        "mean_signed_chunk_delta_quote_rate": mean_or_nan(quote),
        "share_chunks_mt_more_quote": share_positive(quote),
        "mean_abs_chunk_delta_surface_artifact_index": mean_or_nan(
            [abs(value) for value in surface]
        ),
        "mean_signed_chunk_delta_surface_artifact_index": mean_or_nan(surface),
        "share_chunks_mt_more_surface_artifact": share_positive(surface),
        "mean_abs_chunk_delta_mtld": mean_or_nan([abs(value) for value in mtld_deltas]),
        "mean_abs_chunk_delta_ttr": mean_or_nan([abs(value) for value in ttr_deltas]),
        "mean_abs_chunk_delta_median_sentence_words": mean_or_nan(
            [abs(value) for value in median_sentence_deltas]
        ),
        "share_chunks_dash_style_mismatch": mean_or_nan(dash_style_mismatches),
        "share_chunks_quote_style_mismatch": mean_or_nan(quote_style_mismatches),
    }


def build_book_feature_table(
    part1: pd.DataFrame, repo_root: Path, ht_dir: Path = DEFAULT_HT_DIR
) -> pd.DataFrame:
    rows = []
    canonical_books = sorted({canonical_book_id(book_id) for book_id in part1["book_id"]})
    for book_id in canonical_books:
        mt_path = resolve_mt_text_path(book_id, repo_root)
        ht_path = resolve_ht_text_path(book_id, ht_dir)
        mt_text = read_translation_text(mt_path)
        ht_text = read_translation_text(ht_path)
        mt_chunks = read_translation_chunks(mt_path)
        ht_chunks = read_translation_chunks(ht_path)
        mt_features = extract_text_features(mt_text)
        ht_features = extract_text_features(ht_text)
        paired_features = paired_feature_deltas(mt_features, ht_features)
        chunk_aligned_features = aligned_chunk_feature_deltas(mt_chunks, ht_chunks)
        rows.append(
            {
                "book_id": book_id,
                "mt_text_path": str(mt_path.relative_to(repo_root)),
                "ht_text_path": str(ht_path.relative_to(repo_root)),
                "mt_text_char_count": len(mt_text),
                "ht_text_char_count": len(ht_text),
                **mt_features,
                **prefixed_features("mt", mt_features),
                **prefixed_features("ht", ht_features),
                **paired_features,
                **chunk_aligned_features,
            }
        )
    return pd.DataFrame(rows)


def build_mt_judgment_rows(part1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in part1.iterrows():
        for reading_position in ["first", "second"]:
            if row.get(f"{reading_position}_version") != "MT":
                continue
            mt_guess = row.get(f"{reading_position}_q7_decipher", "")
            if mt_guess not in {"HT", "MT"}:
                continue
            rows.append(
                {
                    "user_id": row.get("user_id", ""),
                    "participant_id": row.get("participant_id", ""),
                    "book_id_original": row.get("book_id", ""),
                    "book_id": canonical_book_id(row.get("book_id", "")),
                    "source_lang": row.get("source_lang", ""),
                    "order": f"{row.get('first_version', '')}-first",
                    "reading_position": reading_position,
                    "mt_guess": mt_guess,
                    "mt_guessed_ht": int(mt_guess == "HT"),
                    "acceptability": pd.to_numeric(row.get(f"{reading_position}_q1", ""), errors="coerce"),
                    "smoothness": pd.to_numeric(row.get(f"{reading_position}_q2", ""), errors="coerce"),
                    "immersion": pd.to_numeric(row.get(f"{reading_position}_q3", ""), errors="coerce"),
                    "continue_reading": pd.to_numeric(row.get(f"{reading_position}_q4", ""), errors="coerce"),
                    "origin_confidence": pd.to_numeric(row.get(f"{reading_position}_q8", ""), errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def add_qualitative_features(judgments: pd.DataFrame, qual_path: Path) -> pd.DataFrame:
    result = judgments.copy()
    for scope_prefix in QUAL_SCOPES.values():
        for feature in QUAL_FEATURES:
            result[f"{scope_prefix}_{feature}"] = np.nan
    if not qual_path.exists():
        return result

    qual = read_csv(qual_path)
    qual = qual[
        qual["scope"].isin(QUAL_SCOPES)
        & (qual["current_trans"] == "MT")
        & qual["thought_ai"].isin(["HT", "MT"])
    ].copy()
    if qual.empty:
        return result
    qual["book_id"] = qual["book_id"].map(canonical_book_id)
    for scope, prefix in QUAL_SCOPES.items():
        subset = qual[qual["scope"] == scope]
        columns = ["participant_id", "book_id", *QUAL_FEATURES]
        rename = {feature: f"{prefix}_{feature}" for feature in QUAL_FEATURES}
        result = result.merge(
            subset[columns].rename(columns=rename),
            on=["participant_id", "book_id"],
            how="left",
            suffixes=("", "_from_qual"),
        )
        for feature in QUAL_FEATURES:
            column = f"{prefix}_{feature}"
            duplicate = f"{column}_from_qual"
            if duplicate in result.columns:
                result[column] = result[duplicate].combine_first(result[column])
                result = result.drop(columns=[duplicate])
    return result


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    greater = 0
    less = 0
    for value in a:
        greater += int(np.sum(value > b))
        less += int(np.sum(value < b))
    return (greater - less) / (len(a) * len(b))


def mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0 or mannwhitneyu is None:
        return np.nan
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0
    return float(mannwhitneyu(a, b, alternative="two-sided", method="auto").pvalue)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    valid = [(idx, value) for idx, value in enumerate(p_values) if pd.notna(value)]
    adjusted = [np.nan for _ in p_values]
    if not valid:
        return adjusted
    sorted_valid = sorted(valid, key=lambda item: item[1])
    total = len(sorted_valid)
    running_min = 1.0
    for reverse_rank, (idx, value) in enumerate(reversed(sorted_valid), start=1):
        rank = total - reverse_rank + 1
        running_min = min(running_min, value * total / rank)
        adjusted[idx] = min(running_min, 1.0)
    return adjusted


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    metadata = {
        "mt_guessed_ht",
        "acceptability",
        "smoothness",
        "immersion",
        "continue_reading",
        "origin_confidence",
        "text_char_count",
        "word_count",
    }
    columns = []
    for column in df.columns:
        if column in metadata:
            continue
        if (
            column in TEXT_FEATURE_COLUMNS
            or column in PAIRED_FEATURE_COLUMNS
            or column in CHUNK_ALIGNED_FEATURE_COLUMNS
            or column.startswith(("q1_SG_", "q2_SG_"))
        ):
            columns.append(column)
    return columns


def build_feature_tests(judgments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in numeric_feature_columns(judgments):
        focal = pd.to_numeric(
            judgments[judgments["mt_guess"] == "HT"][feature], errors="coerce"
        ).dropna()
        baseline = pd.to_numeric(
            judgments[judgments["mt_guess"] == "MT"][feature], errors="coerce"
        ).dropna()
        focal_values = focal.to_numpy(float)
        baseline_values = baseline.to_numpy(float)
        rows.append(
            {
                "feature": feature,
                "feature_label": PLOT_LABELS.get(feature, feature),
                "focal_group": "MT guessed HT",
                "baseline_group": "MT guessed MT",
                "focal_n": int(len(focal_values)),
                "baseline_n": int(len(baseline_values)),
                "focal_mean": np.mean(focal_values) if len(focal_values) else np.nan,
                "baseline_mean": np.mean(baseline_values) if len(baseline_values) else np.nan,
                "mean_delta_focal_minus_baseline": (
                    np.mean(focal_values) - np.mean(baseline_values)
                    if len(focal_values) and len(baseline_values)
                    else np.nan
                ),
                "focal_median": np.median(focal_values) if len(focal_values) else np.nan,
                "baseline_median": np.median(baseline_values) if len(baseline_values) else np.nan,
                "median_delta_focal_minus_baseline": (
                    np.median(focal_values) - np.median(baseline_values)
                    if len(focal_values) and len(baseline_values)
                    else np.nan
                ),
                "cliffs_delta": cliffs_delta(focal_values, baseline_values),
                "p_value": mann_whitney_p(focal_values, baseline_values),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["bh_q"] = benjamini_hochberg(result["p_value"].tolist())
        result["abs_cliffs_delta"] = result["cliffs_delta"].abs()
    return result.sort_values(["abs_cliffs_delta", "feature"], ascending=[False, True])


def permutation_spearman_p(
    x: np.ndarray, y: np.ndarray, observed_rho: float, permutations: int = 10000
) -> float:
    if len(x) < 3 or pd.isna(observed_rho):
        return np.nan
    rng = np.random.default_rng(20260522)
    hits = 0
    for _ in range(permutations):
        shuffled = rng.permutation(y)
        rho = spearmanr(x, shuffled).statistic if spearmanr is not None else np.nan
        if pd.notna(rho) and abs(rho) >= abs(observed_rho):
            hits += 1
    return (hits + 1) / (permutations + 1)


def build_book_level_table(judgments: pd.DataFrame, book_features: pd.DataFrame) -> pd.DataFrame:
    agg = (
        judgments.groupby("book_id", as_index=False)
        .agg(
            mt_judgments=("mt_guess", "count"),
            mt_guessed_ht_count=("mt_guessed_ht", "sum"),
            undetected_rate=("mt_guessed_ht", "mean"),
            source_lang=("source_lang", "first"),
        )
        .sort_values("book_id")
    )
    return agg.merge(book_features, on="book_id", how="left")


def build_book_correlations(book_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(book_table["undetected_rate"], errors="coerce")
    candidate_features = [
        column
        for column in [*TEXT_FEATURE_COLUMNS, *PAIRED_FEATURE_COLUMNS]
        + CHUNK_ALIGNED_FEATURE_COLUMNS
        if column in book_table.columns
    ]
    for feature in candidate_features:
        x = pd.to_numeric(book_table[feature], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2 or spearmanr is None:
            rho = np.nan
            p_value = np.nan
            perm_p = np.nan
        else:
            stat = spearmanr(x[mask], y[mask])
            rho = float(stat.statistic)
            p_value = float(stat.pvalue)
            perm_p = permutation_spearman_p(x[mask].to_numpy(float), y[mask].to_numpy(float), rho)
        rows.append(
            {
                "feature": feature,
                "feature_label": PLOT_LABELS.get(feature, feature),
                "n_books": int(mask.sum()),
                "spearman_rho": rho,
                "spearman_p": p_value,
                "permutation_p": perm_p,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["bh_q_spearman"] = benjamini_hochberg(result["spearman_p"].tolist())
        result["abs_spearman_rho"] = result["spearman_rho"].abs()
    return result.sort_values(["abs_spearman_rho", "feature"], ascending=[False, True])


def plot_primary_boxplots(judgments: pd.DataFrame, out_dir: Path) -> None:
    available = [feature for feature in PRIMARY_FEATURES if feature in judgments.columns]
    rows = []
    for feature in available:
        for _, row in judgments.iterrows():
            value = pd.to_numeric(row.get(feature), errors="coerce")
            if pd.notna(value):
                rows.append(
                    {
                        "feature": feature,
                        "feature_label": PLOT_LABELS.get(feature, feature),
                        "guess_group": "MT guessed HT" if row["mt_guess"] == "HT" else "MT guessed MT",
                        "value": float(value),
                    }
                )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return
    grid = sns.catplot(
        data=plot_df,
        x="guess_group",
        y="value",
        col="feature_label",
        col_wrap=4,
        kind="box",
        sharey=False,
        color="#D8D8D8",
        fliersize=0,
        height=3.2,
        aspect=1.0,
    )
    for ax, (_, subset) in zip(grid.axes.flat, plot_df.groupby("feature_label", sort=False)):
        sns.stripplot(
            data=subset,
            x="guess_group",
            y="value",
            hue="guess_group",
            palette={"MT guessed HT": "#E69F00", "MT guessed MT": "#D55E00"},
            size=5.5,
            jitter=0.18,
            alpha=0.8,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=15)
    grid.set_titles("{col_name}")
    grid.fig.suptitle("Primary features by isolated MT origin guess", y=1.03, fontweight="bold")
    save_figure(grid.fig, out_dir, "primary_feature_boxplots")


def plot_effect_summary(tests: pd.DataFrame, out_dir: Path) -> None:
    subset = tests.dropna(subset=["cliffs_delta"]).head(18).copy()
    if subset.empty:
        return
    subset = subset.sort_values("cliffs_delta")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = ["#E69F00" if value > 0 else "#D55E00" for value in subset["cliffs_delta"]]
    ax.barh(subset["feature_label"], subset["cliffs_delta"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Cliff's delta: positive means higher in MT guessed HT")
    ax.set_ylabel("")
    ax.set_title("Largest participant-level feature effects")
    save_figure(fig, out_dir, "participant_feature_effect_summary")


def plot_book_correlations(book_table: pd.DataFrame, correlations: pd.DataFrame, out_dir: Path) -> None:
    selected = [
        feature for feature in PRIMARY_FEATURES if feature in book_table.columns and feature in set(correlations["feature"])
    ]
    selected = selected[:6]
    if not selected:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    for ax, feature in zip(axes, selected):
        sns.scatterplot(
            data=book_table,
            x=feature,
            y="undetected_rate",
            hue="source_lang",
            palette={"French": "#0072B2", "Japanese": "#009E73", "Polish": "#CC79A7"},
            s=55,
            ax=ax,
        )
        for _, row in book_table.iterrows():
            ax.text(row[feature], row["undetected_rate"] + 0.015, short_book_label(row["book_id"]), fontsize=6)
        corr_row = correlations[correlations["feature"] == feature]
        rho = corr_row["spearman_rho"].iloc[0] if not corr_row.empty else np.nan
        ax.set_title(f"{PLOT_LABELS.get(feature, feature)}\nrho={rho:.2f}" if pd.notna(rho) else PLOT_LABELS.get(feature, feature))
        ax.set_xlabel(PLOT_LABELS.get(feature, feature))
        ax.set_ylabel("Undetected rate")
        ax.set_ylim(-0.05, 1.05)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    for ax in axes[len(selected) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.suptitle("Book-level feature values vs MT undetected rate", y=1.02, fontweight="bold")
    save_figure(fig, out_dir, "book_level_feature_scatter")


def plot_paired_book_correlations(
    book_table: pd.DataFrame, correlations: pd.DataFrame, out_dir: Path
) -> None:
    selected = [
        "mt_minus_ht_mtld",
        "abs_mt_minus_ht_mtld",
        "mt_minus_ht_em_dash_rate",
        "abs_mt_minus_ht_em_dash_rate",
        "mt_minus_ht_dash_rate",
        "abs_mt_minus_ht_dash_rate",
        "mt_minus_ht_surface_artifact_index",
        "abs_mt_minus_ht_surface_artifact_index",
    ]
    selected = [
        feature
        for feature in selected
        if feature in book_table.columns and feature in set(correlations["feature"])
    ]
    if not selected:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    for ax, feature in zip(axes, selected):
        sns.scatterplot(
            data=book_table,
            x=feature,
            y="undetected_rate",
            hue="source_lang",
            palette={"French": "#0072B2", "Japanese": "#009E73", "Polish": "#CC79A7"},
            s=55,
            ax=ax,
        )
        for _, row in book_table.iterrows():
            ax.text(row[feature], row["undetected_rate"] + 0.015, short_book_label(row["book_id"]), fontsize=6)
        corr_row = correlations[correlations["feature"] == feature]
        rho = corr_row["spearman_rho"].iloc[0] if not corr_row.empty else np.nan
        title = PLOT_LABELS.get(feature, feature)
        ax.set_title(f"{title}\nrho={rho:.2f}" if pd.notna(rho) else title)
        ax.set_xlabel(title)
        ax.set_ylabel("Undetected rate")
        ax.set_ylim(-0.05, 1.05)
        ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.suptitle("Paired MT-vs-HT feature deltas vs MT undetected rate", y=1.02, fontweight="bold")
    save_figure(fig, out_dir, "paired_book_level_feature_scatter")


def plot_chunk_aligned_book_correlations(
    book_table: pd.DataFrame, correlations: pd.DataFrame, out_dir: Path
) -> None:
    selected = [
        "mean_abs_chunk_delta_em_dash_rate",
        "share_chunks_mt_more_em_dash",
        "mean_abs_chunk_delta_dash_rate",
        "share_chunks_mt_more_dash",
        "mean_abs_chunk_delta_quote_rate",
        "mean_abs_chunk_delta_surface_artifact_index",
        "share_chunks_dash_style_mismatch",
        "mean_abs_chunk_delta_mtld",
    ]
    selected = [
        feature
        for feature in selected
        if feature in book_table.columns and feature in set(correlations["feature"])
    ]
    if not selected:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    for ax, feature in zip(axes, selected):
        sns.scatterplot(
            data=book_table,
            x=feature,
            y="undetected_rate",
            hue="source_lang",
            palette={"French": "#0072B2", "Japanese": "#009E73", "Polish": "#CC79A7"},
            s=55,
            ax=ax,
        )
        for _, row in book_table.iterrows():
            ax.text(row[feature], row["undetected_rate"] + 0.015, short_book_label(row["book_id"]), fontsize=6)
        corr_row = correlations[correlations["feature"] == feature]
        rho = corr_row["spearman_rho"].iloc[0] if not corr_row.empty else np.nan
        title = PLOT_LABELS.get(feature, feature)
        ax.set_title(f"{title}\nrho={rho:.2f}" if pd.notna(rho) else title)
        ax.set_xlabel(title)
        ax.set_ylabel("Undetected rate")
        ax.set_ylim(-0.05, 1.05)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.suptitle("Chunk-aligned MT-vs-HT mismatch features vs MT undetected rate", y=1.02, fontweight="bold")
    save_figure(fig, out_dir, "chunk_aligned_book_level_feature_scatter")


def plot_book_heatmap(book_table: pd.DataFrame, out_dir: Path) -> None:
    selected = [feature for feature in PRIMARY_FEATURES if feature in book_table.columns]
    if not selected:
        return
    heat = book_table[["book_id", "undetected_rate", *selected]].copy()
    heat = heat.sort_values(["undetected_rate", "book_id"], ascending=[False, True])
    values = heat[selected].apply(pd.to_numeric, errors="coerce")
    z = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0)
    z.index = [f"{short_book_label(book)} ({rate:.1f})" for book, rate in zip(heat["book_id"], heat["undetected_rate"])]
    z.columns = [PLOT_LABELS.get(column, column) for column in selected]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    sns.heatmap(z, cmap="vlag", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Primary feature z-scores by book, ordered by undetected rate")
    ax.set_xlabel("")
    ax.set_ylabel("Book (undetected rate)")
    save_figure(fig, out_dir, "book_feature_heatmap")


def plot_paired_book_heatmap(book_table: pd.DataFrame, out_dir: Path) -> None:
    selected = [
        feature
        for feature in [
            "mt_minus_ht_mtld",
            "abs_mt_minus_ht_mtld",
            "mt_minus_ht_ttr",
            "abs_mt_minus_ht_ttr",
            "mt_minus_ht_median_sentence_words",
            "abs_mt_minus_ht_median_sentence_words",
            "mt_minus_ht_em_dash_rate",
            "abs_mt_minus_ht_em_dash_rate",
            "mt_minus_ht_dash_rate",
            "abs_mt_minus_ht_dash_rate",
            "mt_minus_ht_surface_artifact_index",
            "abs_mt_minus_ht_surface_artifact_index",
        ]
        if feature in book_table.columns
    ]
    if not selected:
        return
    heat = book_table[["book_id", "undetected_rate", *selected]].copy()
    heat = heat.sort_values(["undetected_rate", "book_id"], ascending=[False, True])
    values = heat[selected].apply(pd.to_numeric, errors="coerce")
    z = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0)
    z.index = [f"{short_book_label(book)} ({rate:.1f})" for book, rate in zip(heat["book_id"], heat["undetected_rate"])]
    z.columns = [PLOT_LABELS.get(column, column) for column in selected]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.heatmap(z, cmap="vlag", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Paired MT-vs-HT feature z-scores by book, ordered by undetected rate")
    ax.set_xlabel("")
    ax.set_ylabel("Book (undetected rate)")
    save_figure(fig, out_dir, "paired_book_feature_heatmap")


def plot_chunk_aligned_book_heatmap(book_table: pd.DataFrame, out_dir: Path) -> None:
    selected = [
        feature
        for feature in [
            "mean_abs_chunk_delta_em_dash_rate",
            "mean_signed_chunk_delta_em_dash_rate",
            "share_chunks_mt_more_em_dash",
            "mean_abs_chunk_delta_dash_rate",
            "mean_signed_chunk_delta_dash_rate",
            "share_chunks_mt_more_dash",
            "mean_abs_chunk_delta_quote_rate",
            "mean_signed_chunk_delta_quote_rate",
            "share_chunks_mt_more_quote",
            "mean_abs_chunk_delta_surface_artifact_index",
            "share_chunks_dash_style_mismatch",
            "share_chunks_quote_style_mismatch",
            "mean_abs_chunk_delta_mtld",
            "mean_abs_chunk_delta_median_sentence_words",
        ]
        if feature in book_table.columns
    ]
    if not selected:
        return
    heat = book_table[["book_id", "undetected_rate", *selected]].copy()
    heat = heat.sort_values(["undetected_rate", "book_id"], ascending=[False, True])
    values = heat[selected].apply(pd.to_numeric, errors="coerce")
    z = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0)
    z.index = [f"{short_book_label(book)} ({rate:.1f})" for book, rate in zip(heat["book_id"], heat["undetected_rate"])]
    z.columns = [PLOT_LABELS.get(column, column) for column in selected]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.heatmap(z, cmap="vlag", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Chunk-aligned MT-vs-HT mismatch z-scores by book")
    ax.set_xlabel("")
    ax.set_ylabel("Book (undetected rate)")
    save_figure(fig, out_dir, "chunk_aligned_book_feature_heatmap")


def short_book_label(book_id: str) -> str:
    label = re.sub(r"^(french|japanese|polish)_eval_", "", str(book_id))
    words_ = label.split("_")
    if len(words_) <= 3:
        return " ".join(words_)
    return " ".join(words_[:3]) + "..."


def format_number(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    numeric = float(value)
    if math.isclose(numeric, round(numeric)):
        return str(int(round(numeric)))
    return f"{numeric:.{digits}f}"


def write_summary(
    judgments: pd.DataFrame,
    book_table: pd.DataFrame,
    tests: pd.DataFrame,
    correlations: pd.DataFrame,
    out_dir: Path,
) -> None:
    mixed_books = (
        judgments.groupby("book_id")["mt_guess"]
        .nunique()
        .reset_index(name="guess_types")
        .query("guess_types > 1")["book_id"]
        .tolist()
    )
    top_tests = tests.head(10)
    top_corrs = correlations.head(10)
    paired_corrs = correlations[correlations["feature"].isin(PAIRED_FEATURE_COLUMNS)].head(10)
    chunk_corrs = correlations[
        correlations["feature"].isin(CHUNK_ALIGNED_FEATURE_COLUMNS)
    ].head(10)

    lines = [
        "# MT Detection Surface/Lexical Feature Analysis",
        "",
        "## What This Tests",
        "",
        "This analysis compares actual MT excerpts that participants guessed were `HT` "
        "against actual MT excerpts that participants guessed were `MT` in isolated reading.",
        "",
        "The participant-level rows are descriptive because the same MT excerpt can be "
        "judged differently by different readers. Book-level correlations with "
        "`undetected_rate` are therefore the cleaner text-feature view, though n is only 15 books.",
        "",
        "## Data Checks",
        "",
        f"- Isolated MT judgment rows: {judgments.shape[0]}",
        f"- MT guessed HT: {int((judgments['mt_guess'] == 'HT').sum())}",
        f"- MT guessed MT: {int((judgments['mt_guess'] == 'MT').sum())}",
        f"- Canonical books with MT text: {book_table.shape[0]}",
        f"- Canonical books with paired HT eval text: {int(book_table['ht_text_path'].notna().sum())}",
        f"- Books with mixed reader judgments: {len(mixed_books)}",
        "",
        "## Generated Files",
        "",
        "- `mt_judgment_feature_table.csv`: one row per isolated MT judgment.",
        "- `book_level_feature_table.csv`: one row per canonical book with undetected rate.",
        "- `feature_test_summary.csv`: participant-level Mann-Whitney tests and Cliff's delta.",
        "- `book_level_feature_correlations.csv`: book-level Spearman correlations.",
        "- Paired features use `mt_minus_ht_*`, `abs_mt_minus_ht_*`, and `mt_div_ht_*` columns.",
        "- Chunk-aligned features aggregate local MT-vs-HT chunk differences from aligned JSONL chunks.",
        "- `*.png` and `*.pdf`: boxplots, effect summary, book scatterplots, and heatmap.",
        "",
        "## Largest Participant-Level Effects",
        "",
    ]
    if top_tests.empty:
        lines.append("No participant-level feature tests were available.")
    else:
        lines.append("| Feature | Mean delta | Cliff's delta | p | BH q |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in top_tests.iterrows():
            lines.append(
                f"| `{row['feature']}` | "
                f"{format_number(row['mean_delta_focal_minus_baseline'])} | "
                f"{format_number(row['cliffs_delta'])} | "
                f"{format_number(row['p_value'])} | "
                f"{format_number(row['bh_q'])} |"
            )

    lines.extend(["", "## Strongest Book-Level Correlations", ""])
    if top_corrs.empty:
        lines.append("No book-level correlations were available.")
    else:
        lines.append("| Feature | Spearman rho | permutation p | BH q |")
        lines.append("|---|---:|---:|---:|")
        for _, row in top_corrs.iterrows():
            lines.append(
                f"| `{row['feature']}` | "
                f"{format_number(row['spearman_rho'])} | "
                f"{format_number(row['permutation_p'])} | "
                f"{format_number(row['bh_q_spearman'])} |"
            )

    lines.extend(["", "## Strongest Paired MT-vs-HT Book-Level Correlations", ""])
    if paired_corrs.empty:
        lines.append("No paired MT-vs-HT correlations were available.")
    else:
        lines.append("| Feature | Spearman rho | permutation p | BH q |")
        lines.append("|---|---:|---:|---:|")
        for _, row in paired_corrs.iterrows():
            lines.append(
                f"| `{row['feature']}` | "
                f"{format_number(row['spearman_rho'])} | "
                f"{format_number(row['permutation_p'])} | "
                f"{format_number(row['bh_q_spearman'])} |"
            )

    lines.extend(["", "## Strongest Chunk-Aligned MT-vs-HT Correlations", ""])
    if chunk_corrs.empty:
        lines.append("No chunk-aligned MT-vs-HT correlations were available.")
    else:
        lines.append("| Feature | Spearman rho | permutation p | BH q |")
        lines.append("|---|---:|---:|---:|")
        for _, row in chunk_corrs.iterrows():
            lines.append(
                f"| `{row['feature']}` | "
                f"{format_number(row['spearman_rho'])} | "
                f"{format_number(row['permutation_p'])} | "
                f"{format_number(row['bh_q_spearman'])} |"
            )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is hypothesis-generating; do not treat the p-values as confirmatory.",
            "- Text features repeat for participants who read the same book.",
            "- A mixed book means the exact same MT text was guessed HT by one reader and MT by another.",
            "- Paired MT-vs-HT deltas are more interpretable than absolute MT features for comparison-style judgments.",
            "- Chunk-aligned scores are better local surface-style comparisons, but they still aggregate to only 15 book-level points.",
            "- Qualitative subgroup features come from participants' comments, not from the MT text itself.",
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(judgments: pd.DataFrame, book_table: pd.DataFrame) -> None:
    if judgments.shape[0] != 30:
        raise AssertionError(f"Expected 30 isolated MT judgment rows, got {judgments.shape[0]}")
    if book_table.shape[0] != 15:
        raise AssertionError(f"Expected 15 canonical book rows, got {book_table.shape[0]}")
    if book_table["mt_text_path"].isna().any():
        raise AssertionError("Some canonical books did not resolve to MT text")
    if book_table["ht_text_path"].isna().any():
        raise AssertionError("Some canonical books did not resolve to paired HT eval text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, default=DEFAULT_PART1)
    parser.add_argument("--qual-counts", type=Path, default=DEFAULT_QUAL_COUNTS)
    parser.add_argument("--ht-dir", type=Path, default=DEFAULT_HT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_plot_style()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    part1 = read_csv(args.part1)
    judgments = build_mt_judgment_rows(part1)
    book_features = build_book_feature_table(part1, REPO_ROOT, args.ht_dir)
    judgments = judgments.merge(book_features, on="book_id", how="left")
    judgments = add_qualitative_features(judgments, args.qual_counts)
    book_table = build_book_level_table(judgments, book_features)

    validate_outputs(judgments, book_table)

    tests = build_feature_tests(judgments)
    correlations = build_book_correlations(book_table)

    judgments.to_csv(out_dir / "mt_judgment_feature_table.csv", index=False, encoding="utf-8")
    book_table.to_csv(out_dir / "book_level_feature_table.csv", index=False, encoding="utf-8")
    tests.to_csv(out_dir / "feature_test_summary.csv", index=False, encoding="utf-8")
    correlations.to_csv(out_dir / "book_level_feature_correlations.csv", index=False, encoding="utf-8")

    plot_primary_boxplots(judgments, out_dir)
    plot_effect_summary(tests, out_dir)
    plot_book_correlations(book_table, correlations, out_dir)
    plot_paired_book_correlations(book_table, correlations, out_dir)
    plot_chunk_aligned_book_correlations(book_table, correlations, out_dir)
    plot_book_heatmap(book_table, out_dir)
    plot_paired_book_heatmap(book_table, out_dir)
    plot_chunk_aligned_book_heatmap(book_table, out_dir)
    write_summary(judgments, book_table, tests, correlations, out_dir)

    print(f"Wrote MT detection surface-feature outputs to {out_dir}")


if __name__ == "__main__":
    main()
