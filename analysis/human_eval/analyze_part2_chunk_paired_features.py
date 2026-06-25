#!/usr/bin/env python3
"""Chunk-level paired HT-vs-MT feature analysis for Part 2 human-eval choices.

Run from the project root:
    .venv/bin/python analysis/human_eval/analyze_part2_chunk_paired_features.py

Outputs are written to:
    analysis/human_eval/part2_chunk_paired_features/

This is an exploratory, descriptive screen. Participant responses are nested in
books/chunks, so simple Mann-Whitney and Spearman p-values should not be read as
confirmatory mixed-effects inference.
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
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .label_collapse import collapse_label_code
except ImportError:  # pragma: no cover - direct script execution.
    from label_collapse import collapse_label_code

from statistical.utils.text_processing import (  # noqa: E402
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


DEFAULT_PART2 = REPO_ROOT / "human_eval" / "data" / "part2-study-data-full.csv"
DEFAULT_Q5 = REPO_ROOT / "analysis" / "human_eval" / "q5_chunks.csv"
DEFAULT_SPANS = REPO_ROOT / "human_eval" / "data" / "part2-span-study-data-full.csv"
DEFAULT_HT_DIR = REPO_ROOT / "books" / "HT" / "eval"
DEFAULT_MT_DIR = REPO_ROOT / "books" / "MT_chunks"
DEFAULT_OUT = REPO_ROOT / "analysis" / "human_eval" / "part2_chunk_paired_features"

SOURCE_LANG_CODES = {
    "french": "fr",
    "japanese": "ja",
    "polish": "pl",
}

CATEGORY_PREFIXES = ("A", "B", "C", "D")
CATEGORY_COLUMNS = {
    "A": "A. Language-level features",
    "B": "B. Narrative-level features",
    "C": "C. Reader experience",
    "D": "D. Meta-translation",
}
ABCD_SUBGROUPS = {
    "SG_A_grammar": {"codes": {"A1", "A1b"}, "label": "A1: Grammar/tense"},
    "SG_A_word_choice": {"codes": {"A2", "A3", "A4", "A5", "A6"}, "label": "A: Word choice"},
    "SG_A_cultural": {"codes": {"A7"}, "label": "A: Cultural"},
    "SG_A_sentence": {"codes": {"A8"}, "label": "A: Sentence structure"},
    "SG_A_consistency": {"codes": {"A9"}, "label": "A9: Consistency"},
    "SG_A_surface": {"codes": {"A10", "A11", "A12"}, "label": "A: Surface"},
    "SG_B_dialogue": {"codes": {"B1"}, "label": "B: Dialogue"},
    "SG_B_character": {"codes": {"B2", "B3"}, "label": "B2: Character voice/portrayal"},
    "SG_B_imagery": {"codes": {"B4", "B5"}, "label": "B: Imagery"},
    "SG_B_emotion": {"codes": {"B6"}, "label": "B: Emotion"},
    "SG_B_narrative_flow": {"codes": {"B6", "B7", "B8"}, "label": "B: Flow"},
    "SG_C_comprehension": {"codes": {"C1"}, "label": "C: Comprehension"},
    "SG_C_smoothness": {"codes": {"C2", "A9"}, "label": "C2: Smoothness/cadence"},
    "SG_C_engagement": {"codes": {"C3", "C4"}, "label": "C3: Engagement/hook"},
    "SG_C_humanness": {"codes": {"C4"}, "label": "C: Humanness"},
    "SG_C_enjoyment": {"codes": {"C5"}, "label": "C: Enjoyment"},
    "SG_D_translation_relation": {"codes": {"D1", "D2", "D3"}, "label": "D: Translation"},
    "SG_D_ai_mt_verdict": {"codes": {"D4a", "D4b"}, "label": "D4: AI/MT verdict/tell"},
}
CODE_TO_SUBGROUP = {
    code: subgroup for subgroup, spec in ABCD_SUBGROUPS.items() for code in spec["codes"]
}

TEXT_FEATURES = [
    "token_count",
    "word_count",
    "type_count",
    "character_count",
    "sentence_count",
    "paragraph_count",
    "mean_sentence_words",
    "median_sentence_words",
    "std_sentence_words",
    "max_sentence_words",
    "long_sentence_share",
    "ttr",
    "mtld",
    "hapax_rate",
    "repeated_unigram_rate",
    "repeated_bigram_rate",
    "repeated_trigram_rate",
    "comma_rate",
    "semicolon_rate",
    "colon_rate",
    "em_dash_rate",
    "en_dash_rate",
    "hyphen_rate",
    "quote_rate",
    "parenthesis_rate",
    "ellipsis_rate",
    "slash_rate",
    "punctuation_density",
    "quote_style_count",
    "dash_style_count",
    "mixed_quote_style",
    "mixed_dash_style",
    "slash_n_artifact_count",
    "repeated_blank_line_count",
    "single_newline_rate",
    "surface_artifact_index",
    "ly_adverb_rate",
    "average_word_length",
    "long_word_rate",
    "contraction_rate",
    "dialogue_or_quote_density",
    "first_person_pronoun_rate",
    "modality_hedging_rate",
    "formal_marker_rate",
    "formality_proxy",
]
PAIRED_FEATURE_COLUMNS = [
    f"{kind}_{feature}"
    for feature in TEXT_FEATURES
    for kind in ("mt_minus_ht", "abs_mt_minus_ht", "mt_div_ht")
]
PRIMARY_PLOT_FEATURES = [
    "mt_minus_ht_token_count",
    "mt_minus_ht_median_sentence_words",
    "mt_minus_ht_std_sentence_words",
    "mt_minus_ht_mtld",
    "mt_minus_ht_em_dash_rate",
    "mt_minus_ht_quote_rate",
    "mt_minus_ht_surface_artifact_index",
    "mt_minus_ht_ly_adverb_rate",
    "mt_minus_ht_long_word_rate",
]
BOOK_SCATTER_FEATURES = [
    "mt_minus_ht_token_count",
    "mt_minus_ht_median_sentence_words",
    "mt_minus_ht_std_sentence_words",
    "mt_minus_ht_mtld",
    "mt_minus_ht_em_dash_rate",
    "mt_minus_ht_quote_rate",
    "mt_minus_ht_surface_artifact_index",
    "mt_minus_ht_long_word_rate",
]
FEATURE_LABELS = {
    "mt_minus_ht_token_count": "MT minus HT tokens",
    "mt_minus_ht_median_sentence_words": "MT minus HT median sentence words",
    "mt_minus_ht_std_sentence_words": "MT minus HT sentence-length SD",
    "mt_minus_ht_mtld": "MT minus HT MTLD",
    "mt_minus_ht_em_dash_rate": "MT minus HT em dash / 1k words",
    "mt_minus_ht_quote_rate": "MT minus HT quote chars / 1k words",
    "mt_minus_ht_surface_artifact_index": "MT minus HT surface artifact index",
    "mt_minus_ht_ly_adverb_rate": "MT minus HT -ly adverbs / 1k words",
    "mt_minus_ht_long_word_rate": "MT minus HT long-word share",
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
MODALITY_HEDGES = {
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "seem",
    "seemed",
    "seems",
    "might",
    "may",
    "could",
    "would",
    "apparently",
    "almost",
    "rather",
    "somewhat",
    "sort",
    "kind",
}
FORMAL_MARKERS = {
    "therefore",
    "thus",
    "hence",
    "moreover",
    "nevertheless",
    "nonetheless",
    "consequently",
    "accordingly",
    "whereas",
    "whilst",
    "upon",
    "regarding",
    "concerning",
    "shall",
    "indeed",
}

HT_BLUE = "#0072B2"
MT_ORANGE = "#D55E00"
HOOKED_PURPLE = "#7B3294"
GRAY = "#777777"


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


def participant_base(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    match = re.search(r"p0*(\d+)", text)
    if not match:
        return text
    return f"p{int(match.group(1)):03d}"


def infer_source_lang(book_id: object) -> str:
    text = canonical_book_id(book_id).lower()
    if text.startswith("french_"):
        return "French"
    if text.startswith("japanese_"):
        return "Japanese"
    if text.startswith("polish_"):
        return "Polish"
    return ""


def book_id_to_chunk_stem(book_id: str) -> str:
    canonical = canonical_book_id(book_id)
    for prefix, lang_code in SOURCE_LANG_CODES.items():
        marker = f"{prefix}_eval_"
        if canonical.startswith(marker):
            title = canonical.removeprefix(marker)
            return f"{title}_{lang_code}_en"
    raise ValueError(f"Cannot infer chunk filename stem from book_id: {book_id}")


def load_jsonl_chunks(path: Path) -> dict[int, str]:
    chunks: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            chunk_id = int(obj.get("chunk_id", line_number))
            chunks[chunk_id] = str(obj.get("text", ""))
    return chunks


def jsonl_chunk_id_for_part2(part2_chunk_id: object, available_ids: Iterable[int]) -> tuple[int, str]:
    """Map Part 2 chunk IDs to JSONL IDs.

    The current Part 2 CSVs are zero-based and the JSONL files are one-based. If a future
    JSONL sidecar is zero-based, detect that from the available ID range and use direct IDs.
    """
    part2_id = int(str(part2_chunk_id).strip())
    ids = set(available_ids)
    if ids and min(ids) == 1 and 0 not in ids and part2_id + 1 in ids:
        return part2_id + 1, "part2_zero_based_jsonl_one_based"
    if part2_id in ids:
        return part2_id, "same_id"
    if part2_id + 1 in ids:
        return part2_id + 1, "part2_zero_based_jsonl_one_based"
    raise KeyError(f"Part 2 chunk {part2_id} is not present in JSONL IDs {sorted(ids)[:5]}...")


def safe_rate(count: float, denominator: float) -> float:
    return count / denominator if denominator else 0.0


def per_1k_rate(count: float, token_count: float) -> float:
    return safe_rate(count, token_count / 1000.0)


def sentence_lengths(text: str) -> list[int]:
    return [len(words(sentence)) for sentence in sentences(text) if words(sentence)]


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


def count_contractions(text: str) -> int:
    return len(
        re.findall(
            r"\b\w+(?:n't|['’](?:m|re|ve|ll|d|s|t))\b",
            text,
            flags=re.IGNORECASE | re.UNICODE,
        )
    )


def extract_text_features(text: str) -> dict[str, float]:
    norm_tokens = normalized_words(text)
    raw_words = words(text)
    token_count = len(norm_tokens)
    type_counts = Counter(norm_tokens)
    sent_lengths = sentence_lengths(text)
    para_count = len(paragraphs(text))
    quote_count = sum(text.count(char) for char in "\"“”‘’«»")
    dialogue_dash_lines = len(re.findall(r"(?m)^\s*[—–-]\s+", text))
    quote_style_count = count_quote_styles(text)
    dash_style_count = count_dash_styles(text)
    slash_n_artifacts = len(re.findall(r"(?:/n+/?|\\n+)", text, flags=re.IGNORECASE))
    single_newlines = count_single_newlines(text)
    repeated_blank_lines = len(re.findall(r"\n\s*\n+", text))
    surface_artifact_index = (
        slash_n_artifacts
        + int(quote_style_count > 1)
        + int(dash_style_count > 1)
        + repeated_blank_lines
        + per_1k_rate(single_newlines, token_count)
    )
    ly_count = sum(1 for token in norm_tokens if len(token) > 3 and token.endswith("ly"))
    long_word_count = sum(1 for token in norm_tokens if len(token) >= 10)
    first_person_count = sum(1 for token in norm_tokens if token in FIRST_PERSON)
    hedge_count = sum(1 for token in norm_tokens if token in MODALITY_HEDGES)
    formal_marker_count = sum(1 for token in norm_tokens if token in FORMAL_MARKERS)
    contraction_count = count_contractions(text)
    average_word_length = float(np.mean([len(token) for token in norm_tokens])) if norm_tokens else 0.0
    long_word_rate = safe_rate(long_word_count, token_count)
    formal_marker_rate = per_1k_rate(formal_marker_count, token_count)
    contraction_rate = per_1k_rate(contraction_count, token_count)

    return {
        "token_count": float(token_count),
        "word_count": float(len(raw_words)),
        "type_count": float(len(type_counts)),
        "character_count": float(len(text)),
        "sentence_count": float(len(sent_lengths)),
        "paragraph_count": float(para_count),
        "mean_sentence_words": float(np.mean(sent_lengths)) if sent_lengths else 0.0,
        "median_sentence_words": float(np.median(sent_lengths)) if sent_lengths else 0.0,
        "std_sentence_words": float(np.std(sent_lengths)) if sent_lengths else 0.0,
        "max_sentence_words": float(max(sent_lengths)) if sent_lengths else 0.0,
        "long_sentence_share": safe_rate(sum(length >= 30 for length in sent_lengths), len(sent_lengths)),
        "ttr": type_token_ratio(norm_tokens),
        "mtld": mtld(norm_tokens),
        "hapax_rate": safe_rate(sum(1 for count in type_counts.values() if count == 1), token_count),
        "repeated_unigram_rate": repetition_rate(norm_tokens, 1),
        "repeated_bigram_rate": repetition_rate(norm_tokens, 2),
        "repeated_trigram_rate": repetition_rate(norm_tokens, 3),
        "comma_rate": per_1k_rate(text.count(","), token_count),
        "semicolon_rate": per_1k_rate(text.count(";"), token_count),
        "colon_rate": per_1k_rate(text.count(":"), token_count),
        "em_dash_rate": per_1k_rate(text.count("—"), token_count),
        "en_dash_rate": per_1k_rate(text.count("–"), token_count),
        "hyphen_rate": per_1k_rate(text.count("-"), token_count),
        "quote_rate": per_1k_rate(quote_count, token_count),
        "parenthesis_rate": per_1k_rate(text.count("(") + text.count(")"), token_count),
        "ellipsis_rate": per_1k_rate(count_ellipsis(text), token_count),
        "slash_rate": per_1k_rate(text.count("/") + text.count("\\"), token_count),
        "punctuation_density": safe_rate(count_punctuation(text), len(text)),
        "quote_style_count": float(quote_style_count),
        "dash_style_count": float(dash_style_count),
        "mixed_quote_style": float(quote_style_count > 1),
        "mixed_dash_style": float(dash_style_count > 1),
        "slash_n_artifact_count": float(slash_n_artifacts),
        "repeated_blank_line_count": float(repeated_blank_lines),
        "single_newline_rate": per_1k_rate(single_newlines, token_count),
        "surface_artifact_index": float(surface_artifact_index),
        "ly_adverb_rate": per_1k_rate(ly_count, token_count),
        "average_word_length": average_word_length,
        "long_word_rate": long_word_rate,
        "contraction_rate": contraction_rate,
        "dialogue_or_quote_density": per_1k_rate(quote_count + dialogue_dash_lines, token_count),
        "first_person_pronoun_rate": per_1k_rate(first_person_count, token_count),
        "modality_hedging_rate": per_1k_rate(hedge_count, token_count),
        "formal_marker_rate": formal_marker_rate,
        "formality_proxy": average_word_length + long_word_rate + formal_marker_rate - contraction_rate,
    }


def paired_feature_deltas(
    mt_features: dict[str, float], ht_features: dict[str, float]
) -> dict[str, float]:
    paired: dict[str, float] = {}
    for feature in TEXT_FEATURES:
        mt_value = float(mt_features.get(feature, np.nan))
        ht_value = float(ht_features.get(feature, np.nan))
        paired[f"mt_minus_ht_{feature}"] = mt_value - ht_value
        paired[f"abs_mt_minus_ht_{feature}"] = abs(mt_value - ht_value)
        paired[f"mt_div_ht_{feature}"] = mt_value / ht_value if ht_value else np.nan
    return paired


def build_chunk_pair_feature_table(
    part2: pd.DataFrame, ht_dir: Path = DEFAULT_HT_DIR, mt_dir: Path = DEFAULT_MT_DIR
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid = part2[part2["preferred_translation"].isin(["HT", "MT"])].copy()
    valid["canonical_book_id"] = valid["book_id"].map(canonical_book_id)

    chunk_cache: dict[str, tuple[dict[int, str], dict[int, str]]] = {}
    for book_id, chunk_id in sorted(
        valid[["canonical_book_id", "chunk_id"]].drop_duplicates().itertuples(index=False)
    ):
        stem = book_id_to_chunk_stem(book_id)
        if book_id not in chunk_cache:
            ht_chunks = load_jsonl_chunks(ht_dir / f"{stem}.jsonl")
            mt_chunks = load_jsonl_chunks(mt_dir / f"{stem}.jsonl")
            chunk_cache[book_id] = ht_chunks, mt_chunks
        ht_chunks, mt_chunks = chunk_cache[book_id]
        ht_jsonl_id, ht_note = jsonl_chunk_id_for_part2(chunk_id, ht_chunks.keys())
        mt_jsonl_id, mt_note = jsonl_chunk_id_for_part2(chunk_id, mt_chunks.keys())
        if ht_jsonl_id != mt_jsonl_id:
            raise ValueError(f"HT/MT JSONL chunk ID mismatch for {book_id} chunk {chunk_id}")
        ht_text = ht_chunks[ht_jsonl_id]
        mt_text = mt_chunks[mt_jsonl_id]
        ht_features = extract_text_features(ht_text)
        mt_features = extract_text_features(mt_text)
        record: dict[str, object] = {
            "book_id": book_id,
            "source_lang": infer_source_lang(book_id),
            "chunk_id": int(chunk_id),
            "jsonl_chunk_id": ht_jsonl_id,
            "chunk_id_indexing_note": ht_note if ht_note == mt_note else f"HT:{ht_note};MT:{mt_note}",
            "chunk_stem": stem,
            "ht_text_preview": ht_text[:160].replace("\n", "\\n"),
            "mt_text_preview": mt_text[:160].replace("\n", "\\n"),
        }
        record.update({f"ht_{key}": value for key, value in ht_features.items()})
        record.update({f"mt_{key}": value for key, value in mt_features.items()})
        record.update(paired_feature_deltas(mt_features, ht_features))
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["book_id", "chunk_id"]).reset_index(drop=True)


def split_labels(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan"}:
        return []
    return [
        re.sub(r"\s+", " ", raw).strip()
        for raw in text.split(",")
        if re.sub(r"\s+", " ", raw).strip().lower() not in {"", "none", "nan"}
    ]


def label_code(label: str) -> str:
    match = re.match(r"^([A-Z]\d+[a-z]?)\.", label)
    return match.group(1) if match else ""


def collapsed_category_prefix(label: str) -> str:
    return collapse_label_code(label)[:1]


def q5_count_columns(prefix: str) -> dict[str, str]:
    return {code: f"{prefix}\n{column}" for code, column in CATEGORY_COLUMNS.items()}


def count_q5_labels(row: pd.Series, columns_by_prefix: dict[str, str]) -> dict[str, int]:
    counts = {prefix: 0 for prefix in CATEGORY_PREFIXES}
    subgroup_counts = {subgroup: 0 for subgroup in ABCD_SUBGROUPS}
    for column in columns_by_prefix.values():
        if column not in row.index:
            continue
        labels = split_labels(row.get(column, ""))
        for label in labels:
            prefix = collapsed_category_prefix(label)
            if prefix in counts:
                counts[prefix] += 1
            subgroup = CODE_TO_SUBGROUP.get(collapse_label_code(label))
            if subgroup:
                subgroup_counts[subgroup] += 1
    output = {f"{prefix}_count": count for prefix, count in counts.items()}
    output["total_count"] = sum(counts.values())
    output.update({f"{subgroup}_count": count for subgroup, count in subgroup_counts.items()})
    return output


def build_q5_counts(q5: pd.DataFrame) -> pd.DataFrame:
    pos_columns = q5_count_columns("POS")
    neg_columns = q5_count_columns("NEG")
    rows: list[dict[str, object]] = []
    for _, row in q5.iterrows():
        preferred = str(row.get("chunk_preferred", "")).strip().upper()
        if preferred not in {"HT", "MT"}:
            continue
        rejected = "HT" if preferred == "MT" else "MT"
        base = {
            "participant_base": participant_base(row.get("user", "")),
            "assignment_id": row.get("assignment_id", ""),
            "book_id": canonical_book_id(row.get("book_id", "")),
            "chunk_id": int(row.get("chunk_id", 0)),
            "q5_user": row.get("user", ""),
            "chunk_preferred": preferred,
            "q5_comment": row.get("comment", ""),
        }
        pos = count_q5_labels(row, pos_columns)
        neg = count_q5_labels(row, neg_columns)
        record = dict(base)
        for key, value in pos.items():
            record[f"pos_preferred_{key}"] = value
            record[f"{preferred.lower()}_pos_{key}"] = value
            record.setdefault(f"{rejected.lower()}_pos_{key}", 0)
        for key, value in neg.items():
            record[f"neg_rejected_{key}"] = value
            record[f"{rejected.lower()}_neg_{key}"] = value
            record.setdefault(f"{preferred.lower()}_neg_{key}", 0)
        rows.append(record)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    count_columns = [
        column
        for column in df.columns
        if column.endswith("_count") and column not in {"chunk_id"}
    ]
    for column in count_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    return df


def aggregate_spans(spans: pd.DataFrame) -> pd.DataFrame:
    if spans.empty:
        return pd.DataFrame()
    span_df = spans.copy()
    span_df["book_id"] = span_df["book_id"].map(canonical_book_id)
    span_df["chunk_id"] = pd.to_numeric(span_df["chunk_id"], errors="coerce").astype("Int64")
    span_df["participant_base"] = span_df["user_id"].map(participant_base)
    span_df["version"] = span_df["version"].str.upper()
    span_df["label"] = span_df["label"].str.lower()
    start = pd.to_numeric(span_df["start"], errors="coerce")
    end = pd.to_numeric(span_df["end"], errors="coerce")
    span_df["span_chars"] = (end - start).where((end - start) > 0, span_df["text"].str.len())

    rows: list[dict[str, object]] = []
    group_cols = ["user_id", "participant_base", "book_id", "chunk_id"]
    for key, group in span_df.groupby(group_cols, dropna=False):
        record: dict[str, object] = {
            "user_id": key[0],
            "participant_base": key[1],
            "book_id": key[2],
            "chunk_id": int(key[3]) if not pd.isna(key[3]) else np.nan,
        }
        for version in ["HT", "MT"]:
            version_group = group[group["version"] == version]
            for label in ["good", "poor"]:
                subset = version_group[version_group["label"] == label]
                record[f"count_{label}_spans_{version.lower()}"] = int(subset.shape[0])
                record[f"total_{label}_span_chars_{version.lower()}"] = float(
                    subset["span_chars"].sum()
                )
            record[f"count_span_balance_{version.lower()}"] = (
                record[f"count_good_spans_{version.lower()}"]
                - record[f"count_poor_spans_{version.lower()}"]
            )
            record[f"char_span_balance_{version.lower()}"] = (
                record[f"total_good_span_chars_{version.lower()}"]
                - record[f"total_poor_span_chars_{version.lower()}"]
            )
        record["mt_minus_ht_count_span_balance"] = (
            record["count_span_balance_mt"] - record["count_span_balance_ht"]
        )
        record["mt_minus_ht_char_span_balance"] = (
            record["char_span_balance_mt"] - record["char_span_balance_ht"]
        )
        rows.append(record)
    return pd.DataFrame(rows)


def build_response_feature_table(
    part2: pd.DataFrame,
    chunk_features: pd.DataFrame,
    q5_counts: pd.DataFrame,
    span_features: pd.DataFrame,
) -> pd.DataFrame:
    responses = part2[part2["preferred_translation"].isin(["HT", "MT"])].copy()
    responses["book_id_original"] = responses["book_id"]
    responses["book_id"] = responses["book_id"].map(canonical_book_id)
    responses["chunk_id"] = pd.to_numeric(responses["chunk_id"], errors="coerce").astype(int)
    responses["participant_base"] = responses["user_id"].map(participant_base)
    responses["preference_group"] = responses["preferred_translation"].map(
        {"MT": "MT preferred", "HT": "HT preferred"}
    )
    responses["preferred_mt"] = (responses["preferred_translation"] == "MT").astype(int)
    responses["strong_mt_preference"] = (
        (responses["preferred_translation"] == "MT")
        & (responses["difficulty"] == "significantly_better")
    ).astype(int)
    responses["strong_preference"] = (responses["difficulty"] == "significantly_better").astype(int)

    merged = responses.merge(
        chunk_features,
        on=["book_id", "source_lang", "chunk_id"],
        how="left",
        validate="many_to_one",
    )
    if not q5_counts.empty:
        merged = merged.merge(
            q5_counts,
            on=["participant_base", "book_id", "chunk_id"],
            how="left",
            validate="many_to_one",
        )
    if not span_features.empty:
        merged = merged.merge(
            span_features.drop(columns=["participant_base"], errors="ignore"),
            on=["user_id", "book_id", "chunk_id"],
            how="left",
            validate="one_to_one",
        )

    fill_zero_patterns = (
        "_count",
        "count_good_spans_",
        "count_poor_spans_",
        "total_good_span_chars_",
        "total_poor_span_chars_",
        "_span_balance",
    )
    for column in merged.columns:
        if any(pattern in column for pattern in fill_zero_patterns):
            if column not in {"chunk_id", "jsonl_chunk_id"}:
                merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)
    return merged.sort_values(["book_id", "chunk_id", "user_id"]).reset_index(drop=True)


def cliffs_delta(focal: Iterable[float], baseline: Iterable[float]) -> float:
    x = np.asarray(list(focal), dtype=float)
    y = np.asarray(list(baseline), dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    greater = 0
    lesser = 0
    for value in x:
        greater += int(np.sum(value > y))
        lesser += int(np.sum(value < y))
    return (greater - lesser) / (len(x) * len(y))


def bh_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray([np.nan if value is None else value for value in p_values], dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    if len(valid) == 0:
        return q.tolist()
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q.tolist()


def build_feature_test_summary(response_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mt_group = response_features[response_features["preferred_translation"] == "MT"]
    ht_group = response_features[response_features["preferred_translation"] == "HT"]
    for feature in PAIRED_FEATURE_COLUMNS:
        if feature not in response_features.columns:
            continue
        focal = pd.to_numeric(mt_group[feature], errors="coerce").dropna()
        baseline = pd.to_numeric(ht_group[feature], errors="coerce").dropna()
        p_value = np.nan
        u_stat = np.nan
        if mannwhitneyu is not None and len(focal) > 0 and len(baseline) > 0:
            result = mannwhitneyu(focal, baseline, alternative="two-sided")
            u_stat = float(result.statistic)
            p_value = float(result.pvalue)
        rows.append(
            {
                "feature": feature,
                "focal_group": "MT preferred",
                "baseline_group": "HT preferred",
                "n_focal": int(len(focal)),
                "n_baseline": int(len(baseline)),
                "mean_focal": float(focal.mean()) if len(focal) else np.nan,
                "mean_baseline": float(baseline.mean()) if len(baseline) else np.nan,
                "median_focal": float(focal.median()) if len(focal) else np.nan,
                "median_baseline": float(baseline.median()) if len(baseline) else np.nan,
                "mean_delta_focal_minus_baseline": (
                    float(focal.mean() - baseline.mean()) if len(focal) and len(baseline) else np.nan
                ),
                "median_delta_focal_minus_baseline": (
                    float(focal.median() - baseline.median())
                    if len(focal) and len(baseline)
                    else np.nan
                ),
                "cliffs_delta": cliffs_delta(focal, baseline),
                "mannwhitney_u": u_stat,
                "p_value": p_value,
            }
        )
    summary = pd.DataFrame(rows)
    summary["q_value"] = bh_adjust(summary["p_value"])
    return summary.sort_values("q_value", na_position="last").reset_index(drop=True)


def build_book_level_summary(response_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    span_columns = [
        "count_span_balance_ht",
        "count_span_balance_mt",
        "char_span_balance_ht",
        "char_span_balance_mt",
        "mt_minus_ht_count_span_balance",
        "mt_minus_ht_char_span_balance",
    ]
    mean_columns = [col for col in PAIRED_FEATURE_COLUMNS + span_columns if col in response_features]
    for book_id, group in response_features.groupby("book_id"):
        record: dict[str, object] = {
            "book_id": book_id,
            "source_lang": infer_source_lang(book_id),
            "n_responses": int(group.shape[0]),
            "n_chunks": int(group["chunk_id"].nunique()),
            "mt_preference_rate": float(group["preferred_mt"].mean()),
            "ht_preference_rate": float(1.0 - group["preferred_mt"].mean()),
            "significant_mt_preference_rate": float(group["strong_mt_preference"].mean()),
            "strong_preference_rate": float(group["strong_preference"].mean()),
        }
        strong = group[group["strong_preference"] == 1]
        record["mt_share_among_strong_preferences"] = (
            float(strong["preferred_mt"].mean()) if not strong.empty else np.nan
        )
        for column in mean_columns:
            record[f"mean_{column}"] = pd.to_numeric(group[column], errors="coerce").mean()
        rows.append(record)
    return pd.DataFrame(rows).sort_values("mt_preference_rate", ascending=False).reset_index(drop=True)


def permutation_spearman_pvalue(
    x: np.ndarray, y: np.ndarray, observed: float, permutations: int = 5000
) -> float:
    if len(x) < 4 or np.isnan(observed):
        return np.nan
    rng = np.random.default_rng(20260522)
    more_extreme = 0
    for _ in range(permutations):
        shuffled = rng.permutation(y)
        corr = spearmanr(x, shuffled).statistic if spearmanr is not None else np.nan
        if not np.isnan(corr) and abs(corr) >= abs(observed):
            more_extreme += 1
    return (more_extreme + 1) / (permutations + 1)


def build_book_level_correlations(book_summary: pd.DataFrame, permutations: int = 5000) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_columns = [
        col
        for col in book_summary.columns
        if col.startswith("mean_mt_") or col.startswith("mean_abs_mt_")
    ]
    for column in feature_columns:
        subset = book_summary[["mt_preference_rate", column]].replace([np.inf, -np.inf], np.nan)
        subset = subset.dropna()
        if subset.shape[0] < 4 or spearmanr is None:
            rho = np.nan
            p_value = np.nan
            perm_p = np.nan
        else:
            x = subset[column].to_numpy(dtype=float)
            y = subset["mt_preference_rate"].to_numpy(dtype=float)
            if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                rho = np.nan
                p_value = np.nan
                perm_p = np.nan
            else:
                result = spearmanr(x, y)
                rho = float(result.statistic)
                p_value = float(result.pvalue)
                perm_p = permutation_spearman_pvalue(x, y, rho, permutations=permutations)
        rows.append(
            {
                "feature": column.removeprefix("mean_"),
                "n_books": int(subset.shape[0]),
                "spearman_rho": rho,
                "p_value": p_value,
                "permutation_p_value": perm_p,
            }
        )
    df = pd.DataFrame(rows)
    df["q_value"] = bh_adjust(df["p_value"])
    df["permutation_q_value"] = bh_adjust(df["permutation_p_value"])
    return df.sort_values("permutation_q_value", na_position="last").reset_index(drop=True)


def build_hooked_case_study(response_features: pd.DataFrame) -> pd.DataFrame:
    hooked = response_features[
        response_features["book_id"] == "japanese_eval_hooked_a_novel_of_obsession"
    ].copy()
    useful_columns = [
        "chunk_id",
        "jsonl_chunk_id",
        "user_id",
        "participant_id",
        "preferred_translation",
        "difficulty",
        "justification",
        "q5_comment",
        "pos_preferred_A_count",
        "pos_preferred_B_count",
        "pos_preferred_C_count",
        "pos_preferred_D_count",
        "neg_rejected_A_count",
        "neg_rejected_B_count",
        "neg_rejected_C_count",
        "neg_rejected_D_count",
        "count_good_spans_ht",
        "count_poor_spans_ht",
        "count_good_spans_mt",
        "count_poor_spans_mt",
    ]
    useful_columns.extend([column for column in PRIMARY_PLOT_FEATURES if column in hooked.columns])
    useful_columns = [column for column in useful_columns if column in hooked.columns]
    return hooked[useful_columns].sort_values(["chunk_id", "user_id"]).reset_index(drop=True)


def plot_participant_boxplots(response_features: pd.DataFrame, out_dir: Path) -> None:
    plot_df = response_features.copy()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, feature in zip(axes.flat, PRIMARY_PLOT_FEATURES):
        sns.boxplot(
            data=plot_df,
            x="preference_group",
            hue="preference_group",
            y=feature,
            order=["HT preferred", "MT preferred"],
            hue_order=["HT preferred", "MT preferred"],
            palette={"HT preferred": HT_BLUE, "MT preferred": MT_ORANGE},
            showfliers=False,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            x="preference_group",
            y=feature,
            order=["HT preferred", "MT preferred"],
            color="black",
            alpha=0.35,
            size=2.4,
            jitter=0.25,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(FEATURE_LABELS.get(feature, feature))
        ax.tick_params(axis="x", rotation=10)
    save_figure(fig, out_dir, "participant_feature_boxplots")


def plot_effect_summary(feature_summary: pd.DataFrame, out_dir: Path) -> None:
    plot_df = feature_summary.dropna(subset=["cliffs_delta"]).copy()
    plot_df = plot_df[plot_df["feature"].str.startswith("mt_minus_ht_")]
    plot_df = plot_df.reindex(plot_df["cliffs_delta"].abs().sort_values(ascending=False).index).head(25)
    plot_df = plot_df.sort_values("cliffs_delta")
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [MT_ORANGE if value > 0 else HT_BLUE for value in plot_df["cliffs_delta"]]
    ax.barh(plot_df["feature"], plot_df["cliffs_delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cliff's delta (positive = higher in MT-preferred responses)")
    ax.set_ylabel("")
    save_figure(fig, out_dir, "effect_size_summary_cliffs_delta")


def short_book_label(book_id: str) -> str:
    return (
        canonical_book_id(book_id)
        .replace("french_eval_", "FR: ")
        .replace("japanese_eval_", "JA: ")
        .replace("polish_eval_", "PL: ")
        .replace("_", " ")
    )


def plot_book_scatter(book_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, feature in zip(axes.flat, BOOK_SCATTER_FEATURES):
        column = f"mean_{feature}"
        if column not in book_summary.columns:
            ax.axis("off")
            continue
        for _, row in book_summary.iterrows():
            hooked = row["book_id"] == "japanese_eval_hooked_a_novel_of_obsession"
            ax.scatter(
                row[column],
                row["mt_preference_rate"],
                color=HOOKED_PURPLE if hooked else GRAY,
                marker="*" if hooked else "o",
                s=130 if hooked else 45,
                zorder=3 if hooked else 2,
            )
            ax.text(
                row[column],
                row["mt_preference_rate"] + 0.012,
                short_book_label(row["book_id"]).split(": ", 1)[-1][:18],
                fontsize=6.5,
                ha="center",
            )
        ax.set_xlabel(FEATURE_LABELS.get(feature, feature))
        ax.set_ylabel("Book MT preference rate")
        ax.set_ylim(-0.03, 1.03)
    save_figure(fig, out_dir, "book_level_feature_scatters")


def plot_hooked_case(response_features: pd.DataFrame, out_dir: Path) -> None:
    hooked = response_features[
        response_features["book_id"] == "japanese_eval_hooked_a_novel_of_obsession"
    ].copy()
    if hooked.empty:
        return
    chunk = (
        hooked.groupby("chunk_id", as_index=False)
        .agg(
            mt_preference_rate=("preferred_mt", "mean"),
            n=("preferred_mt", "size"),
            both_mt=("preferred_mt", "sum"),
            mt_minus_ht_token_count=("mt_minus_ht_token_count", "mean"),
            mt_minus_ht_median_sentence_words=("mt_minus_ht_median_sentence_words", "mean"),
            mt_minus_ht_mtld=("mt_minus_ht_mtld", "mean"),
            mt_minus_ht_quote_rate=("mt_minus_ht_quote_rate", "mean"),
        )
        .sort_values("chunk_id")
    )
    chunk["both_participants_mt"] = chunk["both_mt"] >= 2

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = np.where(chunk["both_participants_mt"], MT_ORANGE, GRAY)
    axes[0].bar(chunk["chunk_id"], chunk["mt_preference_rate"], color=colors)
    axes[0].set_ylabel("MT preference rate")
    axes[0].set_ylim(0, 1.05)
    long = chunk.melt(
        id_vars=["chunk_id", "both_participants_mt"],
        value_vars=[
            "mt_minus_ht_token_count",
            "mt_minus_ht_median_sentence_words",
            "mt_minus_ht_mtld",
            "mt_minus_ht_quote_rate",
        ],
        var_name="feature",
        value_name="value",
    )
    sns.lineplot(data=long, x="chunk_id", y="value", hue="feature", marker="o", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Hooked chunk ID (Part 2 zero-based)")
    axes[1].set_ylabel("MT minus HT feature delta")
    axes[1].legend(fontsize=7, loc="best")
    save_figure(fig, out_dir, "hooked_chunk_preference_and_feature_deltas")

    span_cols = [
        "count_good_spans_ht",
        "count_poor_spans_ht",
        "count_good_spans_mt",
        "count_poor_spans_mt",
    ]
    available = [col for col in span_cols if col in hooked.columns]
    if available:
        span = hooked.groupby("chunk_id", as_index=False)[available].sum()
        long_span = span.melt("chunk_id", var_name="span_type", value_name="count")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=long_span, x="chunk_id", y="count", hue="span_type", marker="o", ax=ax)
        ax.set_xlabel("Hooked chunk ID (Part 2 zero-based)")
        ax.set_ylabel("Span count across participants")
        save_figure(fig, out_dir, "hooked_span_counts_by_version")


def plot_book_heatmap(book_summary: pd.DataFrame, out_dir: Path) -> None:
    columns = [f"mean_{feature}" for feature in BOOK_SCATTER_FEATURES if f"mean_{feature}" in book_summary]
    if not columns:
        return
    ordered = book_summary.sort_values("mt_preference_rate", ascending=False).copy()
    matrix = ordered.set_index("book_id")[columns].astype(float)
    z = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, np.nan)
    z = z.rename(columns={f"mean_{feature}": FEATURE_LABELS.get(feature, feature) for feature in BOOK_SCATTER_FEATURES})
    z.index = [short_book_label(index) for index in z.index]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(z, cmap="vlag", center=0, linewidths=0.3, ax=ax, cbar_kws={"label": "Book z-score"})
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, out_dir, "book_level_paired_feature_heatmap")


def strongest_rows(df: pd.DataFrame, value_col: str, n: int = 8) -> pd.DataFrame:
    if df.empty or value_col not in df:
        return pd.DataFrame()
    return df.reindex(df[value_col].abs().sort_values(ascending=False).index).head(n)


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    display = df[columns].copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            if column.startswith("n_") or column in {"n_books", "n_focal", "n_baseline"}:
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{int(value)}"
                )
            else:
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{float(value):.{float_digits}f}"
                )
    display = display.fillna("").astype(str)
    widths = {
        column: max(len(column), *(len(value) for value in display[column].tolist()))
        for column in display.columns
    }
    header = "| " + " | ".join(column.ljust(widths[column]) for column in display.columns) + " |"
    separator = "| " + " | ".join("-" * widths[column] for column in display.columns) + " |"
    body = [
        "| "
        + " | ".join(str(row[column]).ljust(widths[column]) for column in display.columns)
        + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *body])


def write_summary(
    out_dir: Path,
    chunk_features: pd.DataFrame,
    response_features: pd.DataFrame,
    feature_summary: pd.DataFrame,
    book_summary: pd.DataFrame,
    book_correlations: pd.DataFrame,
    hooked_case: pd.DataFrame,
) -> None:
    hooked_book = book_summary[
        book_summary["book_id"] == "japanese_eval_hooked_a_novel_of_obsession"
    ]
    hooked_chunks = chunk_features[
        chunk_features["book_id"] == "japanese_eval_hooked_a_novel_of_obsession"
    ]
    hooked_response = response_features[
        response_features["book_id"] == "japanese_eval_hooked_a_novel_of_obsession"
    ]
    top_effects = strongest_rows(
        feature_summary[feature_summary["feature"].str.startswith("mt_minus_ht_")],
        "cliffs_delta",
        n=10,
    )
    top_corrs = strongest_rows(book_correlations, "spearman_rho", n=10)

    hooked_lines: list[str] = []
    if not hooked_book.empty and not hooked_chunks.empty:
        hb = hooked_book.iloc[0]
        hooked_lines.append(
            f"- Hooked MT preference rate: {hb['mt_preference_rate']:.3f} "
            f"({int(hb['n_responses'])} responses, {int(hb['n_chunks'])} chunks)."
        )
        for feature in [
            "mt_minus_ht_token_count",
            "mt_minus_ht_median_sentence_words",
            "mt_minus_ht_std_sentence_words",
            "mt_minus_ht_quote_rate",
            "mt_minus_ht_em_dash_rate",
            "mt_minus_ht_mtld",
            "mt_minus_ht_surface_artifact_index",
        ]:
            value = hooked_chunks[feature].mean()
            hooked_lines.append(f"- Hooked mean {feature}: {value:.3f}.")
        both_mt = (
            hooked_response.groupby("chunk_id")["preferred_mt"].sum().ge(2).sum()
            if not hooked_response.empty
            else 0
        )
        hooked_lines.append(f"- Chunks where both available participants preferred MT: {both_mt}.")
    else:
        hooked_lines.append("- Hooked rows were not found.")

    hooked_comment_terms = ""
    if not hooked_case.empty:
        comments = " ".join(
            hooked_case.get("justification", pd.Series(dtype=str)).fillna("").astype(str).tolist()
        ).lower()
        terms = [
            "voice",
            "narrator",
            "formal",
            "natural",
            "interior",
            "monologue",
            "pacing",
            "wording",
            "flow",
            "awkward",
            "stilted",
        ]
        counts = Counter(term for term in terms for _ in range(comments.count(term)))
        if counts:
            hooked_comment_terms = ", ".join(f"{term}={count}" for term, count in counts.most_common())

    summary = f"""# Part 2 Chunk Paired HT-vs-MT Features

## What Was Counted

- Input responses: `{DEFAULT_PART2.relative_to(REPO_ROOT)}` filtered to HT/MT chunk preferences.
- Coded rationales: `{DEFAULT_Q5.relative_to(REPO_ROOT)}` joined by participant base, canonical book ID, and Part 2 chunk ID.
- Span highlights: `{DEFAULT_SPANS.relative_to(REPO_ROOT)}` aggregated by participant/book/chunk/version.
- Chunk texts: HT from `{DEFAULT_HT_DIR.relative_to(REPO_ROOT)}` and MT from `{DEFAULT_MT_DIR.relative_to(REPO_ROOT)}`.
- Part 2 chunk IDs are zero-based in the CSVs; JSONL chunk IDs are one-based here. The output keeps `chunk_id`, `jsonl_chunk_id`, and `chunk_id_indexing_note`.

The paired feature columns use `mt_minus_ht_*`, `abs_mt_minus_ht_*`, and `mt_div_ht_*`.
Positive `mt_minus_ht_*` values mean the MT chunk has more of that feature than the paired HT chunk.

## Exploratory Cautions

This is a descriptive screen, not a confirmatory model. Participant responses are not independent:
chunks are nested in books, and many chunks have two participant responses. Mann-Whitney and Spearman
p-values are included to rank patterns, but effect sizes and plots should carry more weight.

## Row Counts

- Chunk-pair feature rows: {len(chunk_features)}
- Participant-response rows: {len(response_features)}
- Hooked case-study rows: {len(hooked_case)}
- Books in book-level summary: {len(book_summary)}

## Strongest Participant-Level Feature Deltas

{markdown_table(top_effects, ["feature", "n_focal", "n_baseline", "mean_delta_focal_minus_baseline", "cliffs_delta", "p_value", "q_value"])}

## Strongest Book-Level Correlations

{markdown_table(top_corrs, ["feature", "n_books", "spearman_rho", "p_value", "permutation_p_value", "permutation_q_value"])}

## Hooked Observations

{chr(10).join(hooked_lines)}

Hooked comment keyword counts for narrator voice/register/interior-monologue/pacing/word-choice terms:
{hooked_comment_terms or "_No keyword hits in the selected term list._"}

Inspect `hooked_case_study.csv` for participant-level justifications, coding counts, span counts, and
the primary paired deltas by chunk.
"""
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")


def write_outputs(
    out_dir: Path,
    chunk_features: pd.DataFrame,
    response_features: pd.DataFrame,
    feature_summary: pd.DataFrame,
    book_summary: pd.DataFrame,
    book_correlations: pd.DataFrame,
    hooked_case: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_features.to_csv(out_dir / "chunk_pair_feature_table.csv", index=False, encoding="utf-8")
    response_features.to_csv(
        out_dir / "chunk_response_feature_table.csv", index=False, encoding="utf-8"
    )
    feature_summary.to_csv(out_dir / "feature_test_summary.csv", index=False, encoding="utf-8")
    book_summary.to_csv(out_dir / "book_level_feature_summary.csv", index=False, encoding="utf-8")
    book_correlations.to_csv(
        out_dir / "book_level_feature_correlations.csv", index=False, encoding="utf-8"
    )
    hooked_case.to_csv(out_dir / "hooked_case_study.csv", index=False, encoding="utf-8")
    write_summary(
        out_dir,
        chunk_features,
        response_features,
        feature_summary,
        book_summary,
        book_correlations,
        hooked_case,
    )


def run_analysis(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    setup_plot_style()
    part2 = read_csv(args.part2)
    q5 = read_csv(args.q5)
    spans = read_csv(args.spans)

    chunk_features = build_chunk_pair_feature_table(part2, args.ht_dir, args.mt_dir)
    q5_counts = build_q5_counts(q5)
    span_features = aggregate_spans(spans)
    response_features = build_response_feature_table(
        part2, chunk_features, q5_counts, span_features
    )
    feature_summary = build_feature_test_summary(response_features)
    book_summary = build_book_level_summary(response_features)
    book_correlations = build_book_level_correlations(
        book_summary, permutations=args.permutations
    )
    hooked_case = build_hooked_case_study(response_features)

    write_outputs(
        args.out_dir,
        chunk_features,
        response_features,
        feature_summary,
        book_summary,
        book_correlations,
        hooked_case,
    )
    plot_participant_boxplots(response_features, args.out_dir)
    plot_effect_summary(feature_summary, args.out_dir)
    plot_book_scatter(book_summary, args.out_dir)
    plot_hooked_case(response_features, args.out_dir)
    plot_book_heatmap(book_summary, args.out_dir)

    return {
        "chunk_features": chunk_features,
        "response_features": response_features,
        "feature_summary": feature_summary,
        "book_summary": book_summary,
        "book_correlations": book_correlations,
        "hooked_case": hooked_case,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part2", type=Path, default=DEFAULT_PART2)
    parser.add_argument("--q5", type=Path, default=DEFAULT_Q5)
    parser.add_argument("--spans", type=Path, default=DEFAULT_SPANS)
    parser.add_argument("--ht-dir", type=Path, default=DEFAULT_HT_DIR)
    parser.add_argument("--mt-dir", type=Path, default=DEFAULT_MT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--permutations",
        type=int,
        default=5000,
        help="Permutations for book-level Spearman p-values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_analysis(args)
    print(f"Wrote outputs to {args.out_dir}")
    print(f"chunk_pair_feature_table rows: {len(outputs['chunk_features'])}")
    print(f"chunk_response_feature_table rows: {len(outputs['response_features'])}")
    print(f"hooked_case_study rows: {len(outputs['hooked_case'])}")


if __name__ == "__main__":
    main()
