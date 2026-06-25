from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


QUOTE_CHARS = "\"'`´“”‘’«»„‟‹›"
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
QUOTE_TRANSLATION = str.maketrans({char: " " for char in QUOTE_CHARS})
ENGLISH_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "arent",
    "aren",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "cant",
    "can",
    "could",
    "couldn't",
    "couldnt",
    "couldn",
    "d",
    "did",
    "didn't",
    "didnt",
    "didn",
    "do",
    "does",
    "doesn't",
    "doesnt",
    "doesn",
    "doing",
    "don't",
    "dont",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "hadnt",
    "hadn",
    "has",
    "hasn't",
    "hasnt",
    "hasn",
    "have",
    "haven't",
    "havent",
    "haven",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "hed",
    "hes",
    "her",
    "here",
    "here's",
    "heres",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "hows",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "id",
    "im",
    "ive",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "isnt",
    "isn",
    "it",
    "it's",
    "its",
    "its",
    "itself",
    "let's",
    "lets",
    "ll",
    "me",
    "more",
    "most",
    "mustn't",
    "mustnt",
    "mustn",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "shant",
    "shan",
    "she",
    "she'd",
    "she'll",
    "she's",
    "shes",
    "s",
    "should",
    "shouldn't",
    "shouldnt",
    "shouldn",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "thats",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "theres",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "theyd",
    "theyll",
    "theyre",
    "theyve",
    "t",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "wasnt",
    "wasn",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "wed",
    "were",
    "weve",
    "were",
    "weren't",
    "werent",
    "weren",
    "what",
    "what's",
    "whats",
    "when",
    "when's",
    "whens",
    "where",
    "where's",
    "wheres",
    "which",
    "while",
    "who",
    "who's",
    "whos",
    "whom",
    "why",
    "why's",
    "whys",
    "with",
    "won't",
    "wont",
    "won",
    "would",
    "wouldn't",
    "wouldnt",
    "wouldn",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "youd",
    "youll",
    "youre",
    "youve",
    "m",
    "re",
    "ve",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


@dataclass(frozen=True)
class TextStats:
    words: int
    sentences: int
    paragraphs: int


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(QUOTE_TRANSLATION)
    text = text.translate(PUNCT_TRANSLATION)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if parts:
        return parts
    stripped = text.strip()
    return [stripped] if stripped else []


def sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)


def normalized_words(text: str) -> list[str]:
    return words(normalize_text(text))


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def ngram_counter(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(ngrams(tokens, n))


def content_ngram_counter(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    filtered = [gram for gram in ngrams(tokens, n) if not all(token in ENGLISH_STOPWORDS for token in gram)]
    return Counter(filtered)


def sliding_windows(tokens: list[str], window_size: int = 500, step: int | None = None) -> list[list[str]]:
    if not tokens:
        return []
    step = step or window_size
    if len(tokens) <= window_size:
        return [tokens]
    return [tokens[start : start + window_size] for start in range(0, len(tokens) - window_size + 1, step)]


def type_token_ratio(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    if not tokens:
        return 0.0

    def factor_count(sequence: list[str]) -> float:
        factors = 0.0
        token_count = 0
        types: set[str] = set()
        for token in sequence:
            token_count += 1
            types.add(token)
            ttr = len(types) / token_count
            if ttr <= threshold:
                factors += 1.0
                token_count = 0
                types = set()
        if token_count:
            ttr = len(types) / token_count
            if ttr != 1:
                factors += (1 - ttr) / (1 - threshold)
        return factors

    forward = factor_count(tokens)
    backward = factor_count(list(reversed(tokens)))
    factors = (forward + backward) / 2
    return len(tokens) / factors if factors else float(len(tokens))


def repetition_rate(tokens: list[str], n: int) -> float:
    units = ngrams(tokens, n)
    if not units:
        return 0.0
    counts = Counter(units)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(units)


def text_stats(text: str) -> TextStats:
    return TextStats(
        words=len(words(text)),
        sentences=max(1, len(sentences(text))) if text.strip() else 0,
        paragraphs=len(paragraphs(text)),
    )


def chunk_text(text: str, chunk_words: int = 220, min_chunk_words: int = 40) -> list[str]:
    para_chunks: list[str] = []
    for para in paragraphs(text):
        para_words = words(para)
        if not para_words:
            continue
        if len(para_words) <= chunk_words:
            para_chunks.append(para)
            continue
        for start in range(0, len(para_words), chunk_words):
            slice_words = para_words[start : start + chunk_words]
            if len(slice_words) >= min_chunk_words:
                para_chunks.append(" ".join(slice_words))

    if not para_chunks:
        normalized = text.strip()
        return [normalized] if normalized else []

    merged: list[str] = []
    buffer: list[str] = []
    buffer_count = 0
    for chunk in para_chunks:
        count = len(words(chunk))
        if count < min_chunk_words and merged:
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
            continue
        if buffer_count + count <= chunk_words or not buffer:
            buffer.append(chunk)
            buffer_count += count
        else:
            merged.append("\n\n".join(buffer))
            buffer = [chunk]
            buffer_count = count
    if buffer:
        merged.append("\n\n".join(buffer))
    return merged


def mean_absolute_percentage_difference(values: Iterable[tuple[int, int]]) -> float:
    diffs = []
    for candidate, reference in values:
        if reference == 0:
            continue
        diffs.append(abs(candidate - reference) / reference * 100.0)
    return sum(diffs) / len(diffs) if diffs else 0.0
