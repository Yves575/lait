"""Calculate book- and chunk-level word, token, and optional sentence count stats."""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import tiktoken

SPACY_SENTENCE_MODEL = "en_core_web_trf"
# Same pattern as scripts/chunktext_to_jsonl.py
_CHUNK_TAG_RE = re.compile(r"<chunk>(.*?)</chunk>", re.DOTALL)


def is_empty_chunk(text: str, word_count: int, token_count: int) -> bool:
    """True when a chunk has no substantive text or counts."""
    return not str(text).strip() and word_count <= 0 and token_count <= 0


def _append_chunk(
    rows: list[dict],
    texts: list[str],
    *,
    book: str,
    file: str,
    chunk_id,
    word_count: int,
    text: str,
) -> bool:
    """Append one chunk row. Return False when the chunk is empty and skipped."""
    token_count = count_tokens(text) if text else 0
    if is_empty_chunk(text, word_count, token_count):
        return False
    rows.append(
        {
            "book": book,
            "file": file,
            "chunk_id": chunk_id,
            "word_count": word_count,
            "token_count": token_count,
        }
    )
    texts.append(text)
    return True


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        # Fallback to simple approximation if tiktoken fails
        return len(text.split())


def load_spacy_nlp(model_name: str = SPACY_SENTENCE_MODEL):
    """Load a spaCy model for sentence segmentation (parser + transformer only)."""
    try:
        import spacy
    except ImportError:
        print(
            "Error: spaCy is required for --count-sentences. "
            "Install project requirements and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    used_gpu = False
    try:
        from thinc.api import set_gpu_allocator

        set_gpu_allocator("pytorch")
        used_gpu = bool(spacy.prefer_gpu())
    except Exception as exc:
        print(f"spaCy GPU not enabled ({exc}); using CPU", file=sys.stderr)

    try:
        nlp = spacy.load(
            model_name,
            disable=["ner", "tagger", "lemmatizer", "attribute_ruler"],
        )
    except OSError:
        print(
            f"Error: spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"spaCy {model_name} using {'GPU' if used_gpu else 'CPU'}")
    return nlp


def add_sentence_counts(rows: list[dict], texts: list[str], nlp) -> None:
    """Add spaCy sentence counts to each row, in place."""
    if not rows:
        return
    for row, doc in zip(rows, nlp.pipe(texts, batch_size=8)):
        row["sentence_count"] = sum(1 for sent in doc.sents if sent.text.strip())


def load_chunk_counts(folder: Path, nlp=None) -> pd.DataFrame:
    """Load chunk word and token counts from JSONL files, or tagged ``<chunk>`` .txt.

    Records with empty text and zero word/token counts are skipped.
    When ``nlp`` is provided, also count sentences with that spaCy model.
    """
    rows = []
    texts = []
    skipped = 0
    jsonl_paths = sorted(p for p in folder.glob("*.jsonl") if p.is_file())
    txt_paths = sorted(p for p in folder.glob("*.txt") if p.is_file())

    for path in jsonl_paths:
        if not path.is_file():
            continue

        try:
            df = pd.read_json(path, lines=True)
        except ValueError as exc:
            print(f"Error reading {path}: {exc}", file=sys.stderr)
            raise SystemExit(1)

        if "word_count" not in df.columns:
            print(f"Error: {path} has no word_count column", file=sys.stderr)
            raise SystemExit(1)

        word_counts = pd.to_numeric(df["word_count"], errors="coerce")
        if word_counts.isna().any():
            bad_rows = word_counts[word_counts.isna()].index + 1
            bad_rows_text = ", ".join(str(row) for row in bad_rows[:10])
            print(
                f"Error: {path} has non-numeric word_count values on row(s): {bad_rows_text}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        # Find text column (try common names)
        text_column = None
        for col_name in ["text", "content", "chunk_text", "body"]:
            if col_name in df.columns:
                text_column = col_name
                break

        chunk_ids = df["chunk_id"] if "chunk_id" in df.columns else pd.Series(df.index + 1)
        for idx, (chunk_id, word_count) in enumerate(zip(chunk_ids, word_counts)):
            text = ""
            if text_column is not None and idx < len(df):
                text = str(df.iloc[idx][text_column]) if pd.notna(df.iloc[idx][text_column]) else ""

            if not _append_chunk(
                rows,
                texts,
                book=path.stem,
                file=str(path),
                chunk_id=chunk_id,
                word_count=int(word_count),
                text=text,
            ):
                skipped += 1

    if not rows:
        for path in txt_paths:
            chunks = [
                match.group(1).strip("\n")
                for match in _CHUNK_TAG_RE.finditer(path.read_text(encoding="utf-8"))
            ]
            for chunk_id, text in enumerate(chunks, start=1):
                if not _append_chunk(
                    rows,
                    texts,
                    book=path.stem,
                    file=str(path),
                    chunk_id=chunk_id,
                    word_count=len(text.split()),
                    text=text,
                ):
                    skipped += 1

    if skipped:
        print(f"Skipped {skipped} empty chunk(s) in {folder}", file=sys.stderr)

    if nlp is not None:
        print(
            f"Counting sentences with spaCy {SPACY_SENTENCE_MODEL} "
            f"for {len(rows)} chunk(s) in {folder}..."
        )
        add_sentence_counts(rows, texts, nlp)

    return pd.DataFrame(rows)


def calculate_stats(counts: pd.Series, metric_name: str = "word_count") -> pd.DataFrame:
    """Return max, min, std, mean, median, and total for a count series."""
    stats = counts.agg(["max", "min", "std", "mean", "median", "sum"])
    return stats.rename_axis("stat").reset_index(name=metric_name)


def write_count_stats(chunk_counts: pd.DataFrame, metric_name: str, output_dir: Path) -> None:
    """Write book- and chunk-level stats CSVs for one count column."""
    book_counts = (
        chunk_counts.groupby("book", as_index=False)[metric_name]
        .sum()
        .sort_values("book")
    )
    book_stats = calculate_stats(book_counts[metric_name], metric_name)
    chunk_stats = calculate_stats(chunk_counts[metric_name], metric_name)
    book_path = output_dir / f"book_{metric_name}_stats.csv"
    chunk_path = output_dir / f"chunk_{metric_name}_stats.csv"
    book_stats.to_csv(book_path, index=False)
    chunk_stats.to_csv(chunk_path, index=False)
    label = metric_name.replace("_", " ")
    print(f"Wrote book {label} stats to {book_path}")
    print(f"Wrote chunk {label} stats to {chunk_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate book and chunk word-count and token-count statistics from JSONL "
            "files or tagged <chunk> .txt files. Optionally also count sentences with "
            "spaCy en_core_web_trf."
        )
    )

    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("books/HT/eval"),
        help="Directory containing JSONL chunks or tagged <chunk> .txt files (default: books/HT/eval).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("book_stats/human_translation_counts"),
        help=(
            "Directory for aggregate count CSVs "
            "(default: book_stats/human_translation_counts)."
        ),
    )
    parser.add_argument(
        "--count-sentences",
        action="store_true",
        help=(
            f"Also count sentences with spaCy ({SPACY_SENTENCE_MODEL}) and write "
            "book_sentence_count_stats.csv and chunk_sentence_count_stats.csv. "
            "Intended for English HT/MT text, not source-language chunks."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output_dir)

    if not chunks_dir.is_dir():
        print(f"Error: {chunks_dir} is not a directory", file=sys.stderr)
        raise SystemExit(1)

    nlp = load_spacy_nlp() if args.count_sentences else None
    chunk_counts = load_chunk_counts(chunks_dir, nlp=nlp)
    if chunk_counts.empty:
        print(f"Error: no chunks found in {chunks_dir}", file=sys.stderr)
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_count_stats(chunk_counts, "word_count", output_dir)
    write_count_stats(chunk_counts, "token_count", output_dir)
    if args.count_sentences:
        write_count_stats(chunk_counts, "sentence_count", output_dir)


if __name__ == "__main__":
    main()
