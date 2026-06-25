from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence
import re

import tiktoken

LANGUAGE_CODES = {
    "fr": "French",
    "ja": "Japanese",
    "pl": "Polish",
    "en": "English",
    "sp": "Spanish",
}

TARGET_LANGUAGE_CODES = LANGUAGE_CODES


class Chunk:
    def __init__(self, text: str, start: int, end: int) -> None:
        self.text = text
        self.start = start
        self.end = end


class Book:
    """Agentic-pipeline specific book abstraction with paragraph-preserving token chunking."""

    CHUNK_TOKEN_LIMIT = 1000
    ENCODING_NAME = "o200k_base"
    _ENCODING = tiktoken.get_encoding(ENCODING_NAME)

    def __init__(self, path: Path, src_text: str, chunks: bool = True) -> None:
        self.path = path
        self.translation: Optional[List[Chunk]] = None
        self.chunks = chunks
        self.name = self._derive_name()
        self.source_language = self._detect_book_language()

        if chunks:
            if self._has_chunk_tags(src_text):
                self.src_text = self._split_by_chunks(src_text)
            else:
                self.src_text = self._chunk_src_text(src_text)
        else:
            clean_text = self._remove_chunk_tags(src_text)
            self.src_text = [Chunk(clean_text, 0, len(clean_text))]

    def get_src_text(self, raw: bool = False, chunk: bool = False):
        if raw:
            return "<chunk>\n" + "\n</chunk>\n\n<chunk>\n".join(
                [c.text for c in self.src_text]
            ) + "\n</chunk>"
        if chunk:
            return self.src_text
        return "\n\n".join([c.text for c in self.src_text])

    def get_translation(self, raw: bool = False, chunk: bool = False):
        if raw:
            return "<chunk>\n" + "\n</chunk>\n\n<chunk>\n".join(
                [c.text for c in self.translation]
            ) + "\n</chunk>"
        if chunk:
            return self.translation
        return "\n\n".join([c.text for c in self.translation])

    def set_translation(self, chunk_translations) -> None:
        """Set translation from a raw string, list of strings, or list of Chunk objects."""
        if isinstance(chunk_translations, str):
            text = chunk_translations
            self.translation = [Chunk(text, 0, len(text))]
            return
        if len(chunk_translations) != len(self.src_text):
            raise RuntimeError(
                f"Expected {len(self.src_text)} translated chunk(s) for '{self.name}', "
                f"received {len(chunk_translations)}."
            )
        if chunk_translations and isinstance(chunk_translations[0], Chunk):
            self.translation = list(chunk_translations)
            return

        translations: List[Chunk] = []
        offset = 0
        for chunk_text in chunk_translations:
            translations.append(Chunk(chunk_text, offset, offset + len(chunk_text)))
            offset += len(chunk_text)
        self.translation = translations

    @classmethod
    def _token_count(cls, text: str) -> int:
        return len(cls._ENCODING.encode(text))

    @classmethod
    def _to_chunks(cls, chunk_texts: Sequence[str]) -> List[Chunk]:
        chunks: List[Chunk] = []
        offset = 0
        for idx, chunk_text in enumerate(chunk_texts):
            chunks.append(Chunk(chunk_text, offset, offset + len(chunk_text)))
            offset += len(chunk_text)
            if idx < len(chunk_texts) - 1:
                offset += 2  # join separator used by get_src_text/get_translation
        return chunks

    def _chunk_src_text(self, text: str) -> List[Chunk]:
        paragraphs = text.split("\n\n")
        chunk_texts: List[str] = []
        current_paragraphs: List[str] = []
        current_token_count = 0

        def flush_current() -> None:
            nonlocal current_paragraphs, current_token_count
            if current_paragraphs:
                chunk_texts.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_token_count = 0

        for paragraph in paragraphs:
            token_count = self._token_count(paragraph)
            if token_count > self.CHUNK_TOKEN_LIMIT:
                flush_current()
                chunk_texts.append(paragraph)
                continue

            if current_paragraphs and current_token_count + token_count > self.CHUNK_TOKEN_LIMIT:
                flush_current()

            current_paragraphs.append(paragraph)
            current_token_count += token_count

        flush_current()
        return self._to_chunks(chunk_texts)

    def _derive_name(self) -> str:
        parts = self.path.stem.split("_")
        if len(parts) < 2:
            raise ValueError(f"Could not derive a book name from file name: {self.path.name}")
        return "_".join(parts[:-1])

    def _detect_book_language(self) -> str:
        parts = self.path.stem.split("_")
        if len(parts) < 2:
            raise ValueError(f"Could not detect source language from file name: {self.path.name}")
        lang_code = parts[-1].lower()
        if lang_code not in LANGUAGE_CODES:
            raise ValueError(
                f"Could not detect source language from file name: {self.path.name}. "
                f"Expected one of: {', '.join(sorted(LANGUAGE_CODES))}"
            )
        return LANGUAGE_CODES[lang_code]

    @staticmethod
    def _has_chunk_tags(text: str) -> bool:
        return bool(re.search(r"</?chunk>", text))

    @staticmethod
    def _split_by_chunks(text: str) -> List[Chunk]:
        chunk_texts = [c.strip() for c in re.split(r"<chunk>|</chunk>", text) if c.strip()]
        chunks: List[Chunk] = []
        offset = 0
        for idx, chunk_text in enumerate(chunk_texts):
            chunks.append(Chunk(chunk_text, offset, offset + len(chunk_text)))
            offset += len(chunk_text)
            if idx < len(chunk_texts) - 1:
                offset += 2
        return chunks

    @staticmethod
    def _remove_chunk_tags(text: str) -> str:
        return re.sub(r"</?chunk>", "", text).strip()
