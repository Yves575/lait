from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence

import tiktoken

LANGUAGE_CODES = {
    "fr": "French",
    "ja": "Japanese",
    "pl": "Polish",
    "sp": "Spanish",
    "en": "English",
}

ENCODING_NAME = "o200k_base"
_ENCODING = tiktoken.get_encoding(ENCODING_NAME)


class Chunk:
    def __init__(self, text: str) -> None:
        self.text = text
        self.token_count = len(_ENCODING.encode(text))

    def to_dict(self, chunk_id: int) -> dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "word_count": len(self.text.split()),
            "token_count": self.token_count,
            "text": self.text,
        }


class Book:
    CHUNK_TOKEN_LIMIT = 1000

    def __init__(
        self,
        path: Path,
        src_text: str,
        chunks: bool = True,
        chunk_path: str | Path | None = None,
        chunk_token_limit: int | None = None,
    ) -> None:
        self.path = path
        self.translation: Optional[List[Chunk]] = None
        self.chunks = chunks
        self.chunk_token_limit = chunk_token_limit or self.CHUNK_TOKEN_LIMIT
        self.name = self._derive_name()
        self.source_language = self._detect_book_language()
        self.chunk_path = Path(chunk_path) if chunk_path else None

        if not chunks:
            self.src_text = [Chunk(src_text)]
        elif self.chunk_path is None:
            self.src_text = self._chunk_src_text(src_text)
        elif self.chunk_path.exists():
            self.src_text = self._load_chunks(self.chunk_path)
        else:
            self.src_text = self._chunk_src_text(src_text)
            self.save_chunks(self.chunk_path)

    def get_src_text(self, chunk: bool = False):
        if chunk:
            return self.src_text
        return "\n\n".join([chunk.text for chunk in self.src_text])

    def get_translation(self, chunk: bool = False):
        translation = self._require_translation()
        if chunk:
            return translation
        return "\n\n".join([chunk.text for chunk in translation])

    def set_translation(self, chunk_translations: str | Sequence[str] | Sequence["Chunk"]) -> None:
        """Set translation from a plain string, a list of strings, or Chunk objects."""
        if isinstance(chunk_translations, str):
            self.translation = [Chunk(chunk_translations)]
            return

        items = list(chunk_translations)
        if len(items) != len(self.src_text):
            raise RuntimeError(
                f"Expected {len(self.src_text)} translated chunk(s) for '{self.name}', "
                f"received {len(items)}."
            )
        if items and isinstance(items[0], Chunk):
            self.translation = list(items)
            return
        self.translation = [Chunk(text) for text in items]

    def _chunk_src_text(self, text: str) -> List[Chunk]:
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
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
            token_count = len(_ENCODING.encode(paragraph))
            if token_count > self.chunk_token_limit:
                flush_current()
                chunk_texts.append(paragraph)
                continue

            if current_paragraphs and current_token_count + token_count > self.chunk_token_limit:
                flush_current()

            current_paragraphs.append(paragraph)
            current_token_count += token_count

        flush_current()
        return [Chunk(text) for text in chunk_texts]

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

    def _load_chunks(self, chunk_path: Path) -> List[Chunk]:
        if chunk_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Unsupported chunk file type (expected .jsonl): {chunk_path}")

        records = [
            json.loads(line)
            for line in chunk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [
            Chunk(self._extract_chunk_text(record=record, chunk_path=chunk_path, index=index))
            for index, record in enumerate(records, start=1)
        ]

    @staticmethod
    def _extract_chunk_text(record: Any, chunk_path: Path, index: int) -> str:
        if isinstance(record, dict) and isinstance(record.get("text"), str):
            return record["text"]
        raise ValueError(
            f"Chunk #{index} in {chunk_path} must be an object with a string 'text'."
        )

    def _require_translation(self) -> List[Chunk]:
        if self.translation is None:
            raise RuntimeError(f"No translation is loaded for '{self.name}'.")
        return self.translation

    def save_chunks(self, chunk_path: str | Path, *,
        translation: bool = False,) -> Path:
        chunks = self._require_translation() if translation else self.src_text
        target_path = Path(chunk_path)
        if target_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Chunk file must be .jsonl: {target_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as fh:
            for index, chunk in enumerate(chunks, start=1):
                fh.write(json.dumps(chunk.to_dict(chunk_id=index), ensure_ascii=False) + "\n")
        return target_path

    @classmethod
    def from_path(
        cls,
        book_path: Path,
        chunks: bool = True,
        chunk_path: str | Path | None = None,
        chunk_token_limit: int | None = None,
    ) -> "Book":
        if book_path.suffix.lower() != ".txt":
            raise ValueError(f"Book file must be a .txt file: {book_path}")

        return cls(
            path=book_path,
            src_text=book_path.read_text(encoding="utf-8"),
            chunks=chunks,
            chunk_path=chunk_path,
            chunk_token_limit=chunk_token_limit,
        )

    @classmethod
    def from_chunk_path(cls, chunk_path: Path) -> "Book":
        if chunk_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Chunk file must be a .jsonl file: {chunk_path}")

        return cls(
            path=chunk_path,
            src_text="",
            chunks=True,
            chunk_path=chunk_path,
        )


class BooksList:
    def __init__(
        self,
        path: str | Path,
        chunks: bool,
        chunk_path: str | Path | None = None,
        chunk_token_limit: int | None = None,
    ):
        self.path = Path(path)
        self.chunks = chunks
        self.chunk_path = Path(chunk_path) if chunk_path else None
        self.chunk_token_limit = chunk_token_limit
        self.books: List[Book] = self._load_books()

    def _chunk_path_for(self, book_path: Path) -> Path | None:
        if self.chunk_path is None:
            return None
        return self.chunk_path / f"{book_path.stem}.jsonl"

    def _load_books(self) -> List[Book]:
        path = self.path

        if path.is_file():
            return [
                Book.from_path(
                    path,
                    chunks=self.chunks,
                    chunk_path=self._chunk_path_for(path),
                    chunk_token_limit=self.chunk_token_limit,
                )
            ]

        if path.is_dir():
            book_paths = sorted(
                candidate
                for candidate in path.iterdir()
                if candidate.is_file() and candidate.suffix.lower() == ".txt"
            )
            if not book_paths:
                raise FileNotFoundError(f"No .txt files found in directory: {path}")
            return [
                Book.from_path(
                    book_path,
                    chunks=self.chunks,
                    chunk_path=self._chunk_path_for(book_path),
                    chunk_token_limit=self.chunk_token_limit,
                )
                for book_path in book_paths
            ]

        raise FileNotFoundError(f"Book path not found: {path}")

    def load_translations_from(self, translated_books: "BooksList") -> None:
        translation_map = {book.name: book for book in translated_books.books}
        for book in self.books:
            translated = translation_map.get(book.name)
            if translated is None:
                continue
            book.set_translation(translated.get_src_text(chunk=True))

    def save_books(self, folder: str | Path) -> None:
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        if not self.books:
            raise ValueError("Not books loaded, please first use 'load_books()'")

        for book in self.books:
            file_path = folder_path / f"{book.name}_en.txt"
            file_path.write_text(book.get_translation(), encoding="utf-8")
