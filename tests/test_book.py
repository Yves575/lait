import json
from pathlib import Path

from book import Book


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_book_uses_jsonl_chunks_when_available(tmp_path: Path):
    book_path = tmp_path / "books" / "dev" / "demo_fr_fr.txt"
    chunk_path = tmp_path / "books" / "dev_chunk" / "demo_fr_fr.jsonl"
    _write(book_path, "Paragraph one.\n\nParagraph two.\n\nParagraph three.")
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text(
        "\n".join(
            [
                json.dumps({"chunk_id": 1, "text": "Custom first chunk."}, ensure_ascii=False),
                json.dumps({"chunk_id": 2, "text": "Custom second chunk."}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    book = Book.from_path(book_path, chunks=True, chunk_path=chunk_path)

    assert [chunk.text for chunk in book.get_src_text(chunk=True)] == [
        "Custom first chunk.",
        "Custom second chunk.",
    ]


def test_book_falls_back_to_token_chunking_when_json_is_missing(tmp_path: Path):
    paragraphs = [
        " ".join(["alpha"] * 12),
        " ".join(["beta"] * 12),
        " ".join(["gamma"] * 12),
    ]
    book_path = tmp_path / "books" / "dev" / "demo_fr_fr.txt"
    _write(book_path, "\n\n".join(paragraphs))

    book = Book.from_path(book_path, chunks=True, chunk_token_limit=25)

    assert len(book.get_src_text(chunk=True)) == 2
    assert "alpha" in book.get_src_text(chunk=True)[0].text
    assert "beta" in book.get_src_text(chunk=True)[0].text
    assert "gamma" in book.get_src_text(chunk=True)[1].text


def test_book_without_chunks_keeps_plain_text_as_one_chunk(tmp_path: Path):
    book_path = tmp_path / "books" / "dev" / "demo_fr_fr.txt"
    text = "Paragraph one.\n\nParagraph two."
    _write(book_path, text)

    book = Book.from_path(book_path, chunks=False)

    assert len(book.get_src_text(chunk=True)) == 1
    assert book.get_src_text(chunk=True)[0].text == text


def test_book_getters_do_not_emit_chunk_tags(tmp_path: Path):
    book_path = tmp_path / "books" / "dev" / "demo_fr_fr.txt"
    _write(book_path, "First paragraph.\n\nSecond paragraph.")

    book = Book.from_path(book_path, chunks=True, chunk_token_limit=5)
    book.set_translation(["First translation.", "Second translation."])

    assert "<chunk>" not in book.get_src_text()
    assert "<chunk>" not in book.get_translation()


def test_save_chunks_writes_json_sidecar_instead_of_tagged_text(tmp_path: Path):
    book_path = tmp_path / "books" / "dev" / "demo_fr_fr.txt"
    _write(book_path, "One.\n\nTwo.")

    book = Book.from_path(book_path, chunks=True, chunk_token_limit=3)
    target = tmp_path / "books" / "dev_chunk" / "demo_fr_fr.jsonl"
    saved_path = book.save_chunks(target)

    assert saved_path == target
    saved = [
        json.loads(line)
        for line in saved_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert saved == [
        {"chunk_id": 1, "word_count": 1, "token_count": 2, "text": "One."},
        {"chunk_id": 2, "word_count": 1, "token_count": 2, "text": "Two."},
    ]


def test_book_skips_empty_paragraphs_when_chunking(tmp_path: Path):
    book_path = tmp_path / "books" / "dev" / "demo_fr_fr.txt"
    # Three real paragraphs separated by *extra* blank lines
    # (split("\n\n") on this produces empty strings between them).
    _write(book_path, "Alpha.\n\n\n\nBeta.\n\n\n\nGamma.")

    book = Book.from_path(book_path, chunks=True, chunk_token_limit=1000)
    chunks = book.get_src_text(chunk=True)

    assert all(chunk.text.strip() for chunk in chunks), (
        f"no chunk should be empty/whitespace, got: {[c.text for c in chunks]}"
    )
