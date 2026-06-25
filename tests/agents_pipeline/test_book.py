from pathlib import Path

from agents_pipeline.book import Book


def test_agentic_book_keeps_oversized_paragraph_intact():
    long_paragraph = " ".join(["word"] * 3500)
    text = f"{long_paragraph}\n\nShort ending paragraph."
    book = Book(Path("demo_fr.txt"), text, True)

    assert len(book.src_text) == 2
    assert book.src_text[0].text == long_paragraph
    assert "Short ending paragraph." in book.src_text[1].text
    assert Book._token_count(book.src_text[0].text) > Book.CHUNK_TOKEN_LIMIT


def test_agentic_book_keeps_multiple_paragraphs_under_limit_together():
    paragraphs = [
        " ".join(["alpha"] * 120),
        " ".join(["beta"] * 120),
        " ".join(["gamma"] * 120),
    ]
    text = "\n\n".join(paragraphs)
    book = Book(Path("demo_fr.txt"), text, True)

    assert len(book.src_text) == 1
    assert "alpha" in book.src_text[0].text
    assert "gamma" in book.src_text[0].text


def test_agentic_book_uses_existing_chunk_tags_when_present():
    text = "<chunk>\nFirst chunk.\n</chunk>\n\n<chunk>\nSecond chunk.\n</chunk>"
    book = Book(Path("demo_fr.txt"), text, True)

    assert len(book.src_text) == 2
    assert book.src_text[0].text == "First chunk."
    assert book.src_text[1].text == "Second chunk."
