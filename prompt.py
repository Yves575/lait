from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from book import Book

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_TRANSLATOR = (PROMPTS_DIR / "message_prompt.txt").read_text(encoding="utf-8")
TARGET_LANGUAGE = "English"


def build_translation_user_prompt(chunk_text: str, source_lang: str) -> str:
    return (
        f"{PROMPT_TRANSLATOR.format(text=chunk_text, source_lang=source_lang, target_lang=TARGET_LANGUAGE)}"
    )

def build_gemini_message(books: list[Book], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for book in books:
            for i, chunk_text in enumerate(book.src_text, start=1):
                req = {
                    "key": f"{book.name}_{i}",
                    "request": {
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": build_translation_user_prompt(
                                            chunk_text.text,
                                            book.source_language,
                                        )
                                    }
                                ],
                            }
                        ]
                    },
                }
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
