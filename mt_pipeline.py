import argparse
import json
from collections import defaultdict
from pathlib import Path

from api_model import GPT, APIModel, DeepSeek, Gemini, OpenRouter
from book import Book, BooksList, Chunk
from prompt import build_gemini_message, build_translation_user_prompt

from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_REQUEST_NAME = "structured-output-job-1"
DEFAULT_POLL_INTERVAL = 30
DEFAULT_OUTPUT_FOLDER = Path("books/MT")
DEFAULT_REQUEST_PATH = Path("request/result.jsonl")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_PE = (PROMPTS_DIR / "post-editing_prompt.txt").read_text(encoding="utf-8")

MODEL_REGISTRY = {
    Gemini.name: Gemini,
    GPT.name: GPT,
    OpenRouter.name: OpenRouter,
    DeepSeek.name: DeepSeek,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate one or more text files with Gemini or OpenAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--book_path",
        required=True,
        help="Path to an input .txt file or a directory containing .txt files.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(MODEL_REGISTRY),
        default=Gemini.name,
        help="API provider to use.",
    )
    parser.add_argument(
        "--model_checkpoint",
        required=True,
        type=str,
        help="Provider-specific model name to use.",
    )
    parser.add_argument(
        "--output_folder",
        default=str(DEFAULT_OUTPUT_FOLDER),
        help="Directory where translated text files will be saved.",
    )
    parser.add_argument(
        "--chunks",
        action="store_true",
        help="Split text into chunks for translation. Omit to send the whole text at once.",
    )
    parser.add_argument(
        "--chunk_path",
        default=None,
        help="Path to the chunk folder, either to write or to load chunks",
    )
    parser.add_argument(
        "--reasoning_effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort level for OpenAI models.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of books to translate in parallel.",
    )
    args, _ = parser.parse_known_args()
    if args.provider == Gemini.name:
        parser.add_argument(
            "--request_name",
            default=DEFAULT_REQUEST_NAME,
            help="Batch job request name. Used by Gemini only.",
        )
        parser.add_argument(
            "--poll_interval",
            type=int,
            default=DEFAULT_POLL_INTERVAL,
            help="Seconds between job status checks. Used by Gemini only.",
        )
        parser.add_argument(
            "--request_path",
            default=str(DEFAULT_REQUEST_PATH),
            help="Path of the .jsonl file request will be created",
        )
        parser.add_argument(
            "--given_request",
            default=None,
            help="Path to the request",
        )
    args = parser.parse_args()

    if args.provider == Gemini.name:
        if args.poll_interval <= 0:
            parser.error("--poll_interval must be a positive integer.")

    return args

def _translate_one_book(book, model, folder_path):
    main_folder = Path(folder_path)
    pe_folder = Path(f"{folder_path}/before_PE")

    def save(book: Book, target: Path) -> Path:
        target.mkdir(parents=True, exist_ok=True)
        file_path = target / f"{book.name}_en.txt"
        file_path.write_text(book.get_translation(), encoding="utf-8")
        return file_path
    

    final_path = main_folder / f"{book.name}_en.txt"
    if final_path.exists():
        print(f"Skipping {book.name}, translation already exists at {final_path}")
        return
    print(
        f"Starting the translation of {book.name}, "
        f"number of chunks : {len(book.get_src_text(chunk=True))}, "
        f"lang : {book.source_language}"
    )
    
    chunk_translations = [
            model.direct_message(
                message=build_translation_user_prompt(chunk_src.text, book.source_language),
            )
            for chunk_src in book.get_src_text(chunk=True)
    ]
    book.set_translation(chunk_translations)
    print(f"Finished pre-tranlsation for {book.name}\nSaving book before PE...\n\n")
    save(book, pe_folder)
    src_lines = book.get_src_text().splitlines()
    tgt_lines = book.get_translation().splitlines()
    print(f"[{book.name}] Source: {len(src_lines)} lines, excerpt: {src_lines[0][:100]}...")
    print(f"[{book.name}] Translation: {len(tgt_lines)} lines, excerpt: {tgt_lines[0][:100]}...")
    print("Starting post editing...")
    revised_translation = model.direct_message(
            message=PROMPT_PE.format(
                source_lang=book.source_language,
                source=book.get_src_text(),
                text=book.get_translation(),
            )
    )
    book.translation = [Chunk(revised_translation)]
    print(f"Finished tranlsation for {book.name}\nSaving book...\n\n")
    save(book, main_folder)

def direct_translation(books_list: BooksList, model: APIModel, folder: str | Path, max_workers=1) -> None:
    if max_workers <= 1:
        for book in books_list.books:
            _translate_one_book(book, model, folder)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_translate_one_book, book, model, folder): book
                    for book in books_list.books
                }
                for future in as_completed(futures):
                    book = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"ERROR: {book.name} failed: {exc}")


def post_edit_translation(book: Book, model:APIModel) -> None:##
    revised_translation = model.direct_message(
            message = PROMPT_PE.format(
                    source_lang=book.source_language,
                    source=book.get_src_text(),
                    text=book.get_translation(),
                ))
    book.translation = [Chunk(revised_translation)]

def get_books_from_request(request_path, books):
    reconstructed_books: dict[str, list[tuple[int, str]]] = defaultdict(list)
    # Get the books from the request file
    with request_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            name_key = row["key"]
            book_name, chunk_index = name_key.rsplit("_", 1)
            try:
                text = row["response"]["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"Gemini batch response for key '{name_key}' does not contain text."
                ) from exc
            reconstructed_books[book_name].append((int(chunk_index), text))

    # Sort the text in books per index
    dict_books = {
        book_name: [
            text
            for _, text in sorted(chunks, key=lambda chunk: chunk[0])
        ]
        for book_name, chunks in reconstructed_books.items()
    }

    # Save the books translation result
    for book in books:
        if book.name not in dict_books:
            raise RuntimeError(
                f"No Gemini batch output was found for book '{book.name}'."
            )
        book.set_translation(dict_books[book.name])


def main() -> None:
    
    args = parse_args()

    model_class = MODEL_REGISTRY[args.provider]
    model = model_class(model_checkpoint=args.model_checkpoint, reasoning_effort=args.reasoning_effort)

    list_books = BooksList(path=args.book_path, chunks=args.chunks, chunk_path=args.chunk_path)

    direct_translation(list_books, model, args.output_folder, max_workers=args.parallel)

    print("done")


if __name__ == "__main__":
    main()
