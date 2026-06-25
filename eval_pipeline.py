import argparse
from pathlib import Path
import re, csv

from LiTransProQA.prompting_method.build_dataset_reviwed import build_final_set
from LiTransProQA.prompting_method.eval import evaluate_litransproqa
from api_model import GPT, Gemini
from apply_evaluation import evaluate_file
from book import BooksList

MODEL_REGISTRY = {
    Gemini.name: Gemini,
    GPT.name: GPT,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one or more text files with Gemini or OpenAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--provider", choices=sorted(MODEL_REGISTRY), 
                        default=Gemini.name, help="API provider to use.",)
    parser.add_argument("--model_checkpoint", type=str,
                        required=True, help="Provider-specific model name to use.",
    )
    parser.add_argument("--source_path", type=str, 
                        default="request/source_texts/", help="Controlled-access source text path",)
    parser.add_argument("--target_path", type=str, 
                        default="request/translations/", help="Controlled-access translation path",)
    parser.add_argument("--csv_path", type=str, 
                        default="litransproqa_input.csv", help="Name of the scratch CSV file to create",)
    args = parser.parse_args()
    
    return args


def create_dataset(csv_path: str, source_path: Path, target_path: Path, provider: str,
                    output_dir: str = "request/litransproqa/") -> Path:

    source_path = Path(source_path)
    target_path = Path(target_path)

    src_books = BooksList(path=source_path, chunks=False)
    target_books = BooksList(path=target_path, chunks=False)

    src_books.load_translations_from(target_books)

    input_path = Path(output_dir) / csv_path
    input_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["src", "tgt", "pair", "model", "dataset"])
        for book in src_books.books:
            pair = f"{book.source_language}-{"en"}"
            for src_chunk, trg_chunk in zip(book.src_text, book.translation):
                writer.writerow([src_chunk.text, trg_chunk.text, pair, provider, book.name])

    return input_path

def main():
    args = parse_args()
    csv_dataset_path = create_dataset(args.csv_path, args.source_path, args.target_path, args.provider)
    print(f"CSV created in : \n{csv_dataset_path}\n\n")
    Path("LiTransProQA/prompting_method/final_set/").mkdir(parents=True, exist_ok=True)

    base_dir = Path("LiTransProQA/prompting_method/")
    final_set_path = f"final_set_{args.csv_path}.csv"
    build_final_set(base_dir, str(csv_dataset_path), final_set_path)
    print(f"Final prompt created in : \n{base_dir}/final_set/{final_set_path}\n\n")

    output_path = evaluate_file(
        file_path=f"LiTransProQA/prompting_method/final_set/final_set_{args.csv_path}.csv",
        provider=args.provider,
        model_checkpoint=args.model_checkpoint,
        content_column="QA",
    )
    print(f"Result CSV file saved in :\n{output_path}\n\n")

    evaluate_litransproqa(output_path)


if __name__ == "__main__":
    main()
