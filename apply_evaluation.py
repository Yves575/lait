from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from api_model import GPT, Gemini

MODEL_REGISTRY = {
    Gemini.name: Gemini,
    GPT.name: GPT,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process prompts from a CSV file.")
    parser.add_argument("--file", type=str, 
                        default="prompt_final/prompts.csv", help="Path to the CSV file containing prompts",)
    parser.add_argument("--model", type=str, choices=sorted(MODEL_REGISTRY),
                        default=Gemini.name, help="Model provider to use",)
    parser.add_argument("--model-name", type=str, 
                        default=None, help="Optional API model name override",)
    parser.add_argument("--content-column", type=str,
                        default="QA", help="Column name containing the content",)
    parser.add_argument("--output-dir", type=str,
                        default="LiTransProQA/prompting_method/final_results/", help="Output directory",)
    parser.add_argument("--test-size", type=int, 
                        default=None, help="Only process the first N prompts",)
    return parser.parse_args(argv)


def get_model(provider: str, model_checkpoint: str | None = None) -> Gemini | GPT:
    try:
        model_cls = MODEL_REGISTRY[provider]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model provider '{provider}'. Expected one of: {choices}.") from exc
    return model_cls(model_checkpoint=model_checkpoint)


def evaluate_file(file_path: str | Path, provider: str = Gemini.name, content_column: str = "QA",
    output_dir: str | Path = "LiTransProQA/prompting_method/final_results/", test_size: int | None = None,
    model_checkpoint: str | None = None,) -> Path:
    
    file_path = Path(file_path)
    output_dir = Path(output_dir)

    df = pd.read_csv(file_path)
    if content_column not in df.columns:
        raise ValueError(f"Column '{content_column}' not found in {file_path}.")
    if test_size is not None:
        df = df.head(test_size).copy()

    model = get_model(provider, model_checkpoint=model_checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file_path.name}"

    df["response"] = ""
    df.to_csv(output_file, index=False)

    for idx, row in df.iterrows():
        response = model.direct_message(message=row[content_column])
        df.at[idx, "response"] = response
        df.to_csv(output_file, index=False)

    return output_file


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    output_file = evaluate_file(
        file_path=args.file,
        provider=args.model,
        content_column=args.content_column,
        output_dir=args.output_dir,
        test_size=args.test_size,
        model_checkpoint=args.model_checkpoint,
    )
    print(f"done: {output_file}")
    return output_file


if __name__ == "__main__":
    main()
