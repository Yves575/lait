import ast
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

mapping = {"YES": 1.0, "MAYBE": 0.5, "NO": 0.0}


def normalize_answer_blob(answer_blob):
    if pd.isna(answer_blob):
        return None

    text = str(answer_blob).strip()
    if not text:
        return None

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        text = text[start : end + 1]
    elif start != -1 and end == -1:
        text = text[start:]

    missing_closing_braces = text.count("{") - text.count("}")
    if missing_closing_braces > 0:
        text = text + ("}" * missing_closing_braces)

    return text


def parse_answers(answer_blob):
    text = normalize_answer_blob(answer_blob)
    if text is None:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed response is not a dictionary")
        return parsed


def litr_score(answer_blob):
    answers = parse_answers(answer_blob)
    vals = [mapping[str(v).strip().upper()] for v in answers.values()]
    return sum(vals) / len(vals) if vals else None


def get_score(df):
    df = df.copy()
    df["LiTransProQA_score"] = df["response"].apply(litr_score)
    grouped = df.groupby("dataset")['LiTransProQA_score'].mean()
    return (grouped.index.tolist(), grouped.to_numpy())



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the CSV dataset to evaluate.",
    )
    return parser.parse_args()

def evaluate_litransproqa(file):
    df = pd.read_csv(file)
    books_name, scores = get_score(df)

    for book_name, score in zip(books_name, scores):
        print(f"LiTransProQA score for the translation {book_name}: {score}")
    print()
    print(f"Mean score: {np.mean(scores)}")

def main():
    args = parse_args()
    evaluate_litransproqa(args.file)

if __name__ == "__main__":
    main()
