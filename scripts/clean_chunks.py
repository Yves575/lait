"""Remove <chunk> and </chunk> tags from all .txt files in books/HT/ and save to books/HT_reviewed/."""

import os

INPUT_DIR = "books/eval"
OUTPUT_DIR = "books/eval_chunk"


def clean_chunks(text: str) -> str:
    """Replace <chunk> and </chunk> tags with a newline."""
    return text.replace("<chunk>", "\n").replace("</chunk>", "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    if not txt_files:
        print(f"No .txt files found in {INPUT_DIR}")
        return

    for filename in txt_files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        cleaned = clean_chunks(text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"Saved: {output_path}")

    print(f"\nDone. {len(txt_files)} file(s) processed.")


if __name__ == "__main__":
    main()
