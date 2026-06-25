from pathlib import Path
import pandas as pd

def build_final_set(base_dir: Path, input_csv: Path, output_path: str = "final_set_with_QA.csv"):
    input_path = Path(input_csv)
    if not input_path.exists():
        input_path = base_dir / "../datasets" / input_csv
    df = pd.read_csv(input_path)

    with open(base_dir / "template" / "template_baseline.txt", "r", encoding="utf-8") as f:
        template = f.read()

    with open(base_dir / "template" / "QA_final.txt", "r", encoding="utf-8") as f:
        qa = f.read()

    df_qa = [
        template.format(
            source=row["src"],
            translation=row["tgt"],
            questions=qa
        )
        for _, row in df.iterrows()
    ]

    df["QA"] = df_qa

    output_dir = base_dir / "final_set"
    output_dir.mkdir(exist_ok=True)

    df[["src", "tgt", "QA", "pair", "model", "dataset"]].to_csv(
        output_dir / output_path,
        index=False
    )

def main():
    base_dir = Path(__file__).resolve().parent.parent
    build_final_set(base_dir, "my_input.csv")

if __name__ == "__main__":
    main()
