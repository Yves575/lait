import pandas as pd
import json

df = pd.read_csv("../datasets/my_input.csv") #please merge back the sourece, and target columns based on the ID
print(df.shape)
# Load template file
with open("template/template_baseline.txt", "r") as f:
    template = f.read()

with open("template/QA_final.txt", "r") as f:
    QA = f.read()

df_qa = [template.format(
    source=row["src"],
    translation=row["tgt"],
    questions=QA
) for index, row in df.iterrows()]
df["QA"] = df_qa
df[["src", "tgt", "QA", "pair", "model", "dataset"]].to_csv("final_set/final_set_with_QA.csv", index=False)



