import pandas as pd
import os
with open("model_list.txt", "r") as f:
    models = f.readlines()
models = [model.replace("\n", "") for model in models]
print(models)

for model in models:    #"deepseek/deepseek-chat-v3-0324", 
    print(model)
    os.system(f"python prompt_openrouter.py \
              --file final_set/final_set_with_QA.csv \
              --model {model} \
              --content-column QA \
              --temperature 0.3 \
              --output-dir final_results/")