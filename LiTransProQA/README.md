# LiTransProQA Support Code

This directory contains supporting LiTransProQA code used for the literary MT
evaluation workflow described in the preprint.

The original LiTransProQA project is a question-answering-based literary
translation evaluation metric. This public release keeps the code needed to
inspect the method integration, but does not include text-bearing datasets,
notebooks with out-of-scope metric experiments, or local evaluation inputs.

## Included

- `finetuning_method/`: fine-tuning and inference utilities.
- `prompting_method/`: prompt-building and prompt-scoring utilities.
- `config/question_weights.csv`: public LiTransProQA question weights used by
  `mt_eval.py`.
- `SOTA_metric/`: supporting metric code.
- `Fig/`: upstream summary figure assets.
- `LICENSE`: upstream license.

## Withheld Or Omitted

- Source/target CSV datasets and local run-input tables are not included on
  GitHub.
- Local notebooks and intermediate experiment outputs are not included.
- Reproduction that needs source texts or human translations requires
  controlled-access data.

See `../docs/DATA_ACCESS.md` for the repository-wide controlled-access policy.

## Upstream Citation

```bibtex
@inproceedings{zhang-etal-2025-litransproqa,
    title = "{L}i{T}rans{P}ro{QA}: An {LLM}-based Literary Translation Evaluation Metric with Professional Question Answering",
    author = "Zhang, Ran and Zhao, Wei and Macken, Lieve and Eger, Steffen",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year = "2025",
    url = "https://aclanthology.org/2025.emnlp-main.1482/",
    doi = "10.18653/v1/2025.emnlp-main.1482"
}
```
