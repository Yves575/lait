# Statistical HT/MT Analysis

This module generates research-style tables and plots comparing human translations (HT) with machine translation (MT) systems.

Public-release note: the aggregate outputs are included where possible, but the
HT and source-text inputs are withheld from GitHub. Full reruns require
controlled-access data.

## Inputs

Expected layout:

```text
books/
  dev/*.txt
  eval/*.txt
  HT/*.txt
  MT/<pipeline>/<model>/*.txt
  MT/<pipeline>/*.txt
```

Filenames must follow `<book_name>_<src_lang>_<tgt_lang>.txt`. Matching is done by `<book_name>`, so all comparisons are restricted to books present in HT and the relevant MT systems. Pipelines with a single direct set of `.txt` files, such as `MT/pipeline3/*.txt`, are treated as one implicit system.

## Run

```bash
python -m statistical.run_all
```

If SBERT dependencies or model downloads are not available:

```bash
python -m statistical.run_all --skip-sbert
```

## Outputs

Generated outputs are written under `statistical/outputs/`. Plot artifacts from
earlier broader analyses are not part of this public branch; reruns with
controlled-access data can regenerate local plots and tables as needed.

SBERT embeddings are cached under `statistical/outputs/cache/` when that path is
available locally.

## Additional Analyses

The extended scripts aggregate MT files by pipeline (`P1`, `P2`, etc.) and compare each pipeline against `Human (HT)`.

- `ngram_analysis.py`: computes unigram, bigram, and trigram frequencies, then plots the largest frequency deltas versus HT. Per-pipeline plots pool texts within each pipeline and normalize by n-grams per 10k. The `all_mt_*` plots compare HT against the equal-weight mean normalized frequency across MT systems, which prevents the multiple MT outputs from dominating the single HT by raw count. The default plot names use a content-focused view that removes n-grams made only of English stopwords; raw unfiltered outputs are also saved with `_raw` filenames.
- `token_distribution.py`: builds Zipf-style token frequency rank curves for HT versus each MT pipeline.
- `lexical_diversity.py`: computes TTR and MTLD over ordered 500-token windows and writes both window-level and system-level summaries.
- `repetition_analysis.py`: computes unigram, bigram, and trigram repetition rates within text chunks and plots HT/MT pipeline distributions.

Each script can also be run directly, for example:

```bash
python -m statistical.scripts.ngram_analysis
python -m statistical.scripts.token_distribution
python -m statistical.scripts.lexical_diversity
python -m statistical.scripts.repetition_analysis
```
