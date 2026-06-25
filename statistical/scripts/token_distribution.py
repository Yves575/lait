from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from statistical.config import HT_SYSTEM_NAME, PLOT_DIRS, TABLE_DIRS, ensure_output_dirs
from statistical.utils.loader import load_corpus, pipeline_texts
from statistical.utils.plotting import set_style
from statistical.utils.text_processing import normalized_words


MAX_RANK = 5000


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _token_rank_table(texts: list[str], group: str) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(normalized_words(text))
    total = sum(counts.values())
    rows = []
    for rank, (token, count) in enumerate(counts.most_common(MAX_RANK), start=1):
        rows.append({
            "group": group,
            "rank": rank,
            "token": token,
            "count": count,
            "frequency": count / total if total else 0.0,
        })
    return pd.DataFrame(rows)


def _plot_rank(ht: pd.DataFrame, mt: pd.DataFrame, pipeline: str) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ht["rank"], ht["frequency"], label=HT_SYSTEM_NAME, color="#111111", linewidth=1.8)
    ax.plot(mt["rank"], mt["frequency"], label=pipeline, color="#c0392b", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Token Frequency Rank Distribution: {pipeline} vs HT")
    ax.set_xlabel("Token rank (log scale)")
    ax.set_ylabel("Relative frequency (log scale)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(PLOT_DIRS["token_distribution"] / f"{_safe_name(pipeline)}_token_rank.png", bbox_inches="tight")
    plt.close(fig)


def run() -> pd.DataFrame:
    ensure_output_dirs()
    _, systems, pipelines = load_corpus()
    grouped = pipeline_texts(systems, pipelines)
    tables = {group: _token_rank_table(texts, group) for group, texts in grouped.items()}

    combined = pd.concat(tables.values(), ignore_index=True)
    combined.to_csv(TABLE_DIRS["token_distribution"] / "token_rank_frequencies.csv", index=False)

    ht = tables[HT_SYSTEM_NAME]
    for pipeline in sorted(group for group in grouped if group != HT_SYSTEM_NAME):
        _plot_rank(ht, tables[pipeline], pipeline)
        tables[pipeline].to_csv(TABLE_DIRS["token_distribution"] / f"{_safe_name(pipeline)}_token_ranks.csv", index=False)
    ht.to_csv(TABLE_DIRS["token_distribution"] / "ht_token_ranks.csv", index=False)
    return combined


if __name__ == "__main__":
    run()

