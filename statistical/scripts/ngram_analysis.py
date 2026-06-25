from __future__ import annotations

from collections import Counter
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd

from statistical.config import HT_SYSTEM_NAME, PLOT_DIRS, TABLE_DIRS, ensure_output_dirs
from statistical.utils.loader import load_corpus, mt_systems, pipeline_texts
from statistical.utils.plotting import set_style
from statistical.utils.text_processing import content_ngram_counter, ngram_counter, normalized_words


TOP_K = 30
ALL_MT_LABEL = "All MT (mean system)"


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _combined_counter(texts: list[str], n: int, content_only: bool) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    counter = content_ngram_counter if content_only else ngram_counter
    for text in texts:
        counts.update(counter(normalized_words(text), n))
    return counts


def _frequency_table(counts: Counter[tuple[str, ...]], n: int, label: str, mode: str) -> pd.DataFrame:
    total = sum(counts.values())
    rows = []
    for gram, count in counts.items():
        rows.append({
            "group": label,
            "mode": mode,
            "n": n,
            "ngram": " ".join(gram),
            "count": count,
            "frequency": count / total if total else 0.0,
            "freq_per_10k": count / total * 10000 if total else 0.0,
        })
    return pd.DataFrame(rows)


def _delta_table(ht: pd.DataFrame, mt: pd.DataFrame, pipeline: str, n: int, mode: str) -> pd.DataFrame:
    merged = ht[["ngram", "count", "frequency", "freq_per_10k"]].merge(
        mt[["ngram", "count", "frequency", "freq_per_10k"]],
        on="ngram",
        how="outer",
        suffixes=("_ht", "_mt"),
    ).fillna(0)
    merged.insert(0, "pipeline", pipeline)
    merged.insert(1, "mode", mode)
    merged.insert(2, "n", n)
    merged["delta_frequency"] = merged["frequency_mt"] - merged["frequency_ht"]
    merged["delta_per_10k"] = merged["freq_per_10k_mt"] - merged["freq_per_10k_ht"]
    return merged.sort_values("delta_per_10k", key=lambda series: series.abs(), ascending=False)


def _plot_delta(delta: pd.DataFrame, pipeline: str, n: int, mode: str, suffix: str) -> None:
    set_style()
    subset = delta.head(TOP_K).iloc[::-1]
    colors = ["#c0392b" if value > 0 else "#f4b000" for value in subset["delta_per_10k"]]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(subset) + 1.5)))
    ax.barh(subset["ngram"], subset["delta_per_10k"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"{pipeline} vs HT: Top {n}-gram Frequency Deltas ({mode})")
    ax.set_xlabel("Delta frequency per 10k n-grams (MT pipeline - HT)")
    ax.set_ylabel("")
    fig.tight_layout()
    output = PLOT_DIRS["ngrams"] / f"{_safe_name(pipeline)}_{n}gram_delta{suffix}.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _mean_mt_delta_table(
    ht_counts: Counter[tuple[str, ...]],
    mt_counts: list[Counter[tuple[str, ...]]],
    n: int,
    mode: str,
) -> pd.DataFrame:
    system_count = len(mt_counts)
    aggregates: dict[tuple[str, ...], list[float]] = {}
    for counts in mt_counts:
        total = sum(counts.values())
        if not total:
            continue
        for gram, count in counts.items():
            frequency = count / total
            per_10k = frequency * 10000
            bucket = aggregates.setdefault(gram, [0.0, 0.0, 0.0, 0.0, 0.0])
            bucket[0] += float(count)
            bucket[1] += frequency
            bucket[2] += per_10k
            bucket[3] += per_10k**2
            bucket[4] += 1

    ht_total = sum(ht_counts.values())
    for gram in ht_counts:
        aggregates.setdefault(gram, [0.0, 0.0, 0.0, 0.0, 0.0])

    rows = []
    for gram, values in aggregates.items():
        count_sum, frequency_sum, per_10k_sum, per_10k_sq_sum, systems_with_ngram = values
        ht_count = float(ht_counts.get(gram, 0))
        ht_frequency = ht_count / ht_total if ht_total else 0.0
        ht_per_10k = ht_frequency * 10000
        mt_frequency_mean = frequency_sum / system_count if system_count else 0.0
        mt_per_10k_mean = per_10k_sum / system_count if system_count else 0.0
        variance = per_10k_sq_sum / system_count - mt_per_10k_mean**2 if system_count else 0.0
        rows.append({
            "pipeline": ALL_MT_LABEL,
            "mode": mode,
            "n": n,
            "ngram": " ".join(gram),
            "count_ht": ht_count,
            "frequency_ht": ht_frequency,
            "freq_per_10k_ht": ht_per_10k,
            "count_mt_mean": count_sum / system_count if system_count else 0.0,
            "frequency_mt_mean": mt_frequency_mean,
            "freq_per_10k_mt_mean": mt_per_10k_mean,
            "freq_per_10k_mt_std": sqrt(max(variance, 0.0)),
            "mt_system_count": system_count,
            "mt_systems_with_ngram": int(systems_with_ngram),
            "delta_frequency": mt_frequency_mean - ht_frequency,
            "delta_per_10k": mt_per_10k_mean - ht_per_10k,
        })
    return pd.DataFrame(rows).sort_values(
        "delta_per_10k",
        key=lambda series: series.abs(),
        ascending=False,
    )


def _plot_all_mt_delta(delta: pd.DataFrame, n: int, mode: str, suffix: str) -> None:
    set_style()
    subset = delta.head(TOP_K).iloc[::-1]
    colors = ["#c0392b" if value > 0 else "#f4b000" for value in subset["delta_per_10k"]]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(subset) + 1.5)))
    ax.barh(subset["ngram"], subset["delta_per_10k"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"All MT vs HT: Top {n}-gram Frequency Deltas ({mode})")
    ax.set_xlabel("Delta frequency per 10k n-grams (mean MT system - HT)")
    ax.set_ylabel("")
    fig.tight_layout()
    output = PLOT_DIRS["ngrams"] / f"all_mt_{n}gram_delta{suffix}.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def run() -> pd.DataFrame:
    ensure_output_dirs()
    _, systems, pipelines = load_corpus()
    grouped = pipeline_texts(systems, pipelines)

    all_frequencies: list[pd.DataFrame] = []
    all_deltas: list[pd.DataFrame] = []
    all_mt_deltas: list[pd.DataFrame] = []
    modes = {
        "content": {"content_only": True, "suffix": ""},
        "raw": {"content_only": False, "suffix": "_raw"},
    }
    for mode, options in modes.items():
        mode_frequencies = []
        mode_deltas = []
        mode_all_mt_deltas = []
        for n in (1, 2, 3):
            system_counters = {
                system: _combined_counter(
                    [systems[system][book] for book in sorted(systems[system])],
                    n,
                    content_only=bool(options["content_only"]),
                )
                for system in mt_systems(systems)
            }
            frequency_tables = {
                group: _frequency_table(
                    _combined_counter(texts, n, content_only=bool(options["content_only"])),
                    n,
                    group,
                    mode,
                )
                for group, texts in grouped.items()
            }
            mode_frequencies.extend(frequency_tables.values())
            ht_table = frequency_tables[HT_SYSTEM_NAME]
            ht_counts = _combined_counter(
                grouped[HT_SYSTEM_NAME],
                n,
                content_only=bool(options["content_only"]),
            )
            all_mt_delta = _mean_mt_delta_table(ht_counts, list(system_counters.values()), n, mode)
            all_mt_delta.to_csv(
                TABLE_DIRS["ngrams"] / f"all_mt_{n}gram_delta_{mode}.csv",
                index=False,
            )
            _plot_all_mt_delta(all_mt_delta, n, mode, suffix=str(options["suffix"]))
            mode_all_mt_deltas.append(all_mt_delta)
            for pipeline in sorted(group for group in grouped if group != HT_SYSTEM_NAME):
                delta = _delta_table(ht_table, frequency_tables[pipeline], pipeline, n, mode)
                delta.to_csv(
                    TABLE_DIRS["ngrams"] / f"{_safe_name(pipeline)}_{n}gram_delta_{mode}.csv",
                    index=False,
                )
                _plot_delta(delta, pipeline, n, mode, suffix=str(options["suffix"]))
                mode_deltas.append(delta)

        frequencies_for_mode = pd.concat(mode_frequencies, ignore_index=True)
        deltas_for_mode = pd.concat(mode_deltas, ignore_index=True)
        all_mt_deltas_for_mode = pd.concat(mode_all_mt_deltas, ignore_index=True)
        frequencies_for_mode.to_csv(
            TABLE_DIRS["ngrams"] / f"ngram_frequencies_{mode}.csv",
            index=False,
        )
        deltas_for_mode.to_csv(TABLE_DIRS["ngrams"] / f"ngram_deltas_vs_ht_{mode}.csv", index=False)
        all_mt_deltas_for_mode.to_csv(
            TABLE_DIRS["ngrams"] / f"all_mt_ngram_deltas_vs_ht_{mode}.csv",
            index=False,
        )
        all_frequencies.append(frequencies_for_mode)
        all_deltas.append(deltas_for_mode)
        all_mt_deltas.append(all_mt_deltas_for_mode)

    frequencies = pd.concat(all_frequencies, ignore_index=True)
    deltas = pd.concat(all_deltas, ignore_index=True)
    all_mt = pd.concat(all_mt_deltas, ignore_index=True)
    frequencies.to_csv(TABLE_DIRS["ngrams"] / "ngram_frequencies.csv", index=False)
    deltas.to_csv(TABLE_DIRS["ngrams"] / "ngram_deltas_vs_ht_all_modes.csv", index=False)
    deltas[deltas["mode"] == "content"].to_csv(
        TABLE_DIRS["ngrams"] / "ngram_deltas_vs_ht.csv",
        index=False,
    )
    all_mt.to_csv(TABLE_DIRS["ngrams"] / "all_mt_ngram_deltas_vs_ht_all_modes.csv", index=False)
    all_mt[all_mt["mode"] == "content"].to_csv(
        TABLE_DIRS["ngrams"] / "all_mt_ngram_deltas_vs_ht.csv",
        index=False,
    )
    return deltas


if __name__ == "__main__":
    run()
