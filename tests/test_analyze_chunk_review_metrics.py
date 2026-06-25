import pandas as pd

from scripts.analyze_chunk_review_metrics import kendall_summary


def test_kendall_summary_uses_direct_ht_mt_pair_formula() -> None:
    df = pd.DataFrame(
        [
            {"metric": "metricx-qe", "human": "HT", "metric_pref": "HT"},
            {"metric": "metricx-qe", "human": "MT", "metric_pref": "MT"},
            {"metric": "metricx-qe", "human": "HT", "metric_pref": "MT"},
            {"metric": "metricx-qe", "human": "MT", "metric_pref": "HT"},
            {"metric": "metricx-qe", "human": "HT", "metric_pref": "tie"},
        ]
    )

    summary = kendall_summary(
        df,
        human_col="human",
        metric_col="metric_pref",
        group_cols=["metric"],
        label="test",
    )

    row = summary.iloc[0]
    assert row["total_pairs"] == 5
    assert row["concordant_pairs"] == 2
    assert row["discordant_pairs"] == 2
    assert row["tie_pairs"] == 1
    assert row["kendall_tau"] == 0.0


def test_kendall_summary_keeps_ties_in_denominator() -> None:
    df = pd.DataFrame(
        [
            {"metric": "litransproqa", "human": "HT", "metric_pref": "HT"},
            {"metric": "litransproqa", "human": "HT", "metric_pref": "MT"},
            {"metric": "litransproqa", "human": "tie", "metric_pref": "HT"},
            {"metric": "litransproqa", "human": "MT", "metric_pref": "tie"},
        ]
    )

    summary = kendall_summary(
        df,
        human_col="human",
        metric_col="metric_pref",
        group_cols=["metric"],
        label="test",
    )

    row = summary.iloc[0]
    assert row["total_pairs"] == 4
    assert row["concordant_pairs"] == 1
    assert row["discordant_pairs"] == 1
    assert row["tie_pairs"] == 2
    assert row["kendall_tau"] == 0.0
