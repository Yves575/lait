from scripts.map_paragraph_metrics_to_chunk_review import (
    match_paragraphs_to_chunks,
    metric_means_by_chunk,
    paired_metric_paragraph_indices_by_chunk,
    metric_values_by_chunk,
    normalize_for_match,
    paired_metric_means_by_chunk,
)


def test_normalize_for_match_ignores_punctuation_width_and_case() -> None:
    assert normalize_for_match(" Café -- ＡＢＣ １２３! ") == "caféabc123"


def test_match_paragraphs_to_chunks_and_average_scores() -> None:
    paragraph_rows = [
        {
            "segment_index": 0,
            "source": "First paragraph.",
            "scores": {"comet22": 0.5, "metricx-qe": 4.0},
        },
        {
            "segment_index": 1,
            "source": "Second paragraph with punctuation!",
            "scores": {"comet22": 0.7, "metricx-qe": 6.0},
        },
        {
            "segment_index": 2,
            "source": "Third paragraph",
            "scores": {"comet22": 0.9, "metricx-qe": 8.0},
        },
    ]
    chunk_rows = [
        {"segment_index": 0, "source": "first paragraph\n\nsecond paragraph with punctuation"},
        {"segment_index": 1, "source": "third paragraph."},
    ]

    matches = match_paragraphs_to_chunks(paragraph_rows, chunk_rows, lookahead=2)
    assert [(match.paragraph_index, match.chunk_index) for match in matches] == [
        (0, 0),
        (1, 0),
        (2, 1),
    ]

    means = metric_means_by_chunk(paragraph_rows, matches)
    assert means == {
        0: {"comet22": 0.6, "metricx-qe": 5.0},
        1: {"comet22": 0.9, "metricx-qe": 8.0},
    }


def test_paired_metric_means_use_shared_paragraph_ids_for_shared_metrics() -> None:
    ht_values = {
        0: {
            "cometkiwi": {5: 0.8, 7: 1.0},
            "metricx-qe": {5: 3.0, 6: 5.0, 7: 7.0},
        }
    }
    mt_values = {
        0: {
            "comet22": {5: 0.6, 6: 0.7, 7: 0.8},
            "cometkiwi": {5: 0.2, 6: 0.4, 7: 0.6},
            "metricx-qe": {5: 4.0, 6: 6.0, 7: 8.0},
        }
    }

    means = paired_metric_means_by_chunk({"ht": ht_values, "mt": mt_values})

    assert means["ht"][0]["cometkiwi"] == 0.9
    assert means["mt"][0]["cometkiwi"] == 0.4
    assert means["ht"][0]["metricx-qe"] == 5.0
    assert means["mt"][0]["metricx-qe"] == 6.0
    assert means["mt"][0]["comet22"] == 0.7
    assert paired_metric_paragraph_indices_by_chunk({"ht": ht_values, "mt": mt_values}) == {
        "ht": {
            0: {
                "cometkiwi": [5, 7],
                "metricx-qe": [5, 6, 7],
            }
        },
        "mt": {
            0: {
                "comet22": [5, 6, 7],
                "cometkiwi": [5, 7],
                "metricx-qe": [5, 6, 7],
            }
        },
    }


def test_metric_values_by_chunk_keeps_paragraph_ids() -> None:
    paragraph_rows = [
        {"segment_index": 5, "scores": {"cometkiwi": 0.8}},
        {"segment_index": 6, "scores": {"metricx-qe": 5.0}},
    ]
    matches = [
        type("Match", (), {"paragraph_index": 5, "chunk_index": 0})(),
        type("Match", (), {"paragraph_index": 6, "chunk_index": 0})(),
    ]

    assert metric_values_by_chunk(paragraph_rows, matches) == {
        0: {"cometkiwi": {5: 0.8}, "metricx-qe": {6: 5.0}}
    }
