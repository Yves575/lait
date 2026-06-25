from pathlib import Path

from analysis.human_eval.analyze_mt_detection_surface_features import (
    aligned_chunk_feature_deltas,
    book_id_to_mt_stem,
    count_single_newlines,
    extract_text_features,
    paired_feature_deltas,
    read_translation_text,
    resolve_ht_text_path,
    resolve_mt_text_path,
)


def test_book_id_to_mt_stem_canonicalizes_fixed_needle() -> None:
    assert (
        book_id_to_mt_stem("polish_eval_FIXED_needle_s_eye")
        == "needle_s_eye_pl_en"
    )
    assert book_id_to_mt_stem("french_eval_make_me_famous") == "make_me_famous_fr_en"


def test_extract_text_features_counts_surface_artifacts() -> None:
    text = 'Hello—there. "Quoted" text – with /nn/ artifact...\nSingle newline.'
    features = extract_text_features(text)
    assert features["em_dash_rate"] > 0
    assert features["en_dash_rate"] > 0
    assert features["dash_rate"] >= features["em_dash_rate"] + features["en_dash_rate"]
    assert features["ellipsis_rate"] > 0
    assert features["slash_n_artifact_count"] == 1
    assert features["mixed_dash_style"] == 1
    assert features["surface_artifact_index"] > 0


def test_count_single_newlines_ignores_blank_line_separator() -> None:
    assert count_single_newlines("a\nb\n\nc") == 1


def test_resolve_mt_text_path_prefers_chunks(tmp_path: Path) -> None:
    chunks = tmp_path / "books" / "MT_chunks"
    chunks.mkdir(parents=True)
    expected = chunks / "make_me_famous_fr_en.jsonl"
    expected.write_text('{"chunk_id": 1, "text": "x"}\n', encoding="utf-8")
    assert resolve_mt_text_path("french_eval_make_me_famous", tmp_path) == expected


def test_resolve_ht_text_path_uses_eval_jsonl(tmp_path: Path) -> None:
    ht_dir = tmp_path / "books" / "HT" / "eval"
    ht_dir.mkdir(parents=True)
    expected = ht_dir / "make_me_famous_fr_en.jsonl"
    expected.write_text('{"chunk_id": 1, "text": "x"}\n', encoding="utf-8")
    assert resolve_ht_text_path("french_eval_make_me_famous", ht_dir) == expected


def test_read_translation_text_orders_jsonl_chunks(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        '{"chunk_id": 2, "text": "second"}\n{"chunk_id": 1, "text": "first"}\n',
        encoding="utf-8",
    )
    assert read_translation_text(path) == "first\n\nsecond"


def test_paired_feature_deltas_include_difference_distance_and_ratio() -> None:
    deltas = paired_feature_deltas(
        {"mtld": 12.0, "ttr": 0.4},
        {"mtld": 8.0, "ttr": 0.2},
    )
    assert deltas["mt_minus_ht_mtld"] == 4.0
    assert deltas["abs_mt_minus_ht_mtld"] == 4.0
    assert deltas["mt_div_ht_mtld"] == 1.5
    assert deltas["mt_div_ht_ttr"] == 2.0


def test_aligned_chunk_feature_deltas_aggregate_local_differences() -> None:
    mt_chunks = [
        (1, "A—dash sentence."),
        (2, '"Quote" only.'),
    ]
    ht_chunks = [
        (1, "A sentence."),
        (2, '"Quote" — with dash.'),
    ]
    features = aligned_chunk_feature_deltas(mt_chunks, ht_chunks)
    assert features["aligned_chunk_count"] == 2
    assert features["mean_abs_chunk_delta_em_dash_rate"] > 0
    assert features["mean_abs_chunk_delta_dash_rate"] > 0
    assert features["share_chunks_mt_more_em_dash"] == 0.5
    assert features["share_chunks_dash_style_mismatch"] == 1.0
