import pytest

from analysis.human_eval.analyze_part2_chunk_paired_features import (
    book_id_to_chunk_stem,
    extract_text_features,
    jsonl_chunk_id_for_part2,
    paired_feature_deltas,
)


def test_book_id_to_chunk_stem_canonicalizes_fixed_needle() -> None:
    assert (
        book_id_to_chunk_stem("polish_eval_FIXED_needle_s_eye")
        == "needle_s_eye_pl_en"
    )
    assert (
        book_id_to_chunk_stem("japanese_eval_hooked_a_novel_of_obsession")
        == "hooked_a_novel_of_obsession_ja_en"
    )
    assert book_id_to_chunk_stem("french_eval_make_me_famous") == "make_me_famous_fr_en"


def test_jsonl_chunk_id_for_part2_prefers_zero_to_one_based_mapping() -> None:
    jsonl_id, note = jsonl_chunk_id_for_part2("0", {1, 2, 3})
    assert jsonl_id == 1
    assert note == "part2_zero_based_jsonl_one_based"


def test_jsonl_chunk_id_for_part2_falls_back_to_same_id() -> None:
    jsonl_id, note = jsonl_chunk_id_for_part2("5", {5, 6, 7})
    assert jsonl_id == 5
    assert note == "same_id"


def test_jsonl_chunk_id_for_part2_raises_for_missing_chunk() -> None:
    with pytest.raises(KeyError):
        jsonl_chunk_id_for_part2("9", {1, 2, 3})


def test_extract_text_features_counts_surface_and_voice_proxies() -> None:
    text = (
        "I'm really thinking, perhaps, that this longwordtoken feels formal—very formal.\n"
        '"Yes," I said...'
    )
    features = extract_text_features(text)
    assert features["token_count"] > 0
    assert features["character_count"] == len(text)
    assert features["sentence_count"] >= 2
    assert features["comma_rate"] > 0
    assert features["em_dash_rate"] > 0
    assert features["ellipsis_rate"] > 0
    assert features["contraction_rate"] > 0
    assert features["first_person_pronoun_rate"] > 0
    assert features["modality_hedging_rate"] > 0
    assert features["ly_adverb_rate"] > 0
    assert features["long_word_rate"] > 0


def test_paired_feature_deltas_include_difference_distance_and_ratio() -> None:
    deltas = paired_feature_deltas(
        {"token_count": 80.0, "mtld": 12.0},
        {"token_count": 100.0, "mtld": 8.0},
    )
    assert deltas["mt_minus_ht_token_count"] == -20.0
    assert deltas["abs_mt_minus_ht_token_count"] == 20.0
    assert deltas["mt_div_ht_token_count"] == 0.8
    assert deltas["mt_minus_ht_mtld"] == 4.0
    assert deltas["abs_mt_minus_ht_mtld"] == 4.0
    assert deltas["mt_div_ht_mtld"] == 1.5
