import json
from pathlib import Path

import pytest

import mt_eval
from mt_eval import (
    EvaluationJob,
    build_litransproqa_rows,
    evaluate_batch,
    find_input_files,
    infer_language_pair,
    load_existing_outputs,
    load_jsonl_input,
    output_dataset_name,
    run_litransproqa_judge,
    weighted_litransproqa_score,
    write_outputs,
)


def test_load_jsonl_input_accepts_pretty_chunk_review_records(tmp_path: Path):
    path = tmp_path / "demo_fr_en.jsonl"
    records = [
        {"chunk_id": 1, "SRC": "source one", "HT": "human one", "MT": "machine one"},
        {"chunk_id": 2, "SRC": "source two", "HT": "human two", "MT": "machine two"},
    ]
    path.write_text("\n\n".join(json.dumps(record, indent=2) for record in records), encoding="utf-8")

    loaded = load_jsonl_input(path)

    assert loaded.dataset_name == "demo_fr_en"
    assert loaded.sources == ["source one", "source two"]
    assert loaded.ref_name == "demo_fr_en_ht"
    assert loaded.refs == ["human one", "human two"]
    assert loaded.systems == {
        "demo_fr_en_ht": ["human one", "human two"],
        "demo_fr_en_mt": ["machine one", "machine two"],
    }


def test_load_jsonl_input_requires_src_ht_mt_strings(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"chunk_id": 1, "SRC": "source", "HT": "human"}), encoding="utf-8")

    with pytest.raises(ValueError, match="MT"):
        load_jsonl_input(path)


def test_find_input_files_includes_direct_jsonl_and_par3_pickles(tmp_path: Path):
    direct_jsonl = tmp_path / "demo.jsonl"
    direct_pkl = tmp_path / "demo.pkl"
    nested_pkl = tmp_path / "eval" / "book" / "book.pkl"
    ignored_nested_jsonl = tmp_path / "eval" / "book" / "book.jsonl"
    nested_pkl.parent.mkdir(parents=True)
    for path in (direct_jsonl, direct_pkl, nested_pkl, ignored_nested_jsonl):
        path.write_text("", encoding="utf-8")

    assert find_input_files(tmp_path) == sorted([direct_jsonl, direct_pkl, nested_pkl])


def test_output_dataset_name_collapses_par3_book_book_pickle_path(tmp_path: Path):
    pkl_path = tmp_path / "inner_space" / "inner_space.pkl"
    pkl_path.parent.mkdir()
    pkl_path.write_text("", encoding="utf-8")

    assert output_dataset_name(pkl_path, tmp_path) == "inner_space"


def test_output_dataset_name_preserves_split_when_collapsing_from_dataset_root(tmp_path: Path):
    pkl_path = tmp_path / "eval" / "inner_space" / "inner_space.pkl"
    pkl_path.parent.mkdir(parents=True)
    pkl_path.write_text("", encoding="utf-8")

    assert output_dataset_name(pkl_path, tmp_path) == "eval/inner_space"


def test_infer_language_pair_from_chunk_review_filename():
    assert infer_language_pair("hooked_a_novel_of_obsession_ja_en") == "ja-en"
    assert infer_language_pair("needle_s_eye") == "unknown-en"


def test_build_litransproqa_rows_includes_reference_system_for_adequacy_comparison():
    base_dir = Path("LiTransProQA/prompting_method")
    jobs = [
        EvaluationJob(
            pkl_name="demo_fr_en",
            system_name="demo_fr_en_ht",
            sources=["source"],
            hyps=["human"],
            ref_name="demo_fr_en_ht",
            refs=["human"],
        ),
        EvaluationJob(
            pkl_name="demo_fr_en",
            system_name="demo_fr_en_mt",
            sources=["source"],
            hyps=["machine"],
            ref_name="demo_fr_en_ht",
            refs=["human"],
        ),
    ]

    rows = build_litransproqa_rows(jobs, base_dir)

    assert len(rows) == 2
    assert rows[0]["job_index"] == 0
    assert rows[0]["dataset"] == "demo_fr_en/demo_fr_en_ht"
    assert "Translation: human" in rows[0]["QA"]
    assert rows[1]["job_index"] == 1
    assert rows[1]["segment_index"] == 0
    assert rows[1]["pair"] == "fr-en"
    assert rows[1]["dataset"] == "demo_fr_en/demo_fr_en_mt"
    assert "Source text: source" in rows[1]["QA"]
    assert "Translation: machine" in rows[1]["QA"]


def test_weighted_litransproqa_score_uses_question_weights():
    score = weighted_litransproqa_score(
        "{'1': 'YES', '2': 'MAYBE', '3': 'NO'}",
        [5.0, 3.0, 2.0],
    )

    assert score == pytest.approx((5.0 + 1.5) / 10.0)


def test_run_litransproqa_judge_attaches_scores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    pd = pytest.importorskip("pandas")
    calls = {"count": 0}
    job = EvaluationJob(
        pkl_name="demo_fr_en",
        system_name="demo_fr_en_mt",
        sources=["source"],
        hyps=["machine"],
        ref_name="demo_fr_en_ht",
        refs=["human"],
    )

    class FakeGemini:
        name = "gemini"

        def __init__(self, model_checkpoint):
            self.model_checkpoint = model_checkpoint

        def direct_message(self, message):
            calls["count"] += 1
            return "{'1': 'YES', '2': 'MAYBE', '3': 'NO'}"

    import api_model

    monkeypatch.setattr(api_model, "Gemini", FakeGemini)

    args = type(
        "Args",
        (),
        {
            "out": tmp_path,
            "llm_judge_model": "gemini-test",
            "llm_judge_retries": 0,
            "llm_judge_retry_delay": 0,
        },
    )()

    run_litransproqa_judge([job], args)

    expected_score = weighted_litransproqa_score(
        "{'1': 'YES', '2': 'MAYBE', '3': 'NO'}",
        [4.857142857142857, 4.285714285714286, 4.714285714285714],
    )
    assert job.scores["litransproqa"] == {0: pytest.approx(expected_score)}
    assert calls["count"] == 1
    assert (
        tmp_path / "analysis" / "litransproqa" / "final_results" / "judge_prompts.csv"
    ).exists()


def test_run_litransproqa_judge_records_failed_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    pytest.importorskip("pandas")
    job = EvaluationJob(
        pkl_name="demo_fr_en",
        system_name="demo_fr_en_mt",
        sources=["source"],
        hyps=["machine"],
        ref_name="demo_fr_en_ht",
        refs=["human"],
    )

    class FakeGemini:
        name = "gemini"

        def __init__(self, model_checkpoint):
            self.model_checkpoint = model_checkpoint

        def direct_message(self, message):
            raise RuntimeError("temporary API failure")

    import api_model

    monkeypatch.setattr(api_model, "Gemini", FakeGemini)

    args = type(
        "Args",
        (),
        {
            "out": tmp_path,
            "llm_judge_model": "gemini-test",
            "llm_judge_retries": 1,
            "llm_judge_retry_delay": 0,
        },
    )()

    run_litransproqa_judge([job], args)

    assert "litransproqa" not in job.scores
    assert job.skipped[-1]["reason"] == "judge_request_failed"
    assert "temporary API failure" in job.skipped[-1]["error"]


def test_evaluate_batch_adds_missing_metric_without_recomputing_existing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    write_outputs(
        tmp_path,
        "demo",
        "demo_mt",
        ["source"],
        ["hypothesis"],
        "demo_ht",
        {"cometkiwi": {0: 0.7}},
        [{"metric": "cometkiwi", "segment_index": 3, "reason": "existing_skip"}],
    )
    job = EvaluationJob(
        pkl_name="demo",
        system_name="demo_mt",
        sources=["source"],
        hyps=["hypothesis"],
        ref_name="demo_ht",
        refs=["reference"],
    )
    load_existing_outputs([job], tmp_path)
    calls: list[str] = []

    def fake_run_comet(metric, segments, args, work_dir):
        calls.append(metric)
        assert metric == "comet22"
        return {(0, 0): 0.9}

    monkeypatch.setattr(mt_eval, "run_comet", fake_run_comet)
    args = type(
        "Args",
        (),
        {
            "out": tmp_path,
            "metricx_max_input_length": 1536,
            "comet_max_input_length": 512,
            "llm_as_a_judge": False,
        },
    )()

    evaluate_batch([job], ["cometkiwi", "comet22"], {}, args, {})

    assert calls == ["comet22"]
    rows = [
        json.loads(line)
        for line in (
            tmp_path / "books" / "demo" / "demo_mt" / "segment_scores.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert rows[0]["scores"] == {"comet22": 0.9, "cometkiwi": 0.7}
    system_scores = json.loads(
        (tmp_path / "books" / "demo" / "demo_mt" / "system_scores.json").read_text(
            encoding="utf-8"
        )
    )
    assert sorted(system_scores["metrics"]) == ["comet22", "cometkiwi"]
