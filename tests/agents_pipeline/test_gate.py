import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents_pipeline.core.gate import parse_litrans_answers, run_gate, litr_score


def _make_book_mock(n_chunks: int):
    book = MagicMock()
    book.source_language = "fr"
    book.name = "dev"
    book.src_text = [MagicMock(text=f"Source chunk {i}") for i in range(n_chunks)]
    book.translation = [MagicMock(text=f"Translation chunk {i}") for i in range(n_chunks)]
    return book


def _make_cfg(tmp_path: Path, *, threshold: float = 0.7):
    return SimpleNamespace(
        run_dir=tmp_path,
        max_parallel_jobs=2,
        dry_run=False,
        litrans_threshold=threshold,
    )


def _make_litrans_response(default_judgment: str = "YES") -> str:
    return json.dumps(
        {
            str(i): {
                "judgment": default_judgment,
                "issue": (
                    "No material issue detected for this question."
                    if default_judgment == "YES"
                    else f"Problem detected for question {i}."
                ),
            }
            for i in range(1, 26)
        }
    )


def _noop_build_jobs(*, book, cfg, chunk_ids, **kwargs):
    return [SimpleNamespace(job_id=f"job_{i}") for i in chunk_ids]


def _dispatcher_writing(payloads_by_chunk: dict[int, str]):
    def _dispatch(_jobs, run_dir, _max_workers, _dry_run, _label):
        outputs = run_dir / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        for chunk_id, payload in payloads_by_chunk.items():
            (outputs / f"litrans_answers_{chunk_id:04d}.json").write_text(
                payload,
                encoding="utf-8",
            )

    return _dispatch


def test_parse_litrans_answers_valid_object_format():
    answers = parse_litrans_answers(_make_litrans_response())
    assert len(answers) == 25
    assert answers["1"]["judgment"] == "YES"
    assert "issue" in answers["1"]


def test_parse_litrans_answers_requires_all_25_questions():
    payload = json.dumps({
        "1": {"judgment": "YES", "issue": "ok"},
        "2": {"judgment": "YES", "issue": "ok"},
    })
    with pytest.raises(ValueError, match="exactly questions 1-25"):
        parse_litrans_answers(payload)


def test_parse_litrans_answers_allows_missing_issue_for_yes():
    payload = {
        str(i): {"judgment": "YES"}
        for i in range(1, 26)
    }
    answers = parse_litrans_answers(json.dumps(payload))
    assert answers["1"]["issue"]


def test_run_gate_only_scores_given_chunks(tmp_path):
    book = _make_book_mock(3)
    cfg = _make_cfg(tmp_path, threshold=0.5)
    dispatch = _dispatcher_writing({
        2: _make_litrans_response(),
    })

    gate = run_gate(
        cycle=3, book=book, cfg=cfg,
        dispatch_jobs=dispatch, build_jobs=_noop_build_jobs,
        chunks_to_score=[2],
    )

    assert gate["decision"] == "PASS"
    assert not (tmp_path / "outputs" / "litrans_review_0000.json").is_file()
    assert (tmp_path / "outputs" / "litrans_review_0002.json").is_file()


def test_run_gate_missing_answer_file_raises(tmp_path):
    book = _make_book_mock(1)
    cfg = _make_cfg(tmp_path, threshold=0.5)

    def dispatch(jobs, run_dir, max_workers, dry_run, label):
        # Simulate an agent that failed to produce its required output
        (run_dir / "outputs").mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="litrans_answers_0000.json"):
        run_gate(
            cycle=1, book=book, cfg=cfg,
            dispatch_jobs=dispatch, build_jobs=_noop_build_jobs,
        )


def test_run_gate_invalid_response_raises(tmp_path):
    book = _make_book_mock(1)
    cfg = _make_cfg(tmp_path, threshold=0.5)
    dispatch = _dispatcher_writing({0: "not valid json at all"})

    with pytest.raises((ValueError, KeyError)):
        run_gate(
            cycle=1, book=book, cfg=cfg,
            dispatch_jobs=dispatch, build_jobs=_noop_build_jobs,
        )


def test_litr_score_valid():
    payload = json.loads(_make_litrans_response())
    payload["1"]["judgment"] = "NO"
    payload["1"]["issue"] = "Meaning drift."
    payload["2"]["judgment"] = "MAYBE"
    payload["2"]["issue"] = "Tone slightly off."
    assert litr_score(json.dumps(payload)) == pytest.approx((23.5 / 25))


def test_litr_score_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        litr_score("")


def test_litr_score_unknown_value_raises():
    payload = json.loads(_make_litrans_response())
    payload["1"]["judgment"] = "UNSURE"
    with pytest.raises(KeyError):
        litr_score(json.dumps(payload))


def test_run_gate_all_pass(tmp_path):
    book = _make_book_mock(2)
    cfg = _make_cfg(tmp_path, threshold=0.7)

    def build_jobs(*, book, cfg, chunk_ids):
        return [SimpleNamespace(job_id=f"job_{i}") for i in chunk_ids]

    def dispatch_jobs(_jobs, run_dir, _max_workers, _dry_run, _label):
        outputs = run_dir / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            (outputs / f"litrans_answers_{i:04d}.json").write_text(
                _make_litrans_response(),
                encoding="utf-8",
            )

    gate = run_gate(
        cycle=1,
        book=book,
        cfg=cfg,
        dispatch_jobs=dispatch_jobs,
        build_jobs=build_jobs,
        chunks_to_score=[0, 1],
    )

    assert gate["decision"] == "PASS"
    assert gate["failing_chunk_ids"] == []
    gate_path = tmp_path / "gate" / "litrans_cycle_01" / "gate.json"
    assert gate_path.is_file()


def test_run_gate_some_fail(tmp_path):
    book = _make_book_mock(3)
    cfg = _make_cfg(tmp_path, threshold=0.7)

    def build_jobs(*, book, cfg, chunk_ids):
        return [SimpleNamespace(job_id=f"job_{i}") for i in chunk_ids]

    def dispatch_jobs(_jobs, run_dir, _max_workers, _dry_run, _label):
        outputs = run_dir / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        pass_payload = json.loads(_make_litrans_response())
        fail_payload = json.loads(_make_litrans_response())
        for qid in ("1", "2", "3", "4", "5", "6", "7", "8"):
            fail_payload[qid]["judgment"] = "NO"
            fail_payload[qid]["issue"] = f"Problem for {qid}."
        maybe_payload = json.loads(_make_litrans_response())
        for qid in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"):
            maybe_payload[qid]["judgment"] = "NO"
            maybe_payload[qid]["issue"] = f"Problem for {qid}."
        for idx, payload in enumerate((pass_payload, fail_payload, maybe_payload)):
            (outputs / f"litrans_answers_{idx:04d}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

    gate = run_gate(
        cycle=2,
        book=book,
        cfg=cfg,
        dispatch_jobs=dispatch_jobs,
        build_jobs=build_jobs,
        chunks_to_score=[0, 1, 2],
    )

    assert gate["decision"] == "FAIL"
    assert gate["failing_chunk_ids"] == [1, 2]
    review = json.loads((tmp_path / "outputs" / "litrans_review_0001.json").read_text(encoding="utf-8"))
    assert review["failed_question_ids"] == ["1", "2", "3", "4", "5", "6", "7", "8"]


def test_run_gate_invalid_response_raises(tmp_path):
    book = _make_book_mock(1)
    cfg = _make_cfg(tmp_path, threshold=0.7)

    def build_jobs(*, book, cfg, chunk_ids):
        return [SimpleNamespace(job_id="job_0")]

    def dispatch_jobs(_jobs, run_dir, _max_workers, _dry_run, _label):
        outputs = run_dir / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "litrans_answers_0000.json").write_text("not valid json at all", encoding="utf-8")

    with pytest.raises((ValueError, KeyError)):
        run_gate(
            cycle=1,
            book=book,
            cfg=cfg,
            dispatch_jobs=dispatch_jobs,
            build_jobs=build_jobs,
            chunks_to_score=[0],
        )
