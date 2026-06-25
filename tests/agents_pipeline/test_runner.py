# tests/agents_pipeline/test_runner.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents_pipeline.core.job_spec import JobSpec, RunnerConfig, STAGE_NAMES
from agents_pipeline.runner import (
    _chunk_boundary_context,
    build_translate_jobs,
    build_revise_jobs,
    build_chunk_literary_review_jobs,
    load_translations_into_chunks,
    reconstruct_and_save,
    build_book_review_job,
    build_cross_chunk_audit_job,
    _parse_chunk_literary_review,
    _parse_book_review,
    _parse_cross_chunk_audit,
    build_final_revision_jobs,
    run_final_revision,
    build_style_analysis_job,
    run_style_analysis,
    run_chunk_literary_gate,
    write_master_chunk_gate,
    _persist_run_args,
    _load_run_args,
    _build_stall_contexts,
)


def _make_stage_models():
    return {
        stage: {"provider": "claude", "model": None}
        for stage in STAGE_NAMES
    }


def _make_cfg(tmp_path: Path) -> RunnerConfig:
    stage_models = {
        name: {"provider": "claude", "model": None}
        for name in (
            "style_analysis",
            "translate",
            "litrans_review",
            "revise",
            "book_review",
            "final_revise",
        )
    }
    return RunnerConfig(
        book_path=tmp_path / "data" / "dev_fr.txt",
        run_dir=tmp_path / "runs" / "dev_20260401",
        stage_models=_make_stage_models(),
        max_cycles=3,
        max_global_cycles=2,
        max_parallel_jobs=2,
        litrans_threshold=0.7,
        target_language_code="en",
        target_language="English",
        dry_run=True,
    )


def _make_chunk_mock(text: str):
    c = MagicMock()
    c.text = text
    return c


def _make_book_mock(chunks: list):
    book = MagicMock()
    book.name = "dev"
    book.source_language = "fr"
    book.src_text = chunks
    return book


def test_chunk_boundary_context_uses_paragraph_preserving_token_windows():
    jp_para_1 = "これは最初の段落です。" * 40
    jp_para_2 = "こちらは次の段落です。" * 40
    jp_chunk = f"{jp_para_1}\n\n{jp_para_2}"
    book = MagicMock()
    book.source_language = "Japanese"
    book.src_text = [_make_chunk_mock(jp_chunk) for _ in range(3)]
    book.translation = [_make_chunk_mock(jp_chunk) for _ in range(3)]

    context = _chunk_boundary_context(book=book, chunk_id=1)

    assert context["prev_source_context"] != jp_chunk
    assert context["next_source_context"] != jp_chunk
    assert context["prev_translation_context"] != jp_chunk
    assert context["next_translation_context"] != jp_chunk
    assert "\n\n" not in context["prev_source_context"]
    assert context["prev_source_context"] in {jp_para_1, jp_para_2}


def _make_litrans_review(*, chunk_id: int, score: float, verdict: str, failed_ids=None) -> dict:
    if failed_ids is None:
        failed_ids = []
    answers = {
        str(i): {
            "judgment": "YES",
            "issue": "No material issue detected for this question.",
        }
        for i in range(1, 26)
    }
    for qid in failed_ids:
        answers[qid] = {
            "judgment": "NO",
            "issue": f"Problem detected for question {qid}.",
        }
    return {
        "chunk_id": chunk_id,
        "score": score,
        "verdict": verdict,
        "decision_rule": {
            "allowed_no_count": 0,
            "allowed_maybe_count": 5,
        },
        "yes_count": 25 - len(failed_ids),
        "maybe_count": 0,
        "no_count": len(failed_ids),
        "answers": answers,
        "failed_question_ids": failed_ids,
        "failed_questions": [
            {
                "question_id": qid,
                "group": "Test Group",
                "judgment": answers[qid]["judgment"],
                "issue": answers[qid]["issue"],
            }
            for qid in failed_ids
        ],
    }


def test_build_translate_jobs_count(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock(f"chunk {i}") for i in range(3)]
    book = _make_book_mock(chunks)

    jobs = build_translate_jobs(book=book, cfg=cfg, cycle=1)

    assert len(jobs) == 3
    assert all(isinstance(j, JobSpec) for j in jobs)
    assert all(j.stage == "translate" for j in jobs)
    assert all(j.chunk_id == i for i, j in enumerate(jobs))


def test_build_translate_jobs_writes_source_files(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock("Bonjour le monde")]
    book = _make_book_mock(chunks)

    build_translate_jobs(book=book, cfg=cfg, cycle=1)

    src_file = cfg.run_dir / "inputs" / "source_chunk_0000.txt"
    assert src_file.read_text() == "Bonjour le monde"


def test_build_revise_jobs_only_failing(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock(f"chunk {i}") for i in range(4)]
    book = _make_book_mock(chunks)
    book.translation = [_make_chunk_mock(f"trans {i}") for i in range(4)]

    # Pre-write qe_review files for all chunks
    outputs_dir = cfg.run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    for i in range(4):
        (outputs_dir / f"litrans_review_{i:04d}.json").write_text(
            json.dumps(
                _make_litrans_review(
                    chunk_id=i,
                    score=0.4 if i in (1, 3) else 0.9,
                    verdict="FAIL" if i in (1, 3) else "PASS",
                    failed_ids=["1"] if i in (1, 3) else [],
                )
            )
        )
    # Pre-write translation files
    for i in range(4):
        (outputs_dir / f"segment_translation_{i:04d}.json").write_text(
            json.dumps({"chunk_id": i, "translation": f"Draft {i}"})
        )
        (outputs_dir / f"chunk_literary_review_{i:04d}.json").write_text(
            json.dumps({
                "chunk_id": i,
                "verdict": "PASS",
                "verdicts": {
                    "accuracy": "PASS",
                    "voice": "PASS",
                    "dialogue": "PASS",
                    "prose": "PASS",
                },
                "findings": [],
                "summary": "ok",
            })
        )

    failing_chunk_ids = [1, 3]
    jobs = build_revise_jobs(book=book, cfg=cfg, cycle=2, failing_chunk_ids=failing_chunk_ids)

    assert len(jobs) == 2
    assert all(j.stage == "revise" for j in jobs)
    assert {j.chunk_id for j in jobs} == {1, 3}


def test_build_revise_jobs_switches_stalled_chunk_to_alternate_model(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    cfg.stage_models["revise"] = {"provider": "codex", "model": "gpt-5.4"}
    cfg.stage_models["chunk_literary_review"] = {"provider": "claude", "model": "claude-opus-4-6"}

    chunks = [_make_chunk_mock("chunk 0")]
    book = _make_book_mock(chunks)
    book.translation = [_make_chunk_mock("draft 0")]

    outputs_dir = cfg.run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "litrans_review_0000.json").write_text(
        json.dumps(
            _make_litrans_review(
                chunk_id=0,
                score=0.2,
                verdict="FAIL",
                failed_ids=["1"],
            )
        )
    )
    (outputs_dir / "segment_translation_0000.json").write_text(
        json.dumps({"chunk_id": 0, "translation": "Draft 0"})
    )
    (outputs_dir / "chunk_literary_review_0000.json").write_text(
        json.dumps({
            "chunk_id": 0,
            "verdict": "FAIL",
            "verdicts": {
                "accuracy": "FAIL",
                "voice": "PASS",
                "dialogue": "PASS",
                "prose": "PASS",
            },
            "findings": [
                {
                    "finding_id": "clr_001",
                    "source": "accuracy",
                    "severity": "MEDIUM",
                    "chunk_id": 0,
                    "evidence": "e",
                    "problem": "p",
                    "rewrite_direction": "r",
                    "acceptance_test": "a",
                }
            ],
            "summary": "still failing",
        })
    )

    jobs = build_revise_jobs(
        book=book,
        cfg=cfg,
        cycle=3,
        failing_chunk_ids=[0],
        stall_contexts={
            0: {
                "previous_cycle": 1,
                "current_cycle": 2,
                "repeated_qe_failed_question_ids": ["1"],
                "repeated_literary_finding_keys": ["accuracy | clr_001 | p"],
            }
        },
    )

    assert jobs[0].provider == "claude"
    assert jobs[0].model == "claude-opus-4-6"
    assert "STALLED CHUNK MODE" in jobs[0].prompt_text


def test_build_stall_contexts_requires_three_failures_before_switch(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    book = _make_book_mock([_make_chunk_mock("chunk 0")])

    failures = [
        {
            "cycle": 1,
            "translation": "draft 0",
            "qe_failed_question_ids": ["1"],
            "literary_finding_keys": ["accuracy | clr_001 | p"],
        },
        {
            "cycle": 2,
            "translation": "draft 0",
            "qe_failed_question_ids": ["1"],
            "literary_finding_keys": ["accuracy | clr_001 | p"],
        },
    ]
    summary_path = cfg.run_dir / "metrics" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({
            "jobs": [],
            "chunk_states": {"0": {"local_failures": failures}},
            "updated_at": None,
        }),
        encoding="utf-8",
    )

    assert _build_stall_contexts(
        book=book,
        cfg=cfg,
        failing_chunk_ids=[0],
    ) == {}

    failures.append(
        {
            "cycle": 3,
            "translation": "draft 0",
            "qe_failed_question_ids": ["1"],
            "literary_finding_keys": ["accuracy | clr_001 | p"],
        }
    )
    summary_path.write_text(
        json.dumps({
            "jobs": [],
            "chunk_states": {"0": {"local_failures": failures}},
            "updated_at": None,
        }),
        encoding="utf-8",
    )

    contexts = _build_stall_contexts(
        book=book,
        cfg=cfg,
        failing_chunk_ids=[0],
    )

    assert 0 in contexts
    assert contexts[0]["previous_cycle"] == 2
    assert contexts[0]["current_cycle"] == 3


def test_load_translations_into_chunks(tmp_path):
    cfg = _make_cfg(tmp_path)
    outputs_dir = cfg.run_dir / "outputs"
    outputs_dir.mkdir(parents=True)

    translations = ["Hello world", "Goodbye world", "How are you"]
    for i, t in enumerate(translations):
        (outputs_dir / f"segment_translation_{i:04d}.json").write_text(
            json.dumps({"chunk_id": i, "translation": t})
        )

    book = MagicMock()
    book.src_text = [MagicMock() for _ in range(3)]
    load_translations_into_chunks(book=book, cfg=cfg)

    book.set_translation.assert_called_once_with(translations)


def test_build_book_review_job_single_spec(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock(f"chunk {i}") for i in range(3)]
    book = _make_book_mock(chunks)

    job = build_book_review_job(book=book, cfg=cfg)

    assert isinstance(job, JobSpec)
    assert job.stage == "book_review"
    assert job.chunk_id == -1
    assert job.book_name == "dev"
    assert job.required_outputs == ["outputs/book_review.json"]
    assert "dev_en_draft.txt" in job.allowed_inputs
    for i in range(3):
        assert f"inputs/source_chunk_{i:04d}.txt" in job.allowed_inputs
    provider, model = cfg.stage("book_review")
    assert job.provider == provider
    assert job.model == model
    assert "fr" in job.prompt_text        # source_lang substituted
    assert "dev" in job.prompt_text       # book_name substituted
    assert "3" in job.prompt_text         # num_chunks substituted


def test_build_cross_chunk_audit_job_single_spec(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock(f"chunk {i}") for i in range(3)]
    book = _make_book_mock(chunks)

    job = build_cross_chunk_audit_job(book=book, cfg=cfg)

    assert isinstance(job, JobSpec)
    assert job.stage == "cross_chunk_audit"
    assert job.chunk_id == -1
    assert job.book_name == "dev"
    assert job.required_outputs == ["outputs/cross_chunk_audit.json"]
    assert "dev_en_draft.txt" in job.allowed_inputs
    for i in range(3):
        assert f"inputs/source_chunk_{i:04d}.txt" in job.allowed_inputs


def test_parse_book_review_filters_medium_plus(tmp_path):
    cfg = _make_cfg(tmp_path)
    outputs_dir = cfg.run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    review = {
        "verdict": "FAIL",
        "summary": "several issues",
        "findings": [
            {
                "finding_id": "f_001", "chunk_id": 0, "severity": "LOW",
                "evidence": "e", "problem": "p", "rewrite_direction": "r", "acceptance_test": "a",
            },
            {
                "finding_id": "f_002", "chunk_id": 1, "severity": "MEDIUM",
                "evidence": "e", "problem": "p", "rewrite_direction": "r", "acceptance_test": "a",
            },
            {
                "finding_id": "f_003", "chunk_id": 2, "severity": "HIGH",
                "evidence": "e", "problem": "p", "rewrite_direction": "r", "acceptance_test": "a",
            },
            {
                "finding_id": "f_004", "chunk_id": 1, "severity": "CRITICAL",
                "evidence": "e", "problem": "p", "rewrite_direction": "r", "acceptance_test": "a",
            },
        ],
    }
    (outputs_dir / "book_review.json").write_text(json.dumps(review), encoding="utf-8")

    result = _parse_book_review(cfg.run_dir)

    assert 0 not in result                   # LOW — skipped
    assert sorted(result.keys()) == [1, 2]
    assert len(result[1]) == 2               # MEDIUM + CRITICAL
    assert len(result[2]) == 1               # HIGH


def test_parse_book_review_pass_returns_empty(tmp_path):
    cfg = _make_cfg(tmp_path)
    outputs_dir = cfg.run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    review = {"verdict": "PASS", "summary": "ok", "findings": []}
    (outputs_dir / "book_review.json").write_text(json.dumps(review), encoding="utf-8")

    result = _parse_book_review(cfg.run_dir)

    assert result == {}


def test_parse_cross_chunk_audit_filters_medium_plus(tmp_path):
    cfg = _make_cfg(tmp_path)
    outputs_dir = cfg.run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    review = {
        "verdict": "FAIL",
        "summary": "cross chunk issues",
        "findings": [
            {
                "finding_id": "cca_001", "chunk_id": 0, "severity": "LOW",
                "evidence": "e", "problem": "p", "rewrite_direction": "r", "acceptance_test": "a",
            },
            {
                "finding_id": "cca_002", "chunk_id": 2, "severity": "HIGH",
                "evidence": "e", "problem": "p", "rewrite_direction": "r", "acceptance_test": "a",
            },
        ],
    }
    (outputs_dir / "cross_chunk_audit.json").write_text(json.dumps(review), encoding="utf-8")

    result = _parse_cross_chunk_audit(cfg.run_dir)

    assert 0 not in result
    assert sorted(result.keys()) == [2]


def test_parse_book_review_severity_case_insensitive(tmp_path):
    cfg = _make_cfg(tmp_path)
    outputs_dir = cfg.run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    review = {
        "verdict": "FAIL", "summary": "s",
        "findings": [
            {
                "finding_id": "f_001", "chunk_id": 0, "severity": "medium",
                "evidence": "e", "problem": "p", "rewrite_direction": "r", "acceptance_test": "a",
            },
        ],
    }
    (outputs_dir / "book_review.json").write_text(json.dumps(review), encoding="utf-8")
    result = _parse_book_review(cfg.run_dir)
    assert 0 in result and len(result[0]) == 1


def test_parse_chunk_literary_review_requires_all_lenses(tmp_path):
    cfg = _make_cfg(tmp_path)
    outputs_dir = cfg.run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "chunk_literary_review_0000.json").write_text(
        json.dumps({
            "chunk_id": 0,
            "verdict": "PASS",
            "verdicts": {"accuracy": "PASS"},
            "findings": [],
            "summary": "ok",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="verdicts must contain exactly"):
        _parse_chunk_literary_review(cfg.run_dir, chunk_id=0)


def test_parse_chunk_literary_review_reports_json_location(tmp_path):
    cfg = _make_cfg(tmp_path)
    outputs_dir = cfg.run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "chunk_literary_review_0000.json").write_text(
        '{\n  "chunk_id": 0,\n  "verdict": "PASS"\n  "summary": "oops"\n}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"invalid JSON at line 4 column 3"):
        _parse_chunk_literary_review(cfg.run_dir, chunk_id=0)


def test_run_chunk_literary_gate_fails_when_any_lens_fails(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock("source")]
    book = _make_book_mock(chunks)
    book.translation = [_make_chunk_mock("translation")]

    def fake_run(_jobs, run_dir, _max_workers, _dry_run, label=None):
        outputs = run_dir / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / "chunk_literary_review_0000.json").write_text(
            json.dumps({
                "chunk_id": 0,
                "verdict": "PASS",
                "verdicts": {
                    "accuracy": "FAIL",
                    "voice": "PASS",
                    "dialogue": "PASS",
                    "prose": "PASS",
                },
                "findings": [],
                "summary": "accuracy lens failed",
            }),
            encoding="utf-8",
        )

    with patch("agents_pipeline.runner.run_jobs_parallel", side_effect=fake_run):
        gate = run_chunk_literary_gate(cycle=1, book=book, cfg=cfg, chunk_ids=[0])

    assert gate["decision"] == "FAIL"
    assert gate["failing_chunk_ids"] == [0]


def test_run_chunk_literary_gate_reuses_completed_valid_outputs(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock("source 0"), _make_chunk_mock("source 1")]
    book = _make_book_mock(chunks)
    book.translation = [_make_chunk_mock("translation 0"), _make_chunk_mock("translation 1")]

    outputs = cfg.run_dir / "outputs"
    outputs.mkdir(parents=True)
    valid_review = {
        "chunk_id": 0,
        "verdict": "PASS",
        "verdicts": {
            "accuracy": "PASS",
            "voice": "PASS",
            "dialogue": "PASS",
            "prose": "PASS",
        },
        "findings": [],
        "summary": "ok",
    }
    (outputs / "chunk_literary_review_0000.json").write_text(
        json.dumps(valid_review),
        encoding="utf-8",
    )

    metrics = cfg.run_dir / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "run_summary.json").write_text(
        json.dumps({
            "jobs": [
                {
                    "job_id": "chunk_literary_review_dev_chunk_0000_cycle_01",
                    "status": "success",
                }
            ],
            "chunk_states": {},
            "updated_at": None,
        }),
        encoding="utf-8",
    )

    def fake_run(jobs, run_dir, _max_workers, _dry_run, label=None):
        assert [job.chunk_id for job in jobs] == [1]
        review = dict(valid_review, chunk_id=1)
        (run_dir / "outputs" / "chunk_literary_review_0001.json").write_text(
            json.dumps(review),
            encoding="utf-8",
        )

    with patch("agents_pipeline.runner.run_jobs_parallel", side_effect=fake_run) as run_mock:
        gate = run_chunk_literary_gate(cycle=1, book=book, cfg=cfg, chunk_ids=[0, 1])

    run_mock.assert_called_once()
    assert gate["decision"] == "PASS"
    assert gate["failing_chunk_ids"] == []


def test_write_master_chunk_gate_unions_failures(tmp_path):
    cfg = _make_cfg(tmp_path)
    gate = write_master_chunk_gate(
        cycle=2,
        cfg=cfg,
        qe_gate={"failing_chunk_ids": [0, 2]},
        literary_gate={"failing_chunk_ids": [1, 2]},
    )

    assert gate["decision"] == "FAIL"
    assert gate["failing_chunk_ids"] == [0, 1, 2]
    assert gate["qe_failing_chunk_ids"] == [0, 2]
    assert gate["literary_failing_chunk_ids"] == [1, 2]


def _make_finding(finding_id: str, chunk_id: int, severity: str) -> dict:
    return {
        "finding_id": finding_id,
        "chunk_id": chunk_id,
        "severity": severity,
        "evidence": "some text",
        "problem": "a problem",
        "rewrite_direction": "fix it",
        "acceptance_test": "check it",
    }


def test_build_final_revision_jobs_count_and_structure(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock(f"source {i}") for i in range(4)]
    book = _make_book_mock(chunks)
    book.translation = [_make_chunk_mock(f"trans {i}") for i in range(4)]

    findings_by_chunk = {
        1: {
            "book_review_findings": [_make_finding("f_001", 1, "HIGH")],
            "cross_chunk_audit_findings": [],
        },
        3: {
            "book_review_findings": [],
            "cross_chunk_audit_findings": [_make_finding("f_002", 3, "MEDIUM")],
        },
    }
    jobs = build_final_revision_jobs(
        book=book,
        cfg=cfg,
        affected_chunk_ids=[1, 3],
        findings_by_chunk=findings_by_chunk,
    )

    assert len(jobs) == 2
    assert all(isinstance(j, JobSpec) for j in jobs)
    assert all(j.stage == "final_revise" for j in jobs)
    assert {j.chunk_id for j in jobs} == {1, 3}

    for j in jobs:
        i = j.chunk_id
        assert f"outputs/segment_translation_{i:04d}.json" in j.required_outputs
        assert f"outputs/final_revise_{i:04d}_cycle_01.json" in j.required_outputs
        assert "inputs/full_source.txt" in j.allowed_inputs
        assert "dev_en_draft.txt" in j.allowed_inputs
        assert f"inputs/source_chunk_{i:04d}.txt" in j.allowed_inputs
        assert f"outputs/segment_translation_{i:04d}.json" in j.allowed_inputs
        assert "outputs/style_bible.json" in j.allowed_inputs
        provider, model = cfg.stage("final_revise")
        assert j.provider == provider
        assert j.model == model


def test_build_final_revision_jobs_findings_in_prompt(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock("source text")]
    book = _make_book_mock(chunks)
    book.translation = [_make_chunk_mock("draft translation")]

    findings_by_chunk = {
        0: {
            "book_review_findings": [_make_finding("f_001", 0, "HIGH")],
            "cross_chunk_audit_findings": [],
        }
    }
    jobs = build_final_revision_jobs(
        book=book,
        cfg=cfg,
        affected_chunk_ids=[0],
        findings_by_chunk=findings_by_chunk,
    )

    assert "f_001" in jobs[0].prompt_text
    assert "source text" in jobs[0].prompt_text
    assert "draft translation" in jobs[0].prompt_text


def test_run_final_revision_pass_skips_correction(tmp_path):
    cfg = _make_cfg(tmp_path)
    outputs_dir = cfg.run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    # Write PASS global review outputs (no MEDIUM+ findings)
    (outputs_dir / "book_review.json").write_text(
        json.dumps({"verdict": "PASS", "summary": "ok", "findings": []}),
        encoding="utf-8",
    )
    (outputs_dir / "cross_chunk_audit.json").write_text(
        json.dumps({"verdict": "PASS", "summary": "ok", "findings": []}),
        encoding="utf-8",
    )

    chunks = [_make_chunk_mock("source")]
    book = _make_book_mock(chunks)
    book.get_translation.return_value = "clean translation"

    with patch("agents_pipeline.runner.run_jobs_parallel") as mock_run:
        mock_run.return_value = None
        run_final_revision(book=book, cfg=cfg)

    # Gate written and is PASS
    gate_path = cfg.run_dir / "gate" / "final" / "gate.json"
    assert gate_path.is_file()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["verdict"] == "PASS"
    assert gate["affected_chunk_ids"] == []
    assert gate["reason"] == "all_chunks_clean"

    # Final clean file written
    final = cfg.run_dir / "dev_en.txt"
    assert final.is_file()
    assert final.read_text(encoding="utf-8") == "clean translation"

    # run_jobs_parallel called exactly once (global review jobs only)
    assert mock_run.call_count == 1


def test_run_final_revision_fail_writes_revised_book(tmp_path):
    cfg = _make_cfg(tmp_path)
    (cfg.run_dir / "outputs").mkdir(parents=True)
    (cfg.run_dir / "inputs").mkdir(parents=True)

    # Write source chunk file (needed by load_translations_into_chunks after revision)
    (cfg.run_dir / "inputs" / "source_chunk_0000.txt").write_text("Source text", encoding="utf-8")

    # Write the current segment translation
    (cfg.run_dir / "outputs" / "segment_translation_0000.json").write_text(
        json.dumps({"chunk_id": 0, "translation": "Old translation"}), encoding="utf-8"
    )

    chunks = [_make_chunk_mock("Source text")]
    book = _make_book_mock(chunks)
    book.translation = [_make_chunk_mock("Old translation")]
    book.get_translation.return_value = "Revised translation"

    def fake_run(_jobs, run_dir, _max_workers, _dry_run, label):
        if label == "final_global_review_cycle_01":
            (run_dir / "outputs" / "book_review.json").write_text(
                json.dumps({
                    "verdict": "FAIL",
                    "summary": "one issue",
                    "findings": [
                        {
                            "finding_id": "f_001", "chunk_id": 0, "severity": "HIGH",
                            "evidence": "Old translation", "problem": "mistranslation",
                            "rewrite_direction": "fix it", "acceptance_test": "check it",
                        }
                    ],
                }),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "cross_chunk_audit.json").write_text(
                json.dumps({"verdict": "PASS", "summary": "ok", "findings": []}),
                encoding="utf-8",
            )
        elif label == "final_revise_cycle_01":
            (run_dir / "outputs" / "segment_translation_0000.json").write_text(
                json.dumps({"chunk_id": 0, "translation": "Revised translation"}),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "final_revise_0000_cycle_01.json").write_text(
                json.dumps({
                    "chunk_id": 0,
                    "findings_addressed": [{"finding_id": "f_001", "status_after_revision": "FIXED"}],
                }),
                encoding="utf-8",
            )
        elif label == "final_global_review_cycle_02":
            (run_dir / "outputs" / "book_review.json").write_text(
                json.dumps({"verdict": "PASS", "summary": "ok", "findings": []}),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "cross_chunk_audit.json").write_text(
                json.dumps({"verdict": "PASS", "summary": "ok", "findings": []}),
                encoding="utf-8",
            )

    with patch("agents_pipeline.runner.run_jobs_parallel", side_effect=fake_run):
        run_final_revision(book=book, cfg=cfg)

    # Gate is PASS after confirmation cycle
    gate = json.loads(
        (cfg.run_dir / "gate" / "final" / "gate.json").read_text(encoding="utf-8")
    )
    assert gate["verdict"] == "PASS"
    assert gate["affected_chunk_ids"] == []
    assert gate["book_review_affected_chunk_ids"] == []
    assert gate["cross_chunk_audit_affected_chunk_ids"] == []
    assert gate["confirmed"] is True
    assert gate["global_cycle"] == 2
    assert gate["reason"] == "all_chunks_clean"

    # Final book saved at {book_name}_en.txt (not _en_revised.txt)
    final = cfg.run_dir / "dev_en.txt"
    assert final.is_file()
    assert final.read_text(encoding="utf-8") == "Revised translation"


def test_run_final_revision_merges_cross_chunk_audit_findings(tmp_path):
    cfg = _make_cfg(tmp_path)
    (cfg.run_dir / "outputs").mkdir(parents=True)
    (cfg.run_dir / "inputs").mkdir(parents=True)
    (cfg.run_dir / "inputs" / "source_chunk_0000.txt").write_text("Source text", encoding="utf-8")
    (cfg.run_dir / "outputs" / "segment_translation_0000.json").write_text(
        json.dumps({"chunk_id": 0, "translation": "Old translation"}), encoding="utf-8"
    )
    chunks = [_make_chunk_mock("Source text")]
    book = _make_book_mock(chunks)
    book.translation = [_make_chunk_mock("Old translation")]
    book.get_translation.return_value = "Revised translation"

    def fake_run(_jobs, run_dir, _max_workers, _dry_run, label):
        if label == "final_global_review_cycle_01":
            (run_dir / "outputs" / "book_review.json").write_text(
                json.dumps({"verdict": "PASS", "summary": "ok", "findings": []}),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "cross_chunk_audit.json").write_text(
                json.dumps({
                    "verdict": "FAIL",
                    "summary": "seam issue",
                    "findings": [
                        {
                            "finding_id": "cca_001", "chunk_id": 0, "severity": "HIGH",
                            "evidence": "Old translation", "problem": "boundary seam",
                            "rewrite_direction": "repair the seam", "acceptance_test": "reads smoothly",
                        }
                    ],
                }),
                encoding="utf-8",
            )
        elif label == "final_revise_cycle_01":
            (run_dir / "outputs" / "segment_translation_0000.json").write_text(
                json.dumps({"chunk_id": 0, "translation": "Revised translation"}),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "final_revise_0000_cycle_01.json").write_text(
                json.dumps({
                    "chunk_id": 0,
                    "findings_addressed": [{"finding_id": "cca_001", "status_after_revision": "FIXED"}],
                }),
                encoding="utf-8",
            )
        elif label == "final_global_review_cycle_02":
            (run_dir / "outputs" / "book_review.json").write_text(
                json.dumps({"verdict": "PASS", "summary": "ok", "findings": []}),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "cross_chunk_audit.json").write_text(
                json.dumps({"verdict": "PASS", "summary": "ok", "findings": []}),
                encoding="utf-8",
            )

    with patch("agents_pipeline.runner.run_jobs_parallel", side_effect=fake_run):
        run_final_revision(book=book, cfg=cfg)

    gate = json.loads(
        (cfg.run_dir / "gate" / "final" / "gate.json").read_text(encoding="utf-8")
    )
    assert gate["affected_chunk_ids"] == []
    assert gate["book_review_affected_chunk_ids"] == []
    assert gate["cross_chunk_audit_affected_chunk_ids"] == []


def test_reconstruct_and_save(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)

    book = MagicMock()
    book.name = "dev"
    book.get_translation.return_value = "Hello world.\n\nGoodbye."

    reconstruct_and_save(book=book, cfg=cfg)

    draft_file = cfg.run_dir / "dev_en_draft.txt"
    assert draft_file.read_text(encoding="utf-8") == "Hello world.\n\nGoodbye."
    # No chunk tags in output
    assert "<chunk>" not in draft_file.read_text(encoding="utf-8")


def test_build_style_analysis_job_spec(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock("Bonjour le monde")]
    book = _make_book_mock(chunks)

    job = build_style_analysis_job(book=book, cfg=cfg)

    assert isinstance(job, JobSpec)
    assert job.stage == "style_analysis"
    assert job.chunk_id == -1
    assert job.book_name == "dev"
    assert job.allowed_inputs == ["inputs/full_source.txt"]
    assert job.required_outputs == ["outputs/style_bible.json"]
    provider, model = cfg.stage("style_analysis")
    assert job.provider == provider
    assert job.model == model


def test_build_style_analysis_job_prompt_substitution(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    book = _make_book_mock([_make_chunk_mock("text")])

    job = build_style_analysis_job(book=book, cfg=cfg)

    assert "fr" in job.prompt_text      # source_lang substituted


def test_run_style_analysis_writes_full_source(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    book = _make_book_mock([_make_chunk_mock("Bonjour le monde")])
    book.get_src_text.return_value = "Bonjour le monde"

    # Fake run_jobs_parallel that writes the expected output
    def fake_run(jobs, run_dir, max_workers, dry_run, label):
        out = run_dir / "outputs" / "style_bible.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({
                "character_voice_profiles": [],
                "prose_style_profile": {
                    "register": "literary",
                    "domain": "fiction",
                    "narrative_tense": "past",
                    "rhythm_target": "long",
                    "diction": "elevated",
                },
                "terminology": [],
                "named_entities": [],
            }),
            encoding="utf-8",
        )

    with patch("agents_pipeline.runner.run_jobs_parallel", side_effect=fake_run):
        run_style_analysis(book=book, cfg=cfg)

    src_file = cfg.run_dir / "inputs" / "full_source.txt"
    assert src_file.is_file()
    assert src_file.read_text(encoding="utf-8") == "Bonjour le monde"


def test_run_style_analysis_raises_if_output_missing(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    book = _make_book_mock([_make_chunk_mock("text")])
    book.get_src_text.return_value = "text"

    with patch("agents_pipeline.runner.run_jobs_parallel", return_value=None):
        with pytest.raises(FileNotFoundError, match="style_bible.json"):
            run_style_analysis(book=book, cfg=cfg)


def test_build_translate_jobs_includes_style_bible_in_allowed_inputs(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    book = _make_book_mock([_make_chunk_mock("Bonjour")])

    jobs = build_translate_jobs(book=book, cfg=cfg, cycle=1)

    assert "outputs/style_bible.json" in jobs[0].allowed_inputs


def test_build_revise_jobs_includes_style_bible_in_allowed_inputs(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock("text")]
    book = _make_book_mock(chunks)
    book.translation = [_make_chunk_mock("draft")]
    outputs_dir = cfg.run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "litrans_review_0000.json").write_text(
        json.dumps(
            _make_litrans_review(
                chunk_id=0,
                score=0.4,
                verdict="FAIL",
                failed_ids=["1", "2"],
            )
        )
    )
    (outputs_dir / "segment_translation_0000.json").write_text(
        json.dumps({"chunk_id": 0, "translation": "Draft"})
    )
    (outputs_dir / "chunk_literary_review_0000.json").write_text(
        json.dumps({
            "chunk_id": 0,
            "verdict": "PASS",
            "verdicts": {
                "accuracy": "PASS",
                "voice": "PASS",
                "dialogue": "PASS",
                "prose": "PASS",
            },
            "findings": [],
            "summary": "ok",
        })
    )

    jobs = build_revise_jobs(book=book, cfg=cfg, cycle=2, failing_chunk_ids=[0])

    assert "outputs/style_bible.json" in jobs[0].allowed_inputs


def test_build_chunk_literary_review_jobs_warns_against_raw_ascii_quotes(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    chunks = [_make_chunk_mock("Bonjour")]
    book = _make_book_mock(chunks)
    book.translation = [_make_chunk_mock('He said "hello".')]

    jobs = build_chunk_literary_review_jobs(book=book, cfg=cfg, chunk_ids=[0], cycle=1)

    prompt = jobs[0].prompt_text
    assert "strict parseable JSON" in prompt
    assert "Do not put raw ASCII double quotes" in prompt
    assert "escape the ASCII quote" in prompt


def test_persist_and_load_run_args_roundtrip(tmp_path):
    cfg = _make_cfg(tmp_path)
    run_args_path = cfg.run_dir / "run_args.json"
    run_args_path.parent.mkdir(parents=True, exist_ok=True)

    _persist_run_args(cfg=cfg, path=run_args_path)

    loaded = _load_run_args(run_args_path=run_args_path, run_dir=cfg.run_dir)

    assert loaded.book_path == cfg.book_path
    assert loaded.run_dir == cfg.run_dir
    assert loaded.stage_models == cfg.stage_models
    assert loaded.max_cycles == cfg.max_cycles
    assert loaded.max_global_cycles == cfg.max_global_cycles
    assert loaded.max_parallel_jobs == cfg.max_parallel_jobs
    assert loaded.litrans_threshold == cfg.litrans_threshold
    assert loaded.dry_run == cfg.dry_run
    assert loaded.job_timeout_seconds == cfg.job_timeout_seconds


def test_persist_run_args_is_idempotent_overwrite(tmp_path):
    cfg = _make_cfg(tmp_path)
    path = cfg.run_dir / "run_args.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _persist_run_args(cfg=cfg, path=path)
    first = path.read_text(encoding="utf-8")
    _persist_run_args(cfg=cfg, path=path)
    assert path.read_text(encoding="utf-8") == first


# ---------------------------------------------------------------------------
# _detect_resume_point tests (Tasks 3-6)
# ---------------------------------------------------------------------------

from agents_pipeline.core.job_spec import ResumePoint
from agents_pipeline.runner import _detect_resume_point


def test_detect_resume_point_empty_run_dir_is_fresh_run(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "gate").mkdir()

    rp = _detect_resume_point(cfg)

    assert rp == ResumePoint(
        start_cycle=1,
        initial_failing_chunks=(),
        skip_main_loop=False,
        start_global_cycle=1,
        skip_final_revision=False,
    )


# ---------------------------------------------------------------------------
# Task 4: main-loop FAIL mid-budget
# ---------------------------------------------------------------------------

def _write_main_gate(run_dir: Path, cycle: int, gate: dict) -> None:
    p = run_dir / "gate" / f"cycle_{cycle:02d}" / "gate.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gate), encoding="utf-8")


def test_detect_resume_point_main_loop_fail_mid_budget(tmp_path):
    cfg = _make_cfg(tmp_path)  # max_cycles=3
    _write_main_gate(
        cfg.run_dir,
        cycle=1,
        gate={
            "cycle": 1,
            "decision": "FAIL",
            "failing_chunk_ids": [2, 5, 7],
            "qe_failing_chunk_ids": [2],
            "literary_failing_chunk_ids": [5, 7],
            "reason": "qe_or_chunk_literary_review_failed",
        },
    )

    rp = _detect_resume_point(cfg)

    assert rp.start_cycle == 2
    assert rp.initial_failing_chunks == (2, 5, 7)
    assert rp.skip_main_loop is False
    assert rp.start_global_cycle == 1
    assert rp.skip_final_revision is False


# ---------------------------------------------------------------------------
# Task 5: main-loop PASS or budget exhausted
# ---------------------------------------------------------------------------

def test_detect_resume_point_main_loop_pass_no_final_gate(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_main_gate(
        cfg.run_dir,
        cycle=2,
        gate={
            "cycle": 2,
            "decision": "PASS",
            "failing_chunk_ids": [],
            "qe_failing_chunk_ids": [],
            "literary_failing_chunk_ids": [],
            "reason": "all_chunk_gates_passed",
        },
    )

    rp = _detect_resume_point(cfg)

    assert rp.skip_main_loop is True
    assert rp.start_cycle == 1
    assert rp.initial_failing_chunks == ()
    assert rp.start_global_cycle == 1
    assert rp.skip_final_revision is False


def test_detect_resume_point_main_loop_budget_exhausted(tmp_path):
    cfg = _make_cfg(tmp_path)  # max_cycles=3
    _write_main_gate(
        cfg.run_dir,
        cycle=3,
        gate={
            "cycle": 3,
            "decision": "FAIL",
            "failing_chunk_ids": [4],
            "qe_failing_chunk_ids": [],
            "literary_failing_chunk_ids": [4],
            "reason": "qe_or_chunk_literary_review_failed",
        },
    )

    rp = _detect_resume_point(cfg)

    assert rp.skip_main_loop is True
    assert rp.start_global_cycle == 1
    assert rp.skip_final_revision is False


# ---------------------------------------------------------------------------
# Task 6: final revision branches + inconsistency guard
# ---------------------------------------------------------------------------

def _write_final_gate_file(run_dir: Path, cycle: int, gate: dict) -> None:
    p = run_dir / "gate" / "final" / f"cycle_{cycle:02d}" / "gate.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gate), encoding="utf-8")


def test_detect_resume_point_final_revision_pass(tmp_path):
    cfg = _make_cfg(tmp_path)  # max_global_cycles=2
    _write_main_gate(
        cfg.run_dir,
        cycle=1,
        gate={"cycle": 1, "decision": "PASS", "failing_chunk_ids": [],
              "qe_failing_chunk_ids": [], "literary_failing_chunk_ids": [],
              "reason": "all_chunk_gates_passed"},
    )
    _write_final_gate_file(
        cfg.run_dir,
        cycle=1,
        gate={"global_cycle": 1, "verdict": "PASS",
              "affected_chunk_ids": [],
              "book_review_affected_chunk_ids": [],
              "cross_chunk_audit_affected_chunk_ids": [],
              "confirmed": True, "reason": "all_chunks_clean"},
    )

    rp = _detect_resume_point(cfg)

    assert rp.skip_main_loop is True
    assert rp.skip_final_revision is True


def test_detect_resume_point_final_revision_fail_mid_budget(tmp_path):
    cfg = _make_cfg(tmp_path)  # max_global_cycles=2
    _write_main_gate(
        cfg.run_dir,
        cycle=1,
        gate={"cycle": 1, "decision": "PASS", "failing_chunk_ids": [],
              "qe_failing_chunk_ids": [], "literary_failing_chunk_ids": [],
              "reason": "all_chunk_gates_passed"},
    )
    _write_final_gate_file(
        cfg.run_dir,
        cycle=1,
        gate={"global_cycle": 1, "verdict": "FAIL",
              "affected_chunk_ids": [3],
              "book_review_affected_chunk_ids": [3],
              "cross_chunk_audit_affected_chunk_ids": [],
              "confirmed": False,
              "reason": "book_review_or_cross_chunk_audit_found_medium_plus_issues"},
    )

    rp = _detect_resume_point(cfg)

    assert rp.skip_main_loop is True
    assert rp.skip_final_revision is False
    assert rp.start_global_cycle == 1


def test_detect_resume_point_final_revision_fail_after_completed_revisions(tmp_path):
    cfg = _make_cfg(tmp_path)  # max_global_cycles=2
    _write_main_gate(
        cfg.run_dir,
        cycle=1,
        gate={"cycle": 1, "decision": "PASS", "failing_chunk_ids": [],
              "qe_failing_chunk_ids": [], "literary_failing_chunk_ids": [],
              "reason": "all_chunk_gates_passed"},
    )
    _write_final_gate_file(
        cfg.run_dir,
        cycle=1,
        gate={"global_cycle": 1, "verdict": "FAIL",
              "affected_chunk_ids": [3],
              "book_review_affected_chunk_ids": [3],
              "cross_chunk_audit_affected_chunk_ids": [],
              "confirmed": False,
              "reason": "book_review_or_cross_chunk_audit_found_medium_plus_issues"},
    )
    outputs = cfg.run_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "segment_translation_0003.json").write_text(
        json.dumps({"chunk_id": 3, "translation": "fixed"}),
        encoding="utf-8",
    )
    (outputs / "final_revise_0003_cycle_01.json").write_text(
        json.dumps({"chunk_id": 3, "findings_addressed": []}),
        encoding="utf-8",
    )
    metrics = cfg.run_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "run_summary.json").write_text(
        json.dumps({
            "jobs": [
                {
                    "job_id": "final_revise_dev_chunk_0003_global_cycle_01",
                    "status": "success",
                }
            ],
            "chunk_states": {},
            "updated_at": None,
        }),
        encoding="utf-8",
    )

    rp = _detect_resume_point(cfg)

    assert rp.skip_main_loop is True
    assert rp.skip_final_revision is False
    assert rp.start_global_cycle == 2


def test_detect_resume_point_final_revision_budget_exhausted(tmp_path):
    cfg = _make_cfg(tmp_path)  # max_global_cycles=2
    _write_main_gate(
        cfg.run_dir,
        cycle=1,
        gate={"cycle": 1, "decision": "PASS", "failing_chunk_ids": [],
              "qe_failing_chunk_ids": [], "literary_failing_chunk_ids": [],
              "reason": "all_chunk_gates_passed"},
    )
    _write_final_gate_file(
        cfg.run_dir,
        cycle=2,
        gate={"global_cycle": 2, "verdict": "FAIL",
              "affected_chunk_ids": [3],
              "book_review_affected_chunk_ids": [3],
              "cross_chunk_audit_affected_chunk_ids": [],
              "confirmed": True,
              "reason": "book_review_or_cross_chunk_audit_found_medium_plus_issues"},
    )

    rp = _detect_resume_point(cfg)

    assert rp.skip_main_loop is True
    assert rp.skip_final_revision is True


def test_detect_resume_point_raises_on_inconsistent_run_dir(tmp_path):
    """Final-revision gate exists but no main-loop gate — invalid state."""
    cfg = _make_cfg(tmp_path)
    _write_final_gate_file(
        cfg.run_dir,
        cycle=1,
        gate={"global_cycle": 1, "verdict": "PASS",
              "affected_chunk_ids": [],
              "book_review_affected_chunk_ids": [],
              "cross_chunk_audit_affected_chunk_ids": [],
              "confirmed": True, "reason": "all_chunks_clean"},
    )

    from agents_pipeline.core.job_spec import PipelineError
    with pytest.raises(PipelineError, match="inconsistent"):
        _detect_resume_point(cfg)


def test_run_style_analysis_skips_when_style_bible_exists(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    (cfg.run_dir / "outputs").mkdir(parents=True)
    existing = cfg.run_dir / "outputs" / "style_bible.json"
    existing.write_text('{"already": "here"}', encoding="utf-8")

    called = {"count": 0}

    def fake_run_jobs_parallel(*args, **kwargs):
        called["count"] += 1

    monkeypatch.setattr("agents_pipeline.runner.run_jobs_parallel", fake_run_jobs_parallel)

    book = _make_book_mock([_make_chunk_mock("x")])
    book.get_src_text.return_value = "x"
    run_style_analysis(book=book, cfg=cfg)

    assert called["count"] == 0
    assert existing.read_text(encoding="utf-8") == '{"already": "here"}'


def test_run_final_revision_starts_at_given_global_cycle(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)  # max_global_cycles=2
    (cfg.run_dir / "outputs").mkdir(parents=True)

    labels_seen: list[str] = []

    def fake_run_jobs_parallel(jobs, run_dir, max_parallel, dry_run, label):
        labels_seen.append(label)

    monkeypatch.setattr("agents_pipeline.runner.run_jobs_parallel", fake_run_jobs_parallel)

    monkeypatch.setattr(
        "agents_pipeline.runner.reconstruct_and_save",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "agents_pipeline.runner._parse_book_review",
        lambda run_dir: {},
    )
    monkeypatch.setattr(
        "agents_pipeline.runner._parse_cross_chunk_audit",
        lambda run_dir: {},
    )

    book = _make_book_mock([_make_chunk_mock("x")])
    book.get_translation = MagicMock(return_value="translated text")

    run_final_revision(book=book, cfg=cfg, start_global_cycle=2)

    assert any("cycle_02" in lbl for lbl in labels_seen)
    assert not any("cycle_01" in lbl for lbl in labels_seen)


# ---------------------------------------------------------------------------
# Task 9: run() accepts ResumePoint and honors it
# ---------------------------------------------------------------------------

def _write_segment_translations(cfg: RunnerConfig, n_chunks: int) -> None:
    """Write n dummy segment_translation files so load_translations_into_chunks works."""
    outputs = cfg.run_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    for i in range(n_chunks):
        (outputs / f"segment_translation_{i:04d}.json").write_text(
            json.dumps({"translation": f"translation of chunk {i}"}),
            encoding="utf-8",
        )


def test_run_with_resume_point_skip_main_loop_goes_to_final_revision(tmp_path, monkeypatch):
    from agents_pipeline.core.job_spec import ResumePoint
    from agents_pipeline.runner import run

    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "inputs").mkdir()
    (cfg.run_dir / "inputs" / "full_source.txt").write_text("source text", encoding="utf-8")
    _write_segment_translations(cfg, n_chunks=1)

    calls: list[str] = []

    monkeypatch.setattr(
        "agents_pipeline.runner.run_style_analysis",
        lambda **kwargs: calls.append("style_analysis"),
    )
    monkeypatch.setattr(
        "agents_pipeline.runner.run_jobs_parallel",
        lambda *a, **kw: calls.append(f"run_jobs_parallel:{kw.get('label') or a[-1]}"),
    )
    monkeypatch.setattr(
        "agents_pipeline.runner.load_translations_into_chunks",
        lambda **kwargs: calls.append("load_translations"),
    )
    monkeypatch.setattr(
        "agents_pipeline.runner.reconstruct_and_save",
        lambda **kwargs: calls.append("reconstruct"),
    )
    monkeypatch.setattr(
        "agents_pipeline.runner.run_final_revision",
        lambda **kwargs: calls.append(
            f"final_revision:start={kwargs['start_global_cycle']}"
        ),
    )

    rp = ResumePoint(
        start_cycle=1,
        initial_failing_chunks=(),
        skip_main_loop=True,
        start_global_cycle=2,
        skip_final_revision=False,
    )

    def fake_book_init(path, src_text, chunks):
        book = _make_book_mock([_make_chunk_mock("chunk 0")])
        book.name = "dev"
        return book

    monkeypatch.setattr("agents_pipeline.runner.Book", fake_book_init)

    run(cfg, resume_point=rp)

    assert "style_analysis" in calls
    assert not any("run_jobs_parallel:cycle_" in c for c in calls)
    assert "load_translations" in calls
    assert "final_revision:start=2" in calls


def test_run_with_resume_point_skip_final_revision_exits_after_main_loop(tmp_path, monkeypatch):
    from agents_pipeline.core.job_spec import ResumePoint
    from agents_pipeline.runner import run

    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "inputs").mkdir()
    (cfg.run_dir / "inputs" / "full_source.txt").write_text("source", encoding="utf-8")
    _write_segment_translations(cfg, n_chunks=1)

    calls: list[str] = []

    monkeypatch.setattr("agents_pipeline.runner.run_style_analysis", lambda **kw: None)
    monkeypatch.setattr("agents_pipeline.runner.run_jobs_parallel", lambda *a, **kw: None)
    monkeypatch.setattr(
        "agents_pipeline.runner.load_translations_into_chunks",
        lambda **kw: None,
    )
    monkeypatch.setattr(
        "agents_pipeline.runner.run_final_revision",
        lambda **kwargs: calls.append("final_revision"),
    )

    def fake_book_init(path, src_text, chunks):
        book = _make_book_mock([_make_chunk_mock("chunk 0")])
        book.name = "dev"
        return book

    monkeypatch.setattr("agents_pipeline.runner.Book", fake_book_init)

    rp = ResumePoint(skip_main_loop=True, skip_final_revision=True)
    run(cfg, resume_point=rp)

    assert "final_revision" not in calls


def test_run_without_resume_point_preserves_fresh_run_behavior(tmp_path, monkeypatch):
    """Passing resume_point=None must behave exactly like the pre-resume code."""
    from agents_pipeline.runner import run

    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "inputs").mkdir()
    (cfg.run_dir / "inputs" / "full_source.txt").write_text("source", encoding="utf-8")

    monkeypatch.setattr("agents_pipeline.runner.run_style_analysis", lambda **kw: None)
    monkeypatch.setattr("agents_pipeline.runner.build_translate_jobs", lambda **kw: [])
    monkeypatch.setattr("agents_pipeline.runner.build_revise_jobs", lambda **kw: [])
    monkeypatch.setattr("agents_pipeline.runner.run_jobs_parallel", lambda *a, **kw: None)
    monkeypatch.setattr("agents_pipeline.runner.load_translations_into_chunks", lambda **kw: None)
    monkeypatch.setattr(
        "agents_pipeline.runner.run_gate",
        lambda **kw: {"failing_chunk_ids": []},
    )
    monkeypatch.setattr(
        "agents_pipeline.runner.run_chunk_literary_gate",
        lambda **kw: {"failing_chunk_ids": []},
    )
    monkeypatch.setattr(
        "agents_pipeline.runner.write_master_chunk_gate",
        lambda **kw: {
            "decision": "PASS", "failing_chunk_ids": [],
            "qe_failing_chunk_ids": [], "literary_failing_chunk_ids": [],
        },
    )
    monkeypatch.setattr(
        "agents_pipeline.runner._update_chunk_state_for_cycle",
        lambda **kw: [],
    )
    called = {"final": 0}
    monkeypatch.setattr(
        "agents_pipeline.runner.run_final_revision",
        lambda **kw: called.__setitem__("final", called["final"] + 1),
    )
    monkeypatch.setattr("agents_pipeline.runner.reconstruct_and_save", lambda **kw: None)

    def fake_book_init(path, src_text, chunks):
        book = _make_book_mock([_make_chunk_mock("chunk 0")])
        book.name = "dev"
        return book

    monkeypatch.setattr("agents_pipeline.runner.Book", fake_book_init)

    run(cfg)  # No resume_point.

    assert called["final"] == 1


def test_main_rejects_resume_with_book_path(tmp_path, monkeypatch):
    from agents_pipeline.runner import main

    run_dir = tmp_path / "runs" / "dev_test"
    run_dir.mkdir(parents=True)

    argv = [
        "runner.py",
        "--resume", str(run_dir),
        "--book_path", str(tmp_path / "book.txt"),
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit):
        main()


def test_main_rejects_no_resume_and_no_book_path(monkeypatch):
    from agents_pipeline.runner import main

    argv = ["runner.py"]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit):
        main()


def test_run_resume_at_cycle_gt_1_loads_translations_before_build_revise(tmp_path, monkeypatch):
    """Regression: resuming at cycle > 1 must load translations before build_revise_jobs,
    otherwise book.translation is None and indexing crashes."""
    from agents_pipeline.core.job_spec import ResumePoint
    from agents_pipeline.runner import run

    cfg = _make_cfg(tmp_path)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "inputs").mkdir()
    (cfg.run_dir / "inputs" / "full_source.txt").write_text("src", encoding="utf-8")
    _write_segment_translations(cfg, n_chunks=2)

    load_call_order: list[str] = []

    def fake_load(*, book, cfg):
        load_call_order.append("load")
        # Simulate what the real loader does: populate book.translation.
        book.translation = [_make_chunk_mock("t0"), _make_chunk_mock("t1")]

    def fake_build_revise(*, book, cfg, cycle, failing_chunk_ids, stall_contexts=None):
        load_call_order.append("build_revise")
        # This is the line that crashed in production:
        _ = book.translation[0]
        return []

    monkeypatch.setattr("agents_pipeline.runner.run_style_analysis", lambda **kw: None)
    monkeypatch.setattr("agents_pipeline.runner.load_translations_into_chunks", fake_load)
    monkeypatch.setattr("agents_pipeline.runner.build_revise_jobs", fake_build_revise)
    monkeypatch.setattr("agents_pipeline.runner.run_jobs_parallel", lambda *a, **kw: None)
    monkeypatch.setattr(
        "agents_pipeline.runner.run_gate",
        lambda **kw: {"failing_chunk_ids": []},
    )
    monkeypatch.setattr(
        "agents_pipeline.runner.run_chunk_literary_gate",
        lambda **kw: {"failing_chunk_ids": []},
    )
    monkeypatch.setattr(
        "agents_pipeline.runner.write_master_chunk_gate",
        lambda **kw: {
            "decision": "PASS", "failing_chunk_ids": [],
            "qe_failing_chunk_ids": [], "literary_failing_chunk_ids": [],
        },
    )
    monkeypatch.setattr(
        "agents_pipeline.runner._update_chunk_state_for_cycle",
        lambda **kw: [],
    )
    monkeypatch.setattr("agents_pipeline.runner.reconstruct_and_save", lambda **kw: None)
    monkeypatch.setattr("agents_pipeline.runner.run_final_revision", lambda **kw: None)

    def fake_book_init(path, src_text, chunks):
        book = _make_book_mock([_make_chunk_mock("c0"), _make_chunk_mock("c1")])
        book.name = "dev"
        book.translation = None
        return book

    monkeypatch.setattr("agents_pipeline.runner.Book", fake_book_init)

    rp = ResumePoint(start_cycle=3, initial_failing_chunks=(0, 1))

    run(cfg, resume_point=rp)

    assert load_call_order[0] == "load", (
        f"load_translations_into_chunks must run BEFORE build_revise_jobs. "
        f"Call order: {load_call_order}"
    )
    assert "build_revise" in load_call_order
