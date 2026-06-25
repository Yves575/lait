# tests/agents_pipeline/test_executor.py
import hashlib
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from agents_pipeline.core.job_spec import JobSpec, PipelineError
from agents_pipeline.core.executor import (
    _build_claude_exec_cmd,
    _sha256_file,
    _copy_inputs_to_workspace,
    _prepare_required_output_dirs,
    _validate_required_output_content,
    _validate_job_workspace,
    _list_workspace_files,
    _is_allowed_aux_workspace_file,
    _assert_rel_path,
    _retry_backoff_seconds,
    _is_retryable_error,
    _mock_job_outputs,
    run_jobs_parallel,
)


def _make_job(**overrides) -> JobSpec:
    defaults = dict(
        job_id="test_translate_0000",
        stage="translate",
        chunk_id=0,
        book_name="dev",
        allowed_inputs=["inputs/source_chunk_0000.txt"],
        required_outputs=["outputs/segment_translation_0000.json"],
        prompt_text="Translate this.",
        provider="claude",
        model=None,
    )
    defaults.update(overrides)
    return JobSpec(**defaults)


def test_sha256_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert _sha256_file(f) == expected


def test_copy_inputs_to_workspace(tmp_path):
    run_dir = tmp_path / "run"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    src = run_dir / "inputs" / "source_chunk_0000.txt"
    src.parent.mkdir(parents=True)
    src.write_text("Bonjour le monde", encoding="utf-8")

    hashes = _copy_inputs_to_workspace(
        ["inputs/source_chunk_0000.txt"], workspace, run_dir
    )
    assert (workspace / "inputs" / "source_chunk_0000.txt").read_text() == "Bonjour le monde"
    assert "inputs/source_chunk_0000.txt" in hashes


def test_copy_inputs_missing_raises(tmp_path):
    run_dir = tmp_path / "run"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PipelineError, match="missing required input"):
        _copy_inputs_to_workspace(["inputs/ghost.txt"], workspace, run_dir)


def test_validate_workspace_missing_output_raises(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    job = _make_job()
    with pytest.raises(PipelineError, match="required output missing"):
        _validate_job_workspace(job, workspace, before_hashes={})


def test_validate_workspace_input_mutation_raises(tmp_path):
    run_dir = tmp_path / "run"
    workspace = tmp_path / "ws"
    (workspace / "inputs").mkdir(parents=True)
    (workspace / "outputs").mkdir(parents=True)

    src = workspace / "inputs" / "source_chunk_0000.txt"
    src.write_text("original")

    out = workspace / "outputs" / "segment_translation_0000.json"
    out.write_text(json.dumps({"chunk_id": 0, "translation": "Hello world"}))

    before_hashes = {"inputs/source_chunk_0000.txt": _sha256_file(src)}
    src.write_text("mutated!")  # simulate agent mutating input

    job = _make_job()
    with pytest.raises(PipelineError, match="input file mutated"):
        _validate_job_workspace(job, workspace, before_hashes=before_hashes)


def test_validate_workspace_invalid_segment_translation_json_raises(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "outputs").mkdir(parents=True)
    out = workspace / "outputs" / "segment_translation_0000.json"
    out.write_text('{"chunk_id": 0, "translation": ""broken"}', encoding="utf-8")

    job = _make_job()
    with pytest.raises(PipelineError, match="invalid JSON in required output"):
        _validate_job_workspace(job, workspace, before_hashes={})


def test_validate_required_output_content_rejects_wrong_chunk_id(tmp_path):
    out = tmp_path / "segment_translation_0000.json"
    out.write_text(json.dumps({"chunk_id": 9, "translation": "Hello world"}), encoding="utf-8")

    job = _make_job()
    with pytest.raises(PipelineError, match="expected 0"):
        _validate_required_output_content(job, "outputs/segment_translation_0000.json", out)


def test_validate_required_output_content_rejects_invalid_chunk_literary_review_json(tmp_path):
    out = tmp_path / "chunk_literary_review_0000.json"
    out.write_text('{"chunk_id": 0, "verdict": "PASS", "summary": "broken"', encoding="utf-8")

    job = _make_job(
        stage="chunk_literary_review",
        required_outputs=["outputs/chunk_literary_review_0000.json"],
    )
    with pytest.raises(PipelineError, match="invalid JSON in required output"):
        _validate_required_output_content(job, "outputs/chunk_literary_review_0000.json", out)


def test_validate_required_output_content_rejects_chunk_literary_review_wrong_chunk_id(tmp_path):
    out = tmp_path / "chunk_literary_review_0000.json"
    out.write_text(
        json.dumps(
            {
                "chunk_id": 9,
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
        ),
        encoding="utf-8",
    )

    job = _make_job(
        stage="chunk_literary_review",
        required_outputs=["outputs/chunk_literary_review_0000.json"],
    )
    with pytest.raises(PipelineError, match="expected 0"):
        _validate_required_output_content(job, "outputs/chunk_literary_review_0000.json", out)


def test_is_retryable_error_invalid_json_output():
    assert _is_retryable_error(PipelineError("invalid JSON in required output outputs/segment_translation_0000.json"))


def test_list_workspace_files(tmp_path):
    ws = tmp_path / "ws"
    (ws / "outputs").mkdir(parents=True)
    (ws / "outputs" / "foo.json").write_text("{}")
    (ws / "inputs").mkdir()
    (ws / "inputs" / "bar.txt").write_text("x")
    files = _list_workspace_files(ws)
    assert files == {"outputs/foo.json", "inputs/bar.txt"}


def test_is_allowed_aux_workspace_file():
    assert _is_allowed_aux_workspace_file(".DS_Store")
    assert _is_allowed_aux_workspace_file(".claude.json")
    assert _is_allowed_aux_workspace_file(".codex/cache.json")
    assert not _is_allowed_aux_workspace_file("sneaky_output.json")


def test_assert_rel_path_rejects_absolute():
    with pytest.raises(PipelineError, match="must be relative"):
        _assert_rel_path("/etc/passwd")


def test_assert_rel_path_rejects_dotdot():
    with pytest.raises(PipelineError, match="must not include"):
        _assert_rel_path("../escape.txt")


def test_build_claude_exec_cmd_uses_bare_restricted_tools():
    job = _make_job(model="claude-opus-4-6")

    cmd = _build_claude_exec_cmd(job)

    assert cmd[:2] == ["claude", "--effort"]
    assert "max" in cmd
    assert "--bare" in cmd
    assert "--tools" in cmd
    assert cmd[cmd.index("--tools") + 1] == "Read,Write"
    assert "--allowedTools" in cmd
    assert cmd[cmd.index("--allowedTools") + 1] == "Read,Write"
    assert "--dangerously-skip-permissions" not in cmd
    assert "--output-format" in cmd
    assert "stream-json" in cmd
    assert "--verbose" in cmd
    assert "--no-session-persistence" in cmd
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-opus-4-6"


def test_prepare_required_output_dirs_creates_parent_dirs(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    job = _make_job(
        required_outputs=[
            "outputs/segment_translation_0000.json",
            "outputs/reports/final_revise_0000.json",
        ]
    )

    _prepare_required_output_dirs(job, workspace)

    assert (workspace / "outputs").is_dir()
    assert (workspace / "outputs" / "reports").is_dir()
    assert not (workspace / "outputs" / "segment_translation_0000.json").exists()
    assert not (workspace / "outputs" / "reports" / "final_revise_0000.json").exists()


def test_retry_backoff_zero_base():
    from agents_pipeline.core import executor
    original = executor.JOB_EXEC_RETRY_BASE_SLEEP_SECONDS
    executor.JOB_EXEC_RETRY_BASE_SLEEP_SECONDS = 0
    assert _retry_backoff_seconds(1) == 0
    executor.JOB_EXEC_RETRY_BASE_SLEEP_SECONDS = original


def test_retry_backoff_exponential():
    from agents_pipeline.core import executor
    executor.JOB_EXEC_RETRY_BASE_SLEEP_SECONDS = 2
    executor.JOB_EXEC_RETRY_MAX_SLEEP_SECONDS = 100
    assert _retry_backoff_seconds(1) == 2   # 2 * 2^0
    assert _retry_backoff_seconds(2) == 4   # 2 * 2^1
    assert _retry_backoff_seconds(3) == 8   # 2 * 2^2


def test_is_retryable_error():
    assert _is_retryable_error(PipelineError("claude exec failed rc=1"))
    assert _is_retryable_error(PipelineError("job timed out after 3600s"))
    assert _is_retryable_error(PipelineError("job did not produce required output"))
    assert not _is_retryable_error(PipelineError("unsupported provider: foobar"))


def test_run_jobs_parallel_empty(tmp_path):
    run_jobs_parallel([], tmp_path, max_workers=2, dry_run=True, label="test")


def test_run_jobs_parallel_collects_failures(tmp_path):
    (tmp_path / "workspaces").mkdir()
    job = _make_job()
    with pytest.raises(PipelineError, match="phase had job failures"):
        run_jobs_parallel([job], tmp_path, max_workers=1, dry_run=False, label="test")


def test_mock_job_outputs_book_review(tmp_path):
    job = JobSpec(
        job_id="book_review_dev_global_cycle_01",
        stage="book_review",
        chunk_id=-1,
        book_name="dev",
        allowed_inputs=[],
        required_outputs=["outputs/book_review.json"],
        prompt_text="...",
        provider="claude",
        model=None,
    )
    _mock_job_outputs(job, tmp_path)

    data = json.loads((tmp_path / "outputs" / "book_review.json").read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS"
    assert data["findings"] == []


def test_mock_job_outputs_final_revise(tmp_path):
    job = JobSpec(
        job_id="final_revise_dev_chunk_0002_global_cycle_01",
        stage="final_revise",
        chunk_id=2,
        book_name="dev",
        allowed_inputs=[],
        required_outputs=[
            "outputs/segment_translation_0002.json",
            "outputs/final_revise_0002_cycle_01.json",
        ],
        prompt_text="...",
        provider="claude",
        model=None,
    )
    _mock_job_outputs(job, tmp_path)

    trans = json.loads(
        (tmp_path / "outputs" / "segment_translation_0002.json").read_text(encoding="utf-8")
    )
    assert trans["chunk_id"] == 2
    assert "[DRY RUN" in trans["translation"]

    report = json.loads(
        (tmp_path / "outputs" / "final_revise_0002_cycle_01.json").read_text(encoding="utf-8")
    )
    assert report["chunk_id"] == 2
    assert report["findings_addressed"] == []


def test_mock_job_outputs_litrans_answers(tmp_path):
    job = JobSpec(
        job_id="litrans_review_dev_chunk_0001_cycle_01",
        stage="litrans_review",
        chunk_id=1,
        book_name="dev",
        allowed_inputs=[],
        required_outputs=["outputs/litrans_answers_0001.json"],
        prompt_text="...",
        provider="claude",
        model=None,
    )
    _mock_job_outputs(job, tmp_path)

    data = json.loads(
        (tmp_path / "outputs" / "litrans_answers_0001.json").read_text(encoding="utf-8")
    )
    assert sorted(data.keys(), key=int) == [str(i) for i in range(1, 26)]
    assert data["1"]["judgment"] == "YES"
    assert "issue" in data["1"]


def test_mock_job_outputs_cross_chunk_audit(tmp_path):
    job = JobSpec(
        job_id="cross_chunk_audit_dev_global_cycle_01",
        stage="cross_chunk_audit",
        chunk_id=-1,
        book_name="dev",
        allowed_inputs=[],
        required_outputs=["outputs/cross_chunk_audit.json"],
        prompt_text="...",
        provider="claude",
        model=None,
    )
    _mock_job_outputs(job, tmp_path)

    data = json.loads(
        (tmp_path / "outputs" / "cross_chunk_audit.json").read_text(encoding="utf-8")
    )
    assert data["verdict"] == "PASS"
    assert data["findings"] == []
