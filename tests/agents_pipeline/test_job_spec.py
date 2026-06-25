# tests/agents_pipeline/test_job_spec.py
import pytest
from pathlib import Path
from agents_pipeline.core.job_spec import (
    JobSpec, RunnerConfig, PipelineError, ProviderQuotaPause, STAGE_NAMES
)


def _make_stage_models():
    return {
        stage: {"provider": "claude", "model": None}
        for stage in STAGE_NAMES
    }

def test_jobspec_is_frozen():
    job = JobSpec(
        job_id="test_job",
        stage="translate",
        chunk_id=0,
        book_name="dev",
        allowed_inputs=["inputs/source_chunk_0000.txt"],
        required_outputs=["outputs/segment_translation_0000.json"],
        prompt_text="Translate this.",
        provider="claude",
        model=None,
    )
    with pytest.raises((AttributeError, TypeError)):
        job.chunk_id = 99  # type: ignore

def test_jobspec_default_timeout():
    job = JobSpec(
        job_id="j", stage="translate", chunk_id=0, book_name="b",
        allowed_inputs=[], required_outputs=[], prompt_text="",
        provider="claude", model=None,
    )
    assert job.timeout_seconds == 3600

def test_runner_config_fields():
    cfg = RunnerConfig(
        book_path=Path("data/dev_fr.txt"),
        run_dir=Path("runs/dev_20260401"),
        stage_models=_make_stage_models(),
        max_cycles=3,
        max_global_cycles=2,
        max_parallel_jobs=4,
        litrans_threshold=0.7,
        target_language_code="en",
        target_language="English",
    )
    assert cfg.dry_run is False
    assert cfg.job_timeout_seconds == 3600

def test_pipeline_error_is_runtime_error():
    exc = PipelineError("something broke")
    assert isinstance(exc, RuntimeError)

def test_provider_quota_pause_fields():
    exc = ProviderQuotaPause(
        provider="claude",
        result_text="quota hit",
        reset_at_epoch=9999999,
        sleep_seconds=60,
        rate_limit_type="daily",
    )
    assert exc.provider == "claude"
    assert exc.sleep_seconds == 60
    assert isinstance(exc, PipelineError)

def test_env_non_negative_int_missing(monkeypatch):
    from agents_pipeline.core import job_spec
    monkeypatch.delenv("JOB_EXEC_RETRY_MAX", raising=False)
    assert job_spec._env_non_negative_int("JOB_EXEC_RETRY_MAX", 5) == 5

def test_env_non_negative_int_invalid(monkeypatch):
    from agents_pipeline.core import job_spec
    monkeypatch.setenv("JOB_EXEC_RETRY_MAX", "not_a_number")
    assert job_spec._env_non_negative_int("JOB_EXEC_RETRY_MAX", 5) == 5

def test_env_non_negative_int_negative(monkeypatch):
    from agents_pipeline.core import job_spec
    monkeypatch.setenv("JOB_EXEC_RETRY_MAX", "-3")
    assert job_spec._env_non_negative_int("JOB_EXEC_RETRY_MAX", 5) == 0

def test_runner_config_is_frozen():
    cfg = RunnerConfig(
        book_path=Path("data/dev_fr.txt"),
        run_dir=Path("runs/dev"),
        stage_models=_make_stage_models(),
        max_cycles=3,
        max_global_cycles=2,
        max_parallel_jobs=4,
        litrans_threshold=0.7,
        target_language_code="en",
        target_language="English",
    )
    with pytest.raises((AttributeError, TypeError)):
        cfg.max_cycles = 99  # type: ignore

def test_provider_quota_pause_none_text():
    exc = ProviderQuotaPause(
        provider="codex",
        result_text="",
        reset_at_epoch=0,
        sleep_seconds=0,
        rate_limit_type=None,
    )
    assert "codex" in str(exc)
    assert exc.rate_limit_type == ""


# ResumePoint tests
from agents_pipeline.core.job_spec import ResumePoint


def test_resume_point_defaults_to_fresh_run():
    rp = ResumePoint()
    assert rp.start_cycle == 1
    assert rp.initial_failing_chunks == ()
    assert rp.skip_main_loop is False
    assert rp.start_global_cycle == 1
    assert rp.skip_final_revision is False


def test_resume_point_is_frozen():
    rp = ResumePoint()
    with pytest.raises((AttributeError, Exception)):
        rp.start_cycle = 5  # type: ignore[misc]


def test_resume_point_accepts_custom_values():
    rp = ResumePoint(
        start_cycle=3,
        initial_failing_chunks=(1, 4, 7),
        skip_main_loop=False,
        start_global_cycle=1,
        skip_final_revision=False,
    )
    assert rp.start_cycle == 3
    assert rp.initial_failing_chunks == (1, 4, 7)
