# agents_pipeline/core/job_spec.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_non_negative_int(name: str, default: int) -> int:
    """Read an env var as a non-negative int, falling back to default.

    Returns default if the variable is missing, empty, non-integer, or negative.
    Negative values are clamped to 0, not replaced with default.
    """
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


JOB_EXEC_RETRY_MAX = _env_non_negative_int("JOB_EXEC_RETRY_MAX", 2)
JOB_EXEC_RETRY_BASE_SLEEP_SECONDS = _env_non_negative_int("JOB_EXEC_RETRY_BASE_SLEEP_SECONDS", 2)
JOB_EXEC_RETRY_MAX_SLEEP_SECONDS = _env_non_negative_int("JOB_EXEC_RETRY_MAX_SLEEP_SECONDS", 30)
CLAUDE_QUOTA_RESET_BUFFER_SECONDS = _env_non_negative_int("CLAUDE_QUOTA_RESET_BUFFER_SECONDS", 120)
CLAUDE_QUOTA_RESET_JITTER_SECONDS = _env_non_negative_int("CLAUDE_QUOTA_RESET_JITTER_SECONDS", 30)

PROVIDER_VALUES = {"claude", "codex"}

STAGE_NAMES: frozenset[str] = frozenset({
    "style_analysis",
    "translate",
    "litrans_review",
    "chunk_literary_review",
    "revise",
    "book_review",
    "cross_chunk_audit",
    "final_revise",
})


class PipelineError(RuntimeError):
    """Raised for all recoverable and unrecoverable pipeline failures."""


class ProviderQuotaPause(PipelineError):
    """Raised when a provider signals a quota reset; the runner sleeps and retries."""

    def __init__(
        self,
        *,
        provider: str,
        result_text: str,
        reset_at_epoch: int,
        sleep_seconds: int,
        rate_limit_type: str | None,
    ) -> None:
        self.provider = provider
        self.result_text = result_text
        self.reset_at_epoch = reset_at_epoch
        self.sleep_seconds = sleep_seconds
        self.rate_limit_type = rate_limit_type or ""
        super().__init__(result_text or f"{provider} quota exhausted")


@dataclass(frozen=True)
class JobSpec:
    """Declarative description of one isolated agent job."""

    job_id: str
    stage: str              # pipeline stage name from STAGE_NAMES
    chunk_id: int
    book_name: str
    allowed_inputs: list[str]    # relative paths from run_dir copied into workspace
    required_outputs: list[str]  # relative paths the agent must produce in workspace
    prompt_text: str
    provider: str           # "claude" | "codex"
    model: str | None
    timeout_seconds: int = 3600
    cycle_number: int | None = None
    global_cycle_number: int | None = None


@dataclass(frozen=True)
class RunnerConfig:
    """Full configuration for one MT pipeline run.

    `stage_models` maps each stage name in `STAGE_NAMES` to a dict with
    keys `provider` (in `PROVIDER_VALUES`) and `model` (str or None).
    """

    book_path: Path
    run_dir: Path
    stage_models: dict[str, dict[str, str | None]]
    max_cycles: int
    max_global_cycles: int
    max_parallel_jobs: int
    litrans_threshold: float      # legacy informational QE threshold for reporting
    target_language_code: str
    target_language: str
    dry_run: bool = False
    job_timeout_seconds: int = 3600

    def stage(self, name: str) -> tuple[str, str | None]:
        """Return (provider, model) for a stage; raise KeyError if missing."""
        entry = self.stage_models[name]
        return entry["provider"], entry.get("model")


@dataclass(frozen=True)
class ResumePoint:
    """State needed to resume an interrupted pipeline run at the last
    fully-completed cycle boundary.

    A default-constructed ResumePoint represents a fresh run: start the
    main loop at cycle 1 with no pre-existing failing chunks, and start
    the final-revision loop at cycle 1. Nothing is skipped.
    """

    start_cycle: int = 1
    initial_failing_chunks: tuple[int, ...] = ()
    skip_main_loop: bool = False
    start_global_cycle: int = 1
    skip_final_revision: bool = False
