# agents_pipeline/core/executor.py
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from agents_pipeline.core.job_spec import (
    JobSpec,
    PipelineError,
    ProviderQuotaPause,
)
from agents_pipeline.core.gate import parse_litrans_answers

# These are imported into this module's namespace so tests can patch them directly
# on the executor module (e.g. executor.JOB_EXEC_RETRY_BASE_SLEEP_SECONDS = 0).
from agents_pipeline.core.job_spec import JOB_EXEC_RETRY_MAX  # noqa: F401
from agents_pipeline.core.job_spec import JOB_EXEC_RETRY_BASE_SLEEP_SECONDS  # noqa: F401
from agents_pipeline.core.job_spec import JOB_EXEC_RETRY_MAX_SLEEP_SECONDS  # noqa: F401
from agents_pipeline.core.job_spec import CLAUDE_QUOTA_RESET_BUFFER_SECONDS  # noqa: F401
from agents_pipeline.core.job_spec import CLAUDE_QUOTA_RESET_JITTER_SECONDS  # noqa: F401

import agents_pipeline.core.executor as _self  # used to read module-level vars at call time

POLL_INTERVAL = 10  # seconds between idle-watchdog checks
_RUN_SUMMARY_LOCK = threading.Lock()
_REVIEW_SEVERITIES: frozenset[str] = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
_CHUNK_LITERARY_LENSES: tuple[str, ...] = ("accuracy", "voice", "dialogue", "prose")
_CLAUDE_PIPELINE_PROMPT_PREFIX = (
    "The runner validates outputs after completion. Do not run shell commands "
    "or manual validation. Read only declared inputs, write required outputs, "
    "then stop.\n\n"
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_jobs_parallel(
    jobs: list[JobSpec],
    run_dir: Path,
    max_workers: int,
    dry_run: bool,
    label: str,
) -> None:
    """Dispatch jobs in parallel; raise PipelineError if any job fails."""
    if not jobs:
        return
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_job, job, run_dir, dry_run, label): job for job in jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            job = future_map[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{job.job_id}: {exc}")
    if failures:
        raise PipelineError(f"{label} phase had job failures:\n" + "\n".join(failures[:8]))


# ---------------------------------------------------------------------------
# Single-job execution loop (with retry + quota-pause)
# ---------------------------------------------------------------------------

def _run_job(job: JobSpec, run_dir: Path, dry_run: bool, phase_label: str) -> dict[str, Any]:
    """Run one job in an isolated workspace; retry transient failures."""
    started_at = _utc_now_iso()
    started_mono = time.monotonic()
    _write_manifest(job, run_dir)
    max_retries = _self.JOB_EXEC_RETRY_MAX if not dry_run else 0
    attempt = 0
    quota_pause_count = 0
    max_quota_pauses = 5
    execution_summary: dict[str, Any] = {}
    while True:
        attempt += 1
        (run_dir / "workspaces").mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(
                prefix=f"{job.job_id}_attempt_{attempt}_",
                dir=run_dir / "workspaces",
            )
        )
        try:
            before_hashes = _copy_inputs_to_workspace(
                job.allowed_inputs, workspace, run_dir
            )
            _prepare_required_output_dirs(job, workspace)
            if dry_run:
                _mock_job_outputs(job, workspace)
                execution_summary = {}
            else:
                execution_summary = _execute_agent_job(job, workspace, run_dir)
            _validate_job_workspace(job, workspace, before_hashes)
            _copy_outputs_from_workspace(job, workspace, run_dir)
            finished_at = _utc_now_iso()
            metric = {
                "job_id": job.job_id,
                "phase_label": phase_label,
                "stage": job.stage,
                "chunk_id": job.chunk_id,
                "cycle_number": job.cycle_number,
                "global_cycle_number": job.global_cycle_number,
                "book_name": job.book_name,
                "provider": job.provider,
                "model": job.model,
                "status": "success",
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": round(time.monotonic() - started_mono, 3),
                "attempt_count": attempt,
                "quota_pause_count": quota_pause_count,
                "rate_limit_event_count": int(
                    execution_summary.get("rate_limit_event_count", 0)
                ) + quota_pause_count,
                "input_tokens": execution_summary.get("input_tokens"),
                "output_tokens": execution_summary.get("output_tokens"),
                "cached_input_tokens": execution_summary.get("cached_input_tokens"),
                "total_cost_usd": execution_summary.get("total_cost_usd"),
            }
            append_job_metric(run_dir, metric)
            return metric
        except ProviderQuotaPause as exc:
            quota_pause_count += 1
            if quota_pause_count > max_quota_pauses:
                finished_at = _utc_now_iso()
                append_job_metric(
                    run_dir,
                    {
                        "job_id": job.job_id,
                        "phase_label": phase_label,
                        "stage": job.stage,
                        "chunk_id": job.chunk_id,
                        "cycle_number": job.cycle_number,
                        "global_cycle_number": job.global_cycle_number,
                        "book_name": job.book_name,
                        "provider": job.provider,
                        "model": job.model,
                        "status": "failed",
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "duration_seconds": round(time.monotonic() - started_mono, 3),
                        "attempt_count": attempt,
                        "quota_pause_count": quota_pause_count,
                        "rate_limit_event_count": quota_pause_count,
                        "error": f"job {job.job_id} aborted after {max_quota_pauses} quota pauses",
                    },
                )
                raise PipelineError(
                    f"job {job.job_id} aborted after {max_quota_pauses} quota pauses"
                ) from exc
            if exc.sleep_seconds > 0:
                time.sleep(exc.sleep_seconds)
            continue
        except PipelineError as exc:
            if attempt > max_retries or not _is_retryable_error(exc):
                finished_at = _utc_now_iso()
                append_job_metric(
                    run_dir,
                    {
                        "job_id": job.job_id,
                        "phase_label": phase_label,
                        "stage": job.stage,
                        "chunk_id": job.chunk_id,
                        "cycle_number": job.cycle_number,
                        "global_cycle_number": job.global_cycle_number,
                        "book_name": job.book_name,
                        "provider": job.provider,
                        "model": job.model,
                        "status": "failed",
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "duration_seconds": round(time.monotonic() - started_mono, 3),
                        "attempt_count": attempt,
                        "quota_pause_count": quota_pause_count,
                        "rate_limit_event_count": quota_pause_count,
                        "error": str(exc),
                    },
                )
                raise
            sleep_s = _retry_backoff_seconds(attempt)
            if sleep_s > 0:
                time.sleep(sleep_s)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


# ---------------------------------------------------------------------------
# Agent dispatch
# ---------------------------------------------------------------------------

def _execute_agent_job(job: JobSpec, workspace: Path, run_dir: Path) -> dict[str, Any]:
    """Route to the correct provider executor."""
    if job.provider == "claude":
        return _execute_claude_job(job, workspace, run_dir)
    if job.provider == "codex":
        return _execute_codex_job(job, workspace, run_dir)
    else:
        raise PipelineError(f"unsupported provider: {job.provider}")


def _execute_claude_job(job: JobSpec, workspace: Path, run_dir: Path) -> dict[str, Any]:
    """Run Claude Code CLI in workspace; parse streaming JSON log."""
    cmd = _build_claude_exec_cmd(job)
    log_dir = run_dir / "logs" / "jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    message_file = log_dir / f"{job.job_id}.last_message.txt"

    returncode, log_file, stderr_file = _run_agent_process(
        job=job,
        workspace=workspace,
        cmd=cmd,
        provider_label="claude",
        run_dir=run_dir,
        cwd=workspace,
        prompt_text=_CLAUDE_PIPELINE_PROMPT_PREFIX + job.prompt_text,
    )
    events = _load_provider_events(log_file)
    try:
        result_event = _extract_claude_result_event(events)
        last_message = _extract_claude_last_message_text(events)
    except PipelineError as exc:
        stderr_tail = _tail_file(stderr_file)
        raise PipelineError(
            f"claude exec failed rc={returncode} stderr_tail={stderr_tail} parse_error={exc}"
        ) from exc
    message_file.write_text(last_message + "\n", encoding="utf-8")
    if returncode != 0 or bool(result_event.get("is_error")):
        stderr_tail = _tail_file(stderr_file)
        result_text = str(result_event.get("result", "")).strip()
        quota_pause = _claude_quota_pause_from_events(events, result_text)
        if quota_pause is not None:
            raise quota_pause
        raise PipelineError(
            f"claude exec failed rc={returncode} result={result_text or '<empty>'} "
            f"stderr_tail={stderr_tail}"
        )
    return _extract_usage_summary(provider="claude", events=events)


def _execute_codex_job(job: JobSpec, workspace: Path, run_dir: Path) -> dict[str, Any]:
    """Run Codex CLI in workspace."""
    log_dir = run_dir / "logs" / "jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    message_file = log_dir / f"{job.job_id}.last_message.txt"
    cmd = _build_codex_exec_cmd(job, workspace, message_file)

    returncode, _log_file, stderr_file = _run_agent_process(
        job=job,
        workspace=workspace,
        cmd=cmd,
        provider_label="codex",
        run_dir=run_dir,
    )
    if returncode != 0:
        stderr_tail = _tail_file(stderr_file)
        raise PipelineError(f"codex exec failed rc={returncode} stderr_tail={stderr_tail}")
    events = _load_provider_events(run_dir / "logs" / "jobs" / f"{job.job_id}.jsonl")
    return _extract_usage_summary(provider="codex", events=events)


def _build_claude_exec_cmd(job: JobSpec) -> list[str]:
    """Build the Claude Code CLI command for a job."""
    cmd = [
        "claude",
        "--effort", "max",
        "--bare",
        "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--tools", "Read,Write",
        "--allowedTools", "Read,Write",
    ]
    if job.model:
        cmd.extend(["--model", job.model])
    return cmd


def _build_codex_exec_cmd(job: JobSpec, workspace: Path, message_file: Path) -> list[str]:
    """Build the Codex CLI command for a job."""
    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "-C", str(workspace),
        "--json",
        "-o", str(message_file),
    ]
    if job.model:
        cmd.extend(["-m", job.model])
    return cmd


def _run_agent_process(
    *,
    job: JobSpec,
    workspace: Path,
    cmd: list[str],
    provider_label: str,
    run_dir: Path,
    cwd: Path | None = None,
    prompt_text: str | None = None,
) -> tuple[int, Path, Path]:
    """Launch agent subprocess with wall-clock watchdog. Returns (rc, log, stderr)."""
    log_dir = run_dir / "logs" / "jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job.job_id}.jsonl"
    stderr_file = log_dir / f"{job.job_id}.stderr.txt"

    wall_limit = job.timeout_seconds

    with log_file.open("w", encoding="utf-8") as out, \
            stderr_file.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=err,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
        )
        try:
            if proc.stdin is not None:
                proc.stdin.write(prompt_text if prompt_text is not None else job.prompt_text)
                proc.stdin.close()
        except BrokenPipeError:
            pass

        deadline = time.monotonic() + wall_limit

        while proc.poll() is None:
            time.sleep(POLL_INTERVAL)

            if time.monotonic() >= deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise PipelineError(
                    f"job timed out after {wall_limit}s (wall clock)"
                )

    returncode = proc.returncode
    return returncode, log_file, stderr_file


# ---------------------------------------------------------------------------
# Workspace management
# ---------------------------------------------------------------------------

def _copy_inputs_to_workspace(
    allowed_inputs: list[str], workspace: Path, run_dir: Path
) -> dict[str, str]:
    """Copy declared input files from run_dir into workspace; return sha256 map."""
    before_hashes: dict[str, str] = {}
    for rel in allowed_inputs:
        _assert_rel_path(rel)
        src = run_dir / rel
        if not src.is_file():
            raise PipelineError(f"missing required input file for job: {rel}")
        dst = workspace / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        before_hashes[rel] = _sha256_file(dst)
    return before_hashes


def _copy_outputs_from_workspace(job: JobSpec, workspace: Path, run_dir: Path) -> None:
    """Copy required outputs from workspace back to run_dir."""
    for rel in job.required_outputs:
        _assert_rel_path(rel)
        src = workspace / rel
        if not src.is_file():
            raise PipelineError(f"job did not produce required output: {rel}")
        dst = run_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _prepare_required_output_dirs(job: JobSpec, workspace: Path) -> None:
    """Create required output parent directories before the agent runs."""
    for rel in job.required_outputs:
        _assert_rel_path(rel)
        (workspace / rel).parent.mkdir(parents=True, exist_ok=True)


def _validate_job_workspace(
    job: JobSpec, workspace: Path, before_hashes: dict[str, str]
) -> None:
    """Verify required outputs exist and inputs were not mutated."""
    for rel in job.required_outputs:
        _assert_rel_path(rel)
        path = workspace / rel
        if not path.is_file():
            raise PipelineError(f"required output missing in workspace: {rel}")
        _validate_required_output_content(job, rel, path)

    files_in_workspace = _list_workspace_files(workspace)
    allowed = set(job.allowed_inputs) | set(job.required_outputs)
    undeclared = sorted(
        rel for rel in (files_in_workspace - allowed)
        if not _is_allowed_aux_workspace_file(rel)
    )
    if undeclared:
        preview = ", ".join(undeclared[:10])
        raise PipelineError(f"undeclared files written by job: {preview}")

    for rel, before_hash in before_hashes.items():
        fpath = workspace / rel
        if rel in job.required_outputs:
            continue
        if not fpath.exists():
            raise PipelineError(f"input file unexpectedly removed by job: {rel}")
        if _sha256_file(fpath) != before_hash:
            raise PipelineError(f"input file mutated without declaration: {rel}")


def _validate_required_output_content(job: JobSpec, rel: str, path: Path) -> None:
    """Validate structured outputs that the runner expects to parse later."""
    if not rel.startswith("outputs/") or not rel.endswith(".json"):
        return
    data = _load_required_output_json(rel, path)

    if rel.startswith("outputs/segment_translation_"):
        _validate_segment_translation_output(job, rel, data)
        return
    if rel.startswith("outputs/litrans_answers_"):
        _validate_litrans_answers_output(rel, path)
        return
    if rel.startswith("outputs/chunk_literary_review_"):
        _validate_chunk_literary_review_output(job, rel, data)
        return
    if rel in {"outputs/book_review.json", "outputs/cross_chunk_audit.json"}:
        _validate_global_review_output(rel, data)
        return
    if rel.startswith("outputs/final_revise_"):
        _validate_final_revise_output(job, rel, data)
        return
    if rel == "outputs/style_bible.json":
        _validate_style_bible_output(rel, data)


def _load_required_output_json(rel: str, path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid JSON in required output {rel}: {exc}") from exc


def _validate_segment_translation_output(job: JobSpec, rel: str, data: Any) -> None:
    if not isinstance(data, dict):
        raise PipelineError(f"invalid segment translation output {rel}: top-level JSON must be an object")
    if data.get("chunk_id") != job.chunk_id:
        raise PipelineError(
            f"invalid segment translation output {rel}: chunk_id={data.get('chunk_id')!r}, expected {job.chunk_id}"
        )
    translation = data.get("translation")
    if not isinstance(translation, str) or not translation.strip():
        raise PipelineError(
            f"invalid segment translation output {rel}: missing non-empty string field 'translation'"
        )


def _validate_litrans_answers_output(rel: str, path: Path) -> None:
    try:
        parse_litrans_answers(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PipelineError(f"invalid litrans answers output {rel}: {exc}") from exc


def _validate_chunk_literary_review_output(job: JobSpec, rel: str, data: Any) -> None:
    if not isinstance(data, dict):
        raise PipelineError(f"invalid chunk literary review output {rel}: top-level JSON must be an object")
    if data.get("chunk_id") != job.chunk_id:
        raise PipelineError(
            f"invalid chunk literary review output {rel}: chunk_id={data.get('chunk_id')!r}, expected {job.chunk_id}"
        )
    verdict = data.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().upper() not in {"PASS", "FAIL"}:
        raise PipelineError(f"invalid chunk literary review output {rel}: invalid 'verdict'")
    verdicts = data.get("verdicts")
    if not isinstance(verdicts, dict) or set(verdicts.keys()) != set(_CHUNK_LITERARY_LENSES):
        raise PipelineError(
            f"invalid chunk literary review output {rel}: 'verdicts' must contain exactly {list(_CHUNK_LITERARY_LENSES)}"
        )
    for lens in _CHUNK_LITERARY_LENSES:
        lens_verdict = verdicts.get(lens)
        if not isinstance(lens_verdict, str) or lens_verdict.strip().upper() not in {"PASS", "FAIL"}:
            raise PipelineError(
                f"invalid chunk literary review output {rel}: lens '{lens}' has invalid verdict {lens_verdict!r}"
            )
    findings = data.get("findings")
    if not isinstance(findings, list):
        raise PipelineError(f"invalid chunk literary review output {rel}: missing array field 'findings'")
    for finding in findings:
        _validate_review_finding(finding, rel=rel, expected_chunk_id=job.chunk_id, require_source=True)
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise PipelineError(f"invalid chunk literary review output {rel}: missing non-empty string field 'summary'")


def _validate_global_review_output(rel: str, data: Any) -> None:
    if not isinstance(data, dict):
        raise PipelineError(f"invalid global review output {rel}: top-level JSON must be an object")
    verdict = data.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().upper() not in {"PASS", "FAIL"}:
        raise PipelineError(f"invalid global review output {rel}: invalid 'verdict'")
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise PipelineError(f"invalid global review output {rel}: missing non-empty string field 'summary'")
    findings = data.get("findings")
    if not isinstance(findings, list):
        raise PipelineError(f"invalid global review output {rel}: missing array field 'findings'")
    for finding in findings:
        _validate_review_finding(finding, rel=rel)


def _validate_final_revise_output(job: JobSpec, rel: str, data: Any) -> None:
    if not isinstance(data, dict):
        raise PipelineError(f"invalid final revise output {rel}: top-level JSON must be an object")
    if data.get("chunk_id") != job.chunk_id:
        raise PipelineError(
            f"invalid final revise output {rel}: chunk_id={data.get('chunk_id')!r}, expected {job.chunk_id}"
        )
    findings_addressed = data.get("findings_addressed")
    if not isinstance(findings_addressed, list):
        raise PipelineError(f"invalid final revise output {rel}: missing array field 'findings_addressed'")


def _validate_style_bible_output(rel: str, data: Any) -> None:
    if not isinstance(data, dict):
        raise PipelineError(f"invalid style bible output {rel}: top-level JSON must be an object")


def _validate_review_finding(
    finding: Any,
    *,
    rel: str,
    expected_chunk_id: int | None = None,
    require_source: bool = False,
) -> None:
    if not isinstance(finding, dict):
        raise PipelineError(f"invalid review output {rel}: finding must be an object")
    finding_id = finding.get("finding_id")
    if not isinstance(finding_id, str) or not finding_id.strip():
        raise PipelineError(f"invalid review output {rel}: finding missing non-empty 'finding_id'")
    chunk_id = finding.get("chunk_id")
    if not isinstance(chunk_id, int):
        raise PipelineError(f"invalid review output {rel}: finding '{finding_id}' missing integer 'chunk_id'")
    if expected_chunk_id is not None and chunk_id != expected_chunk_id:
        raise PipelineError(
            f"invalid review output {rel}: finding '{finding_id}' has chunk_id={chunk_id}, expected {expected_chunk_id}"
        )
    severity = finding.get("severity")
    if not isinstance(severity, str) or severity.strip().upper() not in _REVIEW_SEVERITIES:
        raise PipelineError(f"invalid review output {rel}: finding '{finding_id}' has invalid severity {severity!r}")
    for key in ("evidence", "problem", "rewrite_direction", "acceptance_test"):
        value = finding.get(key)
        if not isinstance(value, str) or not value.strip():
            raise PipelineError(
                f"invalid review output {rel}: finding '{finding_id}' missing non-empty '{key}'"
            )
    if require_source:
        source = finding.get("source")
        if not isinstance(source, str) or source.strip() not in _CHUNK_LITERARY_LENSES:
            raise PipelineError(f"invalid review output {rel}: finding '{finding_id}' has invalid source {source!r}")


def _mock_job_outputs(job: JobSpec, workspace: Path) -> None:
    """Write placeholder outputs for dry-run mode."""
    for rel in job.required_outputs:
        out_path = workspace / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if "segment_translation" in rel:
            out_path.write_text(
                json.dumps(
                    {"chunk_id": job.chunk_id, "translation": f"[DRY RUN chunk {job.chunk_id}]"}
                ),
                encoding="utf-8",
            )
        elif "litrans_answers" in rel:
            out_path.write_text(
                json.dumps(
                    {
                        str(i): {
                            "judgment": "YES",
                            "issue": "No material issue detected for this question.",
                        }
                        for i in range(1, 26)
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif "book_review" in rel:
            out_path.write_text(
                json.dumps({
                    "cycle": 1,
                    "verdict": "PASS",
                    "summary": "[DRY RUN]",
                    "findings": [],
                }),
                encoding="utf-8",
            )
        elif "cross_chunk_audit" in rel:
            out_path.write_text(
                json.dumps({
                    "verdict": "PASS",
                    "summary": "[DRY RUN]",
                    "findings": [],
                }),
                encoding="utf-8",
            )
        elif "chunk_literary_review" in rel:
            out_path.write_text(
                json.dumps({
                    "chunk_id": job.chunk_id,
                    "verdict": "PASS",
                    "verdicts": {
                        "accuracy": "PASS",
                        "voice": "PASS",
                        "dialogue": "PASS",
                        "prose": "PASS",
                    },
                    "findings": [],
                    "summary": "[DRY RUN]",
                }),
                encoding="utf-8",
            )
        elif "final_revise" in rel:
            out_path.write_text(
                json.dumps({
                    "chunk_id": job.chunk_id,
                    "findings_addressed": [],
                }),
                encoding="utf-8",
            )
        elif "style_bible" in rel:
            out_path.write_text(
                json.dumps({
                    "character_voice_profiles": [],
                    "prose_style_profile": {
                        "register": "[DRY RUN]",
                        "domain": "[DRY RUN]",
                        "narrative_tense": "past tense",
                        "rhythm_target": "[DRY RUN]",
                        "diction": "[DRY RUN]",
                    },
                    "terminology": [],
                    "named_entities": [],
                }),
                encoding="utf-8",
            )
        else:
            out_path.write_text(f"[DRY RUN: {rel}]", encoding="utf-8")


# ---------------------------------------------------------------------------
# Claude event parsing
# ---------------------------------------------------------------------------

def _load_provider_events(log_file: Path) -> list[dict[str, Any]]:
    """Parse JSONL log file into list of event dicts."""
    events: list[dict[str, Any]] = []
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            events.append(row)
    return events


def _extract_claude_result_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the final result event from Claude streaming output."""
    for event in reversed(events):
        if event.get("type") == "result":
            return event
    raise PipelineError("claude exec log missing final result event")


def _extract_claude_last_message_text(events: list[dict[str, Any]]) -> str:
    """Extract the last assistant message text from Claude streaming output."""
    result_event = _extract_claude_result_event(events)
    result_text = str(result_event.get("result", "")).strip()
    if result_text:
        return result_text
    fragments: list[str] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    fragments.append(text)
    if fragments:
        return "\n".join(fragments).strip()
    raise PipelineError("claude exec log missing assistant text content")


def _extract_claude_rejected_rate_limit_info(
    events: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find a rejected rate-limit event in Claude streaming output."""
    for event in reversed(events):
        if event.get("type") != "rate_limit_event":
            continue
        info = event.get("rate_limit_info")
        if not isinstance(info, dict):
            continue
        if str(info.get("status", "")).strip().lower() == "rejected":
            return info
    return None


def _claude_quota_pause_from_events(
    events: list[dict[str, Any]], result_text: str
) -> ProviderQuotaPause | None:
    """Build a ProviderQuotaPause from rate-limit events if quota was rejected."""
    info = _extract_claude_rejected_rate_limit_info(events)
    if info is None:
        return None
    raw_reset = info.get("resetsAt")
    try:
        reset_at_epoch = int(raw_reset)
    except (TypeError, ValueError):
        return None
    now_epoch = int(time.time())
    base_sleep = max(0, reset_at_epoch - now_epoch)
    buffer_s = max(0, _self.CLAUDE_QUOTA_RESET_BUFFER_SECONDS)
    jitter_cap = max(0, _self.CLAUDE_QUOTA_RESET_JITTER_SECONDS)
    jitter_s = random.randint(0, jitter_cap) if jitter_cap > 0 else 0
    return ProviderQuotaPause(
        provider="claude",
        result_text=result_text or "You've hit your limit",
        reset_at_epoch=reset_at_epoch,
        sleep_seconds=base_sleep + buffer_s + jitter_s,
        rate_limit_type=str(info.get("rateLimitType", "")).strip() or None,
    )


def _extract_usage_summary(*, provider: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Return best-effort token/cost/rate-limit summary from provider events."""
    rate_limit_event_count = sum(1 for event in events if event.get("type") == "rate_limit_event")
    summary: dict[str, Any] = {
        "rate_limit_event_count": rate_limit_event_count,
        "input_tokens": None,
        "output_tokens": None,
        "cached_input_tokens": None,
        "total_cost_usd": None,
    }
    if provider == "claude":
        result_event = _extract_claude_result_event(events)
        usage = result_event.get("usage")
        if isinstance(usage, dict):
            summary["input_tokens"] = usage.get("input_tokens")
            summary["output_tokens"] = usage.get("output_tokens")
            summary["cached_input_tokens"] = usage.get("cache_read_input_tokens")
        summary["total_cost_usd"] = result_event.get("total_cost_usd")
        return summary

    completed_turn = None
    for event in reversed(events):
        if event.get("type") == "turn.completed":
            completed_turn = event
            break
    if isinstance(completed_turn, dict):
        usage = completed_turn.get("usage")
        if isinstance(usage, dict):
            summary["input_tokens"] = usage.get("input_tokens")
            summary["output_tokens"] = usage.get("output_tokens")
            summary["cached_input_tokens"] = usage.get("cached_input_tokens")
    for event in reversed(events):
        total_cost = event.get("total_cost_usd")
        if total_cost is not None:
            summary["total_cost_usd"] = total_cost
            break
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(job: JobSpec, run_dir: Path) -> None:
    """Write a manifest JSON for this job to run_dir/manifests/."""
    manifest = {
        "job_id": job.job_id,
        "stage": job.stage,
        "chunk_id": job.chunk_id,
        "cycle_number": job.cycle_number,
        "global_cycle_number": job.global_cycle_number,
        "book_name": job.book_name,
        "provider": job.provider,
        "model": job.model,
        "allowed_inputs": job.allowed_inputs,
        "required_outputs": job.required_outputs,
        "timeout_seconds": job.timeout_seconds,
    }
    manifest_dir = run_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{job.job_id}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _list_workspace_files(workspace: Path) -> set[str]:
    """Return all file paths in workspace as POSIX relative strings."""
    return {
        p.relative_to(workspace).as_posix()
        for p in workspace.rglob("*")
        if p.is_file()
    }


def _is_allowed_aux_workspace_file(rel: str) -> bool:
    """Return True for auxiliary files agents may create that don't need declaration."""
    if rel in (".DS_Store", ".claude.json", ".codex"):
        return True
    aux_prefixes = (".claude/", ".codex/", ".cache/", ".tmp/", "tmp/")
    if rel.startswith(aux_prefixes):
        return True
    if rel.endswith((".tmp", ".log")):
        return True
    return False


def _assert_rel_path(rel: str) -> None:
    """Raise PipelineError if rel is absolute or contains '..'."""
    posix = PurePosixPath(rel)
    windows = PureWindowsPath(rel)
    if os.path.isabs(rel) or posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise PipelineError(f"path must be relative: {rel}")
    if ".." in posix.parts or ".." in windows.parts:
        raise PipelineError(f"path must not include '..': {rel}")


def _tail_file(path: Path, max_lines: int = 20) -> str:
    """Return the last N lines of a file joined by ' | '."""
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return " | ".join(lines[-max_lines:])


def _is_retryable_error(exc: PipelineError) -> bool:
    """Return True for transient errors that warrant a retry."""
    text = str(exc).lower()
    retryable_tokens = (
        "codex exec failed rc=",
        "codex exec stalled",
        "claude exec failed rc=",
        "claude exec stalled",
        "job timed out after",
        "job did not produce required output",
        "required output missing in workspace",
        "invalid json in required output",
    )
    return any(token in text for token in retryable_tokens)


def _retry_backoff_seconds(attempt: int) -> int:
    """Return exponential backoff sleep duration for a retry attempt.

    Uses JOB_EXEC_RETRY_BASE_SLEEP_SECONDS as the base (attempt 1 = base * 2^0).
    Capped at JOB_EXEC_RETRY_MAX_SLEEP_SECONDS; a cap of 0 means no cap.
    Returns 0 immediately if base is 0.
    """
    base = max(0, _self.JOB_EXEC_RETRY_BASE_SLEEP_SECONDS)
    cap = _self.JOB_EXEC_RETRY_MAX_SLEEP_SECONDS
    if base <= 0:
        return 0
    wait = base * (2 ** max(0, attempt - 1))
    return min(wait, cap) if cap > 0 else wait


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_summary_path(run_dir: Path) -> Path:
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / "run_summary.json"


def get_run_summary(run_dir: Path) -> dict[str, Any]:
    """Return the current run summary, creating an empty shape if needed."""
    path = _run_summary_path(run_dir)
    if not path.exists():
        return {"jobs": [], "chunk_states": {}, "updated_at": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"jobs": [], "chunk_states": {}, "updated_at": None}
    data.setdefault("jobs", [])
    data.setdefault("chunk_states", {})
    data.setdefault("updated_at", None)
    return data


def _mutate_run_summary(run_dir: Path, mutator: Any) -> None:
    with _RUN_SUMMARY_LOCK:
        summary = get_run_summary(run_dir)
        mutator(summary)
        summary["updated_at"] = _utc_now_iso()
        _run_summary_path(run_dir).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def append_job_metric(run_dir: Path, metric: dict[str, Any]) -> None:
    """Append one job metric record to the run summary."""
    def _append(summary: dict[str, Any]) -> None:
        summary.setdefault("jobs", []).append(metric)

    _mutate_run_summary(run_dir, _append)


def set_chunk_state(run_dir: Path, chunk_id: int, chunk_state: dict[str, Any]) -> None:
    """Replace the stored chunk state for one chunk in the run summary."""
    def _set(summary: dict[str, Any]) -> None:
        summary.setdefault("chunk_states", {})[str(chunk_id)] = chunk_state

    _mutate_run_summary(run_dir, _set)
