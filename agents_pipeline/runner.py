#!/usr/bin/env python3
"""MT Agents Pipeline — coordinator entry point.

Per-stage agent provider/model is loaded from a JSON config (default:
``agents_pipeline/config.json``). Override with ``--config path/to/file.json``.

Usage:
    python agents_pipeline/runner.py \\
        --book_path data/dev_fr.txt \\
        --config agents_pipeline/config.json \\
        --max_cycles 3 \\
        --litrans_threshold 0.7 \\
        --max_parallel_jobs 4
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents_pipeline.book import Book, TARGET_LANGUAGE_CODES
from agents_pipeline.core.job_spec import (
    JobSpec,
    RunnerConfig,
    ResumePoint,
    PipelineError,
    STAGE_NAMES,
    PROVIDER_VALUES,
)
from agents_pipeline.core.executor import get_run_summary, run_jobs_parallel, set_chunk_state
from agents_pipeline.core.gate import run_gate

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_TRANSLATE_PROMPT = (_PROMPTS_DIR / "translate.txt").read_text(encoding="utf-8")
_REVISE_PROMPT = (_PROMPTS_DIR / "revise.txt").read_text(encoding="utf-8")
_CHUNK_LITERARY_REVIEW_PROMPT = (
    _PROMPTS_DIR / "chunk_literary_review.txt"
).read_text(encoding="utf-8")
_BOOK_REVIEW_PROMPT = (_PROMPTS_DIR / "book_review.txt").read_text(encoding="utf-8")
_CROSS_CHUNK_AUDIT_PROMPT = (_PROMPTS_DIR / "cross_chunk_audit.txt").read_text(encoding="utf-8")
_FINAL_REVISE_PROMPT = (_PROMPTS_DIR / "final_revise.txt").read_text(encoding="utf-8")
_STYLE_ANALYSIS_PROMPT = (_PROMPTS_DIR / "style_analysis.txt").read_text(encoding="utf-8")
_LITRANS_REVIEW_PROMPT = (_PROMPTS_DIR / "litrans_review.txt").read_text(encoding="utf-8")
_REVIEW_SEVERITIES: frozenset[str] = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
_CHUNK_LITERARY_LENSES: tuple[str, ...] = ("accuracy", "voice", "dialogue", "prose")
_BOUNDARY_CONTEXT_TOKENS = 200
_STALL_CYCLE_THRESHOLD = 3
_STALL_OVERLAP_REQUIRED = True
_MINIMAL_TRANSLATION_CHANGE_RATIO = 0.985


def _head_paragraphs_by_tokens(text: str, limit: int = _BOUNDARY_CONTEXT_TOKENS) -> str:
    """Return first full paragraphs up to the token budget.

    If the first paragraph alone exceeds the budget, keep that paragraph intact.
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ""

    selected: list[str] = []
    token_count = 0
    for paragraph in paragraphs:
        para_tokens = Book._token_count(paragraph)
        if not selected and para_tokens > limit:
            return paragraph
        if selected and token_count + para_tokens > limit:
            break
        selected.append(paragraph)
        token_count += para_tokens
    return "\n\n".join(selected)


def _tail_paragraphs_by_tokens(text: str, limit: int = _BOUNDARY_CONTEXT_TOKENS) -> str:
    """Return last full paragraphs up to the token budget.

    If the last paragraph alone exceeds the budget, keep that paragraph intact.
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ""

    selected: list[str] = []
    token_count = 0
    for paragraph in reversed(paragraphs):
        para_tokens = Book._token_count(paragraph)
        if not selected and para_tokens > limit:
            return paragraph
        if selected and token_count + para_tokens > limit:
            break
        selected.append(paragraph)
        token_count += para_tokens
    return "\n\n".join(reversed(selected))


def _chunk_boundary_context(*, book: Any, chunk_id: int) -> dict[str, str]:
    """Return adjacent source/translation context with first/last-chunk handling."""
    prev_source = "None. This is the first chunk." if chunk_id == 0 else _tail_paragraphs_by_tokens(
        book.src_text[chunk_id - 1].text
    )
    next_source = (
        "None. This is the last chunk."
        if chunk_id >= len(book.src_text) - 1
        else _head_paragraphs_by_tokens(book.src_text[chunk_id + 1].text)
    )

    translations = getattr(book, "translation", None)
    if not translations:
        prev_translation = "Unavailable during the initial translation pass."
        next_translation = "Unavailable during the initial translation pass."
    else:
        prev_translation = (
            "None. This is the first chunk."
            if chunk_id == 0
            else _tail_paragraphs_by_tokens(translations[chunk_id - 1].text)
        )
        next_translation = (
            "None. This is the last chunk."
            if chunk_id >= len(translations) - 1
            else _head_paragraphs_by_tokens(translations[chunk_id + 1].text)
        )

    return {
        "prev_source_context": prev_source or "No previous source context available.",
        "next_source_context": next_source or "No next source context available.",
        "prev_translation_context": prev_translation or "No previous translation context available.",
        "next_translation_context": next_translation or "No next translation context available.",
    }


def _translation_draft_filename(*, book_name: str, target_language_code: str) -> str:
    return f"{book_name}_{target_language_code}_draft.txt"


def _translation_final_filename(*, book_name: str, target_language_code: str) -> str:
    return f"{book_name}_{target_language_code}.txt"


def _review_worker_split(max_parallel_jobs: int) -> tuple[int, int]:
    """Split workers deterministically between QE and literary review."""
    if max_parallel_jobs <= 1:
        return (1, 1)
    qe_workers = max(1, max_parallel_jobs // 2)
    literary_workers = max(1, max_parallel_jobs - qe_workers)
    return (qe_workers, literary_workers)


def _alternate_revise_stage(cfg: RunnerConfig) -> tuple[str, str | None]:
    """Return the alternate provider/model for stalled local revision."""
    for stage_name in ("chunk_literary_review", "final_revise", "style_analysis"):
        provider, model = cfg.stage(stage_name)
        if provider == "claude":
            return provider, model
    return ("claude", None)


def _stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _translation_changed_minimally(previous: str, current: str) -> bool:
    if previous == current:
        return True
    ratio = SequenceMatcher(None, previous, current).ratio()
    return ratio >= _MINIMAL_TRANSLATION_CHANGE_RATIO


def _chunk_state_for(run_dir: Path, chunk_id: int) -> dict[str, Any]:
    summary = get_run_summary(run_dir)
    return dict(summary.get("chunk_states", {}).get(str(chunk_id), {}))


def _failure_signature(
    *,
    litrans_data: dict[str, Any],
    literary_data: dict[str, Any],
) -> dict[str, list[str]]:
    literary_findings = [
        finding for finding in literary_data.get("findings", [])
        if str(finding.get("severity", "")).upper() in _MEDIUM_PLUS
    ]
    literary_keys = sorted(
        {
            " | ".join(
                [
                    str(finding.get("source", "")),
                    str(finding.get("finding_id", "")),
                    str(finding.get("problem", "")).strip().lower(),
                ]
            )
            for finding in literary_findings
        }
    )
    return {
        "qe_failed_question_ids": sorted(str(qid) for qid in litrans_data.get("failed_question_ids", [])),
        "literary_finding_keys": literary_keys,
    }


def _has_failure_overlap(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    prev_qe = set(previous.get("qe_failed_question_ids", []))
    curr_qe = set(current.get("qe_failed_question_ids", []))
    prev_lit = set(previous.get("literary_finding_keys", []))
    curr_lit = set(current.get("literary_finding_keys", []))
    return bool(prev_qe & curr_qe or prev_lit & curr_lit)


def _failure_count(signature: dict[str, Any]) -> int:
    return len(signature.get("qe_failed_question_ids", [])) + len(
        signature.get("literary_finding_keys", [])
    )


def _is_stalled_failure_pair(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    if _STALL_OVERLAP_REQUIRED and not _has_failure_overlap(previous, current):
        return False
    previous_count = _failure_count(previous)
    current_count = _failure_count(current)
    not_materially_reduced = current_count >= previous_count
    minimally_changed = _translation_changed_minimally(
        str(previous.get("translation", "")),
        str(current.get("translation", "")),
    )
    return not_materially_reduced or minimally_changed


def _build_stall_contexts(
    *,
    book: Any,
    cfg: RunnerConfig,
    failing_chunk_ids: list[int],
) -> dict[int, dict[str, Any]]:
    """Return stalled-chunk context for the next revise cycle."""
    contexts: dict[int, dict[str, Any]] = {}
    for chunk_id in failing_chunk_ids:
        chunk_state = _chunk_state_for(cfg.run_dir, chunk_id)
        history = list(chunk_state.get("local_failures", []))
        if len(history) < _STALL_CYCLE_THRESHOLD:
            continue
        previous = history[-2]
        current = history[-1]
        if not _is_stalled_failure_pair(previous, current):
            continue
        contexts[chunk_id] = {
            "repeated_qe_failed_question_ids": sorted(
                set(previous.get("qe_failed_question_ids", []))
                & set(current.get("qe_failed_question_ids", []))
            ),
            "repeated_literary_finding_keys": sorted(
                set(previous.get("literary_finding_keys", []))
                & set(current.get("literary_finding_keys", []))
            ),
            "previous_cycle": previous.get("cycle"),
            "current_cycle": current.get("cycle"),
            "previous_provider": previous.get("provider"),
            "current_provider": current.get("provider"),
        }
    return contexts


def _update_chunk_state_for_cycle(
    *,
    book: Any,
    cfg: RunnerConfig,
    cycle: int,
    chunk_ids: list[int],
    revise_jobs: list[JobSpec],
) -> list[int]:
    """Persist per-chunk failure history and return chunks switched this cycle."""
    switched_chunks: list[int] = []
    revise_job_by_chunk = {job.chunk_id: job for job in revise_jobs}

    for chunk_id in chunk_ids:
        chunk_state = _chunk_state_for(cfg.run_dir, chunk_id)
        litrans_data = _parse_litrans_review(cfg.run_dir, chunk_id=chunk_id)
        literary_data = _parse_chunk_literary_review(cfg.run_dir, chunk_id=chunk_id)
        signature = _failure_signature(litrans_data=litrans_data, literary_data=literary_data)
        translation_obj = book.translation[chunk_id]
        translation = str(getattr(translation_obj, "text", translation_obj))
        failure_record = {
            "cycle": cycle,
            "translation": translation,
            "translation_hash": _stable_text_hash(translation),
            "provider": None,
            "model": None,
            **signature,
        }

        current_failed = bool(signature["qe_failed_question_ids"] or signature["literary_finding_keys"])
        if cycle > 1 and chunk_id in revise_job_by_chunk:
            revise_job = revise_job_by_chunk[chunk_id]
            failure_record["provider"] = revise_job.provider
            failure_record["model"] = revise_job.model
            baseline_provider, baseline_model = cfg.stage("revise")
            if revise_job.provider != baseline_provider or revise_job.model != baseline_model:
                switched_chunks.append(chunk_id)
                model_switches = list(chunk_state.get("model_switches", []))
                model_switches.append(
                    {
                        "cycle": cycle,
                        "provider": revise_job.provider,
                        "model": revise_job.model,
                        "reason": "stalled_chunk",
                    }
                )
                chunk_state["model_switches"] = model_switches
                chunk_state["stalled"] = True

        if current_failed:
            local_failures = list(chunk_state.get("local_failures", []))
            local_failures.append(failure_record)
            chunk_state["local_failures"] = local_failures
            chunk_state["last_local_status"] = "FAIL"
            chunk_state["last_failed_cycle"] = cycle
            chunk_state.setdefault("stalled", False)
        else:
            chunk_state["last_local_status"] = "PASS"
            chunk_state["stalled"] = False
        set_chunk_state(cfg.run_dir, chunk_id, chunk_state)

    return sorted(set(switched_chunks))


def _print_local_cycle_summary(
    *,
    cfg: RunnerConfig,
    cycle: int,
    switched_chunks: list[int],
    gate: dict[str, Any],
) -> None:
    summary = get_run_summary(cfg.run_dir)
    jobs = [
        job for job in summary.get("jobs", [])
        if job.get("cycle_number") == cycle
        and job.get("stage") in {"translate", "revise", "litrans_review", "chunk_literary_review"}
        and job.get("status") == "success"
    ]
    if not jobs:
        return
    stage_totals: dict[str, float] = {}
    provider_models: set[str] = set()
    for job in jobs:
        stage_totals[job["stage"]] = stage_totals.get(job["stage"], 0.0) + float(
            job.get("duration_seconds", 0.0)
        )
        provider_models.add(f"{job.get('stage')}={job.get('provider')}:{job.get('model')}")
    slowest = sorted(jobs, key=lambda row: float(row.get("duration_seconds", 0.0)), reverse=True)[:3]
    slowest_text = [
        f"{row.get('job_id')}:{round(float(row.get('duration_seconds', 0.0)), 2)}"
        for row in slowest
    ]
    print(
        "cycle_summary "
        f"cycle={cycle:02d} "
        f"stage_seconds={json.dumps({k: round(v, 2) for k, v in sorted(stage_totals.items())})} "
        f"providers={sorted(provider_models)} "
        f"qe_fail={gate['qe_failing_chunk_ids']} "
        f"literary_fail={gate['literary_failing_chunk_ids']} "
        f"switched_chunks={switched_chunks} "
        f"slowest={slowest_text}"
    )


def _print_final_cycle_summary(*, cfg: RunnerConfig, global_cycle: int, gate: dict[str, Any]) -> None:
    summary = get_run_summary(cfg.run_dir)
    jobs = [
        job for job in summary.get("jobs", [])
        if job.get("global_cycle_number") == global_cycle
        and job.get("stage") in {"book_review", "cross_chunk_audit", "final_revise"}
        and job.get("status") == "success"
    ]
    if not jobs:
        return
    stage_totals: dict[str, float] = {}
    for job in jobs:
        stage_totals[job["stage"]] = stage_totals.get(job["stage"], 0.0) + float(
            job.get("duration_seconds", 0.0)
        )
    print(
        "final_cycle_summary "
        f"global_cycle={global_cycle:02d} "
        f"stage_seconds={json.dumps({k: round(v, 2) for k, v in sorted(stage_totals.items())})} "
        f"affected_chunks={gate['affected_chunk_ids']}"
    )


def _run_local_review_gates(
    *,
    cycle: int,
    book: Any,
    cfg: RunnerConfig,
    chunk_ids: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run QE and literary gates, overlapping them when workers permit."""
    if cfg.max_parallel_jobs <= 1:
        qe_gate = run_gate(
            cycle=cycle,
            book=book,
            cfg=cfg,
            dispatch_jobs=run_jobs_parallel,
            build_jobs=build_litrans_review_jobs,
            chunks_to_score=chunk_ids,
            max_workers=1,
        )
        literary_gate = run_chunk_literary_gate(
            cycle=cycle,
            book=book,
            cfg=cfg,
            chunk_ids=chunk_ids,
            max_workers=1,
        )
        return qe_gate, literary_gate

    qe_workers, literary_workers = _review_worker_split(cfg.max_parallel_jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        qe_future = executor.submit(
            run_gate,
            cycle=cycle,
            book=book,
            cfg=cfg,
            dispatch_jobs=run_jobs_parallel,
            build_jobs=build_litrans_review_jobs,
            chunks_to_score=chunk_ids,
            max_workers=qe_workers,
        )
        literary_future = executor.submit(
            run_chunk_literary_gate,
            cycle=cycle,
            book=book,
            cfg=cfg,
            chunk_ids=chunk_ids,
            max_workers=literary_workers,
        )
        return qe_future.result(), literary_future.result()


# ---------------------------------------------------------------------------
# Job builders
# ---------------------------------------------------------------------------

def build_translate_jobs(*, book: Any, cfg: RunnerConfig, cycle: int) -> list[JobSpec]:
    """Create one translate JobSpec per source chunk."""
    jobs: list[JobSpec] = []
    inputs_dir = cfg.run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    stage_provider, stage_model = cfg.stage("translate")

    for i, chunk in enumerate(book.src_text):
        src_file = inputs_dir / f"source_chunk_{i:04d}.txt"
        src_file.write_text(chunk.text, encoding="utf-8")
        boundary_context = _chunk_boundary_context(book=book, chunk_id=i)

        prompt = _TRANSLATE_PROMPT.format(
            source_lang=book.source_language,
            target_lang=cfg.target_language,
            source_text=chunk.text,
            prev_source_context=boundary_context["prev_source_context"],
            next_source_context=boundary_context["next_source_context"],
            chunk_id=i,
        )
        jobs.append(JobSpec(
            job_id=f"translate_{book.name}_chunk_{i:04d}_cycle_{cycle:02d}",
            stage="translate",
            chunk_id=i,
            book_name=book.name,
            allowed_inputs=[
                f"inputs/source_chunk_{i:04d}.txt",
                "outputs/style_bible.json",  # produced by run_style_analysis(); must run first
            ],
            required_outputs=[f"outputs/segment_translation_{i:04d}.json"],
            prompt_text=prompt,
            provider=stage_provider,
            model=stage_model,
            timeout_seconds=cfg.job_timeout_seconds,
            cycle_number=cycle,
        ))
    return jobs


def build_revise_jobs(
    *,
    book: Any,
    cfg: RunnerConfig,
    cycle: int,
    failing_chunk_ids: list[int],
    stall_contexts: dict[int, dict[str, Any]] | None = None,
) -> list[JobSpec]:
    """Create one revise JobSpec per failing chunk."""
    jobs: list[JobSpec] = []
    outputs_dir = cfg.run_dir / "outputs"
    baseline_provider, baseline_model = cfg.stage("revise")
    alternate_provider, alternate_model = _alternate_revise_stage(cfg)
    stall_contexts = stall_contexts or {}

    for i in failing_chunk_ids:
        src_chunk = book.src_text[i]
        trans_chunk = book.translation[i]
        boundary_context = _chunk_boundary_context(book=book, chunk_id=i)

        # Read LiTransProQA score for prompt context
        litrans_path = outputs_dir / f"litrans_review_{i:04d}.json"
        score: float = 0.0
        if litrans_path.is_file():
            litrans_data = _parse_litrans_review(cfg.run_dir, chunk_id=i)
            score = litrans_data["score"]
        else:
            litrans_data = {
                "verdict": "UNKNOWN",
                "score": score,
                "decision_rule": {"allowed_no_count": 0, "allowed_maybe_count": 5},
                "yes_count": 0,
                "maybe_count": 0,
                "no_count": 0,
                "failed_question_ids": [],
                "failed_questions": [],
            }

        literary_path = outputs_dir / f"chunk_literary_review_{i:04d}.json"
        literary_findings: list[dict[str, Any]] = []
        literary_verdict = "UNKNOWN"
        if literary_path.is_file():
            literary_data = json.loads(literary_path.read_text(encoding="utf-8"))
            literary_verdict = str(literary_data.get("verdict", "UNKNOWN"))
            for finding in literary_data.get("findings", []):
                if str(finding.get("severity", "")).upper() in _MEDIUM_PLUS:
                    literary_findings.append(finding)

        stall_context = stall_contexts.get(i)
        if stall_context:
            repeated_qe_ids = json.dumps(
                stall_context.get("repeated_qe_failed_question_ids", []),
                ensure_ascii=False,
            )
            repeated_literary = json.dumps(
                stall_context.get("repeated_literary_finding_keys", []),
                ensure_ascii=False,
            )
            stall_context_block = (
                "STALLED CHUNK MODE:\n"
                f"- this chunk failed local cycles {stall_context.get('previous_cycle')} "
                f"and {stall_context.get('current_cycle')}\n"
                f"- repeated QE issues: {repeated_qe_ids}\n"
                f"- repeated literary issues: {repeated_literary}\n"
                "- make targeted but substantive changes only where unresolved issues remain\n"
                "- avoid surface-only edits that preserve the same failure pattern"
            )
            job_provider = alternate_provider
            job_model = alternate_model
        else:
            stall_context_block = "No repeated-failure escalation is active for this chunk."
            job_provider = baseline_provider
            job_model = baseline_model

        prompt = _REVISE_PROMPT.format(
            source_lang=book.source_language,
            target_lang=cfg.target_language,
            source_text=src_chunk.text,
            draft_translation=trans_chunk.text,
            prev_source_context=boundary_context["prev_source_context"],
            next_source_context=boundary_context["next_source_context"],
            prev_translation_context=boundary_context["prev_translation_context"],
            next_translation_context=boundary_context["next_translation_context"],
            score=score,
            litrans_verdict=str(litrans_data.get("verdict", "UNKNOWN")),
            failed_questions=json.dumps(litrans_data.get("failed_questions", []), indent=2, ensure_ascii=False),
            literary_verdict=literary_verdict,
            literary_findings=json.dumps(literary_findings, indent=2, ensure_ascii=False),
            stall_context_block=stall_context_block,
            chunk_id=i,
        )
        jobs.append(JobSpec(
            job_id=f"revise_{book.name}_chunk_{i:04d}_cycle_{cycle:02d}",
            stage="revise",
            chunk_id=i,
            book_name=book.name,
            allowed_inputs=[
                f"inputs/source_chunk_{i:04d}.txt",
                f"outputs/segment_translation_{i:04d}.json",
                f"outputs/litrans_review_{i:04d}.json",
                f"outputs/chunk_literary_review_{i:04d}.json",
                "outputs/style_bible.json",  # produced by run_style_analysis(); must run first
            ],
            required_outputs=[f"outputs/segment_translation_{i:04d}.json"],
            prompt_text=prompt,
            provider=job_provider,
            model=job_model,
            timeout_seconds=cfg.job_timeout_seconds,
            cycle_number=cycle,
        ))
    return jobs


def build_book_review_job(*, book: Any, cfg: RunnerConfig, global_cycle: int = 1) -> JobSpec:
    """Create a single excerpt-level review JobSpec over the reconstructed draft."""
    stage_provider, stage_model = cfg.stage("book_review")
    draft_filename = _translation_draft_filename(
        book_name=book.name,
        target_language_code=cfg.target_language_code,
    )
    prompt = _BOOK_REVIEW_PROMPT.format(
        source_lang=book.source_language,
        target_lang=cfg.target_language,
        book_name=book.name,
        draft_filename=draft_filename,
        num_chunks=len(book.src_text),
    )
    allowed_inputs = [draft_filename, "outputs/style_bible.json"] + [
        f"inputs/source_chunk_{i:04d}.txt" for i in range(len(book.src_text))
    ]
    return JobSpec(
        job_id=f"book_review_{book.name}_global_cycle_{global_cycle:02d}",
        stage="book_review",
        chunk_id=-1,
        book_name=book.name,
        allowed_inputs=allowed_inputs,
        required_outputs=["outputs/book_review.json"],
        prompt_text=prompt,
        provider=stage_provider,
        model=stage_model,
        timeout_seconds=cfg.job_timeout_seconds,
        global_cycle_number=global_cycle,
    )


def build_cross_chunk_audit_job(
    *,
    book: Any,
    cfg: RunnerConfig,
    global_cycle: int = 1,
) -> JobSpec:
    """Create a single cross-chunk audit JobSpec over the reconstructed draft."""
    stage_provider, stage_model = cfg.stage("cross_chunk_audit")
    draft_filename = _translation_draft_filename(
        book_name=book.name,
        target_language_code=cfg.target_language_code,
    )
    prompt = _CROSS_CHUNK_AUDIT_PROMPT.format(
        source_lang=book.source_language,
        target_lang=cfg.target_language,
        book_name=book.name,
        draft_filename=draft_filename,
        num_chunks=len(book.src_text),
    )
    allowed_inputs = [draft_filename, "outputs/style_bible.json"] + [
        f"inputs/source_chunk_{i:04d}.txt" for i in range(len(book.src_text))
    ]
    return JobSpec(
        job_id=f"cross_chunk_audit_{book.name}_global_cycle_{global_cycle:02d}",
        stage="cross_chunk_audit",
        chunk_id=-1,
        book_name=book.name,
        allowed_inputs=allowed_inputs,
        required_outputs=["outputs/cross_chunk_audit.json"],
        prompt_text=prompt,
        provider=stage_provider,
        model=stage_model,
        timeout_seconds=cfg.job_timeout_seconds,
        global_cycle_number=global_cycle,
    )


def build_litrans_review_jobs(
    *,
    book: Any,
    cfg: RunnerConfig,
    chunk_ids: list[int],
    cycle: int = 1,
) -> list[JobSpec]:
    """Create one LiTransProQA review JobSpec per chunk_id."""
    jobs: list[JobSpec] = []
    stage_provider, stage_model = cfg.stage("litrans_review")

    for i in chunk_ids:
        src_chunk = book.src_text[i]
        trans_chunk = book.translation[i]
        boundary_context = _chunk_boundary_context(book=book, chunk_id=i)
        prompt = _LITRANS_REVIEW_PROMPT.format(
            source_lang=book.source_language,
            target_lang=cfg.target_language,
            source_text=src_chunk.text,
            translation=trans_chunk.text,
            prev_source_context=boundary_context["prev_source_context"],
            next_source_context=boundary_context["next_source_context"],
            prev_translation_context=boundary_context["prev_translation_context"],
            next_translation_context=boundary_context["next_translation_context"],
            chunk_id=i,
        )
        jobs.append(JobSpec(
            job_id=f"litrans_review_{book.name}_chunk_{i:04d}_cycle_{cycle:02d}",
            stage="litrans_review",
            chunk_id=i,
            book_name=book.name,
            allowed_inputs=[
                f"inputs/source_chunk_{i:04d}.txt",
                f"outputs/segment_translation_{i:04d}.json",
            ],
            required_outputs=[f"outputs/litrans_answers_{i:04d}.json"],
            prompt_text=prompt,
            provider=stage_provider,
            model=stage_model,
            timeout_seconds=cfg.job_timeout_seconds,
            cycle_number=cycle,
        ))
    return jobs


def build_chunk_literary_review_jobs(
    *,
    book: Any,
    cfg: RunnerConfig,
    chunk_ids: list[int],
    cycle: int = 1,
) -> list[JobSpec]:
    """Create one chunk-level literary review JobSpec per chunk_id."""
    jobs: list[JobSpec] = []
    stage_provider, stage_model = cfg.stage("chunk_literary_review")

    for i in chunk_ids:
        src_chunk = book.src_text[i]
        trans_chunk = book.translation[i]
        boundary_context = _chunk_boundary_context(book=book, chunk_id=i)
        prompt = _CHUNK_LITERARY_REVIEW_PROMPT.format(
            source_lang=book.source_language,
            target_lang=cfg.target_language,
            source_text=src_chunk.text,
            translation=trans_chunk.text,
            prev_source_context=boundary_context["prev_source_context"],
            next_source_context=boundary_context["next_source_context"],
            prev_translation_context=boundary_context["prev_translation_context"],
            next_translation_context=boundary_context["next_translation_context"],
            chunk_id=i,
        )
        jobs.append(JobSpec(
            job_id=f"chunk_literary_review_{book.name}_chunk_{i:04d}_cycle_{cycle:02d}",
            stage="chunk_literary_review",
            chunk_id=i,
            book_name=book.name,
            allowed_inputs=[
                f"inputs/source_chunk_{i:04d}.txt",
                f"outputs/segment_translation_{i:04d}.json",
                "outputs/style_bible.json",
            ],
            required_outputs=[f"outputs/chunk_literary_review_{i:04d}.json"],
            prompt_text=prompt,
            provider=stage_provider,
            model=stage_model,
            timeout_seconds=cfg.job_timeout_seconds,
            cycle_number=cycle,
        ))
    return jobs


def build_style_analysis_job(*, book: Any, cfg: RunnerConfig) -> JobSpec:
    """Create a single style analysis JobSpec that reads the full source text."""
    stage_provider, stage_model = cfg.stage("style_analysis")
    prompt = _STYLE_ANALYSIS_PROMPT.format(
        source_lang=book.source_language,
        target_lang=cfg.target_language,
        book_name=book.name,
    )
    return JobSpec(
        job_id=f"style_analysis_{book.name}",
        stage="style_analysis",
        chunk_id=-1,
        book_name=book.name,
        allowed_inputs=["inputs/full_source.txt"],
        required_outputs=["outputs/style_bible.json"],
        prompt_text=prompt,
        provider=stage_provider,
        model=stage_model,
        timeout_seconds=cfg.job_timeout_seconds,
    )


def run_style_analysis(*, book: Any, cfg: RunnerConfig) -> None:
    """Write full source to inputs/full_source.txt and run the style analysis job.

    Raises FileNotFoundError if the agent does not produce outputs/style_bible.json.
    """
    style_bible_path = cfg.run_dir / "outputs" / "style_bible.json"
    if style_bible_path.exists():
        print(f"style_bible.json already present, skipping style analysis: {style_bible_path}")
        return

    inputs_dir = cfg.run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    full_source_path = inputs_dir / "full_source.txt"
    full_source_path.write_text(book.get_src_text(), encoding="utf-8")

    job = build_style_analysis_job(book=book, cfg=cfg)
    run_jobs_parallel(
        [job],
        cfg.run_dir,
        cfg.max_parallel_jobs,
        cfg.dry_run,
        label="style_analysis",
    )

    style_bible_path = cfg.run_dir / "outputs" / "style_bible.json"
    if not style_bible_path.is_file():
        raise FileNotFoundError(
            f"Style analysis job did not produce outputs/style_bible.json at "
            f"{style_bible_path}. Check that the agent completed successfully."
        )


_MEDIUM_PLUS: frozenset[str] = frozenset({"MEDIUM", "HIGH", "CRITICAL"})


def _require_non_empty_string(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} missing non-empty string field '{key}'")
    return value.strip()


def _parse_litrans_review(
    run_dir: Path,
    *,
    chunk_id: int,
) -> dict[str, Any]:
    """Return validated litrans_review JSON for one chunk."""
    path = run_dir / "outputs" / f"litrans_review_{chunk_id:04d}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    context = f"litrans_review_{chunk_id:04d}.json"
    if not isinstance(data, dict):
        raise ValueError(f"{context} must contain a top-level object")
    if data.get("chunk_id") != chunk_id:
        raise ValueError(f"{context} has chunk_id={data.get('chunk_id')!r}, expected {chunk_id}")

    score = data.get("score")
    if not isinstance(score, (int, float)):
        raise ValueError(f"{context} missing numeric field 'score'")
    verdict = _require_non_empty_string(data, "verdict", context=context).upper()
    if verdict not in {"PASS", "FAIL"}:
        raise ValueError(f"{context} has invalid verdict {verdict!r}")
    decision_rule = data.get("decision_rule")
    if not isinstance(decision_rule, dict):
        raise ValueError(f"{context} missing object field 'decision_rule'")
    for key in ("allowed_no_count", "allowed_maybe_count"):
        if not isinstance(decision_rule.get(key), int):
            raise ValueError(f"{context} decision_rule missing integer field '{key}'")
    for key in ("yes_count", "maybe_count", "no_count"):
        if not isinstance(data.get(key), int):
            raise ValueError(f"{context} missing integer field '{key}'")

    answers = data.get("answers")
    if not isinstance(answers, dict) or len(answers) != 25:
        raise ValueError(f"{context} missing object field 'answers' with 25 entries")
    for qid, answer in answers.items():
        if not isinstance(qid, str):
            raise ValueError(f"{context} answer keys must be strings")
        if not isinstance(answer, dict):
            raise ValueError(f"{context} answer {qid!r} must be an object")
        judgment = answer.get("judgment")
        issue = answer.get("issue")
        if judgment not in {"YES", "NO", "MAYBE"}:
            raise ValueError(f"{context} answer {qid!r} has invalid judgment {judgment!r}")
        if not isinstance(issue, str) or not issue.strip():
            raise ValueError(f"{context} answer {qid!r} missing non-empty issue text")
    failed_question_ids = data.get("failed_question_ids")
    if not isinstance(failed_question_ids, list) or not all(
        isinstance(qid, str) for qid in failed_question_ids
    ):
        raise ValueError(f"{context} missing array field 'failed_question_ids'")
    failed_questions = data.get("failed_questions")
    if not isinstance(failed_questions, list):
        raise ValueError(f"{context} missing array field 'failed_questions'")
    for item in failed_questions:
        if not isinstance(item, dict):
            raise ValueError(f"{context} failed_questions entries must be objects")
        for key in ("question_id", "group", "judgment", "issue"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"{context} failed_questions entry missing non-empty '{key}'")

    return {
        "chunk_id": chunk_id,
        "score": float(score),
        "verdict": verdict,
        "decision_rule": decision_rule,
        "yes_count": data["yes_count"],
        "maybe_count": data["maybe_count"],
        "no_count": data["no_count"],
        "answers": answers,
        "failed_question_ids": failed_question_ids,
        "failed_questions": failed_questions,
    }


def _validate_review_finding(
    finding: Any,
    *,
    context: str,
    expected_chunk_id: int | None = None,
    require_source: bool = False,
) -> dict[str, Any]:
    if not isinstance(finding, dict):
        raise ValueError(f"{context} finding must be an object")

    finding_id = _require_non_empty_string(finding, "finding_id", context=context)
    if "chunk_id" not in finding or not isinstance(finding["chunk_id"], int):
        raise ValueError(f"{context} finding '{finding_id}' missing integer field 'chunk_id'")
    chunk_id = finding["chunk_id"]
    if expected_chunk_id is not None and chunk_id != expected_chunk_id:
        raise ValueError(
            f"{context} finding '{finding_id}' has chunk_id={chunk_id}, expected {expected_chunk_id}"
        )

    severity = _require_non_empty_string(finding, "severity", context=context).upper()
    if severity not in _REVIEW_SEVERITIES:
        raise ValueError(f"{context} finding '{finding_id}' has invalid severity {severity!r}")

    normalized = {
        "finding_id": finding_id,
        "chunk_id": chunk_id,
        "severity": severity,
        "evidence": _require_non_empty_string(finding, "evidence", context=context),
        "problem": _require_non_empty_string(finding, "problem", context=context),
        "rewrite_direction": _require_non_empty_string(
            finding, "rewrite_direction", context=context
        ),
        "acceptance_test": _require_non_empty_string(
            finding, "acceptance_test", context=context
        ),
    }

    if require_source:
        source = _require_non_empty_string(finding, "source", context=context)
        if source not in _CHUNK_LITERARY_LENSES:
            raise ValueError(f"{context} finding '{finding_id}' has invalid source {source!r}")
        normalized["source"] = source

    return normalized


def _parse_chunk_literary_review(
    run_dir: Path,
    *,
    chunk_id: int,
) -> dict[str, Any]:
    """Return validated chunk_literary_review JSON for one chunk."""
    path = run_dir / "outputs" / f"chunk_literary_review_{chunk_id:04d}.json"
    context = f"chunk_literary_review_{chunk_id:04d}.json"
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        bad_line = raw.splitlines()[exc.lineno - 1] if exc.lineno - 1 < len(raw.splitlines()) else ""
        snippet_start = max(0, exc.colno - 41)
        snippet_end = exc.colno + 40
        snippet = bad_line[snippet_start:snippet_end]
        raise ValueError(
            f"{context} contains invalid JSON at line {exc.lineno} column {exc.colno}: "
            f"{exc.msg}. Snippet: {snippet!r}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"{context} must contain a top-level object")
    if data.get("chunk_id") != chunk_id:
        raise ValueError(f"{context} has chunk_id={data.get('chunk_id')!r}, expected {chunk_id}")

    verdict = _require_non_empty_string(data, "verdict", context=context).upper()
    if verdict not in {"PASS", "FAIL"}:
        raise ValueError(f"{context} has invalid verdict {verdict!r}")

    verdicts = data.get("verdicts")
    if not isinstance(verdicts, dict):
        raise ValueError(f"{context} missing object field 'verdicts'")
    if set(verdicts.keys()) != set(_CHUNK_LITERARY_LENSES):
        raise ValueError(
            f"{context} verdicts must contain exactly {list(_CHUNK_LITERARY_LENSES)}"
        )
    normalized_verdicts: dict[str, str] = {}
    for lens in _CHUNK_LITERARY_LENSES:
        lens_verdict = verdicts.get(lens)
        if not isinstance(lens_verdict, str):
            raise ValueError(f"{context} lens '{lens}' must be a string")
        lens_verdict = lens_verdict.strip().upper()
        if lens_verdict not in {"PASS", "FAIL"}:
            raise ValueError(f"{context} lens '{lens}' has invalid verdict {lens_verdict!r}")
        normalized_verdicts[lens] = lens_verdict

    findings_raw = data.get("findings")
    if not isinstance(findings_raw, list):
        raise ValueError(f"{context} missing array field 'findings'")
    findings = [
        _validate_review_finding(
            finding,
            context=context,
            expected_chunk_id=chunk_id,
            require_source=True,
        )
        for finding in findings_raw
    ]
    summary = _require_non_empty_string(data, "summary", context=context)
    return {
        "chunk_id": chunk_id,
        "verdict": verdict,
        "verdicts": normalized_verdicts,
        "findings": findings,
        "summary": summary,
    }


def _completed_job_ids(run_dir: Path) -> set[str]:
    """Return job ids with at least one successful metric entry."""
    summary = get_run_summary(run_dir)
    return {
        str(job.get("job_id"))
        for job in summary.get("jobs", [])
        if job.get("status") == "success" and job.get("job_id")
    }


def _has_valid_completed_chunk_literary_output(
    *,
    run_dir: Path,
    job: JobSpec,
    completed_job_ids: set[str],
) -> bool:
    """Return True when this exact job already succeeded and its output parses."""
    if job.job_id not in completed_job_ids:
        return False
    try:
        _parse_chunk_literary_review(run_dir, chunk_id=job.chunk_id)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return False
    return True


def _parse_segment_translation(run_dir: Path, *, chunk_id: int) -> dict[str, Any]:
    path = run_dir / "outputs" / f"segment_translation_{chunk_id:04d}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    context = f"segment_translation_{chunk_id:04d}.json"
    if not isinstance(data, dict):
        raise ValueError(f"{context} must contain a top-level object")
    if data.get("chunk_id") != chunk_id:
        raise ValueError(f"{context} has chunk_id={data.get('chunk_id')!r}, expected {chunk_id}")
    translation = data.get("translation")
    if not isinstance(translation, str) or not translation.strip():
        raise ValueError(f"{context} missing non-empty string field 'translation'")
    return data


def _parse_final_revise_report(
    run_dir: Path,
    *,
    chunk_id: int,
    global_cycle: int,
) -> dict[str, Any]:
    path = run_dir / "outputs" / f"final_revise_{chunk_id:04d}_cycle_{global_cycle:02d}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    context = f"final_revise_{chunk_id:04d}_cycle_{global_cycle:02d}.json"
    if not isinstance(data, dict):
        raise ValueError(f"{context} must contain a top-level object")
    if data.get("chunk_id") != chunk_id:
        raise ValueError(f"{context} has chunk_id={data.get('chunk_id')!r}, expected {chunk_id}")
    findings_addressed = data.get("findings_addressed")
    if not isinstance(findings_addressed, list):
        raise ValueError(f"{context} missing array field 'findings_addressed'")
    return data


def _has_valid_completed_final_revise_output(
    *,
    run_dir: Path,
    job: JobSpec,
    completed_job_ids: set[str],
) -> bool:
    """Return True when this exact final-revise job succeeded and outputs parse."""
    if job.job_id not in completed_job_ids:
        return False
    try:
        _parse_segment_translation(run_dir, chunk_id=job.chunk_id)
        _parse_final_revise_report(
            run_dir,
            chunk_id=job.chunk_id,
            global_cycle=job.global_cycle_number or 0,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return False
    return True


def _pending_final_revise_jobs(*, cfg: RunnerConfig, jobs: list[JobSpec]) -> list[JobSpec]:
    completed_job_ids = _completed_job_ids(cfg.run_dir)
    return [
        job for job in jobs
        if not _has_valid_completed_final_revise_output(
            run_dir=cfg.run_dir,
            job=job,
            completed_job_ids=completed_job_ids,
        )
    ]


def run_chunk_literary_gate(
    *,
    cycle: int,
    book: Any,
    cfg: RunnerConfig,
    chunk_ids: list[int],
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run chunk-level literary review and write a dedicated gate file."""
    jobs = build_chunk_literary_review_jobs(
        book=book,
        cfg=cfg,
        chunk_ids=chunk_ids,
        cycle=cycle,
    )
    completed_job_ids = _completed_job_ids(cfg.run_dir)
    pending_jobs = [
        job for job in jobs
        if not _has_valid_completed_chunk_literary_output(
            run_dir=cfg.run_dir,
            job=job,
            completed_job_ids=completed_job_ids,
        )
    ]
    if pending_jobs:
        skipped_count = len(jobs) - len(pending_jobs)
        if skipped_count:
            print(
                f"chunk_literary_review_cycle_{cycle:02d}: "
                f"reusing {skipped_count} completed review output(s)"
            )
        run_jobs_parallel(
            pending_jobs,
            cfg.run_dir,
            max_workers if max_workers is not None else cfg.max_parallel_jobs,
            cfg.dry_run,
            label=f"chunk_literary_review_cycle_{cycle:02d}",
        )

    failing_chunk_ids: list[int] = []
    for i in chunk_ids:
        review = _parse_chunk_literary_review(cfg.run_dir, chunk_id=i)
        findings = review.get("findings", [])
        has_blocking_finding = any(
            str(finding.get("severity", "")).upper() in _MEDIUM_PLUS for finding in findings
        )
        verdict = str(review.get("verdict", "PASS")).upper()
        has_failed_lens = any(
            str(lens_verdict).upper() == "FAIL"
            for lens_verdict in review.get("verdicts", {}).values()
        )
        if verdict == "FAIL" or has_failed_lens or has_blocking_finding:
            failing_chunk_ids.append(i)

    gate: dict[str, Any] = {
        "cycle": cycle,
        "decision": "PASS" if not failing_chunk_ids else "FAIL",
        "failing_chunk_ids": failing_chunk_ids,
        "reason": (
            "no_medium_plus_chunk_literary_findings"
            if not failing_chunk_ids
            else "chunk_literary_review_found_medium_plus_issues"
        ),
    }
    gate_path = cfg.run_dir / "gate" / f"chunk_literary_cycle_{cycle:02d}" / "gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate


def write_master_chunk_gate(
    *,
    cycle: int,
    cfg: RunnerConfig,
    qe_gate: dict[str, Any],
    literary_gate: dict[str, Any],
) -> dict[str, Any]:
    """Combine QE and literary-review results into one chunk gate."""
    qe_failing = sorted(set(qe_gate["failing_chunk_ids"]))
    literary_failing = sorted(set(literary_gate["failing_chunk_ids"]))
    failing_chunk_ids = sorted(set(qe_failing) | set(literary_failing))
    gate: dict[str, Any] = {
        "cycle": cycle,
        "decision": "PASS" if not failing_chunk_ids else "FAIL",
        "failing_chunk_ids": failing_chunk_ids,
        "qe_failing_chunk_ids": qe_failing,
        "literary_failing_chunk_ids": literary_failing,
        "reason": (
            "all_chunk_gates_passed"
            if not failing_chunk_ids
            else "qe_or_chunk_literary_review_failed"
        ),
    }
    gate_path = cfg.run_dir / "gate" / f"cycle_{cycle:02d}" / "gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate


def _parse_book_review(run_dir: Path) -> dict[int, list[dict]]:
    """Parse and validate book_review.json; return MEDIUM+ findings by chunk."""
    return _parse_global_review(run_dir, filename="book_review.json")


def _parse_cross_chunk_audit(run_dir: Path) -> dict[int, list[dict]]:
    """Parse and validate cross_chunk_audit.json; return MEDIUM+ findings by chunk."""
    return _parse_global_review(run_dir, filename="cross_chunk_audit.json")


def _parse_global_review(run_dir: Path, *, filename: str) -> dict[int, list[dict]]:
    """Parse and validate a global review JSON file; return MEDIUM+ findings by chunk."""
    path = run_dir / "outputs" / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    context = filename
    if not isinstance(data, dict):
        raise ValueError(f"{context} must contain a top-level object")
    verdict = _require_non_empty_string(data, "verdict", context=context).upper()
    if verdict not in {"PASS", "FAIL"}:
        raise ValueError(f"{context} has invalid verdict {verdict!r}")
    _require_non_empty_string(data, "summary", context=context)
    findings_raw = data.get("findings")
    if not isinstance(findings_raw, list):
        raise ValueError(f"{context} missing array field 'findings'")

    by_chunk: dict[int, list[dict]] = {}
    for finding in findings_raw:
        normalized = _validate_review_finding(finding, context=context)
        if normalized["severity"] not in _MEDIUM_PLUS:
            continue
        chunk_id = normalized["chunk_id"]
        by_chunk.setdefault(chunk_id, []).append(normalized)
    return by_chunk


def _merge_findings_by_chunk(*finding_maps: dict[int, list[dict]]) -> dict[int, list[dict]]:
    """Merge multiple {chunk_id: [finding]} maps preserving per-source order."""
    merged: dict[int, list[dict]] = {}
    for finding_map in finding_maps:
        for chunk_id, findings in finding_map.items():
            merged.setdefault(chunk_id, []).extend(findings)
    return merged


def _group_global_findings_by_chunk(
    *,
    review_findings_by_chunk: dict[int, list[dict]],
    audit_findings_by_chunk: dict[int, list[dict]],
) -> dict[int, dict[str, list[dict]]]:
    """Return grouped excerpt-level findings by chunk."""
    grouped: dict[int, dict[str, list[dict]]] = {}
    all_chunk_ids = sorted(set(review_findings_by_chunk) | set(audit_findings_by_chunk))
    for chunk_id in all_chunk_ids:
        grouped[chunk_id] = {
            "book_review_findings": review_findings_by_chunk.get(chunk_id, []),
            "cross_chunk_audit_findings": audit_findings_by_chunk.get(chunk_id, []),
        }
    return grouped


def build_final_revision_jobs(
    *,
    book: Any,
    cfg: RunnerConfig,
    affected_chunk_ids: list[int],
    findings_by_chunk: dict[int, dict[str, list[dict]]],
    global_cycle: int = 1,
) -> list[JobSpec]:
    """Create one final-revise JobSpec per affected chunk."""
    jobs: list[JobSpec] = []
    stage_provider, stage_model = cfg.stage("final_revise")
    draft_filename = _translation_draft_filename(
        book_name=book.name,
        target_language_code=cfg.target_language_code,
    )
    for i in affected_chunk_ids:
        src_chunk = book.src_text[i]
        trans_chunk = book.translation[i]
        boundary_context = _chunk_boundary_context(book=book, chunk_id=i)
        findings_text = json.dumps(findings_by_chunk[i], indent=2, ensure_ascii=False)
        prompt = _FINAL_REVISE_PROMPT.format(
            source_lang=book.source_language,
            target_lang=cfg.target_language,
            book_name=book.name,
            draft_filename=draft_filename,
            source_text=src_chunk.text,
            draft_translation=trans_chunk.text,
            chunk_id=i,
            findings=findings_text,
            final_revise_report_path=f"outputs/final_revise_{i:04d}_cycle_{global_cycle:02d}.json",
        )
        jobs.append(JobSpec(
            job_id=f"final_revise_{book.name}_chunk_{i:04d}_global_cycle_{global_cycle:02d}",
            stage="final_revise",
            chunk_id=i,
            book_name=book.name,
            allowed_inputs=[
                "inputs/full_source.txt",
                draft_filename,
                f"inputs/source_chunk_{i:04d}.txt",
                f"outputs/segment_translation_{i:04d}.json",
                "outputs/style_bible.json",
            ],
            required_outputs=[
                f"outputs/segment_translation_{i:04d}.json",
                f"outputs/final_revise_{i:04d}_cycle_{global_cycle:02d}.json",
            ],
            prompt_text=prompt,
            provider=stage_provider,
            model=stage_model,
            timeout_seconds=cfg.job_timeout_seconds,
            global_cycle_number=global_cycle,
        ))
    return jobs


# ---------------------------------------------------------------------------
# Final revision coordinator
# ---------------------------------------------------------------------------

def _build_final_gate(
    *,
    global_cycle: int,
    affected_chunk_ids: list[int],
    review_affected_chunk_ids: list[int],
    audit_affected_chunk_ids: list[int],
    confirmed: bool,
) -> dict[str, Any]:
    verdict = "PASS" if not affected_chunk_ids else "FAIL"
    return {
        "global_cycle": global_cycle,
        "verdict": verdict,
        "affected_chunk_ids": affected_chunk_ids,
        "book_review_affected_chunk_ids": review_affected_chunk_ids,
        "cross_chunk_audit_affected_chunk_ids": audit_affected_chunk_ids,
        "confirmed": confirmed,
        "reason": (
            "all_chunks_clean"
            if not affected_chunk_ids
            else "book_review_or_cross_chunk_audit_found_medium_plus_issues"
        ),
    }


def _write_final_gate(*, cfg: RunnerConfig, gate: dict[str, Any], global_cycle: int) -> None:
    cycle_gate_path = cfg.run_dir / "gate" / "final" / f"cycle_{global_cycle:02d}" / "gate.json"
    cycle_gate_path.parent.mkdir(parents=True, exist_ok=True)
    cycle_gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")

    final_gate_path = cfg.run_dir / "gate" / "final" / "gate.json"
    final_gate_path.parent.mkdir(parents=True, exist_ok=True)
    final_gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")


def _existing_incomplete_final_gate(
    *,
    book: Any,
    cfg: RunnerConfig,
    global_cycle: int,
) -> dict[str, Any] | None:
    gate_path = cfg.run_dir / "gate" / "final" / f"cycle_{global_cycle:02d}" / "gate.json"
    if not gate_path.exists():
        return None
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("verdict") != "FAIL" or gate.get("confirmed"):
        return None
    affected_chunk_ids = gate.get("affected_chunk_ids")
    if not isinstance(affected_chunk_ids, list) or not all(
        isinstance(chunk_id, int) for chunk_id in affected_chunk_ids
    ):
        return None
    review_findings_by_chunk = _parse_book_review(cfg.run_dir)
    audit_findings_by_chunk = _parse_cross_chunk_audit(cfg.run_dir)
    grouped_findings_by_chunk = _group_global_findings_by_chunk(
        review_findings_by_chunk=review_findings_by_chunk,
        audit_findings_by_chunk=audit_findings_by_chunk,
    )
    jobs = build_final_revision_jobs(
        book=book,
        cfg=cfg,
        affected_chunk_ids=affected_chunk_ids,
        findings_by_chunk=grouped_findings_by_chunk,
        global_cycle=global_cycle,
    )
    if _pending_final_revise_jobs(cfg=cfg, jobs=jobs):
        return gate
    return None


def run_final_revision(*, book: Any, cfg: RunnerConfig, start_global_cycle: int = 1) -> None:
    """Run iterative excerpt-level review/audit and targeted corrections."""
    final_path = cfg.run_dir / _translation_final_filename(
        book_name=book.name,
        target_language_code=cfg.target_language_code,
    )

    for global_cycle in range(start_global_cycle, cfg.max_global_cycles + 1):
        reconstruct_and_save(book=book, cfg=cfg)

        existing_gate = _existing_incomplete_final_gate(
            book=book,
            cfg=cfg,
            global_cycle=global_cycle,
        )
        if existing_gate is not None:
            print(
                f"final_revision cycle={global_cycle:02d}: "
                "resuming incomplete targeted revisions"
            )
            review_findings_by_chunk = _parse_book_review(cfg.run_dir)
            audit_findings_by_chunk = _parse_cross_chunk_audit(cfg.run_dir)
            grouped_findings_by_chunk = _group_global_findings_by_chunk(
                review_findings_by_chunk=review_findings_by_chunk,
                audit_findings_by_chunk=audit_findings_by_chunk,
            )
            affected_chunk_ids = list(existing_gate["affected_chunk_ids"])
            review_affected_chunk_ids = list(existing_gate["book_review_affected_chunk_ids"])
            audit_affected_chunk_ids = list(existing_gate["cross_chunk_audit_affected_chunk_ids"])
        else:
            review_job = build_book_review_job(book=book, cfg=cfg, global_cycle=global_cycle)
            audit_job = build_cross_chunk_audit_job(book=book, cfg=cfg, global_cycle=global_cycle)
            run_jobs_parallel(
                [review_job, audit_job],
                cfg.run_dir,
                cfg.max_parallel_jobs,
                cfg.dry_run,
                label=f"final_global_review_cycle_{global_cycle:02d}",
            )

            review_findings_by_chunk = _parse_book_review(cfg.run_dir)
            audit_findings_by_chunk = _parse_cross_chunk_audit(cfg.run_dir)
            grouped_findings_by_chunk = _group_global_findings_by_chunk(
                review_findings_by_chunk=review_findings_by_chunk,
                audit_findings_by_chunk=audit_findings_by_chunk,
            )
            affected_chunk_ids = sorted(grouped_findings_by_chunk.keys())
            review_affected_chunk_ids = sorted(review_findings_by_chunk.keys())
            audit_affected_chunk_ids = sorted(audit_findings_by_chunk.keys())

        if not affected_chunk_ids:
            gate = _build_final_gate(
                global_cycle=global_cycle,
                affected_chunk_ids=[],
                review_affected_chunk_ids=review_affected_chunk_ids,
                audit_affected_chunk_ids=audit_affected_chunk_ids,
                confirmed=True,
            )
            _write_final_gate(cfg=cfg, gate=gate, global_cycle=global_cycle)
            _print_final_cycle_summary(cfg=cfg, global_cycle=global_cycle, gate=gate)
            final_path.write_text(book.get_translation(), encoding="utf-8")
            print(f"final_revision verdict={gate['verdict']} affected_chunks=[]")
            print(f"Saved: {final_path}")
            return

        gate = _build_final_gate(
            global_cycle=global_cycle,
            affected_chunk_ids=affected_chunk_ids,
            review_affected_chunk_ids=review_affected_chunk_ids,
            audit_affected_chunk_ids=audit_affected_chunk_ids,
            confirmed=(global_cycle == cfg.max_global_cycles),
        )
        _write_final_gate(cfg=cfg, gate=gate, global_cycle=global_cycle)
        _print_final_cycle_summary(cfg=cfg, global_cycle=global_cycle, gate=gate)
        print(
            f"final_revision cycle={global_cycle:02d} verdict={gate['verdict']} "
            f"affected_chunks={affected_chunk_ids}"
        )

        if global_cycle == cfg.max_global_cycles:
            warnings.warn(
                f"Reached max_global_cycles={cfg.max_global_cycles} with "
                f"{len(affected_chunk_ids)} chunk(s) still flagged at the excerpt level. "
                "Saving best available translation.",
                stacklevel=2,
            )
            final_path.write_text(book.get_translation(), encoding="utf-8")
            print(f"Saved: {final_path}")
            return

        jobs = build_final_revision_jobs(
            book=book,
            cfg=cfg,
            affected_chunk_ids=affected_chunk_ids,
            findings_by_chunk=grouped_findings_by_chunk,
            global_cycle=global_cycle,
        )
        pending_jobs = _pending_final_revise_jobs(cfg=cfg, jobs=jobs)
        if pending_jobs:
            skipped_count = len(jobs) - len(pending_jobs)
            if skipped_count:
                print(
                    f"final_revise_cycle_{global_cycle:02d}: "
                    f"reusing {skipped_count} completed revision output(s)"
                )
            run_jobs_parallel(
                pending_jobs,
                cfg.run_dir,
                cfg.max_parallel_jobs,
                cfg.dry_run,
                label=f"final_revise_cycle_{global_cycle:02d}",
            )
        load_translations_into_chunks(book=book, cfg=cfg)


# ---------------------------------------------------------------------------
# Translation loading + output
# ---------------------------------------------------------------------------

def load_translations_into_chunks(*, book: Any, cfg: RunnerConfig) -> None:
    """Read segment_translation_*.json files and update book.translation."""
    translations: list[str] = []
    for i in range(len(book.src_text)):
        path = cfg.run_dir / "outputs" / f"segment_translation_{i:04d}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing translation output for chunk {i}: {path}. "
                "Did the translate job complete successfully?"
            )
        translations.append(data["translation"])
    book.set_translation(translations)


def reconstruct_and_save(*, book: Any, cfg: RunnerConfig) -> None:
    """Write the reconstructed translation (no chunk tags) as a draft for excerpt review.

    The draft file is consumed by the book_review agent. The final clean output
    is always written by run_final_revision as {book_name}_{target_lang}.txt.
    """
    draft_path = cfg.run_dir / _translation_draft_filename(
        book_name=book.name,
        target_language_code=cfg.target_language_code,
    )
    draft_path.write_text(book.get_translation(), encoding="utf-8")
    print(f"Saved draft: {draft_path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg: RunnerConfig, resume_point: ResumePoint | None = None) -> None:
    """Execute the full translate -> QE/literary-gate -> revise loop.

    When `resume_point` is given, skip completed cycles and stages based
    on gate files already present in `cfg.run_dir`.
    """
    rp = resume_point or ResumePoint()

    persisted_source = cfg.run_dir / "inputs" / "full_source.txt"
    if persisted_source.exists():
        src_text = persisted_source.read_text(encoding="utf-8")
    else:
        src_text = cfg.book_path.read_text(encoding="utf-8")
    book = Book(cfg.book_path, src_text, True)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    run_style_analysis(book=book, cfg=cfg)

    if rp.skip_main_loop:
        # Rebuild book.translation from disk so downstream stages have it.
        load_translations_into_chunks(book=book, cfg=cfg)
    else:
        failing_chunk_ids: list[int] = list(rp.initial_failing_chunks)

        # When resuming mid-loop (cycle > 1), load existing translations so
        # build_revise_jobs can read book.translation for failing chunks.
        if rp.start_cycle > 1:
            load_translations_into_chunks(book=book, cfg=cfg)

        for cycle in range(rp.start_cycle, cfg.max_cycles + 1):
            if cycle == 1:
                jobs = build_translate_jobs(book=book, cfg=cfg, cycle=cycle)
                revise_jobs: list[JobSpec] = []
            else:
                stall_contexts = _build_stall_contexts(
                    book=book,
                    cfg=cfg,
                    failing_chunk_ids=failing_chunk_ids,
                )
                jobs = build_revise_jobs(
                    book=book, cfg=cfg, cycle=cycle,
                    failing_chunk_ids=failing_chunk_ids,
                    stall_contexts=stall_contexts,
                )
                revise_jobs = jobs

            run_jobs_parallel(
                jobs, cfg.run_dir, cfg.max_parallel_jobs, cfg.dry_run,
                label=f"cycle_{cycle:02d}",
            )
            load_translations_into_chunks(book=book, cfg=cfg)

            chunk_ids = list(range(len(book.src_text))) if cycle == 1 else failing_chunk_ids

            qe_gate, literary_gate = _run_local_review_gates(
                cycle=cycle,
                book=book,
                cfg=cfg,
                chunk_ids=chunk_ids,
            )
            gate = write_master_chunk_gate(
                cycle=cycle, cfg=cfg,
                qe_gate=qe_gate, literary_gate=literary_gate,
            )
            switched_chunks = _update_chunk_state_for_cycle(
                book=book,
                cfg=cfg,
                cycle=cycle,
                chunk_ids=chunk_ids,
                revise_jobs=revise_jobs,
            )
            _print_local_cycle_summary(
                cfg=cfg,
                cycle=cycle,
                switched_chunks=switched_chunks,
                gate=gate,
            )
            print(
                f"cycle={cycle:02d} decision={gate['decision']} "
                f"failing_chunks={gate['failing_chunk_ids']} "
                f"qe_fail={gate['qe_failing_chunk_ids']} "
                f"literary_fail={gate['literary_failing_chunk_ids']}"
            )

            if gate["decision"] == "PASS":
                break

            failing_chunk_ids = gate["failing_chunk_ids"]
            if cycle == cfg.max_cycles:
                warnings.warn(
                    f"Reached max_cycles={cfg.max_cycles} with {len(failing_chunk_ids)} "
                    "chunk(s) still failing the QE/literary gate. "
                    "Saving best available translation.",
                    stacklevel=2,
                )

    if rp.skip_final_revision:
        print("Final revision already complete — nothing to do.")
    else:
        reconstruct_and_save(book=book, cfg=cfg)
        run_final_revision(book=book, cfg=cfg, start_global_cycle=rp.start_global_cycle)

    print("Done.")


# ---------------------------------------------------------------------------
# Run persistence helpers
# ---------------------------------------------------------------------------

def _persist_run_args(*, cfg: RunnerConfig, path: Path) -> None:
    """Write the RunnerConfig fields needed to resume this run.

    Called once when a fresh run starts. `path` is overwritten on every
    call (idempotent for the same cfg).
    """
    payload = {
        "book_path": str(cfg.book_path),
        "stage_models": cfg.stage_models,
        "max_cycles": cfg.max_cycles,
        "max_global_cycles": cfg.max_global_cycles,
        "max_parallel_jobs": cfg.max_parallel_jobs,
        "litrans_threshold": cfg.litrans_threshold,
        "target_language_code": cfg.target_language_code,
        "dry_run": cfg.dry_run,
        "job_timeout_seconds": cfg.job_timeout_seconds,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_run_args(*, run_args_path: Path, run_dir: Path) -> RunnerConfig:
    """Reconstruct a RunnerConfig from persisted run_args.json.

    `run_dir` comes from the --resume CLI argument and overrides any
    run_dir that might be encoded in the JSON (we never write run_dir
    to the file; the directory itself is the location).
    """
    data = json.loads(run_args_path.read_text(encoding="utf-8"))
    target_language_code = data.get("target_language_code", "en")
    if target_language_code not in TARGET_LANGUAGE_CODES:
        raise ValueError(
            f"{run_args_path}: target_language_code must be one of "
            f"{', '.join(sorted(TARGET_LANGUAGE_CODES))}, got {target_language_code!r}"
        )
    return RunnerConfig(
        book_path=Path(data["book_path"]),
        run_dir=run_dir,
        stage_models=data["stage_models"],
        max_cycles=data["max_cycles"],
        max_global_cycles=data["max_global_cycles"],
        max_parallel_jobs=data["max_parallel_jobs"],
        litrans_threshold=data["litrans_threshold"],
        target_language_code=target_language_code,
        target_language=TARGET_LANGUAGE_CODES[target_language_code],
        dry_run=data["dry_run"],
        job_timeout_seconds=data["job_timeout_seconds"],
    )



# ---------------------------------------------------------------------------
# Resume-point detection helpers
# ---------------------------------------------------------------------------

def _highest_completed_cycle(
    base_dir: Path, *, prefix: str, gate_filename: str
) -> tuple[int, dict] | None:
    """Return (highest_N, loaded_gate_json) or None if no matching files exist.

    Matches directories named `{prefix}{NN}` containing `{gate_filename}`.
    """
    if not base_dir.exists():
        return None
    highest: tuple[int, dict] | None = None
    for child in base_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        suffix = child.name[len(prefix):]
        try:
            n = int(suffix)
        except ValueError:
            continue
        gate_path = child / gate_filename
        if not gate_path.exists():
            continue
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if highest is None or n > highest[0]:
            highest = (n, gate)
    return highest


def _detect_final_revision_resume(cfg: RunnerConfig) -> tuple[int, bool]:
    """Return (start_global_cycle, skip_final_revision).

    Extended in Task 6 to handle existing final-revision gates. Stub
    returns fresh-start values so Task 5's PASS-no-final-gate case works.
    """
    final_gate_dir = cfg.run_dir / "gate" / "final"
    completed = _highest_completed_cycle(
        final_gate_dir, prefix="cycle_", gate_filename="gate.json"
    )
    if completed is None:
        return (1, False)
    cycle_n, final_gate = completed
    if final_gate["verdict"] == "PASS":
        return (1, True)
    if final_gate["verdict"] == "FAIL" and cycle_n >= cfg.max_global_cycles:
        return (1, True)
    if _final_revision_cycle_has_pending_targeted_revisions(
        cfg=cfg,
        final_gate=final_gate,
        global_cycle=cycle_n,
    ):
        return (cycle_n, False)
    return (cycle_n + 1, False)


def _book_name_from_path(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 2:
        return path.stem
    return "_".join(parts[:-1])


def _final_revision_cycle_has_pending_targeted_revisions(
    *,
    cfg: RunnerConfig,
    final_gate: dict[str, Any],
    global_cycle: int,
) -> bool:
    affected_chunk_ids = final_gate.get("affected_chunk_ids")
    if not isinstance(affected_chunk_ids, list):
        return False
    book_name = _book_name_from_path(cfg.book_path)
    completed_job_ids = _completed_job_ids(cfg.run_dir)
    for chunk_id in affected_chunk_ids:
        if not isinstance(chunk_id, int):
            return True
        job_id = f"final_revise_{book_name}_chunk_{chunk_id:04d}_global_cycle_{global_cycle:02d}"
        if job_id not in completed_job_ids:
            return True
        try:
            _parse_segment_translation(cfg.run_dir, chunk_id=chunk_id)
            _parse_final_revise_report(
                cfg.run_dir,
                chunk_id=chunk_id,
                global_cycle=global_cycle,
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return True
    return False


def _detect_resume_point(cfg: RunnerConfig) -> ResumePoint:
    """Scan an existing run_dir and decide where to pick up.

    See docs/superpowers/specs/2026-04-14-agents-pipeline-resume-design.md
    for the full detection logic. Returns a default ResumePoint (fresh
    run) if no main-loop gate files exist.
    """
    final_gate_root = cfg.run_dir / "gate" / "final"
    has_any_final_gate = final_gate_root.exists() and any(
        (child / "gate.json").exists()
        for child in final_gate_root.iterdir()
        if child.is_dir() and child.name.startswith("cycle_")
    )

    main_gate_dir = cfg.run_dir / "gate"
    completed_main = _highest_completed_cycle(
        main_gate_dir, prefix="cycle_", gate_filename="gate.json"
    )

    if has_any_final_gate and completed_main is None:
        raise PipelineError(
            f"inconsistent run dir {cfg.run_dir}: final-revision gates "
            "exist but no main-loop gate was found"
        )
    if has_any_final_gate and completed_main is not None:
        cycle_n_check, main_gate_check = completed_main
        decision = main_gate_check["decision"]
        if decision == "FAIL" and cycle_n_check < cfg.max_cycles:
            raise PipelineError(
                f"inconsistent run dir {cfg.run_dir}: final-revision gates "
                "exist but main loop is not complete"
            )

    if completed_main is None:
        return ResumePoint()

    cycle_n, main_gate = completed_main

    if main_gate["decision"] == "FAIL" and cycle_n < cfg.max_cycles:
        return ResumePoint(
            start_cycle=cycle_n + 1,
            initial_failing_chunks=tuple(main_gate["failing_chunk_ids"]),
            skip_main_loop=False,
            start_global_cycle=1,
            skip_final_revision=False,
        )

    # Both "PASS" and "FAIL at max_cycles" mean the main loop is done.
    # Either way, proceed to final revision (which may or may not also
    # be partially done — see below).
    final_resume = _detect_final_revision_resume(cfg)
    return ResumePoint(
        start_cycle=1,
        initial_failing_chunks=(),
        skip_main_loop=True,
        start_global_cycle=final_resume[0],
        skip_final_revision=final_resume[1],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_run_dir(*, target_language_code: str) -> Path:
    """Auto-generate run directory name from target language code + UTC timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return Path("runs") / f"{target_language_code}_{ts}"


def load_stage_models(config_path: Path) -> dict[str, dict[str, str | None]]:
    """Load and validate the per-stage provider/model config JSON file.

    The file must define `stages` containing every name in STAGE_NAMES. Each
    stage entry must have a `provider` (in PROVIDER_VALUES) and a `model`
    (string or null).
    """
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    stages = raw.get("stages")
    if not isinstance(stages, dict):
        raise ValueError(f"{config_path}: top-level 'stages' object is required")

    missing = STAGE_NAMES - stages.keys()
    if missing:
        raise ValueError(f"{config_path}: missing stages: {sorted(missing)}")
    extra = stages.keys() - STAGE_NAMES
    if extra:
        raise ValueError(f"{config_path}: unknown stages: {sorted(extra)}")

    out: dict[str, dict[str, str | None]] = {}
    for name, entry in stages.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{config_path}: stage '{name}' must be an object")
        provider = entry.get("provider")
        if provider not in PROVIDER_VALUES:
            raise ValueError(
                f"{config_path}: stage '{name}' provider must be one of "
                f"{sorted(PROVIDER_VALUES)}, got {provider!r}"
            )
        model = entry.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError(f"{config_path}: stage '{name}' model must be a string or null")
        out[name] = {"provider": provider, "model": model}
    return out


def resolve_target_language(code: str) -> tuple[str, str]:
    """Return normalized target language code and display name."""
    normalized = code.lower()
    if normalized not in TARGET_LANGUAGE_CODES:
        raise ValueError(
            f"target language must be one of {', '.join(sorted(TARGET_LANGUAGE_CODES))}; "
            f"got {code!r}"
        )
    return normalized, TARGET_LANGUAGE_CODES[normalized]


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    """Parse command-line arguments for the MT agents pipeline.

    Returns a tuple of (parsed namespace, parser) so that callers can invoke
    parser.error(...) to emit consistent CLI-style error messages.
    """
    parser = argparse.ArgumentParser(
        description="Agentic MT pipeline: translate -> QE + chunk literary review -> revise loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--book_path",
        type=Path,
        default=None,
        help="Source book path (required unless --resume is given).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an existing run directory. When set, --book_path / --config / "
             "--max_cycles etc. must not be passed; they are read from run_args.json.",
    )
    parser.add_argument(
        "--config", default=str(_DEFAULT_CONFIG_PATH),
        help="Path to JSON config defining provider/model per agent stage."
    )
    parser.add_argument(
        "--target_lang",
        default="en",
        help="Target language code for translation output."
    )
    parser.add_argument("--max_cycles", type=int, default=2)
    parser.add_argument("--max_global_cycles", type=int, default=2)
    parser.add_argument("--max_parallel_jobs", type=int, default=6)
    parser.add_argument(
        "--litrans_threshold", type=float, default=0.7,
        help="Legacy informational QE threshold shown in reports; chunk gating is rule-based."
    )
    parser.add_argument("--run_dir", default=None,
                        help="Output directory. Auto-generated if omitted.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip agent calls; write placeholder translations.")
    return parser.parse_args(), parser


def main() -> None:
    """Entry point for the MT agents pipeline CLI."""
    import sys

    args, parser = parse_args()
    cli_tokens = set(sys.argv[1:])

    def _arg_was_passed_on_cli(name: str) -> bool:
        return ("--" + name) in cli_tokens

    if args.resume is not None:
        conflicting = [
            name for name in (
                "book_path", "config", "max_cycles", "max_global_cycles",
                "max_parallel_jobs", "litrans_threshold", "target_lang", "run_dir", "dry_run",
            )
            if _arg_was_passed_on_cli(name)
        ]
        if conflicting:
            parser.error(
                f"--resume cannot be combined with: {', '.join('--' + n for n in conflicting)}. "
                "These are read from run_args.json in the resume directory."
            )

        run_dir = args.resume
        run_args_path = run_dir / "run_args.json"
        if not run_args_path.exists():
            parser.error(
                f"Cannot resume {run_dir}: run_args.json not found. "
                "This run was created before resume support existed, or is not a valid run dir."
            )

        cfg = _load_run_args(run_args_path=run_args_path, run_dir=run_dir)
        resume_point = _detect_resume_point(cfg)
        run(cfg, resume_point=resume_point)
        return

    if args.book_path is None:
        parser.error("--book_path is required when --resume is not given.")

    try:
        target_language_code, target_language = resolve_target_language(args.target_lang)
    except ValueError as exc:
        parser.error(str(exc))
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else _build_run_dir(target_language_code=target_language_code)
    )
    stage_models = load_stage_models(Path(args.config))
    cfg = RunnerConfig(
        book_path=Path(args.book_path),
        run_dir=run_dir,
        stage_models=stage_models,
        max_cycles=args.max_cycles,
        max_global_cycles=args.max_global_cycles,
        max_parallel_jobs=args.max_parallel_jobs,
        litrans_threshold=args.litrans_threshold,
        target_language_code=target_language_code,
        target_language=target_language,
        dry_run=args.dry_run,
    )
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    _persist_run_args(cfg=cfg, path=cfg.run_dir / "run_args.json")
    run(cfg)


if __name__ == "__main__":
    main()
