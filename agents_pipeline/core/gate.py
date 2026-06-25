# agents_pipeline/core/gate.py
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any

_MAPPING = {"YES": 1.0, "MAYBE": 0.5, "NO": 0.0}
_EXPECTED_QIDS: tuple[str, ...] = tuple(str(i) for i in range(1, 26))
_MAX_ALLOWED_MAYBE = 5
_DEFAULT_YES_ISSUE = "No material issue detected for this question."
_QUESTION_GROUPS: dict[str, str] = {
    "1": "Lexical Accuracy and Meaning",
    "2": "Lexical Accuracy and Meaning",
    "3": "Lexical Accuracy and Meaning",
    "4": "Lexical Accuracy and Meaning",
    "5": "Lexical Accuracy and Meaning",
    "6": "Pragmatics and Subtext",
    "7": "Pragmatics and Subtext",
    "8": "Cultural Transfer",
    "9": "Cultural Transfer",
    "10": "Cultural Transfer",
    "11": "Cultural Transfer",
    "12": "Cultural Transfer",
    "13": "Cultural Transfer",
    "14": "Cultural Transfer",
    "15": "Voice, Register, and Style",
    "16": "Voice, Register, and Style",
    "17": "Voice, Register, and Style",
    "18": "Voice, Register, and Style",
    "19": "Voice, Register, and Style",
    "20": "Narrative and Local Consistency",
    "21": "Narrative and Local Consistency",
    "22": "Fluency and Literary Effect",
    "23": "Fluency and Literary Effect",
    "24": "Fluency and Literary Effect",
    "25": "Fluency and Literary Effect",
}


def _parse_litrans_response_dict(response: str) -> dict[str, Any]:
    """Parse the raw LiTrans response text into a JSON object."""
    text = response.strip()
    if not text:
        raise ValueError("LiTransProQA response is empty")

    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        raise ValueError("LiTransProQA response contains no opening brace '{'")
    if end == -1 or end <= start:
        raise ValueError("LiTransProQA response contains no closing brace '}'")
    text = text[start:end + 1]

    if text.count("{") != text.count("}"):
        raise ValueError(
            f"LiTransProQA response has unbalanced braces "
            f"({{ count={text.count('{')}, }} count={text.count('}')})"
        )

    try:
        answers = json.loads(text)
    except json.JSONDecodeError:
        answers = ast.literal_eval(text)
    if not isinstance(answers, dict):
        raise ValueError(f"LiTransProQA response parsed to {type(answers).__name__}, expected dict")
    if not answers:
        raise ValueError("LiTransProQA response contains no answers")
    return answers


def parse_litrans_answers(response: str) -> dict[str, dict[str, str]]:
    """Parse and validate the 25-question LiTrans response.

    Accepts either the current object form
    {"1": {"judgment": "YES", "issue": "..."}}
    or the legacy string form {"1": "YES"} and normalizes to the object form.
    """
    raw_answers = _parse_litrans_response_dict(response)
    keys = {str(k) for k in raw_answers.keys()}
    expected = set(_EXPECTED_QIDS)
    missing = sorted(expected - keys, key=int)
    extra = sorted(keys - expected, key=int)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing question ids: {missing}")
        if extra:
            details.append(f"unexpected question ids: {extra}")
        raise ValueError("LiTransProQA response must contain exactly questions 1-25; " + "; ".join(details))

    normalized: dict[str, dict[str, str]] = {}
    for qid in _EXPECTED_QIDS:
        value = raw_answers[qid]
        if isinstance(value, str):
            judgment = value.strip().upper()
            issue = (
                "No material issue detected for this question."
                if judgment == "YES"
                else "Issue statement missing in legacy answer format."
            )
        elif isinstance(value, dict):
            judgment_raw = value.get("judgment")
            issue_raw = value.get("issue")
            if not isinstance(judgment_raw, str):
                raise ValueError(f"LiTransProQA answer {qid} missing string field 'judgment'")
            judgment = judgment_raw.strip().upper()
            if judgment == "YES":
                if isinstance(issue_raw, str) and issue_raw.strip():
                    issue = issue_raw.strip()
                else:
                    issue = _DEFAULT_YES_ISSUE
            else:
                if not isinstance(issue_raw, str) or not issue_raw.strip():
                    raise ValueError(
                        f"LiTransProQA answer {qid} missing non-empty string field 'issue'"
                    )
                issue = issue_raw.strip()
        else:
            raise ValueError(
                f"LiTransProQA answer {qid} must be a string or object, got {type(value).__name__}"
            )

        if judgment not in _MAPPING:
            raise KeyError(judgment)
        normalized[qid] = {"judgment": judgment, "issue": issue}
    return normalized


def litr_score(response: str) -> float:
    """Parse a LiTransProQA response and return a score in [0, 1].

    Expects all 25 question answers and computes the mean of their judgments.
    """
    answers = parse_litrans_answers(response)
    vals = [_MAPPING[answers[qid]["judgment"]] for qid in _EXPECTED_QIDS]
    return sum(vals) / len(vals)


def run_gate(
    *,
    cycle: int,
    book: Any,
    cfg: Any,
    dispatch_jobs: Any,
    build_jobs: Any,
    chunks_to_score: list[int] | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Dispatch LiTransProQA agent jobs, score answers, write a QE gate.

    `build_jobs(book, cfg, chunk_ids)` returns the JobSpec list for the given
    chunk ids. `dispatch_jobs(jobs, run_dir, max_workers, dry_run, label)`
    runs them in parallel. When chunks_to_score is None (first cycle), all
    chunks are scored.

    For each chunk, reads outputs/litrans_answers_{i:04d}.json produced by the
    agent, computes the score, and writes outputs/litrans_review_{i:04d}.json
    with score/report metadata. Raises if the agent's answer file is missing
    or cannot be parsed into a valid 25-question response.
    """
    run_dir: Path = cfg.run_dir
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if chunks_to_score is None:
        score_ids = list(range(len(book.src_text)))
    else:
        score_ids = list(chunks_to_score)

    if "cycle" in inspect.signature(build_jobs).parameters:
        jobs = build_jobs(book=book, cfg=cfg, chunk_ids=score_ids, cycle=cycle)
    else:
        jobs = build_jobs(book=book, cfg=cfg, chunk_ids=score_ids)
    dispatch_jobs(
        jobs,
        run_dir,
        max_workers if max_workers is not None else cfg.max_parallel_jobs,
        cfg.dry_run,
        f"litrans_review_cycle_{cycle:02d}",
    )

    failing_chunk_ids: list[int] = []
    for i in score_ids:
        answers_path = outputs_dir / f"litrans_answers_{i:04d}.json"
        if not answers_path.is_file():
            raise FileNotFoundError(
                f"LiTransProQA agent did not produce {answers_path}"
            )
        raw_response = answers_path.read_text(encoding="utf-8")
        answers = parse_litrans_answers(raw_response)
        score = litr_score(raw_response)
        no_count = sum(1 for qid in _EXPECTED_QIDS if answers[qid]["judgment"] == "NO")
        maybe_count = sum(1 for qid in _EXPECTED_QIDS if answers[qid]["judgment"] == "MAYBE")
        yes_count = len(_EXPECTED_QIDS) - no_count - maybe_count
        verdict = "PASS" if no_count == 0 and maybe_count <= _MAX_ALLOWED_MAYBE else "FAIL"
        failed_question_ids = [
            qid for qid in _EXPECTED_QIDS if answers[qid]["judgment"] != "YES"
        ]
        failed_questions = [
            {
                "question_id": qid,
                "group": _QUESTION_GROUPS[qid],
                "judgment": answers[qid]["judgment"],
                "issue": answers[qid]["issue"],
            }
            for qid in failed_question_ids
        ]

        review_path = outputs_dir / f"litrans_review_{i:04d}.json"
        review_path.write_text(
            json.dumps(
                {
                    "chunk_id": i,
                    "score": score,
                    "verdict": verdict,
                    "decision_rule": {
                        "allowed_no_count": 0,
                        "allowed_maybe_count": _MAX_ALLOWED_MAYBE,
                    },
                    "yes_count": yes_count,
                    "maybe_count": maybe_count,
                    "no_count": no_count,
                    "answers": answers,
                    "failed_question_ids": failed_question_ids,
                    "failed_questions": failed_questions,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if verdict == "FAIL":
            failing_chunk_ids.append(i)

    decision = "PASS" if not failing_chunk_ids else "FAIL"
    gate: dict[str, Any] = {
        "cycle": cycle,
        "decision": decision,
        "failing_chunk_ids": failing_chunk_ids,
        "decision_rule": {
            "allowed_no_count": 0,
            "allowed_maybe_count": _MAX_ALLOWED_MAYBE,
        },
        "reason": (
            "all_chunks_met_no_and_maybe_limits"
            if decision == "PASS"
            else "chunks_failed_no_or_maybe_limits"
        ),
    }

    gate_path = run_dir / "gate" / f"litrans_cycle_{cycle:02d}" / "gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate
