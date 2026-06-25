#!/usr/bin/env python3
"""Run automatic MT evaluation over LitMT paragraph pickle or chunk-review JSONL files.

Input pickles are expected to contain source_paras and translator_data. Google
Translate fields (gt_paras/gt_sents) are intentionally ignored for metric
scoring.

Input JSONL files are expected to contain chunk-review records with SRC, HT,
and MT string fields. Pretty-printed JSON objects separated by whitespace are
accepted, as are one-object-per-line JSONL files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


COMET_MODELS = {
    "comet22": ("wmt22-comet-da", "comet", False),
    "cometkiwi": ("wmt22-cometkiwi-da", "comet-qe", True),
}
COMET_TOKENIZERS = {
    "comet22": "Unbabel/wmt22-comet-da",
    "cometkiwi": "Unbabel/wmt22-cometkiwi-da",
}
METRICX_MODEL = "google/metricx-24-hybrid-xl-v2p6"
METRICX_TOKENIZER = "google/mt5-xl"
QE_METRICS = {"cometkiwi", "metricx-qe"}
REFERENCE_METRICS = {"comet22", "metricx"}
SUPPORTED_METRICS = QE_METRICS | REFERENCE_METRICS
LITRANSPROQA_METRIC = "litransproqa"
HIGHER_IS_BETTER_METRICS = {"comet22", "cometkiwi", LITRANSPROQA_METRIC}
DEFAULT_LLM_JUDGE_MODEL = "gemini-3.1-pro-preview"
DEFAULT_LLM_JUDGE_RETRIES = 3
DEFAULT_LLM_JUDGE_RETRY_DELAY = 30.0
LITRANSPROQA_WEIGHTS_PATH = Path("LiTransProQA/config/question_weights.csv")
BOOK_RESULTS_DIR_NAME = "books"
ANALYSIS_DIR_NAME = "analysis"


@dataclass(frozen=True)
class Segment:
    job_index: int
    index: int
    source: str
    hypothesis: str
    reference: str | None = None


@dataclass
class EvaluationJob:
    pkl_name: str
    system_name: str
    sources: list[str]
    hyps: list[str]
    ref_name: str | None
    refs: list[str] | None
    scores: dict[str, dict[int, float]] = field(default_factory=dict)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    alignment_diagnostics: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedInput:
    dataset_name: str
    sources: list[str]
    systems: dict[str, list[str]]
    ref_name: str | None
    refs: list[str] | None


AlignmentDiagnosticsMap = dict[tuple[str, str], dict[int, dict[str, Any]]]


def book_results_dir(out_dir: Path) -> Path:
    return out_dir / BOOK_RESULTS_DIR_NAME


def analysis_results_dir(out_dir: Path) -> Path:
    return out_dir / ANALYSIS_DIR_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Input directory containing *.pkl or chunk-review *.jsonl files, "
            "or a PAR3-style tree with */*/*.pkl files."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove any existing output directory contents before recomputing requested metrics.",
    )
    parser.add_argument(
        "--metrics",
        default="comet22,cometkiwi,metricx,metricx-qe",
        help=f"Comma-separated metrics. Supported: {', '.join(sorted(SUPPORTED_METRICS))}",
    )
    parser.add_argument(
        "--llm-as-a-judge",
        action="store_true",
        help="Also evaluate translations with LiTransProQA using Gemini as the judge model.",
    )
    parser.add_argument(
        "--llm-judge-model",
        default=DEFAULT_LLM_JUDGE_MODEL,
        help="Gemini model name used when --llm-as-a-judge is enabled.",
    )
    parser.add_argument(
        "--llm-judge-retries",
        type=int,
        default=DEFAULT_LLM_JUDGE_RETRIES,
        help="Retries per LiTransProQA judge request before recording a failed segment.",
    )
    parser.add_argument(
        "--llm-judge-retry-delay",
        type=float,
        default=DEFAULT_LLM_JUDGE_RETRY_DELAY,
        help="Initial retry delay in seconds for LiTransProQA judge requests; doubles each retry.",
    )
    parser.add_argument(
        "--alignment-diagnostics-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV from par3/verify_alignment.py. Matching rows are copied into "
            "segment_scores.jsonl as alignment_diagnostics; they are not skipped here."
        ),
    )
    parser.add_argument(
        "--misalignment-csv",
        dest="alignment_diagnostics_csv",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device",
        default="0",
        help="Device hint for pymarian-eval, e.g. '0' for GPU 0 or 'cpu' to force CPU.",
    )
    parser.add_argument(
        "--comet-max-input-length",
        type=int,
        default=512,
        help="Maximum tokenized input length for COMET inputs.",
    )
    parser.add_argument(
        "--metricx-max-input-length",
        type=int,
        default=1536,
        help="Maximum tokenized input length for MetricX inputs.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable or "python",
        help="Python executable used for MetricX subprocesses.",
    )
    parser.add_argument("--metricx-dir", type=Path, default=Path("metricx"))
    parser.add_argument("--comet22-tokenizer", default=COMET_TOKENIZERS["comet22"])
    parser.add_argument("--cometkiwi-tokenizer", default=COMET_TOKENIZERS["cometkiwi"])
    parser.add_argument("--metricx-tokenizer", default=METRICX_TOKENIZER)
    parser.add_argument("--metricx-model", default=METRICX_MODEL)
    parser.add_argument(
        "--comet-workspace",
        default=None,
        help="Optional PyMarian workspace value, e.g. -8000 for GPU memory tuning.",
    )
    parser.add_argument(
        "--comet-fp16",
        action="store_true",
        help="Pass --fp16 to pymarian-eval for lower GPU memory use.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a dict")
    return data


def load_json_records(path: Path) -> list[dict[str, Any]]:
    """Parse compact JSONL or pretty-printed, whitespace-separated JSON objects."""
    decoder = json.JSONDecoder()
    content = path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    index = 0

    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break

        try:
            record, next_index = decoder.raw_decode(content, index)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} near character {index}: {exc}") from exc

        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object in {path} near character {index}")
        records.append(record)
        index = next_index

    if not records:
        raise ValueError(f"{path} did not contain any JSON records")
    return records


def require_jsonl_text(record: dict[str, Any], field_name: str, path: Path, record_number: int) -> str:
    value = record.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{path}: record {record_number} is missing string field {field_name!r}")
    return value


def load_jsonl_input(path: Path) -> LoadedInput:
    records = load_json_records(path)
    sources: list[str] = []
    refs: list[str] = []
    hyps: list[str] = []

    for record_number, record in enumerate(records, start=1):
        sources.append(require_jsonl_text(record, "SRC", path, record_number))
        refs.append(require_jsonl_text(record, "HT", path, record_number))
        hyps.append(require_jsonl_text(record, "MT", path, record_number))

    dataset_name = path.stem
    return LoadedInput(
        dataset_name=dataset_name,
        sources=sources,
        systems={
            f"{dataset_name}_ht": refs,
            f"{dataset_name}_mt": hyps,
        },
        ref_name=f"{dataset_name}_ht",
        refs=refs,
    )


def infer_language_pair(dataset_name: str) -> str:
    parts = dataset_name.split("_")
    if len(parts) >= 2 and len(parts[-2]) == 2 and len(parts[-1]) == 2:
        return f"{parts[-2]}-{parts[-1]}"
    return "unknown-en"


def clean_system_name(raw_name: str) -> str:
    return Path(raw_name).stem


def get_translator_paras(
    payload: Any,
    expected_len: int,
) -> list[str] | None:
    if isinstance(payload, dict):
        paras = payload.get("translator_paras")
        if paras:
            return [str(x) for x in paras]
        # all_sents = payload.get("all_sents")
        # if all_sents and len(all_sents) == expected_len:
        #     return [str(x) for x in all_sents]

    return None


def is_empty_text_placeholder(raw_name: str, payload: Any) -> bool:
    return isinstance(payload, dict) and not payload and Path(raw_name).suffix == ".txt"


def load_systems(data: dict[str, Any], data_file: Path) -> tuple[list[str], dict[str, list[str]]]:
    sources = [str(x) for x in data.get("source_paras", [])]
    if not sources:
        raise ValueError(f"{data_file} has no source_paras")

    translator_data = data.get("translator_data", {})
    if not isinstance(translator_data, dict):
        raise ValueError(f"{data_file} translator_data is not a dict")

    systems: dict[str, list[str]] = {}
    for raw_name, payload in translator_data.items():
        system_name = clean_system_name(str(raw_name))
        if system_name == "gt" or system_name.endswith("_gt"):
            logging.info("%s: skipping %s; Google Translate is not evaluated", data_file, system_name)
            continue
        raw_name_str = str(raw_name)
        paras = get_translator_paras(payload, len(sources))
        if paras is None:
            if is_empty_text_placeholder(raw_name_str, payload):
                logging.info(
                    "%s: skipping empty placeholder %s; using aligned translator_data entries",
                    data_file,
                    raw_name,
                )
                continue
            raise ValueError(f"{data_file}: {raw_name} has no translator_paras/all_sents alignment in pickle")
        if len(paras) != len(sources):
            raise ValueError(
                f"{data_file}: {system_name} length mismatch "
                f"source={len(sources)} hypothesis={len(paras)}"
            )
        systems[system_name] = paras
    return sources, systems


def find_reference_system(systems: dict[str, list[str]]) -> tuple[str, list[str]] | None:
    refs = sorted(name for name in systems if name.endswith("_ht"))
    if not refs:
        return None
    if len(refs) > 1:
        logging.warning("multiple *_ht references found; using %s", refs[0])
    return refs[0], systems[refs[0]]


def load_pickle_input(path: Path, data_dir: Path) -> LoadedInput:
    data = load_pickle(path)
    sources, systems = load_systems(data, path)
    ref = find_reference_system(systems)
    ref_name, refs = ref if ref else (None, None)
    return LoadedInput(
        dataset_name=output_dataset_name(path, data_dir),
        sources=sources,
        systems=systems,
        ref_name=ref_name,
        refs=refs,
    )


def load_tokenizer(tokenizer_name: str, *, use_fast: bool = False) -> Any | None:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        logging.warning("transformers is unavailable; using whitespace length checks: %s", exc)
        return None
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast)
    except Exception as exc:
        logging.warning("could not load tokenizer %s; using whitespace length checks: %s", tokenizer_name, exc)
        return None


def load_metric_tokenizers(args: argparse.Namespace, metrics: list[str]) -> dict[str, Any | None]:
    tokenizers: dict[str, Any | None] = {}
    if any(metric in {"metricx", "metricx-qe"} for metric in metrics):
        tokenizers["metricx"] = load_tokenizer(args.metricx_tokenizer, use_fast=False)
    if "comet22" in metrics:
        tokenizers["comet22"] = load_tokenizer(args.comet22_tokenizer, use_fast=False)
    if "cometkiwi" in metrics:
        tokenizers["cometkiwi"] = load_tokenizer(args.cometkiwi_tokenizer, use_fast=False)
    return tokenizers


def input_text_for_length(segment: Segment, qe: bool) -> str:
    if qe:
        return f"source: {segment.source} candidate: {segment.hypothesis}"
    return f"source: {segment.source} candidate: {segment.hypothesis} reference: {segment.reference or ''}"


def normalize_metric_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def normalize_segment_for_metric(metric: str, segment: Segment) -> tuple[Segment | None, dict[str, Any] | None]:
    source = normalize_metric_text(segment.source)
    hypothesis = normalize_metric_text(segment.hypothesis)
    reference = normalize_metric_text(segment.reference) if metric in REFERENCE_METRICS else None

    empty_fields: list[str] = []
    if not source:
        empty_fields.append("source")
    if not hypothesis:
        empty_fields.append("hypothesis")
    if metric in REFERENCE_METRICS and not reference:
        empty_fields.append("reference")

    if empty_fields:
        if "hypothesis" in empty_fields:
            reason = "empty_hypothesis"
        else:
            reason = f"empty_{empty_fields[0]}" if len(empty_fields) == 1 else "empty_required_fields"
        return None, {
            "metric": metric,
            "segment_index": segment.index,
            "reason": reason,
            "empty_fields": empty_fields,
        }

    return (
        Segment(
            segment.job_index,
            segment.index,
            source,
            hypothesis,
            reference,
        ),
        None,
    )


def token_length(text: str, tokenizer: Any | None) -> int:
    if tokenizer is None:
        return len(text.split())
    encoded = tokenizer(text, truncation=False, padding=False)
    return len(encoded["input_ids"])


def validate_segments(
    metric: str,
    segments: Iterable[Segment],
    tokenizer: Any | None,
    max_input_length: int,
    pkl_name: str,
    system_name: str,
) -> tuple[list[Segment], list[dict[str, Any]]]:
    validated: list[Segment] = []
    skipped: list[dict[str, Any]] = []
    qe = metric in QE_METRICS
    for segment in segments:
        length = token_length(input_text_for_length(segment, qe=qe), tokenizer)
        if length > max_input_length:
            skipped.append(
                {
                    "metric": metric,
                    "segment_index": segment.index,
                    "reason": "max_input_length_exceeded",
                    "token_count": length,
                    "max_input_length": max_input_length,
                }
            )
            logging.warning(
                "%s/%s: skipping %s segment %d; input length %d exceeds %d",
                pkl_name,
                system_name,
                metric,
                segment.index,
                length,
                max_input_length,
            )
            continue
        validated.append(segment)
    return validated, skipped


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def parse_score_lines(path: Path, expected: int) -> list[float]:
    scores: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            scores.append(float(line.split()[-1]))
        except ValueError as exc:
            raise RuntimeError(f"could not parse score from {path}: {line!r}") from exc
    if len(scores) != expected:
        raise RuntimeError(f"{path} contained {len(scores)} scores, expected {expected}")
    return scores


def segment_key(segment: Segment) -> tuple[int, int]:
    return segment.job_index, segment.index


def run_comet(
    metric: str,
    segments: list[Segment],
    args: argparse.Namespace,
    work_dir: Path,
) -> dict[tuple[int, int], float]:
    model_name, like, qe = COMET_MODELS[metric]
    if not segments:
        return {}

    with tempfile.TemporaryDirectory(prefix=f"{metric}_", dir=work_dir) as tmp:
        tmp_path = Path(tmp)
        src_file = tmp_path / "src.txt"
        hyp_file = tmp_path / "hyp.txt"
        ref_file = tmp_path / "ref.txt"
        out_file = tmp_path / "scores.txt"
        write_lines(src_file, (s.source for s in segments))
        write_lines(hyp_file, (s.hypothesis for s in segments))
        if not qe:
            write_lines(ref_file, (s.reference or "" for s in segments))

        cmd = [
            "pymarian-eval",
            "-m",
            model_name,
            "-l",
            like,
            "-s",
            str(src_file),
            "-t",
            str(hyp_file),
            "-o",
            str(out_file),
            "-a",
            "skip",
            "--mini-batch",
            str(args.batch_size),
        ]
        if not qe:
            cmd.extend(["-r", str(ref_file)])
        if args.device.lower() == "cpu":
            cmd.extend(["-c", str(max(1, os.cpu_count() or 1))])
        else:
            device = "0" if args.device.lower() == "cuda" else args.device
            cmd.extend(["-d", device])
        if args.comet_workspace is not None:
            cmd.extend(["--workspace", str(args.comet_workspace)])
        if args.comet_fp16:
            cmd.append("--fp16")

        logging.info("running %s on %d segments", metric, len(segments))
        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError("pymarian-eval was not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                logging.error("pymarian-eval stdout:\n%s", exc.stdout.rstrip())
            if exc.stderr:
                logging.error("pymarian-eval stderr:\n%s", exc.stderr.rstrip())
            raise RuntimeError(
                "pymarian-eval failed for "
                f"{metric} with exit code {exc.returncode}. Command: {' '.join(cmd)}"
            ) from exc
        scores = parse_score_lines(out_file, len(segments))
    return {segment_key(segment): score for segment, score in zip(segments, scores)}


def write_metricx_input(path: Path, segments: list[Segment], qe: bool) -> None:
    with path.open("w", encoding="utf-8") as f:
        for batch_index, segment in enumerate(segments):
            job_index, segment_index = segment_key(segment)
            row = {
                "batch_index": batch_index,
                "job_index": job_index,
                "segment_index": segment_index,
                "source": segment.source,
                "hypothesis": segment.hypothesis,
                "reference": "" if qe else segment.reference or "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_metricx_scores(path: Path, segments: list[Segment]) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    segment_by_batch_index = {i: segment for i, segment in enumerate(segments)}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            batch_index = int(row["batch_index"])
            segment = segment_by_batch_index[batch_index]
            scores[segment_key(segment)] = float(row["prediction"])
    if len(scores) != len(segments):
        raise RuntimeError(f"MetricX returned {len(scores)} scores, expected {len(segments)}")
    return scores


def run_metricx(
    metric: str,
    segments: list[Segment],
    args: argparse.Namespace,
    work_dir: Path,
) -> dict[tuple[int, int], float]:
    if not segments:
        return {}

    qe = metric == "metricx-qe"
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{metric}_", dir=work_dir) as tmp:
        tmp_path = Path(tmp)
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        write_metricx_input(input_file, segments, qe=qe)

        cmd = [
            args.python,
            "-m",
            "metricx24.predict",
            "--tokenizer",
            args.metricx_tokenizer,
            "--model_name_or_path",
            args.metricx_model,
            "--max_input_length",
            str(args.metricx_max_input_length),
            "--batch_size",
            str(args.batch_size),
            "--input_file",
            str(input_file.resolve()),
            "--output_file",
            str(output_file.resolve()),
        ]
        if qe:
            cmd.append("--qe")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.metricx_dir.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
        logging.info("running %s on %d segments", metric, len(segments))
        subprocess.run(cmd, cwd=args.metricx_dir, env=env, check=True)

        scores = read_metricx_scores(output_file, segments)
    return scores


def build_litransproqa_prompt(source: str, translation: str, base_dir: Path) -> str:
    template = (base_dir / "template" / "template_baseline.txt").read_text(encoding="utf-8")
    questions = (base_dir / "template" / "QA_final.txt").read_text(encoding="utf-8")
    return template.format(source=source, translation=translation, questions=questions)


def load_litransproqa_weights(path: Path = LITRANSPROQA_WEIGHTS_PATH) -> list[float]:
    with path.open(newline="", encoding="utf-8") as f:
        weights = [float(row["score"]) for row in csv.DictReader(f)]

    if not weights:
        raise ValueError(f"{path} did not contain any LiTransProQA question weights")
    return weights


def weighted_litransproqa_score(answer_blob: Any, weights: list[float]) -> float | None:
    from LiTransProQA.prompting_method.eval import mapping, parse_answers

    answers = parse_answers(answer_blob)
    weighted_scores: list[float] = []
    used_weights: list[float] = []
    for question_number, weight in enumerate(weights, start=1):
        answer = answers.get(str(question_number), answers.get(question_number))
        if answer is None:
            continue
        weighted_scores.append(mapping[str(answer).strip().upper()] * weight)
        used_weights.append(weight)

    if not weighted_scores:
        return None
    return sum(weighted_scores) / sum(used_weights)


def build_litransproqa_rows(
    jobs: list[EvaluationJob],
    base_dir: Path,
    job_indexes: set[int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job_index, job in enumerate(jobs):
        if job_indexes is not None and job_index not in job_indexes:
            continue
        for segment_index, (source, hypothesis) in enumerate(zip(job.sources, job.hyps)):
            source_text = source.strip()
            hypothesis_text = hypothesis.strip()
            if not source_text or not hypothesis_text:
                empty_fields = []
                if not source_text:
                    empty_fields.append("source")
                if not hypothesis_text:
                    empty_fields.append("hypothesis")
                job.skipped.append(
                    {
                        "metric": LITRANSPROQA_METRIC,
                        "segment_index": segment_index,
                        "reason": "empty_required_fields",
                        "empty_fields": empty_fields,
                    }
                )
                continue

            rows.append(
                {
                    "job_index": job_index,
                    "segment_index": segment_index,
                    "src": source_text,
                    "tgt": hypothesis_text,
                    "pair": infer_language_pair(job.pkl_name),
                    "model": "gemini",
                    "dataset": f"{job.pkl_name}/{job.system_name}",
                    "QA": build_litransproqa_prompt(source_text, hypothesis_text, base_dir),
                }
            )
    return rows


def nonempty_csv_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and value != value) and str(value).strip() != ""


def merge_existing_litransproqa_results(df: Any, results_path: Path) -> Any:
    if not results_path.exists():
        return df

    import pandas as pd

    previous = pd.read_csv(results_path)
    if previous.empty or "job_index" not in previous.columns or "segment_index" not in previous.columns:
        return df

    previous_by_key = {
        (int(row["job_index"]), int(row["segment_index"])): row
        for _, row in previous.iterrows()
    }
    for idx, row in df.iterrows():
        previous_row = previous_by_key.get((int(row["job_index"]), int(row["segment_index"])))
        if previous_row is None:
            continue
        for column in ("response", "error"):
            if column in previous_row and nonempty_csv_value(previous_row[column]):
                df.at[idx, column] = previous_row[column]
    return df


def request_litransproqa_response(model: Any, prompt: str, args: argparse.Namespace) -> tuple[str | None, str | None]:
    max_attempts = max(1, args.llm_judge_retries + 1)
    delay = max(0.0, args.llm_judge_retry_delay)
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return model.direct_message(message=prompt), None
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max_attempts:
                break
            sleep_seconds = delay * (2**attempt)
            logging.warning(
                "%s request failed on attempt %d/%d; retrying in %.1fs: %s",
                LITRANSPROQA_METRIC,
                attempt + 1,
                max_attempts,
                sleep_seconds,
                exc,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return None, str(last_error) if last_error is not None else "unknown judge request failure"


def run_litransproqa_judge(
    jobs: list[EvaluationJob],
    args: argparse.Namespace,
    job_indexes: set[int] | None = None,
) -> None:
    try:
        import pandas as pd
        from api_model import Gemini
    except Exception as exc:
        raise RuntimeError("LiTransProQA judge dependencies are unavailable") from exc

    base_dir = Path("LiTransProQA/prompting_method")
    question_weights = load_litransproqa_weights()
    rows = build_litransproqa_rows(jobs, base_dir, job_indexes)
    if not rows:
        logging.info("skipping %s; no eligible segments", LITRANSPROQA_METRIC)
        return

    work_dir = analysis_results_dir(args.out) / "litransproqa"
    prompts_path = work_dir / "judge_prompts.csv"
    results_dir = work_dir / "final_results"
    results_path = results_dir / prompts_path.name
    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["response"] = ""
    df["error"] = ""
    df.to_csv(prompts_path, index=False)
    df = merge_existing_litransproqa_results(df, results_path)
    df.to_csv(results_path, index=False)

    logging.info("running %s with Gemini on %d segments", LITRANSPROQA_METRIC, len(rows))
    model = Gemini(model_checkpoint=args.llm_judge_model)
    for idx, row in df.iterrows():
        if nonempty_csv_value(row.get("response")):
            continue
        response, error = request_litransproqa_response(model, str(row["QA"]), args)
        if response is not None:
            df.at[idx, "response"] = response
            df.at[idx, "error"] = ""
        else:
            df.at[idx, "error"] = error or "unknown judge request failure"
        df.to_csv(results_path, index=False)

    for _, row in df.iterrows():
        job_index = int(row["job_index"])
        segment_index = int(row["segment_index"])
        if not nonempty_csv_value(row.get("response")):
            jobs[job_index].skipped.append(
                {
                    "metric": LITRANSPROQA_METRIC,
                    "segment_index": segment_index,
                    "reason": "judge_request_failed",
                    "error": "" if not nonempty_csv_value(row.get("error")) else str(row["error"]),
                }
            )
            continue
        try:
            score = weighted_litransproqa_score(row["response"], question_weights)
        except Exception as exc:
            jobs[job_index].skipped.append(
                {
                    "metric": LITRANSPROQA_METRIC,
                    "segment_index": segment_index,
                    "reason": "unparseable_judge_response",
                    "error": str(exc),
                }
            )
            logging.warning(
                "%s/%s: could not parse %s response for segment %d: %s",
                jobs[job_index].pkl_name,
                jobs[job_index].system_name,
                LITRANSPROQA_METRIC,
                segment_index,
                exc,
            )
            continue
        if score is None:
            jobs[job_index].skipped.append(
                {
                    "metric": LITRANSPROQA_METRIC,
                    "segment_index": segment_index,
                    "reason": "empty_judge_score",
                }
            )
            continue
        jobs[job_index].scores.setdefault(LITRANSPROQA_METRIC, {})[segment_index] = float(score)

    logging.info("%s raw responses saved to %s", LITRANSPROQA_METRIC, results_path)


def mean_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    return statistics.fmean(vals) if vals else None


def load_existing_outputs_for_job(out_dir: Path, job: EvaluationJob) -> None:
    system_dir = book_results_dir(out_dir) / job.pkl_name / job.system_name
    segment_scores_path = system_dir / "segment_scores.jsonl"
    system_scores_path = system_dir / "system_scores.json"
    skipped_path = system_dir / "skipped_segments.jsonl"

    if segment_scores_path.exists():
        with segment_scores_path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in existing result file {segment_scores_path}:{line_number}"
                    ) from exc

                segment_index = int(row["segment_index"])
                scores = row.get("scores", {})
                if not isinstance(scores, dict):
                    raise ValueError(f"{segment_scores_path}:{line_number}: scores must be an object")
                for metric, score in scores.items():
                    job.scores.setdefault(str(metric), {})[segment_index] = float(score)

                diagnostics = row.get("alignment_diagnostics")
                if isinstance(diagnostics, dict):
                    job.alignment_diagnostics[segment_index] = diagnostics

    if system_scores_path.exists():
        try:
            system_scores = json.loads(system_scores_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in existing result file {system_scores_path}") from exc
        metrics = system_scores.get("metrics", {})
        if isinstance(metrics, dict):
            for metric in metrics:
                job.scores.setdefault(str(metric), {})

    if skipped_path.exists():
        with skipped_path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in existing result file {skipped_path}:{line_number}"
                    ) from exc
                if isinstance(row, dict):
                    job.skipped.append(row)


def load_existing_outputs(jobs: list[EvaluationJob], out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for job in jobs:
        load_existing_outputs_for_job(out_dir, job)


def write_outputs(
    out_dir: Path,
    pkl_name: str,
    system_name: str,
    sources: list[str],
    hyps: list[str],
    ref_name: str | None,
    scores: dict[str, dict[int, float]],
    skipped: list[dict[str, Any]],
    alignment_diagnostics: dict[int, dict[str, Any]] | None = None,
) -> None:
    system_dir = book_results_dir(out_dir) / pkl_name / system_name
    system_dir.mkdir(parents=True, exist_ok=True)
    alignment_diagnostics = alignment_diagnostics or {}

    with (system_dir / "segment_scores.jsonl").open("w", encoding="utf-8") as f:
        for i, (source, hyp) in enumerate(zip(sources, hyps)):
            row: dict[str, Any] = {
                "segment_index": i,
                "source": source,
                "hypothesis": hyp,
                "scores": {},
            }
            if i in alignment_diagnostics:
                row["alignment_diagnostics"] = alignment_diagnostics[i]
            for metric, metric_scores in sorted(scores.items()):
                if i in metric_scores:
                    row["scores"][metric] = metric_scores[i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    system_scores = {
        "pkl": pkl_name,
        "system": system_name,
        "reference_system": ref_name,
        "metrics": {
            metric: {
                "score": mean_or_none(metric_scores.values()),
                "num_scored_segments": len(metric_scores),
                "higher_is_better": metric in HIGHER_IS_BETTER_METRICS,
            }
            for metric, metric_scores in sorted(scores.items())
        },
    }
    (system_dir / "system_scores.json").write_text(
        json.dumps(system_scores, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (system_dir / "skipped_segments.jsonl").open("w", encoding="utf-8") as f:
        for row in skipped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_all_outputs(jobs: list[EvaluationJob], args: argparse.Namespace) -> None:
    for job in jobs:
        write_outputs(
            args.out,
            job.pkl_name,
            job.system_name,
            job.sources,
            job.hyps,
            job.ref_name,
            job.scores,
            job.skipped,
            job.alignment_diagnostics,
        )


def collect_metric_segments(
    metric: str,
    jobs: list[EvaluationJob],
    tokenizer: Any | None,
    max_input_length: int,
    alignment_diagnostics: AlignmentDiagnosticsMap,
    job_indexes: set[int] | None = None,
) -> list[Segment]:
    kept: list[Segment] = []
    for job_index, job in enumerate(jobs):
        if job_indexes is not None and job_index not in job_indexes:
            continue
        if metric in REFERENCE_METRICS and job.refs is None:
            raise ValueError(
                f"{job.pkl_name}/{job.system_name}: {metric} requires a *_ht reference, but none was found"
            )
        if metric in REFERENCE_METRICS and job.system_name == job.ref_name:
            logging.info(
                "%s/%s: not evaluating reference system with %s",
                job.pkl_name,
                job.system_name,
                metric,
            )
            continue

        segments: list[Segment] = []
        job_alignment_diagnostics = alignment_diagnostics.get((job.pkl_name, job.system_name), {})
        job.alignment_diagnostics.update(job_alignment_diagnostics)
        for i, (src, hyp) in enumerate(zip(job.sources, job.hyps)):
            segment = Segment(
                job_index,
                i,
                src,
                hyp,
                job.refs[i] if job.refs is not None and metric in REFERENCE_METRICS else None,
            )
            normalized_segment, validation_skip = normalize_segment_for_metric(metric, segment)
            if validation_skip is not None:
                if metric in REFERENCE_METRICS and "reference" in validation_skip["empty_fields"]:
                    job.skipped.append({**validation_skip, "reason": "empty_reference"})
                    logging.warning(
                        "%s/%s: skipping %s segment %d; empty reference from %s",
                        job.pkl_name,
                        job.system_name,
                        metric,
                        i,
                        job.ref_name,
                    )
                    continue
                if "hypothesis" in validation_skip["empty_fields"]:
                    job.skipped.append(validation_skip)
                    logging.warning(
                        "%s/%s: skipping %s segment %d; empty hypothesis",
                        job.pkl_name,
                        job.system_name,
                        metric,
                        i,
                    )
                    continue
                raise ValueError(
                    f"{job.pkl_name}/{job.system_name}: invalid {metric} segment {i}: "
                    f"{validation_skip['reason']} ({', '.join(validation_skip['empty_fields'])})"
                )
            if normalized_segment is not None:
                segments.append(normalized_segment)

        job_kept, length_skipped = validate_segments(
            metric,
            segments,
            tokenizer,
            max_input_length,
            job.pkl_name,
            job.system_name,
        )
        kept.extend(job_kept)
        job.skipped.extend(length_skipped)
    return kept


def apply_metric_scores(
    metric: str,
    jobs: list[EvaluationJob],
    scores: dict[tuple[int, int], float],
) -> None:
    for (job_index, segment_index), score in scores.items():
        jobs[job_index].scores.setdefault(metric, {})[segment_index] = score


def evaluate_batch(
    jobs: list[EvaluationJob],
    metrics: list[str],
    tokenizers: dict[str, Any | None],
    args: argparse.Namespace,
    alignment_diagnostics: AlignmentDiagnosticsMap,
) -> None:
    wrote_outputs = False
    for metric in metrics:
        pending_job_indexes = {
            job_index for job_index, job in enumerate(jobs) if metric not in job.scores
        }
        if not pending_job_indexes:
            logging.info("skipping %s; existing output already contains this metric", metric)
            continue

        if metric in {"metricx", "metricx-qe"}:
            max_input_length = args.metricx_max_input_length
            tokenizer = tokenizers.get("metricx")
        else:
            max_input_length = args.comet_max_input_length
            tokenizer = tokenizers.get(metric)
        segments = collect_metric_segments(
            metric,
            jobs,
            tokenizer,
            max_input_length,
            alignment_diagnostics,
            pending_job_indexes,
        )
        if not segments:
            logging.info("skipping %s; no eligible segments in batch", metric)
            continue

        if metric in COMET_MODELS:
            scores = run_comet(metric, segments, args, args.out)
        elif metric in {"metricx", "metricx-qe"}:
            scores = run_metricx(metric, segments, args, analysis_results_dir(args.out))
        else:
            raise ValueError(f"unsupported metric: {metric}")
        apply_metric_scores(metric, jobs, scores)
        wrote_outputs = True

    if wrote_outputs:
        write_all_outputs(jobs, args)

    if args.llm_as_a_judge:
        pending_litransproqa_job_indexes = {
            job_index
            for job_index, job in enumerate(jobs)
            if LITRANSPROQA_METRIC not in job.scores
        }
        if not pending_litransproqa_job_indexes:
            logging.info("skipping %s; existing output already contains this metric", LITRANSPROQA_METRIC)
            if wrote_outputs:
                write_all_outputs(jobs, args)
            return
        try:
            run_litransproqa_judge(jobs, args, pending_litransproqa_job_indexes)
        finally:
            write_all_outputs(jobs, args)


def parse_metrics(metrics_arg: str) -> list[str]:
    metrics = [m.strip().lower() for m in metrics_arg.split(",") if m.strip()]
    unknown = sorted(set(metrics) - SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"unsupported metrics: {', '.join(unknown)}")
    return metrics


def find_input_files(data_dir: Path) -> list[Path]:
    direct_pickles = list(data_dir.glob("*.pkl"))
    nested_pickles = list(data_dir.glob("*/*/*.pkl"))
    direct_jsonl = list(data_dir.glob("*.jsonl"))
    return sorted(set(direct_pickles + nested_pickles + direct_jsonl))


def output_dataset_name(pkl_file: Path, data_dir: Path) -> str:
    try:
        relative = pkl_file.relative_to(data_dir)
    except ValueError:
        return pkl_file.stem
    if relative.parent == Path("."):
        return pkl_file.stem
    if relative.parent.name == pkl_file.stem:
        collapsed = relative.parent.parent / pkl_file.stem
        return pkl_file.stem if collapsed.parent == Path(".") else collapsed.as_posix()
    return relative.with_suffix("").as_posix()


def parse_segment_index(row: dict[str, str], csv_path: Path, row_number: int) -> int:
    raw_index = row.get("segment_index") or row.get("paragraph_id")
    if raw_index is None or raw_index == "":
        raise ValueError(f"{csv_path}:{row_number}: missing segment_index/paragraph_id")
    try:
        return int(raw_index)
    except ValueError as exc:
        raise ValueError(f"{csv_path}:{row_number}: invalid segment index {raw_index!r}") from exc


def pkl_names_for_alignment(row: dict[str, str], data_dir: Path) -> set[str]:
    names: set[str] = set()
    pkl_path = row.get("pkl_path", "")
    if pkl_path:
        path = Path(pkl_path)
        names.add(output_dataset_name(path, data_dir))
        names.add(output_dataset_name(path.resolve(), data_dir.resolve()))

    book = row.get("book", "")
    if book:
        names.add(book)
        names.add(f"{book}/{book}")
        for split in ("dev", "test", "train"):
            names.add(f"{split}/{book}/{book}")

    return {name for name in names if name}


def load_alignment_csv(csv_path: Path | None, data_dir: Path) -> AlignmentDiagnosticsMap:
    if csv_path is None:
        return {}
    if not csv_path.exists():
        raise FileNotFoundError(f"Alignment diagnostics CSV does not exist: {csv_path}")

    diagnostics: AlignmentDiagnosticsMap = {}
    source_rows = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_any = {"segment_index", "paragraph_id"}
        missing = {"pkl_path", "pipeline"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(sorted(missing))}")
        if not required_any & set(reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain segment_index or paragraph_id")

        for row_number, row in enumerate(reader, start=2):
            issue_type = row.get("issue_type", "")
            if issue_type and issue_type not in {"length", "bleu", "both"}:
                continue
            source_rows += 1

            system_name = clean_system_name(row.get("system_name") or row.get("pipeline") or "")
            if not system_name:
                raise ValueError(f"{csv_path}:{row_number}: missing pipeline/system_name")

            segment_index = parse_segment_index(row, csv_path, row_number)
            diagnostic = {
                "segment_index": segment_index,
                "alignment_issue_type": issue_type,
                "alignment_report_pkl_path": row.get("pkl_path", ""),
                "alignment_report_pipeline": row.get("pipeline", ""),
            }
            for field in (
                "gt_length",
                "compared_length",
                "ratio",
                "bleu_score",
                "low_bleu_repetitions",
                "total_bleu_comparisons",
                "bleu_scores",
                "message",
            ):
                if row.get(field):
                    diagnostic[field] = row[field]

            for pkl_name in pkl_names_for_alignment(row, data_dir):
                diagnostics.setdefault((pkl_name, system_name), {})[segment_index] = diagnostic

    logging.info("loaded %d alignment diagnostic rows from %s", source_rows, csv_path)
    return diagnostics


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    metrics = parse_metrics(args.metrics)
    if args.overwrite and args.out.exists():
        logging.info("removing existing output directory %s", args.out)
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizers = load_metric_tokenizers(args, metrics)
    alignment_diagnostics = load_alignment_csv(args.alignment_diagnostics_csv, args.data_dir)
    input_files = find_input_files(args.data_dir)
    if not input_files:
        raise SystemExit(f"No .pkl or .jsonl files found in {args.data_dir} or {args.data_dir}/*/*")

    jobs: list[EvaluationJob] = []
    for input_file in input_files:
        logging.info("loading %s", input_file)
        if input_file.suffix.lower() == ".pkl":
            loaded = load_pickle_input(input_file, args.data_dir)
        elif input_file.suffix.lower() == ".jsonl":
            loaded = load_jsonl_input(input_file)
        else:
            logging.info("skipping unsupported input file %s", input_file)
            continue

        if loaded.ref_name:
            logging.info("%s: using %s as reference", input_file.name, loaded.ref_name)
        elif any(metric in REFERENCE_METRICS for metric in metrics):
            raise ValueError(
                f"{input_file.name}: reference-based metrics requested, but no *_ht reference was found"
            )

        for system_name, hyps in sorted(loaded.systems.items()):
            logging.info("%s: queued %s for batch evaluation", input_file.name, system_name)
            jobs.append(
                EvaluationJob(
                    pkl_name=loaded.dataset_name,
                    system_name=system_name,
                    sources=loaded.sources,
                    hyps=hyps,
                    ref_name=loaded.ref_name,
                    refs=loaded.refs,
                )
            )

    if not jobs:
        raise SystemExit("No systems with in-pickle translator_paras found to evaluate")

    if not args.overwrite:
        load_existing_outputs(jobs, args.out)

    logging.info("evaluating %d systems in batch mode", len(jobs))
    evaluate_batch(jobs, metrics, tokenizers, args, alignment_diagnostics)


if __name__ == "__main__":
    main()
