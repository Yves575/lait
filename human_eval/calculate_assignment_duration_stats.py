#!/usr/bin/env python3
"""Calculate active-time statistics matching the mt-book-reader admin UI.

Replicates `/api/admin/participants` in `mt-book-reader/server.js` and the
"Active time" column in `mt-book-reader/public/app.js` (Progress table):

  totalActiveMs = sum over all reading sessions for the participant (user_id)
  of estimateSessionActiveMs(session), then round(totalActiveMs / 60000, 1).

Per-session estimate order (same as server.js):
  1. session.active_ms if truthy
  2. else sum of chunk_exit dwellMs from reading_events
  3. else max(0, (ended_at - started_at) - inactive_ms)

`study-data-full.json` does not include reading_events. Pair it with
`human_eval/data/events.csv` (admin export) or `--db` on production study.db.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "study-data-full.json"
DEFAULT_EVENTS_CSV = Path(__file__).resolve().parent / "data" / "events.csv"
DEFAULT_EXCLUDED_PARTICIPANT_PREFIXES: tuple[str, ...] = ()
MS_PER_MINUTE = 60_000


def parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if "T" in text:
        text = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    else:
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def load_dwell_by_session_from_csv(events_path: Path) -> dict[str, int]:
    """Load per-session dwell sums from admin events.csv (chunk_exit payloads)."""
    totals: dict[str, int] = {}
    with events_path.open("r", encoding="utf-8", newline="") as infile:
        for row in csv.DictReader(infile):
            if row.get("event_type") != "chunk_exit":
                continue
            session_id = str(row.get("session_id") or "")
            if not session_id:
                continue
            try:
                payload = json.loads(row.get("payload_json") or "{}")
            except json.JSONDecodeError:
                continue
            dwell_ms = int(payload.get("dwellMs") or 0)
            if dwell_ms:
                totals[session_id] = totals.get(session_id, 0) + dwell_ms
    return totals


def load_dwell_by_session_from_db(db_path: Path) -> dict[str, int]:
    """Load per-session dwell sums from reading_events in study.db."""
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT session_id,
                   SUM(CAST(json_extract(payload_json, '$.dwellMs') AS INTEGER)) AS dwell_sum
            FROM reading_events
            WHERE event_type = 'chunk_exit'
            GROUP BY session_id
            """
        ).fetchall()
    finally:
        connection.close()
    return {str(session_id): int(dwell_sum or 0) for session_id, dwell_sum in rows}


def estimate_session_active_ms(
    session: dict,
    dwell_by_session: dict[str, int],
) -> int:
    """Mirror estimateSessionActiveMs / admin participants reduce in server.js."""
    active_ms = session.get("active_ms") or 0
    if active_ms:
        return int(active_ms)

    session_id = str(session.get("id") or session.get("session_id") or "")
    dwell_ms = dwell_by_session.get(session_id, 0)
    if dwell_ms:
        return dwell_ms

    started_at = parse_timestamp(session.get("started_at"))
    ended_at = parse_timestamp(session.get("ended_at"))
    if started_at and ended_at:
        elapsed_ms = int((ended_at - started_at).total_seconds() * 1000)
        inactive_ms = int(session.get("inactive_ms") or 0)
        return max(0, elapsed_ms - inactive_ms)
    return 0


def total_active_ms_for_user(
    sessions: list[dict],
    dwell_by_session: dict[str, int],
) -> int:
    return sum(
        estimate_session_active_ms(session, dwell_by_session) for session in sessions
    )


def active_minutes_display(total_active_ms: int) -> float:
    """Match app.js: (totalActiveMs / 60000).toFixed(1) as a float."""
    return round(total_active_ms / MS_PER_MINUTE, 1)


def is_excluded(participant_id: object, excluded_prefixes: tuple[str, ...]) -> bool:
    if not excluded_prefixes:
        return False
    participant_id_text = str(participant_id or "")
    return any(
        participant_id_text.startswith(prefix) for prefix in excluded_prefixes
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return data


def index_sessions_by_user(data: dict) -> dict[str, list[dict]]:
    by_user: dict[str, list[dict]] = {}
    for session in data.get("sessions", []):
        if not isinstance(session, dict):
            continue
        user_id = session.get("user_id")
        if not user_id:
            continue
        by_user.setdefault(str(user_id), []).append(session)
    return by_user


def index_assignments_by_user(data: dict) -> dict[str, dict]:
    by_user: dict[str, dict] = {}
    for assignment in data.get("assignments", []):
        if not isinstance(assignment, dict):
            continue
        user_id = assignment.get("user_id")
        if user_id:
            by_user[str(user_id)] = assignment
    return by_user


def participant_active_minutes(
    data: dict,
    dwell_by_session: dict[str, int],
    excluded_prefixes: tuple[str, ...],
) -> list[dict[str, object]]:
    sessions_by_user = index_sessions_by_user(data)
    assignments_by_user = index_assignments_by_user(data)
    rows: list[dict[str, object]] = []

    for participant in data.get("participants", []):
        if not isinstance(participant, dict):
            continue
        user_id = str(participant.get("id", ""))
        if not user_id:
            continue
        if is_excluded(participant.get("participant_id"), excluded_prefixes):
            continue

        assignment = assignments_by_user.get(user_id, {})
        total_ms = total_active_ms_for_user(
            sessions_by_user.get(user_id, []),
            dwell_by_session,
        )
        rows.append(
            {
                "username": str(participant.get("username") or ""),
                "participant_id": str(participant.get("participant_id") or ""),
                "book_id": str(assignment.get("book_id") or ""),
                "first_version": str(assignment.get("first_version") or ""),
                "second_version": str(assignment.get("second_version") or ""),
                "active_minutes": active_minutes_display(total_ms),
                "total_active_ms": total_ms,
                "session_count": len(sessions_by_user.get(user_id, [])),
            }
        )

    return sorted(rows, key=lambda row: str(row["username"]))


def summarize_durations(durations: list[float]) -> dict[str, float | int]:
    n = len(durations)
    if n == 0:
        return {
            "n": 0,
            "mean_minutes": float("nan"),
            "median_minutes": float("nan"),
            "std_minutes": float("nan"),
            "min_minutes": float("nan"),
            "max_minutes": float("nan"),
        }

    mean_minutes = statistics.mean(durations)
    median_minutes = statistics.median(durations)
    std_minutes = statistics.stdev(durations) if n >= 2 else float("nan")
    return {
        "n": n,
        "mean_minutes": mean_minutes,
        "median_minutes": median_minutes,
        "std_minutes": std_minutes,
        "min_minutes": min(durations),
        "max_minutes": max(durations),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate active-time statistics using the same rules as the "
            "mt-book-reader admin Participant Progress table."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"study-data-full.json path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=None,
        help=(
            "Admin events.csv export for chunk_exit dwell fallback. "
            f"Defaults to {DEFAULT_EVENTS_CSV} when that file exists."
        ),
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Do not load dwell times from events.csv.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite study.db for reading_events dwell fallback (alternative to --events-csv).",
    )
    parser.add_argument(
        "--per-participant",
        action="store_true",
        help="Print each participant's active time (admin table values).",
    )
    parser.add_argument(
        "--exclude-participant-prefix",
        action="append",
        default=None,
        help=(
            "Participant ID prefix to exclude (repeatable). "
            "Default: include all participants."
        ),
    )
    return parser.parse_args()


def resolve_excluded_prefixes(cli_value: list[str] | None) -> tuple[str, ...]:
    if cli_value is None:
        return DEFAULT_EXCLUDED_PARTICIPANT_PREFIXES
    return tuple(prefix for prefix in cli_value if prefix)


def resolve_events_csv_path(args: argparse.Namespace) -> Path | None:
    if args.no_events:
        return None
    if args.events_csv is not None:
        return args.events_csv
    if DEFAULT_EVENTS_CSV.is_file():
        return DEFAULT_EVENTS_CSV
    return None


def load_dwell_by_session(args: argparse.Namespace) -> dict[str, int]:
    events_path = resolve_events_csv_path(args)
    if events_path is not None:
        if not events_path.is_file():
            print(f"Error: {events_path} is not a file", file=sys.stderr)
            raise SystemExit(1)
        dwell_by_session = load_dwell_by_session_from_csv(events_path)
        print(
            f"Loaded dwell sums for {len(dwell_by_session)} sessions from {events_path}"
        )
        return dwell_by_session

    if args.db is not None:
        if not args.db.is_file():
            print(f"Error: {args.db} is not a file", file=sys.stderr)
            raise SystemExit(1)
        dwell_by_session = load_dwell_by_session_from_db(args.db)
        print(f"Loaded dwell sums for {len(dwell_by_session)} sessions from {args.db}")
        return dwell_by_session

    print(
        "Warning: no events.csv or --db; sessions with active_ms=0 will not use "
        "chunk_exit dwell fallback (may undercount vs admin UI).",
        file=sys.stderr,
    )
    return {}


def main() -> None:
    args = parse_args()
    excluded_prefixes = resolve_excluded_prefixes(args.exclude_participant_prefix)

    if not args.input.is_file():
        print(f"Error: {args.input} is not a file", file=sys.stderr)
        raise SystemExit(1)

    dwell_by_session = load_dwell_by_session(args)

    data = load_json(args.input)
    rows = participant_active_minutes(data, dwell_by_session, excluded_prefixes)
    durations = [float(row["active_minutes"]) for row in rows]
    summary = summarize_durations(durations)

    if args.per_participant:
        print()
        print(f"{'Participant':<14} {'Book':<45} {'Order':<10} {'Active min':>10}")
        for row in rows:
            order = (
                f"{row['first_version']} -> {row['second_version']}"
                if row["first_version"] and row["second_version"]
                else "-"
            )
            book = row["book_id"] or "-"
            print(
                f"{row['username']:<14} {book:<45} {order:<10} "
                f"{row['active_minutes']:>10.1f}"
            )
        print()

    print(f"Participants included: {summary['n']}")
    print("Active time unit: minutes (admin UI: sum of estimated session active_ms)")
    print(f"Mean:   {summary['mean_minutes']:.2f}")
    print(f"Median: {summary['median_minutes']:.2f}")
    print(f"Std:    {summary['std_minutes']:.2f}")
    print(f"Min:    {summary['min_minutes']:.2f}")
    print(f"Max:    {summary['max_minutes']:.2f}")


if __name__ == "__main__":
    main()
