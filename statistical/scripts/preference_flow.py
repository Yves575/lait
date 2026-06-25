"""Plot Part 1 final preference to Part 2 chunk preference as an SVG alluvial chart."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


CHOICES = ("HT", "MT")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_flows(
    part1_rows: Iterable[dict[str, str]],
    part2_rows: Iterable[dict[str, str]],
    weighting: str,
) -> tuple[Counter[tuple[str, str]], Counter[str]]:
    chunks_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in part2_rows:
        if row.get("preferred_translation") in CHOICES:
            chunks_by_key[(row["participant_id"], row["book_id"])].append(row)

    flows: Counter[tuple[str, str]] = Counter()
    skipped: Counter[str] = Counter()
    for row in part1_rows:
        source = row.get("comparison_q1_decipher", "")
        if source not in CHOICES:
            skipped[source or "<blank_part1_preference>"] += 1
            continue

        chunks = chunks_by_key.get((row["participant_id"], row["book_id"]), [])
        if not chunks:
            skipped["no_part2_chunks"] += 1
            continue

        weight = 1.0 / len(chunks) if weighting == "fractional" else 1.0
        for chunk in chunks:
            flows[(source, chunk["preferred_translation"])] += weight

    return flows, skipped


def node_totals(flows: Counter[tuple[str, str]]) -> tuple[Counter[str], Counter[str]]:
    left: Counter[str] = Counter()
    right: Counter[str] = Counter()
    for (source, target), value in flows.items():
        left[source] += value
        right[target] += value
    return left, right


def fmt_count(value: float) -> str:
    if abs(value - round(value)) < 0.005:
        return str(int(round(value)))
    return f"{value:.1f}"


def path_between(
    x0: float,
    x1: float,
    y0a: float,
    y0b: float,
    y1a: float,
    y1b: float,
) -> str:
    curve = (x1 - x0) * 0.55
    return (
        f"M {x0:.2f},{y0a:.2f} "
        f"C {x0 + curve:.2f},{y0a:.2f} {x1 - curve:.2f},{y1a:.2f} {x1:.2f},{y1a:.2f} "
        f"L {x1:.2f},{y1b:.2f} "
        f"C {x1 - curve:.2f},{y1b:.2f} {x0 + curve:.2f},{y0b:.2f} {x0:.2f},{y0b:.2f} Z"
    )


def render_svg(
    flows: Counter[tuple[str, str]],
    output_path: Path,
    title: str,
    weighting: str,
) -> None:
    width, height = 900, 520
    top, bottom = 78, 58
    chart_height = height - top - bottom
    gap = 28
    left_x, right_x = 32, width - 32
    bar_width = 32
    flow_left_x = left_x + bar_width
    flow_right_x = right_x - bar_width
    label_pad = 18

    left_totals, right_totals = node_totals(flows)
    total = sum(left_totals.values())
    if total <= 0:
        raise ValueError("No HT/MT flows found after filtering.")

    scale = (chart_height - gap) / total
    colors = {"HT": "#9fb5ee", "MT": "#f7b88f"}
    bar_colors = {"HT": "#3b82f6", "MT": "#f97316"}
    text_color = "#26364f"
    muted_text = "#66738a"

    def positions(totals: Counter[str]) -> dict[str, tuple[float, float]]:
        y = top
        result: dict[str, tuple[float, float]] = {}
        for choice in CHOICES:
            node_height = totals[choice] * scale
            result[choice] = (y, y + node_height)
            y += node_height + gap
        return result

    left_pos = positions(left_totals)
    right_pos = positions(right_totals)
    left_offsets = {choice: left_pos[choice][0] for choice in CHOICES}
    right_offsets = {choice: right_pos[choice][0] for choice in CHOICES}

    # Draw target groups in target order so same-target flows visually merge on the right.
    flow_order = [("HT", "HT"), ("MT", "HT"), ("HT", "MT"), ("MT", "MT")]
    flow_paths: list[str] = []
    for source, target in flow_order:
        value = flows[(source, target)]
        if value <= 0:
            continue
        thickness = value * scale
        y0a = left_offsets[source]
        y0b = y0a + thickness
        y1a = right_offsets[target]
        y1b = y1a + thickness
        left_offsets[source] = y0b
        right_offsets[target] = y1b
        flow_paths.append(
            f'<path d="{path_between(flow_left_x, flow_right_x, y0a, y0b, y1a, y1b)}" '
            f'fill="{colors[target]}" fill-opacity="0.72"/>'
        )

    node_rects: list[str] = []
    node_labels: list[str] = []
    for side, x, totals, pos in [
        ("left", left_x, left_totals, left_pos),
        ("right", right_x - bar_width, right_totals, right_pos),
    ]:
        for choice in CHOICES:
            y0, y1 = pos[choice]
            node_rects.append(
                f'<rect x="{x}" y="{y0:.2f}" width="{bar_width}" height="{y1 - y0:.2f}" '
                f'rx="7" fill="{bar_colors[choice]}"/>'
            )
            label_x = x + bar_width + label_pad if side == "left" else x - label_pad
            anchor = "start" if side == "left" else "end"
            node_labels.append(
                f'<text x="{label_x}" y="{y0 + 36:.2f}" text-anchor="{anchor}" '
                f'font-size="26" font-weight="700" fill="{text_color}">{choice}</text>'
            )
            node_labels.append(
                f'<text x="{label_x}" y="{y0 + 68:.2f}" text-anchor="{anchor}" '
                f'font-size="24" fill="{muted_text}">{fmt_count(totals[choice])}</text>'
            )

    subtitle = (
        "Fractional weighting: each participant/book contributes 1 total vote across chunks"
        if weighting == "fractional"
        else "Chunk weighting: each Part 2 chunk contributes 1 vote"
    )
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" '
            f'font-weight="700" fill="{text_color}">{escape(title)}</text>',
            f'<text x="{width / 2}" y="56" text-anchor="middle" font-size="13" '
            f'fill="{muted_text}">{escape(subtitle)}</text>',
            f'<text x="{left_x}" y="{height - 20}" font-size="14" fill="{muted_text}">'
            "Part 1 final comparison</text>",
            f'<text x="{right_x}" y="{height - 20}" text-anchor="end" font-size="14" '
            f'fill="{muted_text}">Part 2 chunk preference</text>',
            *flow_paths,
            *node_rects,
            *node_labels,
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part1",
        type=Path,
        default=Path("human_eval/data/part1-study-data-full.csv"),
        help="Part 1 full CSV export.",
    )
    parser.add_argument(
        "--part2",
        type=Path,
        default=Path("human_eval/data/part2-study-data-full.csv"),
        help="Part 2 full CSV export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("statistical/outputs/plots/part1_part2_preference_flow.svg"),
        help="Output SVG path.",
    )
    parser.add_argument(
        "--weighting",
        choices=("fractional", "chunk"),
        default="fractional",
        help="How to weight Part 2 chunks relative to each Part 1 participant/book row.",
    )
    args = parser.parse_args()

    flows, skipped = collect_flows(read_csv(args.part1), read_csv(args.part2), args.weighting)
    render_svg(flows, args.output, "Final Preference vs Chunk Preference", args.weighting)

    print(f"Wrote {args.output}")
    print("Flows:")
    for source in CHOICES:
        for target in CHOICES:
            print(f"  {source} -> {target}: {flows[(source, target)]:.3f}")
    if skipped:
        print("Skipped Part 1 rows:")
        for reason, count in skipped.items():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
