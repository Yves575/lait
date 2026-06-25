"""Canonical qualitative labels for human-eval analysis plots.

The Google Sheet schema collapsed several labels and renumbered later labels.
Raw coding tabs may still contain old labels, so canonicalization must inspect
the label text as well as the code. For example, old ``A9. Cadence`` maps to
new ``C2``, while new ``A9. Consistency`` stays in A.
"""

from __future__ import annotations

import re


CODE_RE = re.compile(r"^([A-Z][0-9]+[a-z]?)\.")

CANONICAL_LABELS = {
    "A1": "A1. Grammar & spelling",
    "A2": "A2. Word choice: clarity",
    "A3": "A3. Word choice: naturalness",
    "A4": "A4. Word choice: idiomaticity",
    "A5": "A5. Word choice: register & contextual fit",
    "A6": "A6. Word choice: richness vs. blandness",
    "A7": "A7. Cultural adaptation",
    "A8": "A8. Sentence structure",
    "A9": "A9. Consistency",
    "A10": "A10. Formatting & typography",
    "A11": "A11. Repetition or redundancy",
    "A12": "A12. Untranslated content",
    "B1": "B1. Dialogue: naturalness",
    "B2": "B2. Character voice & portrayal",
    "B3": "B3. Description / vividness / visualizability",
    "B4": "B4. Figurative language",
    "B5": "B5. Emotional conveyance",
    "B6": "B6. Narrative organization",
    "B7": "B7. Narration & POV effects",
    "B8": "B8. Pacing / information delivery / wordiness",
    "C1": "C1. Comprehension / ease of following",
    "C2": "C2. Smoothness / reading effort",
    "C3": "C3. Engagement / immersion",
    "C4": "C4. Humanness / soul",
    "C5": "C5. Enjoyment / overall positive affect",
    "D1": "D1. Doesn't read as translated",
    "D2": "D2. Adaptation vs. literalness",
    "D3": "D3. Faithfulness to original",
    "D4": "D4. MT/AI tell named",
}

COLLAPSED_LABEL_GROUPS = [
    ("A1", CANONICAL_LABELS["A1"], ("A1", "A1b")),
    ("B2", CANONICAL_LABELS["B2"], ("B2", "B3_old")),
    ("C2", CANONICAL_LABELS["C2"], ("C2", "A9_old")),
    ("C3", CANONICAL_LABELS["C3"], ("C3", "C4_old")),
    ("D4", CANONICAL_LABELS["D4"], ("D4a", "D4b")),
]


def label_code(label: str) -> str:
    match = CODE_RE.match(label or "")
    return match.group(1) if match else ""


def label_body(label: str) -> str:
    return re.sub(r"^[A-Z][0-9]+[a-z]?\.\s*", "", label or "").strip().lower()


def collapse_label_code(label: str) -> str:
    code = label_code(label)
    body = label_body(label)

    if code == "A1b":
        return "A1"
    if code == "A9" and ("cadence" in body or "rhythm" in body or "tempo" in body):
        return "C2"
    if code == "A10" and "consistency" in body:
        return "A9"
    if code == "A11" and ("formatting" in body or "typography" in body):
        return "A10"
    if code == "A12" and ("repetition" in body or "redundancy" in body):
        return "A11"
    if code == "A13":
        return "A12"

    if code == "B3" and "character" in body:
        return "B2"
    if code == "B4" and ("description" in body or "vividness" in body or "visual" in body):
        return "B3"
    if code == "B5" and "figurative" in body:
        return "B4"
    if code == "B6" and "emotional" in body:
        return "B5"
    if code == "B7" and "narrative organization" in body:
        return "B6"
    if code == "B8" and ("narration" in body or "pov" in body):
        return "B7"
    if code == "B9":
        return "B8"

    if code == "C4" and "hook" in body:
        return "C3"
    if code == "C5" and "humanness" in body:
        return "C4"
    if code == "C6":
        return "C5"

    if code in {"D4a", "D4b"}:
        return "D4"

    return code


def collapse_code(code: str) -> str:
    code_only_map = {
        "A1b": "A1",
        "A13": "A12",
        "B9": "B8",
        "C6": "C5",
        "D4a": "D4",
        "D4b": "D4",
    }
    return code_only_map.get(code, code)


def collapse_label(label: str) -> str:
    code = collapse_label_code(label)
    return CANONICAL_LABELS.get(code, label)


def collapse_label_with_codes(code: str, label: str) -> tuple[str, tuple[str, ...]]:
    collapsed_code = collapse_label_code(label)
    canonical_label = CANONICAL_LABELS.get(collapsed_code, label)
    return canonical_label, (collapsed_code,)
