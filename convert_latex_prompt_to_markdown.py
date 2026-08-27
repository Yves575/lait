"""Convert latex_prompts/*.tex prompt tables into supplementary_materials/prompts/*.md."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "latex_prompts"
DST_DIR = ROOT / "supplementary_materials" / "prompts"
INDEX_PATH = ROOT / "supplementary_materials" / "Prompts.md"

TABLE_ENVS = ("table*", "table", "longtable")
BEGIN_ENV = re.compile(r"\\begin\{(" + "|".join(re.escape(n) for n in TABLE_ENVS) + r")\}")


def find_matching_brace(s: str, open_idx: int) -> int:
    depth = 0
    i = open_idx
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            i += 2
            continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        cut = len(line)
        i = 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                cut = i
                break
            i += 1
        out.append(line[:cut].rstrip())
    return "\n".join(out)


def uncomment_if_needed(text: str) -> str:
    live = any(
        (not line.lstrip().startswith("%")) and r"\begin{table" in line
        for line in text.splitlines()
    )
    if live:
        return strip_comments(text)
    decommented = []
    for line in text.splitlines():
        if line.startswith("% "):
            decommented.append(line[2:])
        elif line.startswith("%"):
            decommented.append(line[1:])
        else:
            decommented.append(line)
    return strip_comments("\n".join(decommented))


def iter_tables(text: str) -> list[str]:
    tables = []
    for m in BEGIN_ENV.finditer(text):
        end = f"\\end{{{m.group(1)}}}"
        j = text.find(end, m.end())
        if j < 0:
            continue
        tables.append(text[m.start() : j + len(end)])
    return tables


def cmd_arg(text: str, name: str) -> str | None:
    key = f"\\{name}{{"
    idx = text.find(key)
    if idx < 0:
        return None
    start = idx + len(key) - 1
    end = find_matching_brace(text, start)
    if end < 0:
        return None
    return text[start + 1 : end]


def replace_cmd(text: str, name: str, wrap) -> str:
    key = f"\\{name}{{"
    while True:
        idx = text.find(key)
        if idx < 0:
            return text
        start = idx + len(key) - 1
        end = find_matching_brace(text, start)
        if end < 0:
            return text
        text = text[:idx] + wrap(text[start + 1 : end]) + text[end + 1 :]


def unescape_latex(s: str) -> str:
    # Control-word: trailing whitespace is absorbed, not printed.
    s = re.sub(r"\\textbackslash[ \t]*", r"\\", s)
    s = s.replace("\\{\\{", "{")
    s = s.replace("\\}\\}", "}")
    repls = [
        ("\\,", " "),
        ("\\ ", " "),
        ("\\&", "&"),
        ("\\%", "%"),
        ("\\#", "#"),
        ("\\_", "_"),
        ("\\{", "{"),
        ("\\}", "}"),
        ("``", '"'),
        ("''", '"'),
    ]
    for a, b in repls:
        s = s.replace(a, b)
    return s


def finish_title(s: str) -> str:
    return unescape_latex(s).replace("---", "—").replace("--", "–").strip()


def extract_title(header: str) -> str:
    m = re.search(
        r"\\multicolumn\{1\}\{c\}\{\s*\\(?:bf|bfseries)\s+([^}]+)\}",
        header,
    )
    if m:
        return finish_title(m.group(1))
    m = re.search(
        r"\\centering\\bfseries\s+(.+?)(?:\\tabularnewline|\\\\)",
        header,
        re.S,
    )
    if m:
        return finish_title(m.group(1))
    m = re.search(r"\\(?:bf|bfseries)\s+(.+?)(?:\\tabularnewline|\\\\|\})", header)
    if m:
        return finish_title(m.group(1))
    return ""


def convert_lists(text: str) -> str:
    itemize = re.compile(
        r"\\begin\{itemize\}(?:\[[^\]]*\])?\s*(.*?)\\end\{itemize\}",
        re.S,
    )
    enumerate_re = re.compile(
        r"\\begin\{enumerate\}(?:\[[^\]]*\])?\s*(.*?)\\end\{enumerate\}",
        re.S,
    )

    def bullets(inner: str) -> str:
        parts = re.split(r"\\item\s*", inner)
        items = [p.strip() for p in parts[1:] if p.strip()]
        return "\n".join(f"- {it}" for it in items)

    def numbered(inner: str) -> str:
        start_n = 1
        sc = re.search(r"\\setcounter\{enumi\}\{(\d+)\}", inner)
        if sc:
            start_n = int(sc.group(1)) + 1
            inner = inner[: sc.start()] + inner[sc.end() :]
        parts = re.split(r"\\item\s*", inner)
        items = [p.strip() for p in parts[1:] if p.strip()]
        return "\n".join(f"{start_n + i}. {it}" for i, it in enumerate(items))

    while True:
        m = itemize.search(text)
        if not m:
            break
        text = text[: m.start()] + "\n" + bullets(m.group(1)) + "\n" + text[m.end() :]
    while True:
        m = enumerate_re.search(text)
        if not m:
            break
        text = text[: m.start()] + "\n" + numbered(m.group(1)) + "\n" + text[m.end() :]
    return text


def convert_body(body: str) -> str:
    body = re.sub(r"\\begin\{minipage\}(?:\[[^\]]*\])?\{[^}]*\}", "", body)
    body = body.replace("\\end{minipage}", "")
    body = replace_cmd(body, "texttt", lambda inner: inner)
    body = re.sub(r"\\vspace\*?\{[^}]*\}", "\n\n", body)
    body = re.sub(r"\\noalign\{[^}]*\}", "\n\n", body)
    body = convert_lists(body)
    body = replace_cmd(body, "textbf", lambda inner: f"**{inner}**")
    body = replace_cmd(body, "textit", lambda inner: f"*{inner}*")
    body = replace_cmd(body, "emph", lambda inner: f"*{inner}*")
    body = re.sub(r"\\(?:centering|ttfamily|raggedright|scriptsize|small|normalsize)\s*", "", body)
    body = body.replace("\\tabularnewline", "\n\n")
    body = body.replace("\\\\", "\n\n")
    body = re.sub(r"^[ \t]*&\s*", "", body, flags=re.M)
    body = re.sub(r"\\qquad[ \t]*", "    ", body)
    body = re.sub(r"\\quad[ \t]*", "  ", body)
    body = unescape_latex(body)
    body = re.sub(r"\\setcounter\{[^}]*\}\{[^}]*\}", "", body)
    return plain_in_code(wrap_tags(fence_json(tighten(collapse_ws(body)))))


def collapse_ws(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    out: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if out and not blank:
                out.append("")
            blank = True
        else:
            out.append(line)
            blank = False
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _is_tight(line: str) -> bool:
    s = line.lstrip()
    if not s:
        return False
    if s[0] in '{}"[]':
        return True
    if s[0] in "-*+" and len(s) > 1 and s[1] == " ":
        return True
    return bool(re.match(r"\d+\.\s", s))


def tighten(text: str) -> str:
    """Drop blank lines between adjacent list/JSON lines (row-break artifacts)."""
    out: list[str] = []
    for line in text.splitlines():
        if (
            line.strip()
            and out
            and out[-1] == ""
            and _is_tight(line)
            and len(out) >= 2
            and _is_tight(out[-2])
        ):
            out.pop()
        out.append(line)
    return "\n".join(out)


TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9._-]*>")


def _skip_md_code(text: str, i: int) -> int:
    """If text[i] starts a fence or inline code span, return the index after it."""
    if text.startswith("```", i):
        j = text.find("```", i + 3)
        return len(text) if j < 0 else j + 3
    if text[i] == "`":
        j = text.find("`", i + 1)
        return len(text) if j < 0 else j + 1
    return i


def fence_json(text: str) -> str:
    """Wrap JSON object examples in fenced ```json blocks. Skip {template} vars."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = _skip_md_code(text, i)
        if j != i:
            out.append(text[i:j])
            i = j
            continue
        if text[i] == "{" and text[i + 1 :].lstrip().startswith('"'):
            end = find_matching_brace(text, i)
            if end >= 0 and '":' in text[i : end + 1]:
                out.append("```json\n")
                out.append(text[i : end + 1])
                out.append("\n```")
                i = end + 1
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


_EMPH = (
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), r"\1"),
)


def _unwrap_emphasis(s: str) -> str:
    for pat, repl in _EMPH:
        s = pat.sub(repl, s)
    return s


def plain_in_code(text: str) -> str:
    """Drop **/* wrappers inside code; fences and backticks do not render them."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = _skip_md_code(text, i)
        if j != i:
            out.append(_unwrap_emphasis(text[i:j]))
            i = j
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def wrap_tags(text: str) -> str:
    """Wrap HTML-like tags in backticks so markdown does not swallow them."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = _skip_md_code(text, i)
        if j != i:
            out.append(text[i:j])
            i = j
            continue
        m = TAG_RE.match(text, i)
        if m:
            out.append(f"`{m.group(0)}`")
            i = m.end()
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def italicize(caption: str) -> str:
    if "_" in caption:
        return f"*{caption}*"
    return f"_{caption}_"


def convert_table(table: str) -> str:
    header = ""
    i = table.find("\\toprule")
    j = table.find("\\midrule")
    k = table.find("\\bottomrule")
    if i >= 0 and j >= 0:
        header = table[i + len("\\toprule") : j]
    body = table[j + len("\\midrule") : k] if j >= 0 and k >= 0 else ""
    title = extract_title(header)
    caption = cmd_arg(table, "caption") or ""
    caption = collapse_ws(
        unescape_latex(replace_cmd(caption, "textbf", lambda inner: f"**{inner}**"))
    )
    body_md = convert_body(body)
    parts = [f"# {title}"]
    if caption:
        parts += ["", italicize(caption)]
    parts += ["", "---", "", body_md]
    return "\n".join(parts).strip() + "\n"


def convert_tex(text: str) -> str:
    tables = iter_tables(uncomment_if_needed(text))
    sections = [convert_table(t) for t in tables]
    return "\n".join(s.rstrip() + "\n" for s in sections if s.strip())


def convert_file(src: Path) -> str:
    return convert_tex(src.read_text(encoding="utf-8"))


def _strip_italic(s: str) -> str:
    if len(s) >= 2 and s[0] == "_" and s[-1] == "_":
        return s[1:-1]
    if len(s) >= 2 and s.startswith("*") and s.endswith("*") and not s.startswith("**"):
        return s[1:-1]
    return s


def parse_prompt_header(text: str) -> tuple[str, str]:
    title = ""
    caption = ""
    lines = [ln.strip() for ln in text.splitlines()]
    for i, line in enumerate(lines):
        if not line.startswith("# "):
            continue
        title = line[2:].strip()
        for extra in lines[i + 1 :]:
            if extra:
                caption = _strip_italic(extra)
                break
        break
    return title, caption


def _md_table(headers: tuple[str, ...], aligns: str, rows: list[tuple[str, ...]]) -> str:
    def row(vals: tuple[str, ...]) -> str:
        return "| " + " | ".join(vals) + " |"

    seps = []
    for a in aligns:
        if a == "c":
            seps.append(":---:")
        elif a == "r":
            seps.append("---:")
        else:
            seps.append(":---")
    sep = "| " + " | ".join(seps) + " |"
    return "\n".join([row(headers), sep, *[row(r) for r in rows]])


def write_prompts_index(md_paths: list[Path]) -> None:
    rows = []
    for path in md_paths:
        title, caption = parse_prompt_header(path.read_text(encoding="utf-8"))
        rows.append((f"[{path.name}](prompts/{path.name})", title, caption))
    table = _md_table(("File", "Name", "Description"), "lcl", rows)
    INDEX_PATH.write_text(
        "# Prompts\n\n"
        "Index of all prompts used for translation pipelines.\n\n"
        f"{table}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {INDEX_PATH.relative_to(ROOT)} ({len(rows)} prompts)")


def _self_check() -> None:
    example = DST_DIR / "prompt-Template.md"
    got = convert_file(SRC_DIR / "prompt-Template.tex")
    expected = example.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert got == expected, (
        "prompt-Template.md mismatch\n"
        f"--- got ---\n{got!r}\n--- expected ---\n{expected!r}"
    )
    ht = convert_file(SRC_DIR / "chunk_alignment_HT_to_SRC.tex")
    assert r"(\n)" in ht, ht
    api = convert_file(SRC_DIR / "chunk_alignment_api.tex")
    assert "---CANDIDATE CHUNK---" in api, api
    style = convert_file(SRC_DIR / "style_analysis.tex")
    assert '\n    "character_voice_profiles"' in style, style
    assert "```json" in style, style
    assert style.count("```json") == 1, style
    tr = convert_file(SRC_DIR / "translate.tex")
    assert '```json\n{"chunk_id": {chunk_id}, "translation":' in tr, tr
    assert "**{chunk_id}**" not in tr.split("```json", 1)[1].split("```", 1)[0], tr
    fr = convert_file(SRC_DIR / "final_revise.tex")
    json_bits = re.findall(r"```json\n(.*?)```", fr, re.S)
    assert json_bits and all("**" not in bit and not re.search(r"(?<!\*)\*(?!\*)", bit) for bit in json_bits), fr
    assert "**{chunk_id}**" in fr, fr
    rev = convert_file(SRC_DIR / "revise.tex")
    assert "\n{source_text}\n" in rev, rev
    assert "```json\n{source_text}" not in rev, rev
    br = convert_file(SRC_DIR / "book_review.tex")
    assert br.count("```json") == 2, br
    ht_mt = convert_file(SRC_DIR / "chunk_alignment_HT_to_MT.tex")
    assert "`<chunk>`" in ht_mt and "`</chunk>`" in ht_mt, ht_mt
    assert re.sub(r"`[^`]+`", "", ht_mt).find("<chunk>") < 0, ht_mt
    pe = convert_file(SRC_DIR / "post-editing_prompt.tex")
    assert "`<SOURCE>`" in pe and "`</TRANSLATION>`" in pe, pe
    tmpl = convert_file(SRC_DIR / "prompt-Template.tex")
    assert "```json" not in tmpl, tmpl


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    _self_check()
    written: list[Path] = []
    for src in sorted(SRC_DIR.glob("*.tex")):
        md = convert_file(src)
        dst = DST_DIR / f"{src.stem}.md"
        dst.write_text(md, encoding="utf-8", newline="\n")
        written.append(dst)
        print(f"wrote {dst.relative_to(ROOT)} ({len(md.splitlines())} lines)")
    write_prompts_index(written)
    index = INDEX_PATH.read_text(encoding="utf-8")
    assert "[book_review.md](prompts/book_review.md)" in index, index
    assert all(p.name in index for p in written), index


if __name__ == "__main__":
    main()
