"""Load a LaTeX project into one searchable document with line mapping.

Everything here is offset-preserving. Masking replaces text with spaces of the same
length, so a position found in the masked text maps straight back to a file and line.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LoadError(RuntimeError):
    pass


@dataclass
class MathSpan:
    start: int
    end: int
    content: str
    display: bool


@dataclass
class Float:
    kind: str  # figure or table
    label: str
    caption: str
    start: int
    end: int
    body: str


@dataclass
class Doc:
    root: Path
    text: str = ""
    lines: List[Tuple[str, int]] = field(default_factory=list)  # (file, lineno) per output line
    starts: List[int] = field(default_factory=list)  # offset where each line starts
    body_start: int = 0
    body_end: int = 0
    _cache: Dict[str, object] = field(default_factory=dict)

    def locate(self, offset: int) -> Tuple[str, int]:
        i = max(0, bisect_right(self.starts, offset) - 1)
        f, n = self.lines[i]
        try:
            f = str(Path(f).relative_to(self.root))
        except ValueError:
            pass
        return f, n

    # -- derived views, computed once --------------------------------------------

    @property
    def math(self) -> List[MathSpan]:
        if "math" not in self._cache:
            self._cache["math"] = find_math(self.text)
        return self._cache["math"]  # type: ignore[return-value]

    @property
    def prose(self) -> str:
        """The body with math, citations, tables and markup blanked out, same length as ``text``."""
        if "prose" not in self._cache:
            self._cache["prose"] = make_prose(self)
        return self._cache["prose"]  # type: ignore[return-value]

    @property
    def floats(self) -> List[Float]:
        if "floats" not in self._cache:
            self._cache["floats"] = find_floats(self)
        return self._cache["floats"]  # type: ignore[return-value]

    @property
    def sections(self) -> List[Tuple[int, str]]:
        if "sections" not in self._cache:
            self._cache["sections"] = [
                (m.start(), m.group(2).strip())
                for m in re.finditer(r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}", self.text)
            ]
        return self._cache["sections"]  # type: ignore[return-value]

    def section_at(self, offset: int) -> str:
        name = ""
        for pos, title in self.sections:
            if pos > offset:
                break
            name = title
        return name

    @property
    def abstract(self) -> Optional[Tuple[int, int]]:
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", self.text, re.S)
        return (m.start(1), m.end(1)) if m else None

    def in_abstract(self, offset: int) -> bool:
        a = self.abstract
        return bool(a and a[0] <= offset < a[1])

    @property
    def ref_pattern(self) -> "re.Pattern[str]":
        """Matches reference commands, including project macros that wrap \\ref, such as \\figref."""
        if "ref_pattern" not in self._cache:
            names = {"ref", "cref", "Cref", "autoref", "eqref", "pageref", "vref", "labelcref", "cpageref"}
            for m in re.finditer(r"\\(?:re)?newcommand\*?\{\\(\w+)\}\s*(?:\[1\])?\s*\{[^{}]*?\\(?:ref|cref|Cref|autoref|eqref)\{#1\}", self.text):
                names.add(m.group(1))
            self._cache["ref_pattern"] = re.compile(r"\\(?:%s)\*?\{([^}#]*)\}" % "|".join(sorted(names)))
        return self._cache["ref_pattern"]  # type: ignore[return-value]

    @property
    def labels(self) -> Dict[str, int]:
        if "labels" not in self._cache:
            self._cache["labels"] = {m.group(1): m.start() for m in re.finditer(r"\\label\{([^}]*)\}", self.text)}
        return self._cache["labels"]  # type: ignore[return-value]


def _strip_comment(line: str) -> str:
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\":
            out.append(line[i:i + 2])
            i += 2
            continue
        if c == "%":
            break
        out.append(c)
        i += 1
    return "".join(out)


def find_main(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise LoadError(f"{path} does not exist")
    candidates = [p for p in sorted(path.rglob("*.tex")) if "\\documentclass" in p.read_text(errors="replace")]
    if not candidates:
        raise LoadError(f"no .tex file with \\documentclass found in {path}")
    for name in ("main.tex", "paper.tex"):
        for c in candidates:
            if c.name == name:
                return c
    return max(candidates, key=lambda p: len(p.read_text(errors="replace")))


def load(path: Path) -> Doc:
    main = find_main(path).resolve()
    doc = Doc(root=main.parent)
    out_lines: List[str] = []
    seen = set()

    def read(p: Path) -> None:
        if p in seen:
            return
        seen.add(p)
        for i, raw in enumerate(p.read_text(errors="replace").splitlines(), 1):
            line = _strip_comment(raw)
            m = re.fullmatch(r"\s*\\(?:input|include|subfile)\{([^}]+)\}\s*", line)
            if m:
                target = p.parent / m.group(1)
                for cand in (target, target.with_suffix(".tex"), Path(str(target) + ".tex")):
                    if cand.is_file():
                        read(cand.resolve())
                        break
                continue
            out_lines.append(line)
            doc.lines.append((str(p), i))

    read(main)
    pos = 0
    for line in out_lines:
        doc.starts.append(pos)
        pos += len(line) + 1
    doc.text = "\n".join(out_lines)
    b = re.search(r"\\begin\{document\}", doc.text)
    e = re.search(r"\\end\{document\}", doc.text)
    doc.body_start = b.end() if b else 0
    doc.body_end = e.start() if e else len(doc.text)
    bib = re.search(r"\\begin\{thebibliography\}|\\bibliography\{|\\printbibliography", doc.text[doc.body_start:])
    if bib:
        doc.body_end = min(doc.body_end, doc.body_start + bib.start())
    return doc


_ENVS = r"equation|align|gather|multline|eqnarray|displaymath|flalign|math"
_MATH_RE = re.compile(
    r"\\begin\{(?P<env>(?:%s)\*?)\}.*?\\end\{(?P=env)\}"
    r"|\$\$.*?\$\$"
    r"|\\\[.*?\\\]"
    r"|\\\(.*?\\\)"
    r"|(?<![\\$])\$(?!\$)(?:[^$\\]|\\.)+?\$" % _ENVS,
    re.S,
)


def find_math(text: str) -> List[MathSpan]:
    spans = []
    for m in _MATH_RE.finditer(text):
        s = m.group(0)
        display = not (s.startswith("$") and not s.startswith("$$")) and not s.startswith("\\(")
        inner = re.sub(r"^\\begin\{[^}]*\}|\\end\{[^}]*\}$", "", s)
        inner = inner.strip("$").removeprefix("\\[").removesuffix("\\]").removeprefix("\\(").removesuffix("\\)")
        spans.append(MathSpan(m.start(), m.end(), inner.strip(), display))
    return spans


def _blank(text: str, start: int, end: int) -> str:
    seg = text[start:end]
    return text[:start] + re.sub(r"[^\n]", " ", seg) + text[end:]


_MASK_PATTERNS = [
    re.compile(r"\\begin\{(tabular\*?|tabularx|lstlisting|verbatim|algorithmic|algorithm\*?|minted|tikzpicture)\}.*?\\end\{\1\}", re.S),
    re.compile(r"\\(?:cite[a-z]*|ref|cref|Cref|autoref|eqref|label|url|href|includegraphics|bibliographystyle|usepackage|input)\*?(?:\[[^\]]*\])*\{[^}]*\}"),
    re.compile(r"\\(?:texttt|verb|path)\*?\{[^}]*\}"),
    re.compile(r"\\(?:sub)*(?:section|paragraph)\*?\{[^}]*\}"),
    re.compile(r"\\begin\{[a-z*]+\}|\\end\{[a-z*]+\}"),
]


def make_prose(doc: Doc) -> str:
    t = doc.text
    for span in doc.math:
        t = _blank(t, span.start, span.end)
    for pat in _MASK_PATTERNS:
        for m in list(pat.finditer(t)):
            t = _blank(t, m.start(), m.end())
    # Drop command names but keep their arguments, so \emph{GP} still reads as GP.
    t = re.sub(r"\\[A-Za-z]+\*?", lambda m: " " * len(m.group(0)), t)
    t = re.sub(r"[{}~]", " ", t)
    # Everything outside the body is not prose.
    return " " * doc.body_start + t[doc.body_start:doc.body_end] + " " * (len(t) - doc.body_end)


def _match_brace(text: str, start: int) -> int:
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def find_floats(doc: Doc) -> List[Float]:
    out = []
    for m in re.finditer(r"\\begin\{(figure\*?|table\*?|wrapfigure|wraptable|sidewaysfigure|sidewaystable)\}(.*?)\\end\{\1\}", doc.text, re.S):
        body = m.group(2)
        cap = ""
        c = re.search(r"\\caption(?:\[[^\]]*\])?\{", body)
        if c:
            end = _match_brace(body, c.end() - 1)
            cap = body[c.end():end] if end > 0 else body[c.end():]
        lab = re.search(r"\\label\{([^}]*)\}", body)
        kind = "table" if "table" in m.group(1) else "figure"
        out.append(Float(kind, lab.group(1) if lab else "", re.sub(r"\s+", " ", cap).strip(), m.start(), m.end(), body))
    return out


def sentences(doc: Doc) -> List[Tuple[int, int]]:
    """Sentence spans as offsets into ``doc.text``, split on the math-and-citation-masked prose."""
    p = doc.prose
    protected = re.compile(r"\b(?:e\.g|i\.e|et al|cf|vs|Fig|Figs|Eq|Eqs|Sec|Secs|Tab|Thm|Lem|Prop|approx|resp|Dr|No|etc)\.")
    p2 = p
    for m in protected.finditer(p):
        p2 = p2[:m.end() - 1] + "," + p2[m.end():]
    spans = []
    start = None
    for m in re.finditer(r"[.!?](?=\s|$)|\n\s*\n", p2):
        s = start if start is not None else doc.body_start
        spans.append((s, m.end()))
        start = m.end()
    if start is not None and start < doc.body_end:
        spans.append((start, doc.body_end))
    out = []
    for s, e in spans:
        # trim leading whitespace and skip empty or markup-only spans
        while s < e and doc.text[s].isspace():
            s += 1
        if len(p[s:e].strip()) >= 20:
            out.append((s, e))
    return out
