"""Notation checks.

Rules extract where the paper defines symbols and how it styles them. Qwen3 is asked
only two questions, whether two definitions of one symbol conflict, and whether
several symbols name one concept.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .findings import ERROR, INFO, WARN, Finding
from .llm import LLMUnavailable, Ollama
from .tex import Doc, sentences


@dataclass
class Definition:
    symbol: str
    desc: str
    offset: int


_SYM = r"\$(?P<sym>[^$]{1,40})\$"
_DESC = r"(?P<desc>[^.;$\\]{3,80})"
_PATTERNS = [
    re.compile(_SYM + r"\s*(?:\\in[^$]{0,30}\s*)?,?\s*(?:is|are|denotes?|represents?|refers to|stands for)\s+(?:the\s+|a\s+|an\s+)?" + _DESC),
    re.compile(r"[Ll]et\s+" + _SYM + r"\s+(?:be|denote)\s+(?:the\s+|a\s+|an\s+)?" + _DESC),
    re.compile(r"[Ww]e\s+(?:denote|write)\s+(?:the\s+|a\s+|an\s+)?(?P<desc>[^$.;]{3,80}?)\s+(?:by|as)\s+" + _SYM),
    re.compile(r"(?P<desc>[^.;$]{3,80}?),?\s+(?:denoted|written)\s+(?:by\s+|as\s+)?" + _SYM),
]


def _clean_desc(d: str) -> str:
    d = re.sub(r"\s+", " ", d).strip(" ,:")
    d = re.sub(r"\b(?:and|which|that|where|with|so|such)\b.*$", "", d).strip(" ,:")
    return d


def _norm_symbol(s: str) -> str:
    return re.sub(r"\s+|\\[,;:! ]", "", s)


def definitions(doc: Doc) -> List[Definition]:
    out: List[Definition] = []
    seen = set()
    for s, e in sentences_with_math(doc):
        text = doc.text[s:e]
        for pat in _PATTERNS:
            for m in pat.finditer(text):
                desc = _clean_desc(m.group("desc"))
                sym = _norm_symbol(m.group("sym"))
                if len(desc) < 3 or not sym or (sym, desc.lower(), s) in seen:
                    continue
                seen.add((sym, desc.lower(), s))
                out.append(Definition(sym, desc, s + m.start()))
    return out


def sentences_with_math(doc: Doc):
    """Sentence spans over the body, split on text where math is blanked but kept in ``doc.text``."""
    return sentences(doc)


GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "iota", "kappa",
         "lambda", "mu", "nu", "xi", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", "varphi",
         "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega"}
BOLD = {"mathbf", "boldsymbol", "bm"}
_STRIP_TEXT = re.compile(r"\\(?:text|mathrm|operatorname|mbox|textrm|textit|textsf)\*?\{[^{}]*\}")
_TOKEN = re.compile(
    r"\\(?P<w>mathbf|boldsymbol|bm|vec|mathcal|mathbb|mathsf|mathit|tilde|hat|bar)\s*\{\s*(?P<a>\\?[A-Za-z]+)\s*\}"
    r"|\\(?P<g>[A-Za-z]+)"
    r"|(?P<l>[A-Za-z])"
)


def symbol_styles(doc: Doc) -> Dict[str, Counter]:
    """Count how often each base symbol appears with each font wrapper."""
    styles: Dict[str, Counter] = defaultdict(Counter)
    for span in doc.math:
        content = _STRIP_TEXT.sub(" ", span.content)
        for m in _TOKEN.finditer(content):
            if m.group("w"):
                styles[m.group("a").lstrip("\\")][m.group("w")] += 1
            elif m.group("g") and m.group("g") in GREEK:
                styles[m.group("g")]["plain"] += 1
            elif m.group("l"):
                styles[m.group("l")]["plain"] += 1
    return styles


VARIANTS = [("epsilon", "varepsilon"), ("phi", "varphi"), ("theta", "vartheta"), ("rho", "varrho"), ("pi", "varpi"), ("sigma", "varsigma")]


def _first_offset(doc: Doc, pattern: str) -> int:
    m = re.search(pattern, doc.text)
    return m.start() if m else 0


def check(doc: Doc, llm: Optional[Ollama] = None) -> List[Finding]:
    out: List[Finding] = []
    defs = definitions(doc)
    loc = doc.locate

    # 1. One symbol defined more than once with different descriptions.
    by_symbol: Dict[str, List[Definition]] = defaultdict(list)
    for d in defs:
        by_symbol[d.symbol].append(d)
    for sym, ds in by_symbol.items():
        descs = {d.desc.lower() for d in ds}
        if len(descs) < 2:
            continue
        lines = ", ".join(str(loc(d.offset)[1]) for d in ds)
        listing = [f'"{d.desc}" (line {loc(d.offset)[1]})' for d in ds]
        verdict = None
        if llm:
            try:
                verdict = llm.symbol_conflict(sym, [f"{d.desc}" for d in ds])
            except LLMUnavailable:
                llm = None
        if verdict is not None:
            if verdict.get("conflict"):
                f, n = loc(ds[1].offset)
                out.append(Finding("notation", WARN, f"${sym}$ is defined with different meanings, lines {lines}", f, n, "qwen",
                                   listing + [f"Qwen3: {verdict.get('reason', '')}"]))
        elif _word_overlap(list(descs)) < 0.34:
            f, n = loc(ds[1].offset)
            out.append(Finding("notation", INFO, f"${sym}$ may be defined with different meanings, lines {lines} (no model, unconfirmed)", f, n, "rule", listing))

    # 2. One concept written with several symbols.
    groups: Dict[str, List[Definition]] = defaultdict(list)
    for d in defs:
        groups[re.sub(r"^(?:the|a|an)\s+", "", d.desc.lower())].append(d)
    reported = set()
    for desc, ds in groups.items():
        syms = sorted({d.symbol for d in ds})
        if len(syms) > 1:
            f, n = loc(ds[1].offset)
            out.append(Finding("notation", WARN, f'"{desc}" is written as {_fmt(syms)}', f, n, "rule",
                               [f"${d.symbol}$ at line {loc(d.offset)[1]}" for d in ds]))
            reported.add(tuple(syms))
    if llm and len(defs) >= 2:
        items = [{"id": i + 1, "symbol": d.symbol, "desc": d.desc} for i, d in enumerate(defs[:60])]
        try:
            for g in llm.group_concepts(items):
                members = [defs[i - 1] for i in g["ids"]]
                syms = sorted({d.symbol for d in members})
                if len(syms) < 2 or tuple(syms) in reported:
                    continue
                reported.add(tuple(syms))
                f, n = loc(members[1].offset)
                out.append(Finding("notation", WARN, f'"{g["concept"]}" may be written as {_fmt(syms)}', f, n, "qwen",
                                   [f"${d.symbol}$ is {d.desc}, line {loc(d.offset)[1]}" for d in members]))
        except LLMUnavailable:
            pass

    # 3. Font style mixing and glyph variants for the same letter.
    styles = symbol_styles(doc)
    for base, counts in sorted(styles.items()):
        bold = {w: c for w, c in counts.items() if w in BOLD and c >= 1}
        if len(bold) > 1:
            m = re.search(r"\\(?:%s)\s*\{\s*\\?%s\s*\}" % ("|".join(BOLD), re.escape(base)), doc.text)
            f, n = loc(m.start() if m else 0)
            detail = ", ".join(f"\\{w}{{{base}}} x{c}" for w, c in sorted(bold.items(), key=lambda kv: -kv[1]))
            out.append(Finding("notation", WARN, f"bold {base} is written in several ways, {detail}", f, n))
    for a, b in VARIANTS:
        if styles.get(a) and styles.get(b) and sum(styles[a].values()) >= 1 and sum(styles[b].values()) >= 1:
            m = re.search(r"\\%s(?![A-Za-z])" % b, doc.text)
            f, n = loc(m.start() if m else 0)
            out.append(Finding("notation", INFO, f"both \\{a} ({sum(styles[a].values())}x) and \\{b} ({sum(styles[b].values())}x) are used, pick one glyph", f, n))
    return out


def _fmt(syms: List[str]) -> str:
    return ", ".join(f"${s}$" for s in syms)


def _word_overlap(descs: List[str]) -> float:
    sets = [set(re.findall(r"[a-z]{3,}", d)) for d in descs]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 1.0
    best = 1.0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            best = min(best, len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
    return best
