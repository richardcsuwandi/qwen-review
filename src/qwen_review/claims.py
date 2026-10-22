"""Claim and evidence checks.

Rules find undefined references, floats nobody cites, and claim sentences. Qwen3 is
asked only whether a claim that cites nothing is backed by one of the paper's own
tables, figures or theorems.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from .findings import ERROR, INFO, WARN, Finding
from .llm import LLMUnavailable, Ollama
from .tex import Doc, sentences

_STRONG = re.compile(
    r"\b(outperform\w*|better than|state[- ]of[- ]the[- ]art|SOTA|superior|significant(?:ly)?|substantially|dramatically|"
    r"considerably|orders? of magnitude|we (?:also |further |additionally )?(?:show|prove|demonstrate) that)\b",
    re.I,
)
_WEAK = re.compile(
    r"\b((?<!expected )improv(?:es|ed|ing|ements?)\b(?! (?:probability|criterion))|achiev\w+|reduc(?:es|ed|ing)|faster|slower|"
    r"more accurate|consistent(?:ly)?|best|lowest|highest)\b",
    re.I,
)
_COMPARE = re.compile(r"\b(than|over|compared (?:with|to)|relative to|versus|vs\.?|against|baselines?)\b", re.I)


class _Claim:
    """Callable that answers whether a sentence reads as a claim worth backing."""

    @staticmethod
    def search(text: str):
        if _STRONG.search(text):
            return True
        return bool(_WEAK.search(text) and _COMPARE.search(text))


_CLAIM = _Claim()
_NUMBER_CLAIM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\\%|%|\\times|×|x\b|percent|\\,\\%)")
_CITE = re.compile(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
_THEOREM = re.compile(r"\\begin\{(theorem|lemma|proposition|corollary)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}", re.S)


def _ref_labels(doc: Doc, text: str) -> List[str]:
    return [lab.strip() for m in doc.ref_pattern.finditer(text) for lab in m.group(1).split(",")]


def _table_text(body: str) -> str:
    body = re.sub(r"\\caption(?:\[[^\]]*\])?\{.*", "", body, flags=re.S)
    body = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|centering|begin\{[^}]*\}(?:\{[^}]*\})?|end\{[^}]*\}|label\{[^}]*\})", " ", body)
    body = body.replace("&", " | ").replace("\\\\", " ; ")
    return re.sub(r"\s+", " ", body).strip()


def evidence_items(doc: Doc) -> List[Dict[str, str]]:
    items = []
    for f in doc.floats:
        if f.label:
            text = f.caption[:300]
            if f.kind == "table":
                text += " Values: " + _table_text(f.body)[:500]
            items.append({"label": f.label, "kind": f.kind, "text": text})
    for m in _THEOREM.finditer(doc.text):
        lab = re.search(r"\\label\{([^}]*)\}", m.group(2))
        if lab:
            body = re.sub(r"\\label\{[^}]*\}", "", m.group(2))
            items.append({"label": lab.group(1), "kind": m.group(1), "text": re.sub(r"\s+", " ", body).strip()[:300]})
    return items


def check(doc: Doc, llm: Optional[Ollama] = None, max_llm: int = 25) -> List[Finding]:
    out: List[Finding] = []
    loc = doc.locate
    labels = doc.labels

    # References that point nowhere.
    for m in doc.ref_pattern.finditer(doc.text[doc.body_start:doc.body_end]):
        for lab in m.group(1).split(","):
            lab = lab.strip()
            if lab and lab not in labels:
                f, n = loc(doc.body_start + m.start())
                out.append(Finding("reference", ERROR, f"\\ref to undefined label {lab}", f, n))

    # Floats that are never referenced from the text.
    cited: Set[str] = set(_ref_labels(doc, doc.text))
    for fl in doc.floats:
        if fl.label and fl.label not in cited:
            f, n = loc(fl.start)
            out.append(Finding("reference", WARN, f"{fl.kind} {fl.label} is never referenced in the text", f, n))
        elif not fl.label:
            f, n = loc(fl.start)
            out.append(Finding("reference", INFO, f"{fl.kind} has no \\label, so the text cannot refer to it", f, n))

    for m in _THEOREM.finditer(doc.text):
        lab = re.search(r"\\label\{([^}]*)\}", m.group(2))
        if lab and lab.group(1) not in cited:
            f, n = loc(m.start())
            out.append(Finding("reference", INFO, f"{m.group(1)} {lab.group(1)} is never referenced in the text", f, n))

    # Claim sentences that cite no float, theorem or source.
    evidence = evidence_items(doc)
    ev_labels = {e["label"] for e in evidence}
    asked = 0
    handled: Set[int] = set()
    corpus = " ".join(fl.body for fl in doc.floats)
    unchecked: List[tuple] = []
    sents = sentences(doc)
    for idx, (s, e) in enumerate(sents):
        text = doc.text[s:e]
        prose = doc.prose[s:e]
        if not (_CLAIM.search(prose) or _NUMBER_CLAIM.search(text)):
            continue
        window = text
        if idx + 1 < len(sents):
            window += doc.text[sents[idx + 1][0]:sents[idx + 1][1]]
        if _ref_labels(doc, window) or _CITE.search(text):
            continue
        if doc.section_at(s).lower().startswith(("related", "background", "preliminar")):
            continue
        f, n = loc(s)
        claim = re.sub(r"\s+", " ", re.sub(r"[{}]", "", re.sub(r"\\[A-Za-z]+\*?", "", text))).strip()
        short = claim if len(claim) <= 140 else claim[:137] + "..."
        verdict = None
        if llm and asked < max_llm:
            asked += 1
            try:
                verdict = llm.claim_support(claim, evidence)
            except LLMUnavailable:
                llm = None
        if verdict is None and llm and asked >= max_llm:
            pass
        if verdict is not None and verdict.get("supported"):
            # A number in the claim has to appear in the evidence it is credited to.
            item = next((e for e in evidence if e["label"] == verdict["label"]), None)
            missing = [x for x in re.findall(r"\d+(?:\.\d+)?", claim) if item and not re.search(r"(?<![\d.])" + re.escape(x) + r"(?![\d])", item["text"])]
            if missing:
                verdict = {"supported": False, "label": "", "reason": f"{verdict['label']} does not contain {', '.join(missing)}"}
        if verdict is not None:
            if verdict.get("supported") and verdict.get("label") in ev_labels:
                out.append(Finding("claim", INFO, f'claim cites nothing, but {verdict["label"]} appears to support it, add a \\ref: "{short}"',
                                   f, n, "qwen", [f"Qwen3: {verdict.get('reason', '')}"]))
            elif _NUMBER_CLAIM.search(text):
                # A derived number, such as a relative improvement, is rarely stated verbatim in a caption.
                absent = _absent_numbers(text, corpus)
                handled.add(s)
                if absent:
                    out.append(Finding("claim", WARN, f'{", ".join(absent)} is in no table or caption and nothing supports this claim, check how it is derived: "{short}"',
                                       f, n, "qwen", [f"Qwen3: {verdict.get('reason', '')}"]))
                else:
                    out.append(Finding("claim", INFO, f'numeric claim cites nothing, check that a table supports it: "{short}"', f, n, "qwen",
                                       [f"Qwen3: {verdict.get('reason', '')}"]))
            else:
                out.append(Finding("claim", WARN, f'claim with no supporting table, figure or theorem: "{short}"', f, n, "qwen",
                                   [f"Qwen3: {verdict.get('reason', '')}"]))
        else:
            unchecked.append((f, n, short))

    if unchecked:
        f, n, _ = unchecked[0]
        why = "no model" if not llm else f"over the --max-claims limit of {max_llm}"
        out.append(Finding("claim", INFO, f"{len(unchecked)} claim sentence{'s' if len(unchecked) != 1 else ''} cite no table, figure or source and were not checked ({why})",
                           f, n, "rule", [f'line {a[1]}: "{a[2]}"' for a in unchecked[:12]] + ([f"and {len(unchecked) - 12} more"] if len(unchecked) > 12 else [])))

    # Numbers in claims that appear in no table or caption, one finding per sentence.
    for s, e in sents:
        if s in handled or not (doc.in_abstract(s) or doc.section_at(s).lower().startswith(("intro", "conclu", "discussion"))):
            continue
        absent = _absent_numbers(doc.text[s:e], corpus)
        if absent:
            f, n = loc(s)
            out.append(Finding("claim", INFO, f"{', '.join(absent)} not found in any table or caption, check that it is derived from reported results", f, n))
    return out


def _absent_numbers(text: str, corpus: str) -> List[str]:
    absent = []
    for m in _NUMBER_CLAIM.finditer(text):
        num = m.group(1)
        if float(num) == 0 or re.search(r"(?<![\d.])" + re.escape(num) + r"(?![\d])", corpus):
            continue
        absent.append(m.group(0).strip().replace("\\", ""))
    return absent
