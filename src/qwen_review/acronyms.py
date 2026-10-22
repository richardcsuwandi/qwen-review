"""Acronym checks, done with rules only."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

from .findings import ERROR, INFO, WARN, Finding
from .tex import Doc

# Widely understood terms that a reader does not need defined. Extend with --allow.
COMMON = {
    "AI", "ML", "PDF", "URL", "HTTP", "API", "CPU", "GPU", "TPU", "RAM", "USA", "UK", "EU", "UN", "IEEE", "ACM",
    "SGD", "MSE", "RMSE", "MAE", "IID", "I.I.D", "CNN", "RNN", "LSTM", "GAN", "NLP", "OK", "ID", "FAQ", "AM", "PM",
    "TODO", "NB", "SOTA", "ICML", "NEURIPS", "ICLR", "AAAI", "IJCAI", "ACL", "EMNLP", "CVPR", "ICCV", "ECCV", "KDD",
    "TMLR", "JMLR", "ArXiv", "ARXIV", "DOI", "ISBN", "LATEX", "BIBTEX", "UCI", "MNIST", "CIFAR", "IMAGENET", "GPT",
    "BERT", "LLM", "LLMS", "SQL", "JSON", "CSV", "HTML", "CSS", "GPS", "RGB", "ODE", "PDE", "SDE", "KL", "MCMC",
    "PCA", "SVM", "ROC", "AUC", "OOD", "RL", "IF", "OR", "AND", "NOT", "XOR", "II", "III", "IV", "VI", "VII", "VIII",
}
SMALL = {"of", "the", "and", "for", "in", "on", "to", "a", "an", "with", "via", "by", "at", "from", "per"}

_TOKEN = re.compile(r"(?<![A-Za-z0-9\\_-])([A-Z][A-Z0-9]{1,7})(?:s|'s)?(?![A-Za-z0-9_-])")


def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "xes", "ches", "shes")):
        return w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _form_key(lf: str) -> str:
    """Compare long forms ignoring case, punctuation and plural endings."""
    return " ".join(_singular(w) for w in re.findall(r"[a-z0-9]+", lf.lower()))


def _norm_words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9]*", s)


def _initials_match(abbr: str, words: List[str]) -> bool:
    """True when ``abbr`` is built from the initials of ``words`` (skipping small words)."""
    a = abbr.lower().rstrip("s") if abbr.endswith("s") and len(abbr) > 2 else abbr.lower()
    sig = [w for w in words if w.lower() not in SMALL]
    for seq in (sig, words):
        # initials of every word, or of the words with one-word hyphenated/compound pieces split off
        init = "".join(w[0].lower() for w in seq)
        if init == a:
            return True
        # allow an acronym that uses more than one letter of a word, such as "BO" for "Bayesian optimization" with "OPT"
        if a and init and (a.startswith(init) or init.startswith(a)) and abs(len(a) - len(init)) <= 1 and len(init) >= 2:
            return True
    return False


def _long_form_before(prose: str, paren_start: int, abbr: str) -> str:
    """Find a long form in the words just before ``paren_start`` that matches ``abbr``."""
    before = prose[max(0, paren_start - 120):paren_start]
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", before)
    n = len(abbr.rstrip("s")) if abbr.endswith("s") and len(abbr) > 2 else len(abbr)
    for k in range(max(1, n - 3), min(len(words), n + 4) + 1):
        cand = words[-k:]
        cand_words = [p for w in cand for p in w.split("-")]
        if _initials_match(abbr, cand) or _initials_match(abbr, cand_words):
            return " ".join(cand)
    return ""


def find_definitions(prose: str) -> List[Tuple[int, str, str]]:
    """Return (offset, ACRONYM, long form) for 'Long Form (ABBR)' and 'ABBR (Long Form)'."""
    defs: List[Tuple[int, str, str]] = []
    for m in re.finditer(r"\(\s*([A-Z][A-Za-z0-9]{1,7}s?)\s*\)", prose):
        abbr = m.group(1)
        if not re.fullmatch(r"[A-Z][A-Z0-9]+s?|[A-Z][a-z]?[A-Z]+s?", abbr):
            continue
        lf = _long_form_before(prose, m.start(), abbr)
        if lf:
            defs.append((m.start(1), abbr.rstrip("s") if abbr.endswith("s") and abbr[:-1].isupper() and len(abbr) > 2 else abbr, lf))
    for m in re.finditer(r"\b([A-Z][A-Z0-9]{1,7})s?\s*\(([A-Za-z][^()]{3,60})\)", prose):
        words = _norm_words(m.group(2))
        if _initials_match(m.group(1), words):
            defs.append((m.start(1), m.group(1), " ".join(words)))
    return sorted(set(defs))


def _acro_macros(doc: Doc) -> Set[str]:
    """Acronyms managed by the acronym or glossaries packages, which define them automatically."""
    managed: Set[str] = set()
    for m in re.finditer(r"\\(?:newacronym|DeclareAcronym)(?:\[[^\]]*\])?\{[^}]*\}\{([^}]*)\}", doc.text):
        managed.add(m.group(1))
    return managed


def check(doc: Doc, allow: Iterable[str] = ()) -> List[Finding]:
    prose = doc.prose
    allowed = {a.upper() for a in COMMON} | {a.upper() for a in allow} | {a.upper() for a in _acro_macros(doc)}
    defs = find_definitions(prose)
    def_by_acr: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for off, abbr, lf in defs:
        def_by_acr[abbr].append((off, lf))

    uses: Dict[str, List[int]] = defaultdict(list)
    def_offsets = {(off, abbr) for off, abbr, _ in defs}
    for m in _TOKEN.finditer(prose):
        a = m.group(1)
        if not re.search(r"[A-Z]{2}", a):
            continue
        # skip the occurrence that is itself the definition
        if (m.start(1), a) in def_offsets:
            continue
        uses[a].append(m.start(1))

    out: List[Finding] = []
    loc = doc.locate

    for a, ds in def_by_acr.items():
        forms = {}
        for off, lf in ds:
            forms.setdefault(_form_key(lf), (off, lf))
        if len(forms) > 1:
            (o1, f1), (o2, f2) = list(forms.values())[:2]
            file, line = loc(o2)
            out.append(Finding("acronym", ERROR, f'"{a}" is defined two different ways, "{f1}" (line {loc(o1)[1]}) and "{f2}"', file, line))

    flat: Dict[str, str] = {}
    for off, abbr, lf in defs:
        flat.setdefault(_form_key(lf), abbr)
    seen_long: Dict[str, Set[str]] = defaultdict(set)
    for off, abbr, lf in defs:
        seen_long[_form_key(lf)].add(abbr)
    for lf, abbrs in seen_long.items():
        if len(abbrs) > 1:
            off = min(o for o, a, l in defs if a in abbrs and _form_key(l) == lf)
            file, line = loc(off)
            out.append(Finding("acronym", WARN, f'one term has several abbreviations ({", ".join(sorted(abbrs))})', file, line))

    for a, offs in sorted(uses.items()):
        if a in allowed:
            continue
        first_use = min(offs)
        if a not in def_by_acr:
            file, line = loc(first_use)
            out.append(Finding("acronym", WARN, f'"{a}" is never defined ({len(offs)} use{"s" if len(offs) != 1 else ""})', file, line))
            continue
        first_def = min(o for o, _ in def_by_acr[a])
        if first_use < first_def:
            file, line = loc(first_use)
            out.append(Finding("acronym", WARN, f'"{a}" is used before it is defined (defined at line {loc(first_def)[1]})', file, line))

    for a, ds in sorted(def_by_acr.items()):
        if a in allowed:
            continue
        # Count uses anywhere in the body, including tables and captions that are not prose.
        total_uses = len(uses.get(a, [])) or max(0, len(re.findall(r"(?<![A-Za-z0-9])%s(?:s)?(?![A-Za-z0-9])" % re.escape(a), doc.text[doc.body_start:doc.body_end])) - len(ds))
        # An abstract definition followed by a second one in the body is normal. Two body definitions are not.
        body_defs = [o for o, _ in ds if not doc.in_abstract(o)]
        if len(body_defs) > 1:
            file, line = loc(body_defs[1])
            out.append(Finding("acronym", INFO, f'"{a}" is defined more than once in the body (first at line {loc(body_defs[0])[1]})', file, line))
        if total_uses == 0:
            file, line = loc(ds[0][0])
            out.append(Finding("acronym", INFO, f'"{a}" is defined but never used again, so the abbreviation can go', file, line))
    return out
