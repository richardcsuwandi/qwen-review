"""Render findings as text, Markdown or JSON."""
from __future__ import annotations

import json
from collections import Counter
from typing import List

from .findings import ERROR, INFO, WARN, Finding

MARK = {ERROR: "error", WARN: "warn ", INFO: "info "}


def counts(findings: List[Finding]) -> Counter:
    return Counter(f.severity for f in findings)


def _summary(findings: List[Finding]) -> str:
    c = counts(findings)
    parts = [f"{c[s]} {s}" + ("s" if c[s] != 1 and s != "info" else "") for s in (ERROR, WARN, INFO) if c[s]]
    return ", ".join(parts) if parts else "no findings"


def to_text(findings: List[Finding]) -> str:
    out = []
    for f in sorted(findings, key=Finding.sort_key):
        tag = " [qwen]" if f.by == "qwen" else ""
        out.append(f"{f.file}:{f.line}  {MARK[f.severity]}  {f.check:9} {f.message}{tag}")
        for d in f.detail:
            out.append(f"      {d}")
    out.append("")
    out.append(_summary(findings))
    return "\n".join(out)


def to_markdown(findings: List[Finding]) -> str:
    lines = ["# Pre-submission review", "", _summary(findings) + ".", ""]
    for check in ("acronym", "notation", "claim", "reference"):
        group = [f for f in sorted(findings, key=Finding.sort_key) if f.check == check]
        if not group:
            continue
        lines += [f"## {check.capitalize()}", ""]
        for f in group:
            by = " *(Qwen3)*" if f.by == "qwen" else ""
            lines.append(f"- **{f.severity}** `{f.file}:{f.line}` {f.message}{by}")
            for d in f.detail:
                lines.append(f"  - {d}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_json(findings: List[Finding]) -> str:
    return json.dumps(
        {
            "summary": dict(counts(findings)),
            "findings": [
                {"check": f.check, "severity": f.severity, "message": f.message, "file": f.file, "line": f.line,
                 "by": f.by, "detail": f.detail}
                for f in sorted(findings, key=Finding.sort_key)
            ],
        },
        indent=2,
        ensure_ascii=False,
    )
