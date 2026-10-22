"""Command line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, acronyms, claims, notation, report
from .findings import ERROR, WARN, Finding
from .llm import DEFAULT_MODEL, LLMUnavailable, Ollama
from .tex import LoadError, load

CHECKS = ("acronyms", "notation", "claims")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qwen-review", description="Local pre-submission checks for a LaTeX paper.")
    p.add_argument("--version", action="version", version=f"qwen-review {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check a .tex file or a project directory")
    c.add_argument("path", type=Path)
    c.add_argument("--checks", default=",".join(CHECKS), help=f"comma separated, from {', '.join(CHECKS)}")
    c.add_argument("--allow", default="", help="comma separated acronyms that need no definition")
    c.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    c.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    c.add_argument("--no-llm", action="store_true", help="rules only, no Ollama needed")
    c.add_argument("--fail-on", choices=["error", "warn", "never"], default="error")
    c.add_argument("--max-claims", type=int, default=25, help="most claims to send to the model")
    return p


def run_checks(doc, checks: List[str], allow: List[str], llm: Optional[Ollama], max_claims: int = 25) -> List[Finding]:
    findings: List[Finding] = []
    if "acronyms" in checks:
        findings += acronyms.check(doc, allow)
    if "notation" in checks:
        findings += notation.check(doc, llm)
    if "claims" in checks:
        findings += claims.check(doc, llm, max_claims)
    return findings


def cmd_check(args) -> int:
    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    bad = [c for c in checks if c not in CHECKS]
    if bad:
        print(f"error: unknown check {', '.join(bad)}", file=sys.stderr)
        return 2
    try:
        doc = load(args.path)
    except (LoadError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    llm: Optional[Ollama] = None
    if not args.no_llm and ("notation" in checks or "claims" in checks):
        llm = Ollama(model=args.model)
        try:
            llm.check()
        except LLMUnavailable as e:
            print(f"warning: {e}. Continuing with rules only.", file=sys.stderr)
            llm = None
    findings = run_checks(doc, checks, [a for a in args.allow.split(",") if a], llm, args.max_claims)
    print({"text": report.to_text, "markdown": report.to_markdown, "json": report.to_json}[args.format](findings))
    if args.fail_on == "never":
        return 0
    levels = {ERROR} if args.fail_on == "error" else {ERROR, WARN}
    return 1 if any(f.severity in levels for f in findings) else 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_check(args) if args.cmd == "check" else 2


if __name__ == "__main__":
    sys.exit(main())
