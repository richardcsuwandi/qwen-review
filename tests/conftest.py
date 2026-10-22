from pathlib import Path

import pytest

from qwen_review.tex import load


@pytest.fixture
def doc(tmp_path):
    """Build a Doc from a LaTeX body string."""

    def make(body: str, preamble: str = "", extra: dict = None):
        for name, text in (extra or {}).items():
            f = tmp_path / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
        (tmp_path / "main.tex").write_text(
            "\\documentclass{article}\n" + preamble + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
        )
        return load(tmp_path / "main.tex")

    return make


class FakeLLM:
    """Stub model with scripted answers."""

    def __init__(self, conflict=True, groups=None, support=None):
        self._conflict = conflict
        self._groups = groups or []
        self._support = support
        self.claims = []

    def symbol_conflict(self, symbol, definitions):
        return {"conflict": self._conflict, "reason": "stub"}

    def group_concepts(self, items):
        return self._groups

    def claim_support(self, claim, evidence):
        self.claims.append(claim)
        return self._support or {"supported": False, "label": "", "reason": "stub"}
