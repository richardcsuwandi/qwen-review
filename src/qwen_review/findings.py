from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

ERROR, WARN, INFO = "error", "warn", "info"
RANK = {ERROR: 0, WARN: 1, INFO: 2}


@dataclass
class Finding:
    check: str  # acronym, notation, claim or reference
    severity: str
    message: str
    file: str
    line: int
    by: str = "rule"  # rule or qwen
    detail: List[str] = field(default_factory=list)

    def sort_key(self):
        return (RANK[self.severity], self.file, self.line)
