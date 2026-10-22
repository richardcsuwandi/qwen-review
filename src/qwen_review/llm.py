"""Qwen3 on Ollama, used only for judgments that rules cannot make.

Every answer is checked against the input before it is trusted. A symbol, label or
index the model returns that was not in the prompt is discarded.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "qwen3:8b"


class LLMUnavailable(RuntimeError):
    pass


CONFLICT_SCHEMA = {
    "type": "object",
    "properties": {"conflict": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["conflict", "reason"],
}
GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"concept": {"type": "string"}, "ids": {"type": "array", "items": {"type": "integer"}}},
                "required": ["concept", "ids"],
            },
        }
    },
    "required": ["groups"],
}
CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "label": {"type": "string"},
        "quote": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["supported", "label", "quote", "reason"],
}

CONFLICT_SYSTEM = (
    "You review notation in a research paper. You get one math symbol and the places where the paper "
    "defines it. Decide whether the definitions give the symbol different meanings. Different wording "
    "of the same meaning is not a conflict. A different object, dimension or role is. "
    "Reply with JSON only, with a reason under 25 words."
)
GROUP_SYSTEM = (
    "You review notation in a research paper. You get numbered definitions, each a symbol with a "
    "description. Find concepts that the paper writes with more than one symbol. Only group entries that "
    "clearly describe the same object, for example a kernel written as k in one place and K in another. "
    "Do not group related but different objects. Reply with JSON only. Use only the given ids."
)
CLAIM_SYSTEM = (
    "You check whether a claim in a research paper is backed by the paper's own tables, figures or "
    "theorems. You get the claim and labelled evidence items with their captions, statements or table "
    "values. An item supports the claim only if it directly shows the same comparison, on the same "
    "metric, at the same size. A table of regret does not support a claim about speed. A table with two "
    "baselines does not support a claim about all baselines or state of the art. A number in the claim "
    "must appear in the evidence. If unsure, answer not supported. When supported, give the label "
    "exactly as listed and copy the supporting words or numbers verbatim into quote. Reply with JSON "
    "only, with a reason under 25 words."
)


class Ollama:
    def __init__(self, model: str = DEFAULT_MODEL, host: Optional[str] = None, timeout: float = 180.0):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout

    def check(self) -> None:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=5) as r:
                names = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
            raise LLMUnavailable(f"Ollama at {self.host} is not reachable: {e}") from e
        if self.model not in names and f"{self.model}:latest" not in names:
            raise LLMUnavailable(f"model {self.model} is not pulled. Run: ollama pull {self.model}")

    def chat_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self.model, "stream": False, "think": False, "format": schema,
            "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
            return json.loads(data["message"]["content"])
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, ValueError) as e:
            raise LLMUnavailable(f"Ollama request failed: {e}") from e

    def symbol_conflict(self, symbol: str, definitions: List[str]) -> Dict[str, Any]:
        body = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(definitions))
        return self.chat_json(CONFLICT_SYSTEM, f"Symbol: {symbol}\nDefinitions:\n{body}", CONFLICT_SCHEMA)

    def group_concepts(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        body = "\n".join(f"{it['id']}. {it['symbol']} : {it['desc']}" for it in items)
        out = self.chat_json(GROUP_SYSTEM, body, GROUP_SCHEMA)
        valid = {it["id"] for it in items}
        groups = []
        for g in out.get("groups", []):
            ids = [i for i in g.get("ids", []) if i in valid]
            if len(set(ids)) >= 2:
                groups.append({"concept": g.get("concept", ""), "ids": sorted(set(ids))})
        return groups

    def claim_support(self, claim: str, evidence: List[Dict[str, str]]) -> Dict[str, Any]:
        body = "\n".join(f"[{e['label']}] ({e['kind']}) {e['text']}" for e in evidence) or "(none)"
        out = self.chat_json(CLAIM_SYSTEM, f"Claim:\n{claim}\n\nEvidence:\n{body}", CLAIM_SCHEMA)
        by_label = {e["label"]: e for e in evidence}
        item = by_label.get(out.get("label", ""))
        quote = " ".join((out.get("quote") or "").lower().split())
        # Trust the model only when it points at a real item and quotes it verbatim.
        if not item or not quote or quote not in " ".join(item["text"].lower().split()):
            if out.get("supported"):
                out["reason"] = "no verifiable quote in the evidence it pointed to"
            out["supported"] = False
            out["label"] = ""
        return out
