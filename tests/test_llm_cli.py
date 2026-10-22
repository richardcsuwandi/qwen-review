import json

from qwen_review import cli, llm


def test_claim_support_requires_a_verbatim_quote(monkeypatch):
    o = llm.Ollama()
    ev = [{"label": "tab:r", "kind": "table", "text": "Regret. Values: A | 0.5 ; B | 0.2"}]
    monkeypatch.setattr(o, "chat_json", lambda *a: {"supported": True, "label": "tab:r", "quote": "B | 0.2", "reason": "ok"})
    assert o.claim_support("B is best", ev)["supported"] is True
    monkeypatch.setattr(o, "chat_json", lambda *a: {"supported": True, "label": "tab:r", "quote": "made up words", "reason": "ok"})
    assert o.claim_support("B is best", ev)["supported"] is False
    monkeypatch.setattr(o, "chat_json", lambda *a: {"supported": True, "label": "tab:other", "quote": "B | 0.2", "reason": "ok"})
    assert o.claim_support("B is best", ev)["label"] == ""


def test_group_concepts_drops_unknown_ids(monkeypatch):
    o = llm.Ollama()
    monkeypatch.setattr(o, "chat_json", lambda *a: {"groups": [{"concept": "kernel", "ids": [1, 2, 99]}, {"concept": "x", "ids": [1, 77]}]})
    out = o.group_concepts([{"id": 1, "symbol": "k", "desc": "kernel"}, {"id": 2, "symbol": "K", "desc": "kernel"}])
    assert out == [{"concept": "kernel", "ids": [1, 2]}]


def test_cli_json_exit_codes(tmp_path, capsys):
    (tmp_path / "main.tex").write_text("\\documentclass{article}\\begin{document}See \\ref{nope}.\\end{document}")
    code = cli.main(["check", str(tmp_path), "--no-llm", "--format", "json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["summary"]["error"] == 1
    assert cli.main(["check", str(tmp_path), "--no-llm", "--fail-on", "never"]) == 0
    capsys.readouterr()
    (tmp_path / "main.tex").write_text("\\documentclass{article}\\begin{document}Clean text here.\\end{document}")
    assert cli.main(["check", str(tmp_path), "--no-llm"]) == 0


def test_cli_rejects_bad_input(tmp_path, capsys):
    assert cli.main(["check", str(tmp_path / "missing.tex"), "--no-llm"]) == 2
    (tmp_path / "main.tex").write_text("\\documentclass{article}\\begin{document}x\\end{document}")
    assert cli.main(["check", str(tmp_path), "--checks", "bogus", "--no-llm"]) == 2
