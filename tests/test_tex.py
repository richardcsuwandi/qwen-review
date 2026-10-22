from qwen_review.tex import load, find_math


def test_inputs_are_flattened_and_lines_map_back(doc):
    d = doc("Hello.\n\\input{sec/a}\nBye.", extra={"sec/a.tex": "First line\nSecond line % comment\n"})
    assert "Second line" in d.text and "comment" not in d.text
    off = d.text.index("Second line")
    assert d.locate(off) == ("sec/a.tex", 2)
    assert d.locate(d.text.index("Bye")) == ("main.tex", 6)


def test_prose_masks_math_and_keeps_offsets(doc):
    d = doc("The value $x^2$ is \\emph{GP} and \\cite{a}.")
    assert len(d.prose) == len(d.text)
    assert "x^2" not in d.prose and "GP" in d.prose and "cite" not in d.prose


def test_math_spans_cover_inline_and_display(doc):
    d = doc("Inline $a+b$ and\n\\begin{equation}\nc=d\n\\end{equation}\nand \\[ e \\].")
    kinds = [(m.display, m.content) for m in d.math]
    assert (False, "a+b") in kinds and (True, "c=d") in kinds and (True, "e") in kinds


def test_escaped_dollar_is_not_math():
    assert find_math(r"It costs \$5 and \$6.") == []


def test_ref_pattern_includes_project_macros(doc):
    d = doc("See \\figref{fig:a}.", preamble="\\newcommand{\\figref}[1]{Figure~\\ref{#1}}")
    assert [m.group(1) for m in d.ref_pattern.finditer(d.text)] == ["fig:a"]


def test_floats_have_captions_and_labels(doc):
    d = doc("\\begin{figure}\\caption{A {nested} caption.}\\label{fig:x}\\end{figure}")
    f = d.floats[0]
    assert f.label == "fig:x" and f.caption == "A {nested} caption." and f.kind == "figure"


def test_directory_loading_finds_main(tmp_path):
    (tmp_path / "notes.tex").write_text("no class here")
    (tmp_path / "paper.tex").write_text("\\documentclass{article}\\begin{document}x\\end{document}")
    assert load(tmp_path).text.startswith("\\documentclass")
