from conftest import FakeLLM
from qwen_review import notation


def test_definitions_are_extracted(doc):
    d = doc("Let $f$ be the objective function. Here $k$ is the kernel function, where $n$ denotes the number of points. "
            "We denote the noise variance by $\\sigma^2$.")
    got = {x.symbol: x.desc for x in notation.definitions(d)}
    assert got["f"] == "objective function"
    assert got["k"] == "kernel function"
    assert "number of points" in got["n"]
    assert got["\\sigma^2"] == "noise variance"


def test_bold_style_mixing_is_flagged(doc):
    d = doc("We have $\\mathbf{x}$ and $\\boldsymbol{x}$ and $\\bm{y}$ and $\\mathbf{y}$.")
    out = notation.check(d)
    assert any("bold x" in f.message for f in out)
    assert any("bold y" in f.message for f in out)


def test_glyph_variants(doc):
    d = doc("Noise $\\epsilon$ and also $\\varepsilon$.")
    assert any("epsilon" in f.message and "varepsilon" in f.message for f in notation.check(d))


def test_conflicting_definitions_need_the_model_or_low_overlap(doc):
    body = "Here $n$ is the number of observations. Later $n$ is the input dimension."
    with_model = notation.check(doc(body), FakeLLM(conflict=True))
    assert any(f.by == "qwen" and "different meanings" in f.message for f in with_model)
    benign = notation.check(doc(body), FakeLLM(conflict=False))
    assert not any("different meanings" in f.message for f in benign)
    rules_only = notation.check(doc(body))
    assert any("unconfirmed" in f.message for f in rules_only)


def test_same_concept_with_two_symbols_exact_and_model(doc):
    body = "Here $k$ is the kernel function. Later $K$ is the kernel function."
    out = notation.check(doc(body))
    assert any('"kernel function" is written as' in f.message for f in out)
    body2 = "Here $k$ is the kernel. Later $K$ denotes the covariance function."
    d = doc(body2)
    out2 = notation.check(d, FakeLLM(groups=[{"concept": "kernel", "ids": [1, 2]}]))
    assert any(f.by == "qwen" and "kernel" in f.message for f in out2)
