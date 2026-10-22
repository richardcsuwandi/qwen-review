from conftest import FakeLLM
from qwen_review import claims

TABLE = "\\begin{table}\\caption{Regret.}\\label{tab:r}\\begin{tabular}{lc}A & 0.5 \\\\ B & 0.2\\end{tabular}\\end{table}"


def test_undefined_ref_and_unreferenced_float(doc):
    d = doc("See \\ref{fig:nope}.\n" + TABLE)
    out = claims.check(d)
    assert any(f.severity == "error" and "fig:nope" in f.message for f in out)
    assert any("tab:r is never referenced" in f.message for f in out)


def test_referenced_float_is_quiet(doc):
    out = claims.check(doc("See Table~\\ref{tab:r}.\n" + TABLE))
    assert not any("never referenced" in f.message for f in out)


def test_project_ref_macros_count_as_references(doc):
    d = doc("See \\tabref{tab:r}.\n" + TABLE, preamble="\\newcommand{\\tabref}[1]{Table~\\ref{#1}}")
    assert not any("never referenced" in f.message for f in claims.check(d))


def test_claim_with_ref_or_cite_is_fine(doc):
    body = "Our method outperforms all baselines (Table~\\ref{tab:r}).\n" + TABLE
    assert not any(f.check == "claim" for f in claims.check(doc(body), FakeLLM()))
    assert not any(f.check == "claim" for f in claims.check(doc("Our method outperforms prior work~\\cite{x}.")))


def test_expected_improvement_is_not_a_claim(doc):
    out = claims.check(doc("We use expected improvement to pick the next point."), FakeLLM())
    assert not any(f.check == "claim" for f in out)


def test_unsupported_claim_is_a_warning_with_the_model(doc):
    llm = FakeLLM()
    out = claims.check(doc("Our method achieves state-of-the-art results.\n" + TABLE), llm)
    assert any(f.severity == "warn" and "state-of-the-art" in f.message for f in out)
    assert llm.claims


def test_supported_claim_suggests_a_ref(doc):
    llm = FakeLLM(support={"supported": True, "label": "tab:r", "reason": "table"})
    out = claims.check(doc("Our method outperforms the baseline by a wide margin.\n" + TABLE), llm)
    assert any(f.severity == "info" and "add a \\ref" in f.message for f in out)


def test_numeric_claim_absent_from_tables_is_a_warning(doc):
    out = claims.check(doc("Our method reduces regret by 41\\% on average.\n" + TABLE), FakeLLM())
    assert any(f.severity == "warn" and "41%" in f.message for f in out)


def test_no_model_collapses_claims_into_one_finding(doc):
    body = "Our method outperforms A. Our method outperforms B. Our method outperforms C."
    out = [f for f in claims.check(doc(body)) if f.check == "claim"]
    assert len(out) == 1 and "3 claim sentences" in out[0].message


def test_related_work_claims_are_skipped(doc):
    body = "\\section{Related Work}\nSmith et al. outperform earlier methods on every task.\n"
    assert not [f for f in claims.check(doc(body), FakeLLM()) if f.check == "claim"]
