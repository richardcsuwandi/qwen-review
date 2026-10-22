from qwen_review import acronyms


def msgs(doc, body, **kw):
    return [f.message for f in acronyms.check(doc(body), **kw)]


def test_undefined_acronym_is_reported(doc):
    out = msgs(doc, "We use KRR here. KRR is a baseline.")
    assert any('"KRR" is never defined (2 uses)' in m for m in out)


def test_defined_acronym_is_fine_and_used_before_definition_is_flagged(doc):
    ok = msgs(doc, "Gaussian processes (GPs) are useful. We then use GPs again.")
    assert not any('"GP"' in m and "never" in m for m in ok)
    late = msgs(doc, "We use GP here. Gaussian processes (GP) are useful. GP again.")
    assert any('"GP" is used before it is defined' in m for m in late)


def test_hyphenated_long_form_and_plural_variants(doc):
    out = msgs(doc, "Period-guided graph attention (PGGA) is a baseline. PGGA wins. Gaussian process (GP) and Gaussian processes (GP). GP.")
    assert not any("PGGA" in m for m in out)
    assert not any("defined two different ways" in m for m in out)


def test_conflicting_expansions_are_an_error(doc):
    fs = acronyms.check(doc("Expected improvement (EI) is one. Expected information (EI) is another. EI."))
    assert any(f.severity == "error" and "two different ways" in f.message for f in fs)


def test_allowlist_and_common_terms(doc):
    assert not msgs(doc, "We use a GPU and an API with KRR.", allow=["KRR"])


def test_defined_but_unused_counts_table_uses(doc):
    unused = msgs(doc, "Bayesian optimization (BO) is useful.")
    assert any("defined but never used again" in m for m in unused)
    in_table = msgs(doc, "Bayesian optimization (BO) is useful.\n\\begin{tabular}{l}BO\\end{tabular}")
    assert not any("never used again" in m for m in in_table)


def test_abstract_then_body_definition_is_not_a_redefinition(doc):
    body = "\\begin{abstract}Gaussian processes (GPs) matter.\\end{abstract}\n\\section{Intro}Gaussian processes (GPs) matter. GPs."
    assert not any("more than once" in m for m in msgs(doc, body))


def test_acronym_package_definitions_are_respected(doc):
    d = doc("The ZZZ works. ZZZ.", preamble="\\newacronym{zzz}{ZZZ}{Zed Zed Zed}")
    assert not acronyms.check(d)
