# qwen-review

A pre-submission checker for LaTeX papers that runs entirely on your machine. It flags undefined acronyms, inconsistent notation, claims with no supporting table or figure, and broken references. Rules do the extraction and Qwen3 on Ollama handles the few judgments that need language understanding, so an unpublished manuscript never leaves your laptop.

```text
$ qwen-review check examples/paper
main.tex:45  error  reference \ref to undefined label fig:missing
main.tex:10  warn   acronym   "AKO" is never defined (10 uses)
main.tex:10  warn   claim     41% is in no table or caption and nothing supports this claim, check how it is derived: "AKO reduces regret by 41\% on average and outperforms all baselines." [qwen]
      Qwen3: No specific percentage or comparison to all baselines is provided in the table.
main.tex:22  warn   claim     claim with no supporting table, figure or theorem: "On the hardest benchmark AKO is significantly faster than random search." [qwen]
      Qwen3: The claim is about speed, but the evidence is about regret.
main.tex:39  warn   reference figure fig:budget is never referenced in the text
sections/method.tex:2  warn   notation  bold x is written in several ways, \mathbf{x} x3, \boldsymbol{x} x1, \bm{x} x1
sections/method.tex:4  warn   notation  "kernel function" may be written as $K$, $k$ [qwen]
      $k$ is kernel function, line 3
      $K$ is kernel function evaluated on the training inputs, line 4
sections/method.tex:10  warn   notation  $n$ is defined with different meanings, lines 3, 10 [qwen]
      "number of observations" (line 3)
      "input dimension" (line 10)
      Qwen3: Symbol 'n' is used for both number of observations and input dimension, implying different meanings.
main.tex:9  info   acronym   "BO" is defined but never used again, so the abbreviation can go
sections/method.tex:10  info   acronym   "EI" is defined more than once in the body (first at line 15)
...
1 error, 11 warns, 5 info
```

The sample in `examples/paper` has these problems planted in it. Findings come with `file:line` positions, and `\input` files are followed.

## Install

```sh
pip install git+https://github.com/richardcsuwandi/qwen-review.git
ollama pull qwen3:8b
```

There are no Python dependencies. Without Ollama the tool still runs the rule-based checks and says so.

## Usage

```sh
qwen-review check main.tex               # one file, following \input and \include
qwen-review check my-paper/              # a project directory, finds the main file
qwen-review check main.tex --no-llm      # rules only
qwen-review check main.tex --checks acronyms,notation
qwen-review check main.tex --allow GP,BO # acronyms that need no definition
qwen-review check main.tex --format markdown > review.md
qwen-review check main.tex --fail-on warn
```

Other flags are `--model` (default `qwen3:8b`), `--format json` and `--max-claims`. The exit code is 1 when a finding reaches `--fail-on`, which defaults to `error`.

## What it checks

### Acronyms (rules only)

- An acronym that is never defined, or is used before its definition.
- One acronym defined with two different expansions, or one term with two abbreviations.
- A definition in the body that repeats an earlier body definition. An abstract definition followed by one in the introduction is accepted.
- An abbreviation that is defined and never used again. Uses in tables and captions count.

Definitions are recognized in the forms `Long Form (ABBR)` and `ABBR (Long Form)`, checked against the initials. Plurals and hyphenated words are handled, `\newacronym` entries are respected, and common terms such as GPU and API need no definition.

### Notation (rules plus Qwen3)

- One symbol defined with different meanings. Rules find the definitions from phrases such as "where $k$ is the kernel", "let $n$ denote" and "we denote the noise by $\sigma$". Qwen3 decides whether two definitions of one symbol actually conflict.
- One concept written with several symbols, such as a kernel that is $k$ in one place and $K$ in another. Exact matches are found by rules, and Qwen3 finds looser semantic matches. Ids it returns must exist in the input.
- Bold vectors written as `\mathbf`, `\boldsymbol` and `\bm` for the same letter, and both `\epsilon` and `\varepsilon` in one paper.

### Claims and references (rules plus Qwen3)

- `\ref` to a label that does not exist, and figures, tables and theorems that nothing refers to. Project macros such as `\figref` that wrap `\ref` are detected.
- Claim sentences such as "outperforms", "state-of-the-art" and "reduces error by 12%" that cite no table, figure, theorem or source. Qwen3 then checks whether the paper's own tables, figures or theorems back the claim.
- Numbers in the abstract, introduction and conclusion that appear in no table or caption.

## How Qwen3 is kept honest

Small local models over-agree, so each answer is verified before it is used.

- A claim counts as supported only if the model names a real label and **quotes words or numbers that appear verbatim in that table, caption or theorem**. An invented quote makes the claim unsupported.
- A number in the claim must also appear in the evidence it is credited to.
- The model is shown table values, so it can tell that a regret table does not support a claim about speed.
- Numeric claims that no table states are warnings only when the number is absent from every table. Derived quantities such as relative improvements are common, so otherwise they stay informational.
- Without a model, the unchecked claims collapse into one summary finding instead of dozens of notes.

## Limits

- The checks are heuristics. Run them as a second pair of eyes before a human read, not as a verdict.
- Claim checking sends at most `--max-claims` sentences to the model, 25 by default. Evidence is each caption and table plus theorem statements, so claims backed only by long prose in the results section may be flagged.
- `qwen3:8b` is the tested default. Smaller models are noticeably more lenient.
- Definitions are found with patterns for English phrasing. Unusual wording will be missed.
- Only LaTeX sources are read, not PDFs.

## Develop

```sh
uv venv && uv pip install -e ".[dev]"
pytest -q
```

The tests use a stub model and need no network or Ollama.

## License

[Apache-2.0](LICENSE).
