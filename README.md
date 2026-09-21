# Mathematics for Machine Learning, Applied

Working through *Mathematics for Machine Learning* (Deisenroth, Faisal & Ong) chapter by chapter,
implementing each method from scratch in Python/PyTorch and pointing it at a problem from AI
safety research — interpretability, probing, steering, robustness, evaluation.

Every section produces one notebook in three tiers:

1. **Core** — implement the method from scratch, verify against `torch.linalg` / `scipy` / `sklearn`.
2. **Intuition** — a small experiment showing why the result holds, or where it breaks.
3. **Safety application** — the method used on a toy model or a small real model's activations,
   tied to a named paper or technique.

See [ROADMAP.md](ROADMAP.md) for the chapter-by-chapter plan and progress, and
[CLAUDE.md](CLAUDE.md) for how Claude Code is meant to behave in this repo (in short: it writes
stubs and hints, not solutions).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                      # core dependencies
uv sync --extra interp       # adds transformer-lens + datasets, for the model-based exercises
uv run pre-commit install    # strips notebook outputs, syncs jupytext twins, runs ruff
```

Check it works:

```bash
uv run pytest
uv run ruff check .
```

Then register the kernel and start JupyterLab:

```bash
uv run python -m ipykernel install --user --name mml-applied
uv run jupyter lab
```

Everything is expected to run on CPU. `mml.utils.get_device()` picks CUDA or MPS when available,
and models stay small — toy models, GPT-2 small, Pythia-70m.

## The book

The book is **not** committed to this repo; `book/` is gitignored, so your PDF stays local:

```bash
cp ~/path/to/mml-book.pdf book/
```

`book/mml-book.pdf` is already in place (2024-01-15 draft, 417 pages). The chapter READMEs list
every section with its page range — printed page numbers and PDF page numbers, which differ by a
constant +6 — so `/section` reads the right pages before writing anything, and exercises match the
book's notation and equation numbers. If you swap in a different edition, re-check that offset.

`data/` is also gitignored — cached activations and downloaded datasets live there.

## How to work through a section

```
/section 3.8
```
Claude reads the book section, the chapter README and the roadmap, then **proposes a plan** —
objectives plus a list of exercises across the three tiers — and waits for approval. On approval
it writes the notebook (with its jupytext `.py` twin), stubs in `src/` that raise
`NotImplementedError`, and tests in `tests/`. It does not write solutions.

Then work through the notebook. When stuck:

```
/hint 3.8 ex2           # level 1: points at the concept or equation
/hint 3.8 ex2 level 2   # the approach in words
/hint 3.8 ex2 level 3   # pseudocode, key step left to you
```

When an attempt is ready:

```
/review chapters/03_analytic_geometry/src/projections.py
```
Runs the tests and gives feedback on correctness, then numerical stability, then idiom — without
rewriting the code.

And only if you actually want the answer:

```
/solution 3.8 ex2
```
Writes a commented reference solution into that chapter's `solutions/` folder, leaving your
attempt untouched.

## Layout

```
chapters/NN_name/
  README.md     goals, safety connections, section checklist, reflections
  notebooks/    one notebook per book section (+ .py twin)
  src/          implementations worth keeping
  tests/        pytest tests for those implementations
  project/      end-of-chapter safety mini-project
mml/            shared library — grows as chapters are completed
tests/          tests for the shared library
book/           gitignored — your PDF of the book
data/           gitignored — cached activations, datasets
```

`mml/` starts with `utils.py` (device, seeding, timing) and `checks.py` (`assert_close`, shape
checks, finite-difference gradient checks). `plotting.py` and `models.py` are deliberately thin
and fill in as the chapters need them. When something written in a chapter's `src/` proves
generally useful, promote it into `mml/` with tests.

## Notebooks

Notebooks are paired with percent-format `.py` twins via jupytext, so diffs are readable and the
`.py` side can be edited directly. `nbstripout` runs in pre-commit, so outputs are never
committed. If the two sides ever drift:

```bash
uv run jupytext --sync chapters/**/notebooks/*.ipynb
```
