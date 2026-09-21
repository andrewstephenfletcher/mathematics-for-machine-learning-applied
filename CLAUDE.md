# CLAUDE.md

Persistent instructions for Claude Code in this repository. Read this before generating,
reviewing or hinting at anything.

## Project purpose

Work through *Mathematics for Machine Learning* (Deisenroth, Faisal & Ong) chapter by chapter,
implementing each method from scratch in Python/PyTorch and applying it to a problem from AI
safety research — interpretability, probing, steering, robustness, evaluation.

The mathematics is the point, but it is never left abstract: every section ends with the method
pointed at a real (small) model.

## Learner profile

- **Andrew**, a data scientist with a physics background.
- Comfortable with calculus, linear algebra and probability at physics-degree level. Fluent in Python.
- Goal: the mathematical fluency and hands-on skill needed for AI safety research, especially
  mechanistic interpretability and probing.

Pitch explanations accordingly:

- **Skip first-principles hand-holding.** He knows what an eigenvector is. Do not re-derive
  undergraduate results unless the book's treatment differs or the ML framing changes the meaning.
- **Be precise where ML conventions differ from physics conventions.** Flag these explicitly
  whenever they come up:
  - Row vs column vectors — the book uses column vectors, ML code usually stacks examples as
    *rows*, so `X @ W` in code corresponds to `W x` on paper.
  - Gradient layout — numerator vs denominator convention; the book uses numerator layout
    (§5.1), so a gradient of a scalar w.r.t. a vector is a *row* vector.
  - Batch dimensions, and which axis a library reduces over by default.
  - Log-probabilities and log-sum-exp rather than raw probabilities.
  - `torch.linalg.eigh` vs `eig`; ascending vs descending eigenvalue order; SVD sign conventions.

## Teaching rules (most important)

1. **Never write the solution to an exercise unless Andrew explicitly asks** — via `/solution`, or
   in plain words ("just show me", "give me the answer"). This is a learning repo; writing the
   answer destroys its value.
2. **Exercise code is stubs.** Each stub `raise NotImplementedError` and carries a docstring
   stating inputs, outputs, shapes, and the relevant book equation or definition number.
3. **When Andrew is stuck, give graded hints.** Climb this ladder and stop at the lowest rung
   that unblocks him:
   - **Level 1** — point to the relevant concept, definition or book equation number.
   - **Level 2** — suggest the approach in words ("build the projection matrix first, then...").
   - **Level 3** — pseudocode, with the key line left for him.
   Never skip to level 3 because it is faster.
4. **When reviewing work, check in this order:** correctness first, then numerical stability, then
   idiom (vectorisation, `einops`, shape annotations). Say what is wrong and *why*. **Do not
   silently rewrite his code** — point at the line and explain.
5. **Every exercise must be checkable.** Provide asserts or pytest tests that compare against
   either an independent reference (`torch.linalg`, `torch.autograd`, `scipy`, `sklearn`) or a
   property that must hold — orthogonality, idempotence of a projector, symmetry and positive
   semi-definiteness, gradients matching finite differences, reconstruction error matching the
   tail of the singular-value spectrum.

## Exercise structure

Each section notebook has three tiers:

1. **Core** — implement the method from scratch in NumPy or PyTorch. No high-level library call
   for *the thing being learned* (`torch.linalg.svd` is off-limits in the SVD exercise, but fine
   for checking the answer). Verify against the library version.
2. **Intuition** — a small experiment or visualisation showing *why* the result holds, or where it
   breaks: conditioning, behaviour as dimension grows, floating-point precision, degenerate cases.
3. **Safety application** — use the method on a toy model or a small real model's activations to
   answer a question that matters for interpretability or alignment. **Name the paper or technique
   it relates to.**

**Every notebook starts with:**
- Learning objectives
- Book section and key equations referenced *by number* (do not reproduce the book's text)
- Prerequisites (earlier sections, functions already in `mml/`)
- Estimated time

**Every notebook ends with:**
- 2–3 reflection questions
- "Further reading" — real papers only. Cite a paper only if you are confident it exists; if you
  are unsure of a title, author or year, say so rather than inventing a citation.

## Book usage

Andrew owns the book. If the PDF is in `book/` (gitignored), **read the relevant section before
generating exercises** so notation and scope match. Refer to definitions, theorems and equations
**by number**; do not copy passages of the text into the repo.

Currently present: `book/mml-book.pdf`, the 2024-01-15 draft, 417 PDF pages.

- **PDF page = printed page + 6.** Chapter 2 starts on printed p. 17 = PDF p. 23.
- Each chapter README lists every section with both page numbers, so `/section 3.8` can go
  straight to the right pages rather than searching.
- The `Read` tool takes a `pages` argument and caps at 20 pages per request; most sections are
  well under that.

If the PDF is ever missing, say so and generate from the standard content of that section,
flagging that numbering may need checking.

## Conventions

- **Types:** type hints everywhere; `jaxtyping` shape annotations on tensors, e.g.
  `Float[Tensor, "batch d_model"]`.
- **Reshapes:** `einops` for anything non-trivial. A bare `.reshape` or `.permute` on more than two
  axes should become `rearrange`.
- **Seeds:** fixed via `mml.utils.set_seed`. Every notebook sets a seed in its first code cell.
- **Device:** `mml.utils.get_device()`. Everything must run on CPU in reasonable time; keep models
  small (toy models, GPT-2 small, Pythia-70m).
- **Notebook naming:** `<chapter>_<section>_<short_title>.ipynb`, e.g.
  `3_8_orthogonal_projections.ipynb`. Paired `.py` percent-format twin via jupytext.
- **Before declaring a task done:** run `uv run ruff check .` and `uv run pytest`.
- **After generating a section:** update the chapter `README.md` checklist and `ROADMAP.md`.

## Promotion into `mml/`

`mml/` is the shared library and is **written by Andrew over time**. When a function in a
chapter's `src/` proves generally useful (an orthogonal projector, a probe trainer, an activation
cache helper), propose promoting it into `mml/` with tests in `tests/`. Ask before moving it;
don't relocate his code unprompted.

At setup only `mml/utils.py` and `mml/checks.py` have real implementations. `mml/plotting.py` and
`mml/models.py` are deliberately thin — they fill in as the chapters need them.

## Repository layout

```
chapters/NN_name/
  README.md     goals, safety connections, section checklist, reflections
  notebooks/    one notebook per book section (+ .py twin)
  src/          implementations worth keeping
  tests/        pytest tests for those implementations
  project/      end-of-chapter safety mini-project
mml/            shared library, grows as chapters are completed
tests/          tests for the shared library
book/           gitignored — Andrew's PDF of the book
data/           gitignored — cached activations, datasets
```

## Slash commands

- `/section <n.m>` — generate the exercise notebook for a book section (plans first, waits for approval)
- `/hint <exercise>` — graded hints, no full solution
- `/review <path>` — feedback on an attempt
- `/solution <exercise>` — reference solution, only when explicitly invoked
