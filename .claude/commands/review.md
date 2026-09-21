---
description: Review my attempt at an exercise
argument-hint: <path to notebook, .py twin or src file>
---
Review my attempt at: $ARGUMENTS

1. Read the file, plus the stub docstrings and tests it is meant to satisfy.
2. Run the relevant tests (`uv run pytest <chapter tests path>`) and `uv run ruff check` on the
   file. Report what actually passed and what failed, with the output.
3. Give feedback in this order:
   - **Correctness** — does it compute what the book's definition says? Check edge cases:
     non-square matrices, rank deficiency, repeated eigenvalues, batch dimensions, empty inputs.
   - **Numerical stability** — catastrophic cancellation, explicit inverses where a `solve` would
     do, normal equations where QR/SVD is safer, missing log-sum-exp, tolerances that are too
     tight or too loose for float32.
   - **Idiom** — vectorisation over Python loops, `einops` for non-trivial reshapes, jaxtyping
     annotations, seeds, naming.
4. Point out what is wrong and why — name the line and the invariant it breaks. **Do not silently
   rewrite my code.** If a fix needs showing, show the smallest possible fragment and say what it
   changes.
5. Call out what I got *right*, especially where I chose a more stable or more general approach
   than the obvious one.
6. Finish with a suggested next step: the next exercise, or an extension that probes whether I
   really understood it (a harder case, a degenerate input, a connection to a later chapter).
