# Chapter 5 — Vector Calculus

## Goals

**Maths.** Partial derivatives, gradients, Jacobians, gradients of matrices, the chain rule,
backpropagation and automatic differentiation, Hessians, Taylor expansion and linearisation.

**Safety connection.** Gradient-based attribution — saliency, gradient x input, integrated gradients — is just the
first-order term of a Taylor expansion. Attribution patching is a first-order approximation to
activation patching, so its failure modes come straight from linearisation: it is wrong exactly
where the model is most nonlinear in the patched component.

## Sections

Section numbers, titles and page ranges taken from the PDF in `book/`
(2024-01-15 draft, 417 pages). Printed page numbers first, PDF page numbers in
brackets — the offset is +6.

- [ ] **5.1** Differentiation of Univariate Functions — pp. 141–145 (PDF 147–151) — `notebooks/5_1_*.ipynb`
- [ ] **5.2** Partial Differentiation and Gradients — pp. 146–148 (PDF 152–154) — `notebooks/5_2_*.ipynb`
- [ ] **5.3** Gradients of Vector-Valued Functions — pp. 149–154 (PDF 155–160) — `notebooks/5_3_*.ipynb`
- [ ] **5.4** Gradients of Matrices — pp. 155–157 (PDF 161–163) — `notebooks/5_4_*.ipynb`
- [ ] **5.5** Useful Identities for Computing Gradients — pp. 158–158 (PDF 164–164) — `notebooks/5_5_*.ipynb`
- [ ] **5.6** Backpropagation and Automatic Differentiation — pp. 159–163 (PDF 165–169) — `notebooks/5_6_*.ipynb`
- [ ] **5.7** Higher-Order Derivatives — pp. 164–164 (PDF 170–170) — `notebooks/5_7_*.ipynb`
- [ ] **5.8** Linearization and Multivariate Taylor Series — pp. 165–169 (PDF 171–175) — `notebooks/5_8_*.ipynb`

Not exercise notebooks, but worth a pass: **5.9 Further Reading** — p. 170 (PDF 176) and the chapter's **Exercises** — pp. 170–171 (PDF 176–177).

## Mini-project

Build a tiny reverse-mode autodiff engine and verify it against `torch.autograd`. Then implement
attribution patching on an IOI-style prompt, compare it with true activation patching, and explain
where the linear approximation breaks.

`project/` — status: not started.

## Watch out for

Layout convention. The book uses numerator layout, so the gradient of a scalar with respect to a
vector is a *row* vector and Jacobians are `(out, in)`. `torch.autograd.grad` returns a tensor
shaped like the *input*. Both are correct; mixing them silently transposes your chain rule.
`mml.checks.numerical_jacobian` follows the book.

## Reflections

Filled in as sections are completed. One or two lines per section: what was surprising, what
clicked, what still feels shaky.
