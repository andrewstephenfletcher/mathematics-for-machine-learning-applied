# Chapter 7 — Continuous Optimisation

## Goals

**Maths.** Gradient descent and step size, momentum, stochastic gradient descent, constrained optimisation
and Lagrange multipliers, convexity, duality.

**Safety connection.** Adversarial attacks are constrained optimisation: maximise a loss inside an L-p norm ball, which
is what PGD does by alternating a gradient step with a projection (Ch 3 again). Probe training is
convex, so its optimum is well defined; network training is not, which is why "the" solution is
not a meaningful object. Soft-prompt optimisation to trigger or evade a behaviour is the same
machinery pointed at the input.

## Sections

Section numbers, titles and page ranges taken from the PDF in `book/`
(2024-01-15 draft, 417 pages). Printed page numbers first, PDF page numbers in
brackets — the offset is +6.

- [ ] **7.1** Optimization Using Gradient Descent — pp. 227–232 (PDF 233–238) — `notebooks/7_1_*.ipynb`
- [ ] **7.2** Constrained Optimization and Lagrange Multipliers — pp. 233–235 (PDF 239–241) — `notebooks/7_2_*.ipynb`
- [ ] **7.3** Convex Optimization — pp. 236–245 (PDF 242–251) — `notebooks/7_3_*.ipynb`

Not exercise notebooks, but worth a pass: **7.4 Further Reading** — p. 246 (PDF 252) and the chapter's **Exercises** — pp. 247–248 (PDF 253–254).

## Mini-project

Implement SGD, momentum and Adam from scratch. Then implement PGD to optimise an input
perturbation that flips a linear probe's prediction under a norm budget, and plot success rate
against budget.

`project/` — status: not started.

## Watch out for

The book minimises; adversarial work usually maximises a loss — keep the sign explicit. Also note
that projection onto an L-infinity ball is a clamp, onto an L-2 ball a rescale: the "projection"
in PGD is the orthogonal projection of Ch 3 only in the L-2 case.

## Reflections

Filled in as sections are completed. One or two lines per section: what was surprising, what
clicked, what still feels shaky.
