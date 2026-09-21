# Chapter 9 — Linear Regression

## Goals

**Maths.** Least squares, the normal equations, maximum likelihood as orthogonal projection, ridge as MAP,
Bayesian linear regression and predictive uncertainty.

**Safety connection.** Linear probes for continuous quantities — board position, numeric values, dates, sentiment
intensity — in residual-stream activations. The Bayesian version gives calibrated uncertainty on a
probe readout, which matters when the probe is being used as a monitor rather than as evidence
about representation.

## Sections

Section numbers, titles and page ranges taken from the PDF in `book/`
(2024-01-15 draft, 417 pages). Printed page numbers first, PDF page numbers in
brackets — the offset is +6.

- [ ] **9.1** Problem Formulation — pp. 291–291 (PDF 297–297) — `notebooks/9_1_*.ipynb`
- [ ] **9.2** Parameter Estimation — pp. 292–302 (PDF 298–308) — `notebooks/9_2_*.ipynb`
- [ ] **9.3** Bayesian Linear Regression — pp. 303–312 (PDF 309–318) — `notebooks/9_3_*.ipynb`
- [ ] **9.4** Maximum Likelihood as Orthogonal Projection — pp. 313–314 (PDF 319–320) — `notebooks/9_4_*.ipynb`

Not exercise notebooks, but worth a pass: **9.5 Further Reading** — p. 315 (PDF 321).

## Mini-project

Regress a continuous attribute from residual-stream activations layer by layer, compare OLS /
ridge / Bayesian versions, and show that the MLE solution is a projection onto the column space.

`project/` — status: not started.

## Watch out for

Never form `(X.T @ X).inverse()`. It squares the condition number; with `d_model = 768` and a few
hundred prompts the design matrix is wide and the normal equations are singular. Use
`torch.linalg.lstsq` (QR) or the SVD, and note that ridge is what makes the wide case well posed.

## Reflections

Filled in as sections are completed. One or two lines per section: what was surprising, what
clicked, what still feels shaky.
