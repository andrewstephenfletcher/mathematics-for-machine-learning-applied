# Chapter 10 — Dimensionality Reduction with PCA

## Goals

**Maths.** The maximum-variance and minimum-reconstruction-error perspectives, eigenvector computation,
PCA in high dimensions, probabilistic PCA, the link to autoencoders.

**Safety connection.** PCA is the first tool anyone reaches for to visualise representation geometry — and it fails
under superposition, because the features a model packs into a space are more numerous than its
dimensions and are not orthogonal. That failure is precisely what motivates sparse autoencoders
and dictionary learning.

## Sections

Section numbers, titles and page ranges taken from the PDF in `book/`
(2024-01-15 draft, 417 pages). Printed page numbers first, PDF page numbers in
brackets — the offset is +6.

- [ ] **10.1** Problem Setting — pp. 318–319 (PDF 324–325) — `notebooks/10_1_*.ipynb`
- [ ] **10.2** Maximum Variance Perspective — pp. 320–324 (PDF 326–330) — `notebooks/10_2_*.ipynb`
- [ ] **10.3** Projection Perspective — pp. 325–332 (PDF 331–338) — `notebooks/10_3_*.ipynb`
- [ ] **10.4** Eigenvector Computation and Low-Rank Approximations — pp. 333–334 (PDF 339–340) — `notebooks/10_4_*.ipynb`
- [ ] **10.5** PCA in High Dimensions — pp. 335–335 (PDF 341–341) — `notebooks/10_5_*.ipynb`
- [ ] **10.6** Key Steps of PCA in Practice — pp. 336–338 (PDF 342–344) — `notebooks/10_6_*.ipynb`
- [ ] **10.7** Latent Variable Perspective — pp. 339–342 (PDF 345–348) — `notebooks/10_7_*.ipynb`

Not exercise notebooks, but worth a pass: **10.8 Further Reading** — p. 343 (PDF 349).

## Mini-project

Reproduce a toy-models-of-superposition setup, then compare PCA with a small sparse autoencoder
at recovering the ground-truth features.

`project/` — status: not started.

## Watch out for

PCA requires centring, and on activations the mean is not a nuisance — it often carries most of
the norm. Decide deliberately whether to centre, and report it. Also: with `n < d` (fewer prompts
than dimensions) the covariance is rank deficient; use the SVD of the centred data, not an
eigendecomposition of a `768 x 768` covariance estimated from 200 points.

## Reflections

Filled in as sections are completed. One or two lines per section: what was surprising, what
clicked, what still feels shaky.
