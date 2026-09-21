# Chapter 11 — Density Estimation with Gaussian Mixture Models

## Goals

**Maths.** Gaussian mixture models, maximum likelihood with latent variables, derivation of the EM
algorithm, the latent-variable perspective.

**Safety connection.** Clustering activations to discover distinct model "modes" or behaviours, and to separate
anomalous inputs from typical ones. EM and the ELBO are the groundwork for variational methods,
and the responsibilities of a mixture are a soft, probabilistic version of the hard cluster
assignments usually used to summarise activation structure.

## Sections

Section numbers, titles and page ranges taken from the PDF in `book/`
(2024-01-15 draft, 417 pages). Printed page numbers first, PDF page numbers in
brackets — the offset is +6.

- [ ] **11.1** Gaussian Mixture Model — pp. 349–349 (PDF 355–355) — `notebooks/11_1_*.ipynb`
- [ ] **11.2** Parameter Learning via Maximum Likelihood — pp. 350–359 (PDF 356–365) — `notebooks/11_2_*.ipynb`
- [ ] **11.3** EM Algorithm — pp. 360–362 (PDF 366–368) — `notebooks/11_3_*.ipynb`
- [ ] **11.4** Latent-Variable Perspective — pp. 363–367 (PDF 369–373) — `notebooks/11_4_*.ipynb`

Not exercise notebooks, but worth a pass: **11.5 Further Reading** — p. 368 (PDF 374).

## Mini-project

Implement EM from scratch, fit a GMM to activations from a mix of prompt types, and test whether
the clusters align with semantic categories or with anomalous inputs.

`project/` — status: not started.

## Watch out for

EM on high-dimensional data collapses: a component that captures a single point drives its
covariance to zero and the likelihood to infinity. Regularise the covariance, work with
log-responsibilities via `logsumexp`, and check that the log-likelihood is *monotonically
non-decreasing* every iteration — that is the property EM guarantees, and it makes a perfect
test.

## Reflections

Filled in as sections are completed. One or two lines per section: what was surprising, what
clicked, what still feels shaky.
