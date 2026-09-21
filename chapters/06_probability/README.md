# Chapter 6 — Probability and Distributions

## Goals

**Maths.** Sum and product rules, Bayes' theorem, expectations and covariance, independence, the Gaussian
(marginals, conditionals, products), conjugacy, the exponential family, change of variables.

**Safety connection.** A language model's output *is* a categorical distribution. Entropy, cross-entropy and KL
divergence are how interventions are measured — and the KL penalty is what holds an RLHF policy
near its base model. Gaussian fits to activations give Mahalanobis-distance anomaly detection,
one of the standard baselines for backdoor and OOD detection.

## Sections

Section numbers, titles and page ranges taken from the PDF in `book/`
(2024-01-15 draft, 417 pages). Printed page numbers first, PDF page numbers in
brackets — the offset is +6.

- [ ] **6.1** Construction of a Probability Space — pp. 172–177 (PDF 178–183) — `notebooks/6_1_*.ipynb`
- [ ] **6.2** Discrete and Continuous Probabilities — pp. 178–182 (PDF 184–188) — `notebooks/6_2_*.ipynb`
- [ ] **6.3** Sum Rule, Product Rule, and Bayes' Theorem — pp. 183–185 (PDF 189–191) — `notebooks/6_3_*.ipynb`
- [ ] **6.4** Summary Statistics and Independence — pp. 186–196 (PDF 192–202) — `notebooks/6_4_*.ipynb`
- [ ] **6.5** Gaussian Distribution — pp. 197–204 (PDF 203–210) — `notebooks/6_5_*.ipynb`
- [ ] **6.6** Conjugacy and the Exponential Family — pp. 205–213 (PDF 211–219) — `notebooks/6_6_*.ipynb`
- [ ] **6.7** Change of Variables/Inverse Transform — pp. 214–220 (PDF 220–226) — `notebooks/6_7_*.ipynb`

Not exercise notebooks, but worth a pass: **6.8 Further Reading** — p. 221 (PDF 227) and the chapter's **Exercises** — pp. 222–224 (PDF 228–230).

## Mini-project

Fit class-conditional Gaussians to activations and use Mahalanobis distance to flag
out-of-distribution or backdoor-triggered inputs in a toy model. Measure the KL between clean and
ablated model outputs.

`project/` — status: not started.

## Watch out for

Work in log-space. Every probability a model produces is a logit first; use `log_softmax` and
`logsumexp` rather than normalising then taking a log. A Gaussian log-density needs
`torch.linalg.slogdet` and a `cholesky_solve`, never an explicit `inv`.

## Reflections

Filled in as sections are completed. One or two lines per section: what was surprising, what
clicked, what still feels shaky.
