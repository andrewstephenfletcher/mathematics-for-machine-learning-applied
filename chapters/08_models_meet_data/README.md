# Chapter 8 — When Models Meet Data

## Goals

**Maths.** Empirical risk minimisation, maximum likelihood and MAP estimation, regularisation,
cross-validation, model selection, directed graphical models.

**Safety connection.** This is the chapter that makes probing *rigorous*. A probe that reaches 90% accuracy tells you
nothing without a control task: if a probe trained on random labels also reaches 85%, the probe is
memorising, not reading. Selectivity is the difference. Graphical models are the language in which
causal interventions and causal abstraction are stated.

## Sections

Section numbers, titles and page ranges taken from the PDF in `book/`
(2024-01-15 draft, 417 pages). Printed page numbers first, PDF page numbers in
brackets — the offset is +6.

- [ ] **8.1** Data, Models, and Learning — pp. 251–257 (PDF 257–263) — `notebooks/8_1_*.ipynb`
- [ ] **8.2** Empirical Risk Minimization — pp. 258–264 (PDF 264–270) — `notebooks/8_2_*.ipynb`
- [ ] **8.3** Parameter Estimation — pp. 265–271 (PDF 271–277) — `notebooks/8_3_*.ipynb`
- [ ] **8.4** Probabilistic Modeling and Inference — pp. 272–277 (PDF 278–283) — `notebooks/8_4_*.ipynb`
- [ ] **8.5** Directed Graphical Models — pp. 278–282 (PDF 284–288) — `notebooks/8_5_*.ipynb`
- [ ] **8.6** Model Selection — pp. 283–288 (PDF 289–294) — `notebooks/8_6_*.ipynb`

## Mini-project

Build a probe evaluation harness with train/val/test splits, a random-label control task and a
selectivity score, and use it to compare probes across layers.

`project/` — status: not started.

## Watch out for

Activations from the same prompt are not independent samples. Splitting at the token level leaks
across train and test; split by prompt. This is the single most common way a probing result turns
out to be nothing.

## Reflections

Filled in as sections are completed. One or two lines per section: what was surprising, what
clicked, what still feels shaky.
