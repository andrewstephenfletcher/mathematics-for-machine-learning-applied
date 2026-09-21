# Chapter 12 — Classification with Support Vector Machines

## Goals

**Maths.** Separating hyperplanes, the max-margin principle, hinge loss, primal and dual formulations,
kernels.

**Safety connection.** Margin is a measure of probe robustness: the distance from an activation to the decision boundary
is exactly the perturbation budget needed to flip the probe, which connects straight back to the
PGD work in Ch 7. Linear versus kernel probes sharpens the question of what it means for a
representation to be "linearly available" to the model rather than merely present.

## Sections

Section numbers, titles and page ranges taken from the PDF in `book/`
(2024-01-15 draft, 417 pages). Printed page numbers first, PDF page numbers in
brackets — the offset is +6.

- [ ] **12.1** Separating Hyperplanes — pp. 372–373 (PDF 378–379) — `notebooks/12_1_*.ipynb`
- [ ] **12.2** Primal Support Vector Machine — pp. 374–382 (PDF 380–388) — `notebooks/12_2_*.ipynb`
- [ ] **12.3** Dual Support Vector Machine — pp. 383–387 (PDF 389–393) — `notebooks/12_3_*.ipynb`
- [ ] **12.4** Kernels — pp. 388–389 (PDF 394–395) — `notebooks/12_4_*.ipynb`
- [ ] **12.5** Numerical Solution — pp. 390–391 (PDF 396–397) — `notebooks/12_5_*.ipynb`

Not exercise notebooks, but worth a pass: **12.6 Further Reading** — p. 392 (PDF 398).

## Mini-project

Train logistic and SVM probes on the same activations, relate margin to the minimum perturbation
needed to evade the probe (connecting back to Ch 7), and compare linear against RBF-kernel
probes.

`project/` — status: not started.

## Watch out for

A kernel probe that beats a linear one does not show the model represents the concept better — it
shows *your* classifier is stronger. The interpretability claim is about linear availability, so
the kernel result is a control, not an improvement.

## Reflections

Filled in as sections are completed. One or two lines per section: what was surprising, what
clicked, what still feels shaky.
