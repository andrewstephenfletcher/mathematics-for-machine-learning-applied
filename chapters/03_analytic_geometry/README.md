# Chapter 3 — Analytic Geometry

## Goals

**Maths.** Norms, inner products, cosine similarity, orthogonality, orthonormal bases, Gram-Schmidt,
orthogonal complements, orthogonal projections, rotations.

**Safety connection.** Projections are the core operation behind concept erasure (INLP, LEACE), steering vectors and
the logit lens. Near-orthogonality in high dimensions — the fact that random vectors in `R^768`
are almost always nearly perpendicular — is the geometric root of superposition. Rotation
invariance raises the question of whether a network has a privileged basis at all.

## Sections

Section numbers, titles and page ranges taken from the PDF in `book/`
(2024-01-15 draft, 417 pages). Printed page numbers first, PDF page numbers in
brackets — the offset is +6.

- [ ] **3.1** Norms — pp. 71–71 (PDF 77–77) — `notebooks/3_1_*.ipynb`
- [ ] **3.2** Inner Products — pp. 72–74 (PDF 78–80) — `notebooks/3_2_*.ipynb`
- [ ] **3.3** Lengths and Distances — pp. 75–75 (PDF 81–81) — `notebooks/3_3_*.ipynb`
- [ ] **3.4** Angles and Orthogonality — pp. 76–77 (PDF 82–83) — `notebooks/3_4_*.ipynb`
- [ ] **3.5** Orthonormal Basis — pp. 78–78 (PDF 84–84) — `notebooks/3_5_*.ipynb`
- [ ] **3.6** Orthogonal Complement — pp. 79–79 (PDF 85–85) — `notebooks/3_6_*.ipynb`
- [ ] **3.7** Inner Product of Functions — pp. 80–80 (PDF 86–86) — `notebooks/3_7_*.ipynb`
- [ ] **3.8** Orthogonal Projections — pp. 81–90 (PDF 87–96) — `notebooks/3_8_*.ipynb`
- [ ] **3.9** Rotations — pp. 91–93 (PDF 97–99) — `notebooks/3_9_*.ipynb`

Not exercise notebooks, but worth a pass: **3.10 Further Reading** — p. 94 (PDF 100) and the chapter's **Exercises** — pp. 96–97 (PDF 102–103).

## Mini-project

Find a difference-of-means direction for a binary concept in GPT-2 small activations, project
it out, and measure how a probe's accuracy changes. Then compare against iterative nullspace
projection.

`project/` — status: not started.

## Watch out for

An inner product induces a geometry: "orthogonal" is only meaningful relative to one. Activation
space has no canonical metric, so whitening (Ch 4) changes which directions count as orthogonal —
this is exactly the difference between INLP and LEACE.

## Reflections

Filled in as sections are completed. One or two lines per section: what was surprising, what
clicked, what still feels shaky.
