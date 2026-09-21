# Chapter 2 — Linear Algebra

## Goals

**Maths.** Solving linear systems, Gaussian elimination, vector spaces, linear independence, basis
and rank, linear maps, kernel and image, change of basis, affine spaces.

**Safety connection.** A transformer's residual stream is a vector space, and every weight matrix is a linear map
on it. Rank and null space tell you what information a layer *cannot* read — a rank-`r` map
throws away everything outside an `r`-dimensional image, and anything in the kernel is invisible
to whatever reads downstream. The "features as directions" view of interpretability starts here.

## Sections

Section numbers, titles and page ranges taken from the PDF in `book/`
(2024-01-15 draft, 417 pages). Printed page numbers first, PDF page numbers in
brackets — the offset is +6.

- [ ] **2.1** Systems of Linear Equations — pp. 19–21 (PDF 25–27) — `notebooks/2_1_*.ipynb`
- [ ] **2.2** Matrices — pp. 22–26 (PDF 28–32) — `notebooks/2_2_*.ipynb`
- [ ] **2.3** Solving Systems of Linear Equations — pp. 27–34 (PDF 33–40) — `notebooks/2_3_*.ipynb`
- [ ] **2.4** Vector Spaces — pp. 35–39 (PDF 41–45) — `notebooks/2_4_*.ipynb`
- [ ] **2.5** Linear Independence — pp. 40–43 (PDF 46–49) — `notebooks/2_5_*.ipynb`
- [ ] **2.6** Basis and Rank — pp. 44–47 (PDF 50–53) — `notebooks/2_6_*.ipynb`
- [ ] **2.7** Linear Mappings — pp. 48–60 (PDF 54–66) — `notebooks/2_7_*.ipynb`
- [ ] **2.8** Affine Spaces — pp. 61–62 (PDF 67–68) — `notebooks/2_8_*.ipynb`

Not exercise notebooks, but worth a pass: **2.9 Further Reading** — p. 63 (PDF 69) and the chapter's **Exercises** — pp. 64–69 (PDF 70–75).

## Mini-project

Take a trained linear probe (or GPT-2's unembedding) and show that adding any vector from
its null space leaves the output unchanged. Discuss what this implies for probe evasion, and for
how much a probe really "sees".

`project/` — status: not started.

## Watch out for

The book writes vectors as columns and maps as `Ax`. PyTorch stacks examples as *rows*, so
the same map is `X @ A.T` (or `X @ W` for a weight matrix stored transposed). Decide on a
convention in the first notebook and annotate every shape.

## Reflections

Filled in as sections are completed. One or two lines per section: what was surprising, what
clicked, what still feels shaky.
