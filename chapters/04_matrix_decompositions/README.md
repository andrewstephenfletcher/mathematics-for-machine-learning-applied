# Chapter 4 — Matrix Decompositions

## Goals

**Maths.** Determinant and trace, eigenvalues and eigenvectors, Cholesky decomposition, eigendecomposition
and diagonalisation, SVD, low-rank approximation (Eckart-Young).

**Safety connection.** SVD of attention OV and QK circuits is the "Mathematical Framework for Transformer Circuits"
view of a head: what it reads and what it writes, stripped of the basis. Effective rank measures
how many directions a representation really uses; LoRA is a deliberate low-rank update; Cholesky
gives cheap whitening for Mahalanobis distance in Ch 6.

## Sections

Section numbers, titles and page ranges taken from the PDF in `book/`
(2024-01-15 draft, 417 pages). Printed page numbers first, PDF page numbers in
brackets — the offset is +6.

- [ ] **4.1** Determinant and Trace — pp. 99–104 (PDF 105–110) — `notebooks/4_1_*.ipynb`
- [ ] **4.2** Eigenvalues and Eigenvectors — pp. 105–113 (PDF 111–119) — `notebooks/4_2_*.ipynb`
- [ ] **4.3** Cholesky Decomposition — pp. 114–114 (PDF 120–120) — `notebooks/4_3_*.ipynb`
- [ ] **4.4** Eigendecomposition and Diagonalization — pp. 115–118 (PDF 121–124) — `notebooks/4_4_*.ipynb`
- [ ] **4.5** Singular Value Decomposition — pp. 119–128 (PDF 125–134) — `notebooks/4_5_*.ipynb`
- [ ] **4.6** Matrix Approximation — pp. 129–133 (PDF 135–139) — `notebooks/4_6_*.ipynb`
- [ ] **4.7** Matrix Phylogeny — pp. 134–134 (PDF 140–140) — `notebooks/4_7_*.ipynb`

Not exercise notebooks, but worth a pass: **4.8 Further Reading** — p. 135 (PDF 141) and the chapter's **Exercises** — pp. 137–138 (PDF 143–144).

## Mini-project

SVD of GPT-2 small OV matrices; project the top singular vectors through the unembedding and
inspect which tokens they promote. Measure how much of each head is captured by a rank-k
approximation.

`project/` — status: not started.

## Watch out for

`torch.linalg.eigh` (symmetric) returns eigenvalues *ascending*; `eig` does not sort at all and
returns complex tensors. SVD singular vectors are only defined up to a sign, and up to an
arbitrary rotation within a repeated-singular-value subspace — so never compare `U` to a reference
`U` elementwise; compare subspaces or reconstructions.

## Reflections

Filled in as sections are completed. One or two lines per section: what was surprising, what
clicked, what still feels shaky.
