# Roadmap

Chapter-by-chapter goals for *Mathematics for Machine Learning* (Deisenroth, Faisal & Ong),
each paired with the AI safety work it unlocks. Chapter 1 is motivational and gets no directory.

These are starting points, not a contract — refine them as the work makes the real questions
clearer. Update the progress table and tick the section boxes in each chapter README as sections
are generated and completed.

## Progress

| Ch | Title | Sections done | Mini-project | Status |
|----|-------|---------------|--------------|--------|
| 2  | [Linear Algebra](chapters/02_linear_algebra/) | 0 | ☐ | Not started |
| 3  | [Analytic Geometry](chapters/03_analytic_geometry/) | 0 | ☐ | Not started |
| 4  | [Matrix Decompositions](chapters/04_matrix_decompositions/) | 0 | ☐ | Not started |
| 5  | [Vector Calculus](chapters/05_vector_calculus/) | 0 | ☐ | Not started |
| 6  | [Probability and Distributions](chapters/06_probability/) | 0 | ☐ | Not started |
| 7  | [Continuous Optimisation](chapters/07_continuous_optimization/) | 0 | ☐ | Not started |
| 8  | [When Models Meet Data](chapters/08_models_meet_data/) | 0 | ☐ | Not started |
| 9  | [Linear Regression](chapters/09_linear_regression/) | 0 | ☐ | Not started |
| 10 | [Dimensionality Reduction with PCA](chapters/10_pca/) | 0 | ☐ | Not started |
| 11 | [Density Estimation with GMMs](chapters/11_gmm_em/) | 0 | ☐ | Not started |
| 12 | [Classification with SVMs](chapters/12_svm/) | 0 | ☐ | Not started |

---

## Ch 2 — Linear Algebra

**Maths.** Solving linear systems, Gaussian elimination, vector spaces, linear independence, basis
and rank, linear maps, kernel and image, change of basis, affine spaces.

**Safety connection.** A transformer's residual stream is a vector space, and every weight matrix
is a linear map on it. Rank and null space tell you what information a layer *cannot* read. The
"features as directions" view of interpretability starts here.

**Mini-project.** Take a trained linear probe (or GPT-2's unembedding) and show that adding any
vector from its null space leaves the output unchanged. Discuss what this implies for probe
evasion, and for how much a probe really "sees".

## Ch 3 — Analytic Geometry

**Maths.** Norms, inner products, cosine similarity, orthogonality, orthonormal bases,
Gram–Schmidt, orthogonal complements, orthogonal projections, rotations.

**Safety connection.** Projections are the core operation behind concept erasure (INLP, LEACE),
steering vectors and the logit lens. Near-orthogonality in high dimensions is the geometric root
of superposition. Rotation invariance raises the question of privileged bases.

**Mini-project.** Find a difference-of-means direction for a binary concept in GPT-2 small
activations, project it out, and measure how a probe's accuracy changes. Then compare against
iterative nullspace projection.

## Ch 4 — Matrix Decompositions

**Maths.** Determinant and trace, eigendecomposition, Cholesky, diagonalisation, SVD, low-rank
approximation (Eckart–Young).

**Safety connection.** SVD of attention OV and QK circuits (the "Mathematical Framework for
Transformer Circuits" view), effective rank of representations, LoRA as a low-rank update,
whitening via Cholesky.

**Mini-project.** SVD of GPT-2 small OV matrices; project the top singular vectors through the
unembedding and inspect which tokens they promote. Measure how much of each head is captured by a
rank-k approximation.

## Ch 5 — Vector Calculus

**Maths.** Partial derivatives, gradients, Jacobians, gradients of matrices, chain rule,
backpropagation and automatic differentiation, Hessians, Taylor expansion and linearisation.

**Safety connection.** Gradient-based attribution (saliency, gradient × input, integrated
gradients). Attribution patching is a first-order Taylor approximation of activation patching, so
its failure modes come straight from linearisation.

**Mini-project.** Build a tiny reverse-mode autodiff engine and verify it against
`torch.autograd`. Then implement attribution patching on an IOI-style prompt, compare it with true
activation patching, and explain where the linear approximation breaks.

## Ch 6 — Probability and Distributions

**Maths.** Sum and product rules, Bayes' theorem, expectations and covariance, independence,
Gaussians (marginals, conditionals, products), conjugacy, the exponential family, change of
variables.

**Safety connection.** A language model outputs a categorical distribution; entropy, cross-entropy
and KL divergence are the standard ways to measure how an intervention changes behaviour — and the
KL penalty is what holds an RLHF policy near its base model. Gaussian fits to activations enable
Mahalanobis-distance anomaly detection.

**Mini-project.** Fit class-conditional Gaussians to activations and use Mahalanobis distance to
flag out-of-distribution or backdoor-triggered inputs in a toy model. Measure the KL between clean
and ablated model outputs.

## Ch 7 — Continuous Optimisation

**Maths.** Gradient descent, step size, momentum, stochastic gradient descent, constrained
optimisation and Lagrange multipliers, convexity, duality.

**Safety connection.** Adversarial attacks as constrained optimisation (PGD in norm balls). Probe
training is convex while network training is not. Optimising inputs or soft prompts to trigger or
evade a behaviour.

**Mini-project.** Implement SGD, momentum and Adam from scratch. Then implement PGD to optimise an
input perturbation that flips a linear probe's prediction under a norm budget, and plot success
rate against budget.

## Ch 8 — When Models Meet Data

**Maths.** Empirical risk minimisation, MLE and MAP, regularisation, cross-validation, model
selection, directed graphical models.

**Safety connection.** Rigorous probe evaluation — overfitting, control tasks and selectivity,
baselines. Graphical models are the language of causal interventions and causal abstraction in
interpretability.

**Mini-project.** Build a probe evaluation harness with train/val/test splits, a random-label
control task and a selectivity score, and use it to compare probes across layers.

## Ch 9 — Linear Regression

**Maths.** Least squares, normal equations, MLE as orthogonal projection, ridge as MAP, Bayesian
linear regression and predictive uncertainty.

**Safety connection.** Linear probes for continuous quantities — position, numeric values, dates —
in activations, and uncertainty in probe readouts.

**Mini-project.** Regress a continuous attribute from residual-stream activations layer by layer,
compare OLS / ridge / Bayesian versions, and show that the MLE solution is a projection onto the
column space.

## Ch 10 — Dimensionality Reduction with PCA

**Maths.** Maximum-variance and minimum-reconstruction-error views, eigenvector computation, PCA
in high dimensions, probabilistic PCA, the link to autoencoders.

**Safety connection.** PCA is the first tool for visualising representation geometry, but it fails
under superposition — which is exactly what motivates sparse autoencoders and dictionary learning.

**Mini-project.** Reproduce a toy-models-of-superposition setup, then compare PCA with a small
sparse autoencoder at recovering the ground-truth features.

## Ch 11 — Density Estimation with Gaussian Mixture Models

**Maths.** Gaussian mixture models, maximum likelihood with latent variables, derivation of the EM
algorithm, the latent-variable perspective.

**Safety connection.** Clustering activations to discover distinct model "modes" or behaviours. EM
and the ELBO are the groundwork for variational methods.

**Mini-project.** Implement EM from scratch, fit a GMM to activations from a mix of prompt types,
and test whether the clusters align with semantic categories or with anomalous inputs.

## Ch 12 — Classification with Support Vector Machines

**Maths.** Separating hyperplanes, max-margin, hinge loss, primal and dual formulations, kernels.

**Safety connection.** Margin as a measure of probe robustness. Linear versus kernel probes, and
the question of what it means for a representation to be "linearly available".

**Mini-project.** Train logistic and SVM probes on the same activations, relate margin to the
minimum perturbation needed to evade the probe (connecting back to Ch 7), and compare linear
against RBF-kernel probes.
