# Counterfactuals-for-k-means-and-Gaussian-Clustering
This repository provides functionality for the computation of counterfactual explanations for k-means clustering and Gaussian (GMM) clustering models.

Given a factual data point assigned to one cluster, the code computes a counterfactual instance that minimally changes the original point so that it would be assigned to a target cluster instead.

The implementation supports both Euclidean and Mahalanobis distance–based counterfactuals and allows for actionability constraints through feature masks.

## Supported Models

The repository currently supports counterfactual explanation generation for:

### k-means clustering

- Counterfactuals based on distances to cluster centroids  
- Optional Mahalanobis distance formulation  
- Feature-level actionability constraints  

### Gaussian Mixture Models (GMMs)

- Full-covariance Gaussian clusters  
- Cluster priors included in the decision boundary  
- Actionable counterfactuals under plausibility constraints

## How the Explainers Work

This repository contains two demonstration notebooks, one for k-means clustering and one for Gaussian Mixture Models (GMMs), that show how to compute counterfactual explanations using the provided explainers.

### k-means Explainer (Euclidean / Mahalanobis)

**File:** `kmeans_explainer.py`

The k-means counterfactual explainer computes the smallest actionable change required to move a data point from a source cluster to a target cluster.

#### Initialization

The explainer is initialized with:
- `m_source`: centroid of the current (source) cluster
- `m_target`: centroid of the desired (target) cluster

From these centroids, the method defines the **decision boundary** between the two clusters, which corresponds to a hyperplane equidistant from both centroids.

---

#### `compute_counterfactual(y, mask, d_eps_center_dist_ratio)`

This method computes a counterfactual under **Euclidean distance**.

- Splits features into:
  - **Free (actionable)** features where `mask = 1`
  - **Fixed (immutable)** features where `mask = 0`
- Projects the factual point `y` onto the decision hyperplane, modifying **only** the free features.
- The parameter `d_eps_center_dist_ratio` controls how far the counterfactual is pushed past the decision boundary:
  - `0.0`: exactly on the boundary
  - `1.0`: deeper into the target cluster’s region

---

#### `compute_mahalanobis_counterfactual(y, R, d_eps_center_dist_ratio)`

This method computes a counterfactual under a **Mahalanobis distance**.

- Replaces the Euclidean distance with a Mahalanobis distance defined by the positive-definite matrix `R`.
- Solves for the point that is **closest to `y` in Mahalanobis distance** while satisfying the constraint that the target cluster is preferred over the source cluster.
- The parameter `d_eps_center_dist_ratio` again controls the margin beyond the boundary.

---

**In short:**  
For k-means clustering, the explainer finds the **closest actionable point** that crosses from the source cluster’s side of the separating hyperplane to the target cluster’s side.

---

### GMM Explainer (Gaussian Mixture Models)

**File:** `gmm_explainer.py`

The GMM counterfactual explainer computes the minimal actionable change needed for a point to be reassigned from one Gaussian component to another in a Gaussian Mixture Model.

#### Initialization

The explainer is initialized with:
- `m_source`, `m_target`: means of the source and target Gaussian components
- `S_sourse`, `S_target`: covariance matrices of the source and target components
- `prior_s`, `prior_t`: mixture weights (priors) of the source and target components

The explainer operates on the **log-posterior** of each component, combining likelihood and prior information.

---

#### `compute_counterfactual(y, M, epsilon)`

- Splits features into:
  - **Free (actionable)** features where `M = 1`
  - **Fixed (immutable)** features where `M = 0`
- Derives a constraint ensuring that the **posterior probability of the target Gaussian exceeds that of the source Gaussian**.
- The parameter `epsilon ≥ 0` acts as a **plausibility margin**, controlling how confidently the counterfactual should belong to the target component.
- The resulting constrained optimization problem is solved by:
  - Formulating a nonlinear equation in a scalar Lagrange multiplier
  - Solving it numerically using root finding
- The solution is used to compute the optimal values of the free features `z_F`, while fixed features remain unchanged.

---

**In short:**  
For Gaussian Mixture Models, the explainer finds the **closest actionable point** whose probability under the target Gaussian (including priors) exceeds that under the source Gaussian, while respecting a user-defined plausibility margin `epsilon`.

---


### Reference
Georgios Vardakas, Antonia Karra, Evaggelia Pitoura and Aristidis Likas. "Counterfactual explanations for k-means and Gaussian clustering." In IEEE 37th International Conference on Tools with Artificial Intelligence (ICTAI), 2025.



### Acknowledgments
The research project is implemented in the framework of H.F.R.I. call ``Basic research Financing (Horizontal support of all Sciences)'' under the National Recovery and Resilience Plan ``Greece 2.0'' funded by the European Union - NextGenerationEU (H.F.R.I. ProjectNumber: 15940).
