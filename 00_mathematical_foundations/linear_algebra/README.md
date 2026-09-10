# Singular Value Decomposition (SVD) & Low-Rank Projection

A first-principles implementation of Singular Value Decomposition (SVD) using NumPy. This module covers core matrix factorization, coordinate subspace projection, low-rank reconstruction, and the theoretical proof-verification of the Eckart–Young–Mirsky Theorem.

---

## 1. What is SVD?

Singular Value Decomposition states that **any** real matrix $X \in \mathbb{R}^{m \times n}$ (regardless of whether it is square, rectangular, full-rank, or singular) can be factorized into three constituent linear transformations:

$$X = U \Sigma V^T$$

### Geometric Interpretation: What is Happening?
Every linear transformation maps a geometric shape (such as a unit hyper-sphere) into an aligned or rotated hyper-ellipse. SVD splits that transformation into three sequential geometric operations:
1. **$V^T$ (Initial Rotation/Reflection):** Rotates the coordinate system in the input space $\mathbb{R}^n$ to align with the axes of maximal variance.
2. **$\Sigma$ (Axis-Aligned Scaling):** Stretches or compresses the vectors along each coordinate axis by a factor of $\sigma_i$.
3. **$U$ (Final Rotation/Reflection):** Rotates the scaled ellipse into the orientation of the output space $\mathbb{R}^m$.

---

## 2. Term-by-Term Component Breakdown

Let our data matrix be $X \in \mathbb{R}^{m \times n}$, where:
* $m$ is the number of observations (rows/samples).
* $n$ is the number of features (columns/variables).
* $r = \min(m, n)$ is the maximum possible rank under compact (thin) SVD.

| Symbol | Name | Dimensions | Explicit Mathematical Properties | Intuitive Role in Machine Learning |
| :--- | :--- | :--- | :--- | :--- |
| **$X$** | Data Matrix | $m \times n$ | Arbitrary real entries $X_{ij} \in \mathbb{R}$. | Raw dataset where each row is an instance and each column is a recorded attribute. |
| **$U$** | Left Singular Vectors | $m \times r$ | Columns are orthonormal: $U^T U = I_r$. They are the eigenvectors of $X X^T$. | **Sample-space geometry.** Expresses how individual data points (rows) relate to the global latent patterns. |
| **$\Sigma$** | Singular Values Matrix | $r \times r$ | Diagonal matrix where $\Sigma_{ii} = \sigma_i \ge 0$ and $\Sigma_{ij} = 0$ for $i \neq j$. Sorted: $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$. They are the square roots of the eigenvalues of $X^T X$ (or $X X^T$). | **Energy and scale.** Quantifies the amount of variance or "signal" present along each principal axis. A small $\sigma_i$ means the corresponding axis contains mostly noise. |
| **$V^T$** | Right Singular Vectors (Transposed) | $r \times n$ | Rows of $V^T$ (columns of $V$) are orthonormal: $V^T V = I_r$. They are the eigenvectors of $X^T X$. | **Feature-space orientation.** Defines the directions (linear combinations of the original $n$ features) along which the data varies the most. |

---

## 3. Subspace Projection vs. Ambient Reconstruction

In dimensionality reduction, it is critical to distinguish between **coordinates in a reduced space** and **reconstruction in the original space**.

### Step 1: Projection onto the Subspace (Dimensionality Reduction)
We take our original data $X \in \mathbb{R}^{m \times n}$ and project it onto the top-$k$ principal axes ($V_k \in \mathbb{R}^{n \times k}$):

$$Z = X V_k = U_k \Sigma_k \in \mathbb{R}^{m \times k}$$

* **$Z$** is the **compressed representation**.
* Each row is now represented using only $k$ numbers instead of $n$ numbers ($k \ll n$).
* This is the exact representation passed into downstream models (e.g., clustering or classification).

### Step 2: Reconstruction Back to Ambient Space (Approximation)
To inspect the compressed data or evaluate information loss, we lift the low-dimensional coordinates $Z$ back into the original $n$-dimensional ambient space:

$$X_k = Z V_k^T = U_k \Sigma_k V_k^T \in \mathbb{R}^{m \times n}$$

* **$X_k$** has the same shape as $X$ ($m \times n$), but its mathematical rank is strictly $k$.
* Any variation along the remaining $(r - k)$ dimensions is discarded.

---

## 4. The Eckart–Young–Mirsky Theorem

When truncating a matrix to rank $k$, why do we retain the top-$k$ singular components rather than using another linear projection?

The **Eckart–Young–Mirsky Theorem** proves that the truncated SVD matrix $X_k = U_k \Sigma_k V_k^T$ is the **unique global minimizer** of the reconstruction error among all possible matrices of rank at most $k$:

$$\min_{\text{rank}(B) \le k} \Vert{}X - B\Vert{}_F = \Vert{}X - X_k\Vert{}_F$$

### Understanding the Error Metrics

**1. Empirical Frobenius Error**
The Frobenius norm represents the total Euclidean discrepancy across every single cell in the matrix:

$$\Vert{}X - X_k\Vert{}_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n \left( X_{ij} - (X_k)_{ij} \right)^2}$$

**2. Theoretical Eckart–Young Value**
The theorem guarantees that you do not need to calculate every matrix element difference to know the reconstruction error. The error is identical to the root sum of squares of the omitted singular values:

$$\text{Theoretical Error} = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$

In this implementation, the script asserts:

$$\Vert{}X - X_k\Vert{}_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2} \quad (\text{within numerical tolerance } \epsilon \approx 10^{-12})$$

---

## 5. Preserved Energy (Variance Ratio)

The "energy" (total sum of squared variance) of matrix $X$ is defined as the squared Frobenius norm:

$$\Vert{}X\Vert{}_F^2 = \sum_{i=1}^r \sigma_i^2$$

When keeping $k$ singular values, the proportion of total information preserved is:

$$\text{Energy Preserved} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2} \times 100\%$$

* **100% Energy:** $k = r$ (lossless reconstruction).
* **Threshold Selection:** In real-world data pipelines, $k$ is selected dynamically by setting a variance threshold (e.g., $95\%$ or $99\%$) and solving for the smallest $k$ where the cumulative sum exceeds that value.

---

## 6. Practical ML Relevance

* **Principal Component Analysis (PCA):** If $X$ is centered by subtracting its column means ($\tilde{X} = X - \mu$), the right singular vectors $V$ are the exact principal component loading vectors, and $Z = \tilde{X} V$ contains the principal component scores.
* **Latent Semantic Analysis (LSA / NLP):** Decomposes term-document frequency matrices to uncover latent topics and compute semantic similarity.
* **Low-Rank Adaptation (LoRA):** Instead of fine-tuning a full parameter weight update $\Delta W \in \mathbb{R}^{d \times k}$, LoRA decomposes it into two low-rank matrices $B \cdot A$ (where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$), drastically reducing trainable parameters.

---

## 7. Execution & Verification


### Running the Script
```powershell
python 00_mathematical_foundations/linear_algebra/svd_projection.py

Expected terminal output:

```text
X shape: (6, 4)
U shape: (6, 4)
S shape: (4,)
Vt shape: (4, 4)
Projected coordinates (rank-4) shape: (6, 4)
Reconstructed ambient shape: (6, 4)
Rank-4 Frobenius error: 0.0000
Theoretical Eckart-Young error: 0.0000
Energy preserved: 100.00%
Errors match: True
```
