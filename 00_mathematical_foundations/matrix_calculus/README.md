# Gradient Checker (Numerical vs. Analytical)

A beginner-friendly tool to verify backpropagation calculations.

---

## 1. Why Do We Need This?

When building neural networks from scratch, we must calculate gradients by hand using calculus:

* **Analytical Gradient:** Fast, exact mathematical formulas (what we write in `.backward()`).
* **The Problem:** A tiny typo (like a missed minus sign or wrong index) can quietly break training without ever raising an error.

The **Gradient Checker** solves this by comparing your code against the fundamental definition of a derivative.

---

## 2. The Core Idea: Finite Differences

Calculus defines a derivative as the slope between two infinitely close points. Computationally, we approximate this using a tiny number $\epsilon$ (epsilon, e.g., $10^{-5}$):

$$f'(x) \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

Because this calculation only evaluates the forward pass $f(x)$, it is straightforward, automated, and practically error-proof.
---

## 3. How the Algorithm Works

For every single scalar value inside an input array:

1. **Save:** Record the original value $x_i$.
2. **Nudge Forward:** Set $x_i = x_i + \epsilon$ and run the forward pass $\rightarrow f(x + \epsilon)$.
3. **Nudge Backward:** Set $x_i = x_i - \epsilon$ and run the forward pass $\rightarrow f(x - \epsilon)$.
4. **Compute Slope:** $\text{grad}_i = \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$.
5. **Restore:** Reset $x_i$ to its original value so the rest of the array isn't corrupted.

---

## 4. Comparing the Results: Relative Error

Direct differences $\vert{}A - B\vert{}$ fail if numbers are extremely large or small. Instead, we compute the **relative error**:

$$\text{Relative Error} = \frac{\vert{}G_{\text{analytic}} - G_{\text{num}}\vert{}}{\max(\vert{}G_{\text{analytic}}\vert{}, \vert{}G_{\text{num}}\vert{}) + 10^{-15}}$$

### How to Read the Error

* **$< 10^{-7}$:** Perfect match. Our derivative code is correct.
* **$10^{-5}$ to $10^{-7}$:** Usually acceptable (often due to floating-point rounding).
* **$> 10^{-4}$:** The backward pass contains a bug.

---

## 5. How to Run

Execute the script from this project root:

```bash
python 00_mathematical_foundations/matrix_calculus/grad_checker.py

Relative error: 5.10e-11
Gradcheck passed successfully!
