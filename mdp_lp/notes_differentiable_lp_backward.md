# How the Differentiable LP Backward Pass Works

Plain-language explanation of the backward pass in `solvers/differentiable_lp.py`, aligned with **ICML 2026 DFL/SA Appendix A**.

---

## 0. How the PDF derives the gradient (Appendix A.2)

**The LP in the PDF:**

- Maximize $r_\theta^\top y$ subject to $H_\theta y = \gamma$, $y \geq 0$
- Optimal value: $v^*_\theta = r_\theta^\top y^*_\theta$

They want $dv^*_\theta / d\theta$. By the product rule you get **Equation (1)**:

$$
\frac{dv^*_\theta}{d\theta}
= \Big( \frac{\partial r_\theta}{\partial \theta} \Big)^\top y^*_\theta
\;+\;
r_\theta^\top \frac{dy^*_\theta}{d\theta}
$$

- First term = **direct**: how rewards change with $\theta$, weighted by occupancy.
- Second term = **indirect**: how optimal occupancy $y^*$ changes with $\theta$, weighted by rewards.

**Getting $dy^*/d\theta$ from the constraint**

Constraint: $H_\theta y^*_\theta = \gamma$. Assume $d\gamma/d\theta = 0$. Differentiate:

**Equation (2):**
$$
\frac{\partial H_\theta}{\partial \theta} y^*_\theta + H_\theta \frac{dy^*_\theta}{d\theta} = 0
$$

Split $y^*$ into **basic** $y^B$ (the $S$ nonzero entries) and **non-basic** $y^N = 0$. Split $H_\theta = [H^B_\theta \mid H^N_\theta]$. Only the basic part matters:

$$
\frac{\partial H^B_\theta}{\partial \theta} y^B_\theta + H^B_\theta \frac{dy^B_\theta}{d\theta} = 0
$$

So **Equation (3):**
$$
\frac{dy^B_\theta}{d\theta} = -(H^B_\theta)^{-1} \frac{\partial H^B_\theta}{\partial \theta} y^B_\theta
$$

Plug that into (1) to get **Equation (4)** (full DFL gradient):

$$
\frac{dv^*_\theta}{d\theta}
= \Big( \frac{\partial r^B_\theta}{\partial \theta} \Big)^\top y^B_\theta
\;-\; (r^B_\theta)^\top (H^B_\theta)^{-1} \frac{\partial H^B_\theta}{\partial \theta} y^B_\theta
$$

So the PDF gradient has **two parts**: one through $r_\theta$ (direct), one through $y^*_\theta$ via $H_\theta$ (indirect).

**What our code does:** We only implement the **indirect** part (the part that goes through $y^*$). We compute $dL/dH$ and $dL/d\gamma$ from $dL/dy^*$, and we **set $dL/dr = 0$**. So the direct term (through $\partial r/\partial \theta$) is not used. The direct term would need gradient through $r$ and then $r(\theta)$, which we do not do.

---

## 1. What the LP is

- **Variables:** vector $y$ of length $S \times A$ (one number per state–action pair; occupancy).
- **Objective:** maximize $r^\top y$.
- **Constraint:** $H y = \gamma$ ($S$ equations), $y \geq 0$.

The solver returns optimal $y^*$.

---

## 2. What “basis” means

When the solver finishes, only **$S$** of the $S \times A$ variables are nonzero. The rest are 0.

- **Basis $B$** = set of indices of those nonzero variables.
- **$y_B$** = the $S$ nonzero values.
- All other entries of $y^*$ are 0.

The constraint $H y = \gamma$ then reduces to a **square** system on the basic columns. If $H_B$ is the $S \times S$ submatrix of $H$ with columns $B$, then:

$$
y_B = H_B^{-1} \gamma
$$

So the **values** of the basic variables are determined only by $H$ and $\gamma$, not by $r$. ($r$ decided *which* variables are in the basis; once the basis is fixed, the numbers come from $H$ and $\gamma$.)

---

## 3. What the backward pass has to do

We receive **grad_y** = $dL/dy^*$ (gradient of the loss w.r.t. $y^*$).

We must return:

- **grad_H** = $dL/dH$
- **grad_r** = $dL/dr$
- **grad_gamma** = $dL/d\gamma$

Only the basic entries of $y^*$ are nonzero, so we only use **grad_y_B** = `grad_y` at the basis indices.

---

## 4. Why grad_r = 0

We assume the **basis does not change** for small perturbations (fixed-basis assumption).

Then $y_B = H_B^{-1} \gamma$ — no $r$ in that formula. So $y^*$ does not depend on $r$, and we set **grad_r = 0**.

In words: *$r$ chose the basis; the numbers in $y_B$ are fixed by $H$ and $\gamma$ only.*

---

## 5. Why grad_gamma = $H_B^{-T}$ grad_y_B

We have $y_B = H_B^{-1} \gamma$, so a small change in $\gamma$ gives:

$$
d(y_B) = H_B^{-1} \, d(\gamma)
$$

So $dL = (\text{grad\_y\_B})^\top d(y_B) = (\text{grad\_y\_B})^\top H_B^{-1} d(\gamma)$, hence:

$$
\text{grad\_gamma} = (H_B^{-1})^\top \text{grad\_y\_B} = H_B^{-T} \text{grad\_y\_B}
$$

**In the code:** Solve $H_B^\top x = \text{grad\_y\_B}$ for $x$; then $\text{grad\_gamma} = x$.

```python
x = torch.linalg.solve(H_B.T, grad_y_B)
grad_gamma[b] = x
```

---

## 6. Why grad_H_B = minus (x outer y_B)

From $y_B = H_B^{-1} \gamma$, a small change in $H_B$ gives (chain rule for the inverse):

$$
d(y_B) = -H_B^{-1} \, d(H_B) \, y_B
$$

So $dL = (\text{grad\_y\_B})^\top d(y_B) = -(\text{grad\_y\_B})^\top H_B^{-1} d(H_B) y_B$.

With $x = H_B^{-T} \text{grad\_y\_B}$ (same $x$ as above), this is $dL = -x^\top d(H_B) y_B$, so the gradient w.r.t. $H_B$ is:

$$
\text{grad\_H\_B} = -x \, y_B^\top
$$

i.e. minus the outer product of $x$ and $y_B$.

**In the code:**

```python
x = torch.linalg.solve(H_B.T, grad_y_B)
grad_H_B = -torch.outer(x, y_B)
```

Then we write `grad_H_B` into the correct columns of **grad_H** (those in the basis $B$).

---

## 7. Summary table

| Output      | Formula                         | Reason |
|------------|-----------------------------------|--------|
| grad_r     | 0                                 | $y_B = H_B^{-1}\gamma$ does not depend on $r$. |
| grad_gamma | $H_B^{-T}$ grad_y_B              | $d(y_B) = H_B^{-1} d(\gamma)$. |
| grad_H     | $-x y_B^\top$ with $x = H_B^{-T}$ grad_y_B | $d(y_B) = -H_B^{-1} d(H_B) y_B$. |

The backward pass is just **differentiating $y_B = H_B^{-1} \gamma$** w.r.t. $H$ and $\gamma$, and passing **grad_y_B** through. No gradient through $r$.

---

## 8. PDF vs our code (same math)

- **PDF:** They write the full gradient (4) = direct term + indirect term. The indirect term uses Equation (3): $dy^B/d\theta = -(H^B)^{-1} (\partial H^B/\partial \theta) y^B$.

- **Our code:** We implement the backward of the **LP layer** $(H, r, \gamma) \to y^*$. We get **grad_y** = $dL/dy^*$ (e.g. for $L = -v^*$, grad_y is proportional to $r$). We set **grad_r = 0**.

- **Same math:** In the PDF’s indirect term, the factor $(r^B)^\top$ in front is exactly the upstream gradient we get on the basic indices: **grad_y_B** (which equals $r$ on $B$ when $L = -v^*$). So our $x = (H_B^\top)^{-1}$ grad_y_B is $(H^B)^{-\top} r^B$ in their notation, and **grad_H_B = − outer(x, y_B)** is the derivative of the indirect term w.r.t. $H$. We therefore compute the **indirect part** of Equation (4). The **direct part** $(\partial r/\partial \theta)^\top y^*$ would come from backprop through $r$; we drop it by setting grad_r = 0.
