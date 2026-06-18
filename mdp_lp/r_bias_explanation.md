# Reward-Bias Modification in Soft State Aggregation

## What Changed

### Original: W depends only on P

The cluster membership logits for state `s` in batch element `b`:

```
ℓ_{b,s} = W_linear · vec( Σ_{s'} P_{b,s,a,s'} · E_{s'} )

W_{b,s,m} = softmax_m( ℓ_{b,s} / τ )
```

where `E ∈ R^{S × r}` is a learned embedding (free parameter).

### Modified: add a small r-dependent bias

```
ℓ_{b,s} = W_linear · vec( Σ_{s'} P_{b,s,a,s'} · E_{s'} )  +  W_r · r_{b,s}
           \_________________________________________/           \____________/
                   main logits (from P)                      small bias (from r)
```

where `W_r ∈ R^{M × A}` is initialized with `N(0, 10^{-3})`.

That's the only change — one additive term.

---

## Why This Matters: The Gradient Bottleneck

The training loss is:

```
L = -r_true^T · y*    where  y* = H_B^{-1} · γ
```

The LP backward pass computes:

```
∂L/∂H = -(H_B^{-T} · ∂L/∂y*_B) · y_B^T

∂L/∂r = 0
```

`∂L/∂r = 0` because `y* = H_B^{-1} · γ` doesn't depend on `r` at all (fixed basis). So the entire gradient to `λ_pred` must flow through `H`.

### Original gradient chain (P-only)

```
L  →  y*  →  H  →  P_agg(W, P)  →  P(λ_pred)  →  λ_pred
```

This path exists but is very indirect: `P_agg` is a doubly-weighted average of `P`, and `H = I - β · P_agg` compresses the signal further.

### New gradient chain (with r_bias)

The r_bias adds a **second path**:

```
L  →  y*  →  H  →  P_agg(W, P)  →  W(ℓ)  →  ℓ = f(P) + W_r · r  →  r(λ_pred)  →  λ_pred
```

By chain rule through the new term:

```
∂ℓ_{b,s} / ∂λ_pred = W_r · ∂r_{b,s} / ∂λ_pred
```

Since `r` is a direct function of `λ_pred` (rewards depend on demand rate → prices), this path carries a strong, direct gradient.

---

## Intuitive Example

Imagine 2 states, 1 action, 2 clusters. Two instances with `λ_A = 3` (low demand) and `λ_B = 20` (high demand).

**Without r_bias:** Both instances have similar Poisson transition matrices (both are stochastic over `S` states). After `P · E`, the features `φ_A` and `φ_B` are close — the embedding `E` can only amplify small differences in transition probabilities. The cluster assignments `W_A ≈ W_B`, so `P_agg` is nearly identical → `H` is nearly identical → `y*` is nearly identical → `λ_pred` gets nearly the same gradient for both. The model can't tell them apart.

**With r_bias:** The rewards encode selling price × expected sales, which directly depends on `λ`. Instance A has `r_A = [5.2]`, instance B has `r_B = [34.7]`. Even with tiny `W_r`:

```
ℓ_A = f(P_A) + 0.001 × 5.2
ℓ_B = f(P_B) + 0.001 × 34.7
```

The logits differ slightly → `W_A ≠ W_B` → `P_agg,A ≠ P_agg,B` → different `H` → different `y*` → the loss can now push `λ_pred` differently for each instance.

And crucially, `∂P_agg/∂r` is nonzero, so the gradient flows through the reward pathway:

```
∂L/∂λ_pred  ∝  (∂L/∂H · ∂H/∂P_agg) · ∂P_agg/∂W · ∂W/∂ℓ · W_r · ∂r/∂λ_pred
                \____________________/   \_______________________________/
                     from LP                    new r-bias path
```

The small initialization (`σ = 10^{-3}`) ensures `E` still dominates the cluster structure early on (no chicken-and-egg collapse), while `W_r` grows organically as training progresses.

---

## Results

| Metric      | P-only (embed) | P + r concat | **P embed + r_bias** |
|-------------|---------------|-------------|---------------------|
| Regret      | 53.3          | 265.1       | **12.9**            |
| Lambda MSE  | 398.3         | 64.4        | **30.4**            |
| pred_MSE    | 357           | 103         | **28.2**            |
| λ_bias      | +17.8         | -7.4        | **+1.7**            |
