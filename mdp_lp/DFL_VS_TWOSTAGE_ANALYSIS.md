# DFL vs Twostage: Results Scan and Recommendation

## 1. What was scanned

- All `results.json` files under the project (excluding `old_results/`) that could be loaded.
- Only **experiment_type = b** runs were compared (exp_b: beta_2=0, p lognormal).
- For each run we extracted: algo, cost_case, n_samples, s_max, p_lognormal_std, sigma_epsilon, n_epochs, loss_weight, aug_lag_rho, train_optimal_reward, train_reward_per_epoch (epoch0, epoch1, final), test_obj_values (mean).

---

## 2. Why DFL “starts at optimal” and then goes **down** (e.g. ps2)

### 2.1 Two different metrics on the same plot

The **first point** (epoch 0) and **all later points** (epochs 1, 2, …) are **not** the same quantity:

| Point | What is actually computed |
|--------|----------------------------|
| **Epoch 0** | `eval_pi_on_true_mdp(mdp_true, pi_true, gamma)` — i.e. the **true value** of the current (init) policy when run on the **true** MDP (correct flow balance, true dynamics). |
| **Epochs 1+** | `(r_true * y_star).sum(dim=1).mean()` — i.e. **r_true · y_star**, where `y_star` is the **LP solution under the predicted λ** (and thus under the **predicted** MDP). So this is **not** the value of any policy in the true MDP; it mixes “predicted-MDP occupancy” with “true reward.” |

So the curve mixes:

1. One number: “value of init policy on true MDP.”
2. Rest: “hybrid” reward r_true · y_star(pred λ).

They are not comparable. The “optimal” horizontal line is **mean over train of the true LP optimal value** (true MDP), so it is comparable to a **true** policy value (e.g. epoch 0), but **not** to r_true · y_star.

### 2.2 What happens in ps2 (p_lognormal_std = 2)

- **Epoch 0:** Init policy is “use λ = mean(λ_true)” for every sample. That policy is evaluated **on the true MDP** → you get **~21,376** (train_optimal_reward ≈ 21,438). So the **first point** is “value of the mean-λ policy on true MDP,” and it lands **at** optimal.
- **Epochs 1+:** The plot switches to the **hybrid** metric r_true · y_star. After the first gradient steps, λ starts to move away from the mean. The occupancy y_star is then computed from the **predicted** λ (and predicted MDP), and r_true · y_star is **not** the value of that policy in the true MDP. As DFL updates λ, this hybrid number **drops** (to ~15,077) because:
  - The DFL loss is **not** “maximize r_true · y_star”; it is (-v* + aug_lag + loss_weight * pred_loss). So minimizing DFL loss can **reduce** r_true · y_star.
  - With high p variance (ps2), true λ’s are spread out; gradients push λ in directions that improve the loss (e.g. satisfy constraints, reduce violation) but can make the hybrid metric worse.

So:

- “Starts at optimal” = **epoch 0** is the **true** value of the init (mean-λ) policy on the true MDP, which in this run is close to the optimal line.
- “Goes down” = **epochs 1+** use a **different, hybrid** metric that drops as λ moves away from the mean and the solver/gradients don’t maximize that hybrid number.

So the behavior is consistent once you account for the **metric switch** and the **objective** DFL is actually minimizing.

### 2.3 Recommendation for a fair plot

To avoid the misleading “start at optimal then drop”:

- Use **one** metric for the whole curve. Best option: **always** use “value of current policy on true MDP” (i.e. the same as epoch 0) for every epoch, e.g. at the end of each epoch run an eval pass with `eval_pi_on_true_mdp(mdp_true, pi_from_y_star, gamma)` and log that. Then the curve is comparable to the “optimal” line and reflects real decision quality.

---

## 3. Locked-in recommendation: best settings for DFL vs twostage

From the scanned Exp B runs (paired by same root folder, cost_case, n_epochs):

### 3.1 Where DFL wins or ties (test set mean LP value under predicted λ)

| Run (root / cost / epochs) | DFL test_mean | Twostage test_mean | Main params |
|----------------------------|---------------|--------------------|-------------|
| sanity_20_30_se01_ps05, high_margin, 100 | **4054** | 3769 | lw=100, rho=10, p_std=0.5 |
| sanity_20_30_se01_ps05, low_margin, 100 | **3112** | 2849 | lw=100, rho=10, p_std=0.5 |
| sanity_20_30_se01_ps05_dfl_initmean, high_margin, 50 | **6738** | 6730 | lw=0, rho=100, p_std=1, init=mean |
| sanity_20_30_se01_ps05_dfl_initmean, low_margin, 50 | **3484** | 3482 | lw=0, rho=100, p_std=1, init=mean |
| mdp_lp (sanity_20_30_se2_lw30_100), high_margin, 100 | **8776** | 8770 | lw=30, rho=1, p_std=1 |

### 3.2 Where DFL collapses (avoid these)

- **loss_weight = 0** (DFL-only): sanity_20_30_se01_ps05_w0, sanity_20_30_se01_ps05_dfl_only_rho100k/rho10k → DFL test_mean very negative or very low; twostage much better.
- **p_lognormal_std = 2** (ps2) with current setup: DFL train curve drops from near-optimal to ~15k (hybrid metric), and you only have DFL run (no twostage pair in ps2 folder).

### 3.3 Recommended “solution” for best DFL vs twostage

Use these settings so DFL is **competitive or better** than twostage and stable:

1. **Init λ at mean** (you already do this).
2. **loss_weight > 0** (e.g. **20–100**). Do **not** use loss_weight=0 unless you are only experimenting with pure DFL (then DFL often collapses).
3. **p_lognormal_std**: **0.5 or 1.0**. Avoid 2.0 for now; with ps2 the hybrid metric drops a lot and the curve is hard to interpret (and epoch 0 uses a different metric).
4. **aug_lag_rho**: **10** (default) is fine; 100 or 1 also appear in good runs.
5. **n_samples = 20, s_max = 30, sigma_epsilon = 0.1** (your sanity setup) are fine; the **mdp_lp** run with n_samples=20, p_std=1, lw=30, rho=1 gives DFL ≈ twostage.
6. **Epochs**: 50–100; DFL often needs a bit of time to settle.

**Concrete “best” setup to lock in:**

- **experiment_type b**, **cost_case** high_margin or low_margin  
- **n_samples** 20 (or 100 if you want), **s_max** 30  
- **p_lognormal_mean** 10, **p_lognormal_std** **0.5** or **1.0**  
- **sigma_epsilon** 0.1 (or 2.0 as in dfl_initmean if you want more λ variance)  
- **loss_weight** **20** or **100**  
- **aug_lag_rho** 10 (or 100)  
- **Init λ at mean** (already in code)  
- **n_epochs** 50 or 100  

This matches your **sanity_20_30_se01_ps05** (lw=100) and **sanity_20_30_se01_ps05_dfl_initmean** (lw=0, rho=100, p_std=1) style runs where DFL beats or ties twostage.

### 3.4 Why ps2 made DFL “start at optimal and go down”

- **“Start at optimal”:** Epoch 0 is the **true** value of the “λ = mean” policy on the true MDP; with your data that happens to be ~21,376 vs optimal 21,438.  
- **“Go down”:** From epoch 1 onward the plot uses the **hybrid** r_true · y_star; DFL minimizes a different objective, so λ drifts and that hybrid number drops. So it’s a combination of **(1) metric switch** and **(2) DFL objective ≠ “maximize r_true · y_star”.**

---

## 4. Summary table (Exp B, no old_results)

| Root | Cost | Epochs | DFL test_mean | TS test_mean | DFL lw | DFL rho | p_std |
|------|------|--------|---------------|--------------|--------|---------|-------|
| sanity_20_30_se01_ps05 | high_margin | 100 | 4054 | 3769 | 100 | 10 | 0.5 |
| sanity_20_30_se01_ps05 | low_margin | 100 | 3112 | 2849 | 100 | 10 | 0.5 |
| sanity_20_30_se01_ps05_dfl_initmean | high_margin | 50 | 6738 | 6730 | 0 | 100 | 1.0 |
| sanity_20_30_se01_ps05_dfl_initmean | low_margin | 50 | 3484 | 3482 | 0 | 100 | 1.0 |
| sanity_20_30_se01_ps05_dfl_initmean_10k_attempt2 | low_margin | 50 | 3484 | 3482 | 0 | 10000 | 1.0 |
| mdp_lp (sanity_20_30_se2_lw30_100) | high_margin | 100 | 8776 | 8770 | 30 | 1 | 1.0 |
| mdp_lp | low_margin | 100 | 7376 | **7844** | 30 | 1 | 1.0 |
| sanity_20_30_se01_ps2 | high_margin | 50 | 4794 | (no TS) | 20 | 10 | 2.0 |
| sanity_20_30_se01_ps05_w0 | high/low | 100 | negative | 3769/2849 | 0 | 10 | 0.5 |
| sanity_20_30_se01_ps05_dfl_only_rho* | high/low | 100 | negative/low | 3769/2849 | 0 | 10k/100k | 0.5 |

**Bottom line:** Use **loss_weight in [20, 100]**, **p_std in [0.5, 1.0]**, **init λ at mean**, and (optionally) **one consistent “value on true MDP” metric** for the whole reward curve so “start at optimal and go down” is not misleading.
