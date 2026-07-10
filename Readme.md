# Reproducing the Paper Experiments

Run from the repository root.

## Inventory Problem (Section 6.1)

```bash
bash mdp_lp/run_dfl_seeds.sh          # runs all methods over seeds 1–10
python make_figures.py inventory      # renders the figures
```

`run_dfl_seeds.sh` loops over seeds 1–10 and trains each method (Two-stage,
DFL-LP, DFL-QP, DFL-Sketch, DFL-SA) by calling `run_elasticity_experiments`.
The method list is the `CONFIGS=(...)` array near the top of the script —
uncomment the rows you want to run. `make_figures.py inventory` then reads the
saved results and draws the training-curve, regret, and compute-cost panels.

## CartPole (Section 6.3)

```bash
bash mdp_continuous/run_dfl_seeds.sh  # runs the experiments + renders the figures
```

`run_dfl_seeds.sh` runs CartPole over seeds 1–10 in one shot: the baselines
(Two-stage / DFL-LP / DFL-QP / DFL-Sketch) via `run_dfl_vs_qpth`, and the
DFL-SA feature-dimension sweeps (d = 30, 60, 90) via `sweep_n_features`, then
renders the figures with `make_figures.py cartpole`. Note the baseline step is
commented out by default — uncomment step [1/4] to regenerate those curves.

Figures are written to `figures/`.
