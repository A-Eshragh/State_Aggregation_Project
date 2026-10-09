# Reproducing the Paper Experiments

Run from the repository root.


## Figure 1

```bash
python -m mdp_lp.plot_fig1_sa_learned                    # columns 1-2 (~17 min)
python -c "import mdp_lp.plot_fig1_sketch_col as m; m.K_SKETCH=128; m.DLAM=0.02; m.main()"   # column 3
python mdp_lp/replot_fig1_sa.py                          # draws the figure
```

Column 3 must use `K_SKETCH=128, DLAM=0.02`; the script's own defaults
(512, 0.005) give a different curve.

## Inventory Problem (Section 6.1)

```bash
bash mdp_lp/run_dfl_seeds.sh
python make_figures.py inventory
```

`run_dfl_seeds.sh` loops over seeds 1–10 and trains each method (Two-stage,
DFL-LP, DFL-QP, DFL-Feas, DFL-Sketch, DFL-SA) by calling
`run_elasticity_experiments`. The method list is the `CONFIGS=(...)` array near
the top of the script — uncomment the rows you want to run. `make_figures.py
inventory` then reads the saved results and draws the training-curve, regret,
and compute-cost panels. This is the |S| = 36 setting.

## Larger Inventory Problems

```bash
bash mdp_lp/run_dfl_seeds_S_50.sh
bash mdp_lp/run_dfl_seeds_S_100.sh
```

Same methods, bigger state space, seeds 1–10.

## Cliff Walking

```bash
bash mdp_cw/run_dfl_seed.sh
bash mdp_cw/run_dfl_seed.sh ssa18 ssa24
```

Runs every method, then prints the summary table and redraws the figure. For
wall-clock times use `TIMING=1 bash mdp_cw/run_dfl_seed.sh ssa12t2`: it runs one
job at a time. The default mode runs 10 seeds at once, which is fine for regret
but makes timings meaningless.

## CartPole (Section 6.3)

```bash
bash mdp_continuous/run_dfl_seeds.sh
```

`run_dfl_seeds.sh` runs CartPole over seeds 1–10 in one shot: the baselines
(Two-stage / DFL-LP / DFL-QP / DFL-Feas / DFL-Sketch) via `run_dfl_vs_qpth`, and
the DFL-SA feature-dimension sweeps (d = 30, 60, 90) via `sweep_n_features`, then
renders the figures with `make_figures.py cartpole`. The last step re-derives
every row from the caches and compares it against the published table, so a
clean run is the check itself.

Figures are written to `figures/`.
