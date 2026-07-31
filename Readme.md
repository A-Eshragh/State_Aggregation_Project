# Reproducing the Paper Experiments

Run from the repository root.

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

## LP Layer Ablation

Three ways to build the M-state LP: hard memberships (`hard`), the exact
projected LP (`projlp`), and the aggregated MDP we use (`projmdp`). Only
`--sa_layer` changes — the rest is the `dfl_sketch_agg10` row of the inventory
run above.

```bash
for arm in hard projlp projmdp; do
  for seed in 1 2 3 4 5 6 7 8 9 10; do
    python -m mdp_lp.run_elasticity_experiments \
      --experiment_type b --cost_case normal_margin --bc 10 \
      --algo dfl_sketch --n_agg 10 --n_epochs 50 --n_samples 10 --lr 0.01 \
      --pretrain_epochs 30 --b_max 5 --s_max 30 --q_max 30 \
      --n_sketches 8 --sketch_ratio 0.2 --sketch_mode gaussian \
      --aug_lag_rho 0.1 --loss_weight 1 --sa_tau 0.5 --sa_layer "$arm" \
      --seed "$seed" --tag "layerabl_${arm}_seed${seed}" \
      --results_root mdp_lp/results_projection_ablation_tab1
  done
done
```

`projmdp` is the published DFL-SA (M = 10) row, so it doubles as the check.
The share of infeasible training solves is `lp_infeasible_per_epoch` in
`checkpoint_final.json`. Run the arms one at a time if you want wall-clock;
run them together and they share the machine.

## Reading the Numbers

Each run writes a `results.json`. `test_mean_regret` is the absolute regret and
`100 * test_mean_regret / test_mean_oracle_reward` is the percentage. Average
over seeds:

```bash
python - <<'EOF'
import json, glob, numpy as np
runs = glob.glob('mdp_lp/results_dflwin/**/tag_S51_twostage_high_margin_seed*/**/results.json',
                 recursive=True)
r = np.array([json.load(open(p))['test_mean_regret'] for p in runs])
print(f'{r.mean():.3f} +/- {r.std(ddof=1)/np.sqrt(len(r)):.3f}  (n={len(r)})')
EOF
```

The cliff walking and CartPole scripts print their tables themselves.

Figures are written to `figures/`.
