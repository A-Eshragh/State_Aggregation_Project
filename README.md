# Decision-Focused Learning in MDPs: An Occupancy Measure Approach

Code for the NeurIPS 2026 paper
[*Decision-Focused Learning in MDPs: An Occupancy Measure Approach*](https://arxiv.org/abs/2610.08384).

The repository trains a predictive model end-to-end through the occupancy-measure
LP of an MDP and compares it against Two-stage, DFL-LP, DFL-QP, DFL-Feas,
DFL-Sketch, and our state-aggregation method (DFL-SA).

## Installation

```bash
conda create -n dfl-mdp python=3.10
conda activate dfl-mdp
pip install -r requirements.txt
pip install gymnasium   # only needed for the geometry test and Hopper env
```

The LP baselines use [Gurobi](https://www.gurobi.com/) (`gurobipy`), which
needs a license; free academic licenses are available.

The run scripts call the interpreter in `$PY` where it is set, so point it at
your environment if `python3` is not it:

```bash
export PY=$(which python)
```

All commands below are run from the repository root. The scripts raise the
stack limit (`ulimit -s 65536`) themselves, since Gurobi can segfault on large
LPs with the default limit.

## Reproducing the Experiments

### Figure 1

```bash
python -m mdp_lp.plot_fig1_sa_learned                    # columns 1-2 (~17 min)
python -c "import mdp_lp.plot_fig1_sketch_col as m; m.K_SKETCH=128; m.DLAM=0.02; m.main()"   # column 3
python mdp_lp/replot_fig1_sa.py                          # draws the figure
```

Column 3 must use `K_SKETCH=128, DLAM=0.02`; the script's own defaults
(512, 0.005) give a different curve.

### Inventory Problem (Section 6.1)

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

### Larger Inventory Problems

```bash
bash mdp_lp/run_dfl_seeds_S_50.sh
bash mdp_lp/run_dfl_seeds_S_100.sh
```

Same methods, bigger state space, seeds 1–10.

### Cliff Walking (Section 6.2)

```bash
bash mdp_cw/run_dfl_seed.sh
bash mdp_cw/run_dfl_seed.sh ssa18 ssa24
```

Runs every method, then prints the summary table and redraws the figure. For
wall-clock times use `TIMING=1 bash mdp_cw/run_dfl_seed.sh ssa12t2`: it runs one
job at a time. The default mode runs 10 seeds at once, which is fine for regret
but makes timings meaningless.

### CartPole (Section 6.3)

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

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{zke+26,
title={Decision-Focused Learning in {MDP}s: An Occupancy Measure Approach},
author={Zihao Zhao and Ashwath K. Karunakaram and Ali Eshragh and Yuexing Li and Kai Wang},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=rzZPOKJV34}
}
```
