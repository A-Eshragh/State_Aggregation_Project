# mdp_cw

Cliff Walking code and final NeurIPS submission artifacts.

## What is kept

- `cliff_mdp.py`, `weather_dynamics.py`, `features_model.py`
- `prepare_cliff_data.py`, `train_cliff.py`, `run_cliff_experiments.py`
- `run_v4_hard_experiments.py`, `summarize_paired_runs.py`
- `cliff_run_utils.py`, `load_run_for_eval.py`, `verify_policies.py`
- `test_geometry.py`
- `plot_report_three_panel_best_nk.py`
- `plot_v4_hard_publication.py`, `generate_reviewer_figures.py`
- `plot_cliff_run.py`
- `selected_plots/`

## Final submission snapshot

The NeurIPS figures are stored in `selected_plots/`:

- grid: `4x12`
- states/actions: `S = 48`, `|A| = 4`
- discount: `gamma = 0.95`
- training skew: `k = 15`
- sample budget: `n = 200`
- seeds: 10 paired random seeds
- methods:
  - `Two-Stage`
  - `DFL-LP`
  - `DFL-QP`
  - `DFL-Sketch`
  - `DFL-SA (M=6,12,24)`

The canonical figures are:

- `selected_plots/training_curves.png`
- `selected_plots/compute_cost.png`
- `selected_plots/decision_quality.png`
- `selected_plots/paired_ts_vs_dfl.png`

## Regenerating the final plots

From the repository root:

```bash
python3 mdp_cw/plot_report_three_panel_best_nk.py \
  --flat_root /path/to/the/raw/results_random_all5_k15_sketchstab \
  --k 15 \
  --n_seeds 10 \
  --n_label 200 \
  --out_dir mdp_cw/selected_plots
```
