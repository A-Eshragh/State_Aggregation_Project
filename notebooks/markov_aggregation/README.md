# Markov-Guided State Aggregation

This directory is the self-contained Markov aggregation experiment package.
It keeps the executed notebook, the Markov-specific training and measurement
code, and only the compact artifacts required to inspect or re-run the
reported measurements.

## Contents

- `markov_aggregation_execution_protocol.ipynb`: executed reader-facing
  notebook. It compares ordinary learned soft aggregation with the additive
  Markov variants and the anchored-prototype implementation.
- `scripts/markov_protocol_runner.py`: isolated training runner for the
  additive Markov variants and ordinary-SA controls.
- `scripts/markov_prototype_anchored_smoke.py`: three-seed paired training
  runner for the anchored prototype implementation.
- `scripts/compute_cliff_markov_measurements.py`: checkpoint-based Cliff
  geometry measurement helper.
- `scripts/build_notebook.py`: generates the notebook from the current local
  package layout.
- `results/`: compact baseline context, paired-run summaries, aggregation
  checkpoints, and saved geometry measurements used by the notebook.

## Re-running

Run `scripts/build_notebook.py` from the repository root, then execute the
notebook in the `dfl_sa_cuda` environment. The notebook uses the included
artifacts by default. If the three paired prototype rows are absent, it calls
the bundled prototype runner to regenerate them from fresh generated data.

The notebook writes generated data caches to `.cache/markov_aggregation/`,
outside this directory. Those caches are not experimental results and are not
versioned.

## Scope

The additive archive contains the existing ten-seed Markov runs for
Inventory/basic, Cliff/basic, and Cliff/medium. The anchored-prototype study
is a distinct matched three-seed experiment. The notebook reports their
decision deltas and assignment measurements separately; it does not treat the
two implementation families as interchangeable.
