#!/bin/bash
# Run all elasticity experiments locally
# This will run all experiments sequentially (one at a time)

echo "=========================================="
echo "Running All Elasticity Experiments"
echo "=========================================="
echo ""
echo "Parameters:"
echo "  - 100 epochs"
echo "  - 200 samples"
echo "  - β₀ = 4.0, β₁ = -0.3"
echo "  - Experiment C: β₂ = ±0.09"
echo ""
echo "This will run 16 experiments total:"
echo "  - 3 experiment types (a, b, c)"
echo "  - 2 cost cases (high_margin, low_margin)"
echo "  - 2 algorithms (dfl, twostage)"
echo "  - Experiment C has 2 β₂ values (±0.09)"
echo ""
echo "Starting experiments..."
echo ""

cd "$(dirname "$0")/.."
python -m mdp_lp.run_all_elasticity

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved in: mdp_lp/results_elasticity/"
echo "=========================================="
