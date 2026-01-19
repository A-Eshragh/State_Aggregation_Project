#!/bin/bash
# Run all elasticity experiments in parallel with nohup (runs in background)

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Starting all elasticity experiments with nohup"
echo "=========================================="
echo ""
echo "This will run 16 experiments in parallel:"
echo "  - Experiments: a, b, c"
echo "  - Cost cases: high_margin, low_margin"
echo "  - Algorithms: dfl, twostage"
echo "  - Experiment C: β₂ = ±0.09"
echo ""
echo "Parameters:"
echo "  - 100 epochs"
echo "  - 200 samples"
echo "  - β₀ = 4.0, β₁ = -0.3"
echo ""
echo "Output will be saved to:"
echo "  - nohup.out (main log)"
echo "  - mdp_lp/results_elasticity/logs/*.log (individual experiment logs)"
echo ""
echo "To monitor progress:"
echo "  tail -f nohup.out"
echo ""
echo "Starting in 3 seconds..."
sleep 3

nohup python -m mdp_lp.run_all_elasticity > nohup.out 2>&1 &

echo "Experiments started in background!"
echo "Process ID: $!"
echo ""
echo "Monitor with: tail -f nohup.out"
echo "Check status: ps aux | grep run_all_elasticity"
echo "Results will be in: mdp_lp/results_elasticity/"
