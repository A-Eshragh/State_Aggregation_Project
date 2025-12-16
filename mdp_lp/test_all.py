#!/usr/bin/env python3
"""
Simple test runner for mdp_lp codebase.
Run this to test all components including trainers.
"""

import sys
from mdp_lp.tests.tests import (
    # Core tests
    test_H_matrix,
    test_r_flatten,
    test_lp_formulation,
    test_dense_lp_solver,
    test_gradients,
    # Trainer tests
    test_dfl_trainer_forward,
    test_dfl_trainer_train_with_target,
    test_dfl_trainer_train_maximize,
    test_twostage_trainer_train,
    test_twostage_trainer_eval,
    test_trainer_gradients,
    # Sparse solver tests
    test_sparse_lp_solver,
    test_sparse_lp_solver_batched,
    print_section
)

def main():
    """Run all tests."""
    print("=" * 70)
    print("RUNNING ALL MDP_LP TESTS")
    print("=" * 70)
    
    tests = [
        ("Core MDP Tests", [
            test_H_matrix,
            test_r_flatten,
            test_lp_formulation,
        ]),
        ("LP Solver Tests", [
            test_dense_lp_solver,
            test_gradients,
            test_sparse_lp_solver,
            test_sparse_lp_solver_batched,
        ]),
        ("Trainer Tests", [
            test_dfl_trainer_forward,
            test_dfl_trainer_train_with_target,
            test_dfl_trainer_train_maximize,
            test_twostage_trainer_train,
            test_twostage_trainer_eval,
            test_trainer_gradients,
        ]),
    ]
    
    passed = 0
    failed = 0
    
    for category, test_funcs in tests:
        print_section(f"{category}")
        for test_func in test_funcs:
            try:
                test_func()
                passed += 1
            except Exception as e:
                print(f"✗ {test_func.__name__} FAILED: {e}")
                failed += 1
                import traceback
                traceback.print_exc()
    
    print_section("TEST SUMMARY")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print_section("ALL TESTS PASSED ✓")
        return 0
    else:
        print_section("SOME TESTS FAILED ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())

