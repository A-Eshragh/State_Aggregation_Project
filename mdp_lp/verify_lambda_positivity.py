#!/usr/bin/env python3
"""
Audit script: verify that all saved lambda values are nonnegative.

Usage (from repo root):
  python -m mdp_lp.verify_lambda_positivity

What it checks:
  - For every mdp_lp/results_elasticity/**/results.json:
      train_lambda_pred, train_lambda_true, test_lambda_pred, test_lambda_true
    are all >= -tol (tol defaults to 1e-12) and finite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _coerce_1d_float_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if not isinstance(x, list) or len(x) == 0:
        return None
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr.reshape(-1)


def _scan_results_json(root: Path) -> List[Path]:
    return sorted(root.glob("mdp_lp/results_elasticity/**/results.json"))


def audit_file(path: Path, *, tol: float) -> List[Tuple[str, float]]:
    d: Dict[str, Any] = json.loads(path.read_text())
    keys = [
        "train_lambda_pred",
        "train_lambda_true",
        "test_lambda_pred",
        "test_lambda_true",
    ]
    offenders: List[Tuple[str, float]] = []
    for k in keys:
        arr = _coerce_1d_float_array(d.get(k))
        if arr is None:
            continue
        if not np.all(np.isfinite(arr)):
            offenders.append((k, float(np.nanmin(arr))))
            continue
        m = float(arr.min())
        if m < -tol:
            offenders.append((k, m))
    return offenders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-12, help="tolerance for negativity check")
    args = ap.parse_args()

    root = Path(".")
    files = _scan_results_json(root)
    print(f"[verify_lambda_positivity] results.json files: {len(files)}")

    bad: List[Tuple[Path, str, float]] = []
    for p in files:
        offenders = audit_file(p, tol=args.tol)
        for k, m in offenders:
            bad.append((p, k, m))

    if bad:
        print(f"[verify_lambda_positivity] FAIL: {len(bad)} negative/non-finite entries")
        for p, k, m in bad[:50]:
            print(f"  {m:.6g}\t{k}\t{p}")
        raise SystemExit(2)

    print("[verify_lambda_positivity] PASS: no negative/non-finite lambda values found")


if __name__ == "__main__":
    main()

