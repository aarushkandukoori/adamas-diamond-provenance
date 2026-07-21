#!/usr/bin/env python3
"""Re-run ADAMAS simulation and assert committed headline values within 1e-9."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-9
COMMITTED_PATH = ROOT / "results" / "results.json"
SIM = ROOT / "src" / "adamas_simulation.py"

# Headline values from the committed results (deterministic, not statistical).
EXPECTED = {
    "eer_learned": 0.01881666666666667,
    "eer_pca": 0.08875,
    "stageA_best_key_bits": -124.28434129972646,
    "pl_gate_acc": 0.9978750000000001,
    "lab_admitted": 200,
    "lab_n": 200,
    "G_for_128": 22,
    "vault_choice.C": 10000,
    "vault_choice.D": 8,
    "vault_choice.sec": 90.16602947289002,
    "vault_choice.unlock": 0.9700447419773197,
}


def _get(d: dict, key: str):
    if "." not in key:
        return d[key]
    cur = d
    for part in key.split("."):
        cur = cur[part]
    return cur


def _diff(got, exp) -> float:
    return abs(float(got) - float(exp))


def _match(got, exp) -> bool:
    if isinstance(exp, int) and not isinstance(exp, bool):
        return got == exp
    return _diff(got, exp) <= TOL


def main() -> int:
    # Inherit / force single-threaded BLAS for bit-exact linear algebra.
    for key, val in (
        ("OPENBLAS_NUM_THREADS", "1"),
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("PYTHONHASHSEED", "0"),
        (
            "NPY_DISABLE_CPU_FEATURES",
            "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_BF16",
        ),
    ):
        os.environ.setdefault(key, val)

    with open(COMMITTED_PATH) as f:
        committed = json.load(f)

    print("Running simulation ...", flush=True)
    proc = subprocess.run(
        [sys.executable, str(SIM)],
        cwd=str(ROOT),
        check=False,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        print(f"Simulation exited with code {proc.returncode}", file=sys.stderr)
        return proc.returncode or 1

    with open(ROOT / "results" / "results.json") as f:
        fresh = json.load(f)

    rows = []
    ok = True
    for key, exp in EXPECTED.items():
        got = _get(fresh, key)
        cval = _get(committed, key)
        match = _match(got, exp) and _match(cval, exp)
        if not match:
            ok = False
        rows.append((key, exp, got, cval, _diff(got, exp), match))

    print()
    print(f"{'key':<28} {'expected':>22} {'fresh':>22} {'committed':>22} {'|diff|':>14} {'ok':>5}")
    print("-" * 120)
    for key, exp, got, cval, diff, match in rows:
        print(
            f"{key:<28} {exp!s:>22} {got!s:>22} {cval!s:>22} {diff:14.3e} {'PASS' if match else 'FAIL':>5}"
        )

    if not ok:
        print(
            "\nVERIFY FAILED: one or more headline values differ by more than 1e-9.",
            file=sys.stderr,
        )
        return 1

    print("\nVERIFY PASSED: all headline values match within 1e-9.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
