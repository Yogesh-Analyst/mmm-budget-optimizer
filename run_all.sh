#!/usr/bin/env bash
# Rebuild data, refit both models, optimise budgets and redraw charts (~1-2 min on a laptop).
set -euo pipefail
cd "$(dirname "$0")/src"
# Numba backend: avoids PyTensor's C compiler step, which fails on recent macOS toolchains.
export PYTENSOR_FLAGS="mode=NUMBA,cxx="
python prepare_data.py
for d in robyn edtech; do
  python fit_mmm.py "$d"
  python optimizer.py "$d"
  python report.py "$d"
done
