#!/usr/bin/env bash
# Regenerates every result file and figure used in the paper.
# Run from this directory. Total time is about 33 minutes on a laptop core.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python3}

$PY expDE_window_ff.py      | tee expDE.log        # window and free-fermion scaling
$PY expG_hubbard.py         | tee expG.log         # Fermi-Hubbard orderings
$PY expB_sector.py          | tee expB.log         # symmetry-sector certificate
$PY expB2_sector_scaling.py | tee expB2.log        # sector enhancement vs n
$PY expC_joint.py           | tee expC.log         # joint multi-cut bracket
$PY expF_angle_sched.py     | tee expF.log         # schedules, XXZ, core tables
$PY expI_angle_budget.py    | tee expI_angle_budget.log   # angle-budget certificate
$PY expA_allocation.py 0 16 | tee expA.log         # TFIM allocation ensemble
$PY expA_analyse.py         | tee expA_analyse.log
$PY expA_mechanism.py       | tee expA_mechanism.log
$PY expH_multiblock.py      | tee expH_multiblock.log     # two-sided joint bound (needs expA and expC)
$PY expJ_heisenberg.py      | tee expJ_heisenberg.log     # connected Heisenberg allocation control
$PY make_figures.py
