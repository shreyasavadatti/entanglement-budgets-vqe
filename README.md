# Entanglement budgets and certified energy floors for variational quantum circuits

Paper and code for *Entanglement budgets and certified energy floors for
variational quantum circuits* (Shreya Savadatti, Information Networking
Institute, Carnegie Mellon University).

The paper bounds, before any training, the best energy a fixed VQE circuit can
reach. It reads two budgets off each cut of the register.

The rank budget is the number of entangling gates `g_C` crossing cut `C`. It gives

```
E*_A - E_0  >=  max_C  Delta * (1 - Lambda_C(2^{g_C}))
```

where `Delta` is the spectral gap and `Lambda_C(R)` is the largest squared
fidelity a Schmidt-rank-`R` state can have with the ground state across `C`.
Within the gap-and-fidelity framework, this single-cut maximum comes within a
factor of at most the number of constrained cuts of the strongest floor implied
jointly by all per-cut rank constraints. It switches off once `g_C` reaches
roughly `log2` of the target's effective Schmidt rank, and placing gates to
minimise it does not reliably improve trained VQE energies on either a
disordered Ising ensemble or a connected disordered-Heisenberg control.

The angle budget is the sum of reduced entangling angles of the gates crossing
the cut. A gate `exp(-i theta P x Q)` moves any state at most
`min_{k in Z} |theta - k*pi/2|` in Fubini-Study angle from every Schmidt-rank
class, yielding a hybrid rank-and-angle certificate for any number of entanglers
per cut. It contains the rank floor, is tight for one entangler per cut and for
rank-two targets under arbitrary cut-local control, is met to four decimals by
trained circuits on a four-qubit singlet test, and remains informative on the
critical Ising chain at depths where rank counting certifies nothing. A
symmetry-sector version replaces the global gap by the larger in-sector gap.

## Layout

```
paper/
  main.tex        REVTeX 4.2 source (bibliography embedded in the file)
  main.pdf        compiled paper
  figures/        the nine figures used in the paper
code/
  certlib.py      Hamiltonians, Schmidt spectra, certificates, VQE with
                  adjoint gradients, parity sectors, free-fermion spectra,
                  MPS fitting, multi-block joint bound,
                  angle-budget certificate
  exp*.py         one script per experiment (see below)
  make_figures.py builds every figure (including schematic.png) from result files
  run_all.sh      runs everything in order
  walkthrough.ipynb  short notebook that reads the results
  *.json, *.log   results and console output from the run used in the paper
  figures/        figures written by make_figures.py
```

| Script | What it produces |
|---|---|
| `expDE_window_ff.py` | informative window, free-fermion scaling to n = 128 |
| `expB_sector.py` | symmetry-sector certificate at n = 8 |
| `expB2_sector_scaling.py` | sector enhancement for n = 6 to 12 |
| `expF_angle_sched.py` | gate schedules, XXZ chain, core tables |
| `expI_angle_budget.py` | angle-budget certificate: random-circuit check, singlet tightness, critical TFIM |
| `expC_joint.py` | joint multi-cut bracket from MPS fitting |
| `expA_allocation.py` | disordered TFIM allocation ensemble (16 chains, 5 budgets, 4 layouts) |
| `expA_analyse.py` | paired statistics for the TFIM allocation ensemble |
| `expA_mechanism.py` | within-instance test and fragmentation test |
| `expH_multiblock.py` | two-sided bound on joint multi-cut capacity (multi-block and sum bounds) |
| `expJ_heisenberg.py` | connected-budget allocation control on 16 disordered Heisenberg chains |
| `expG_hubbard.py` | Fermi-Hubbard ordering comparison and particle-number check |

## Running

```
pip install -r requirements.txt
cd code
bash run_all.sh
```

Only NumPy, SciPy and Matplotlib are needed. The full run takes about 33
minutes on a single core; the two allocation ensembles account for most of that.
Random seeds are fixed in each script, so the results match the files included
here.

To rebuild the paper, copy `code/figures/*.png` into `paper/figures/` and run
`pdflatex main && bibtex main && pdflatex main && pdflatex main` inside `paper/`.

## Scope of the numbers

Exact diagonalisation is used up to n = 12. The large-n results use the
free-fermion solution of the open transverse-field Ising chain, which needs no
truncation but is evaluated in floating point, so those values are accurate to
the digits shown rather than interval-certified. Trained VQE energies come from
finitely many restarts and are upper bounds on the best energy each circuit can
reach.

## License

MIT, see `LICENSE`.
