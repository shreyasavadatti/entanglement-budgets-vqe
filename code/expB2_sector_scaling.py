"""Experiment B2. Growth of the sector enhancement with system size.

Ordered periodic TFIM (h/J = 0.5). For n = 6..12 we compare the global gap
(set by the finite-size Z2 partner) with the gap inside the parity sector that
holds the ground state, and fit log(Delta_q / Delta) against n.
"""
import json
import numpy as np
import certlib as C

rows = []
for n in (6, 8, 10, 12):
    H = C.build_TFIM(n, h=0.5, periodic=True)
    w = np.linalg.eigvalsh(H)
    gap = w[1] - w[0]
    sec = C.sector_spectrum(H, n, k=2)
    s0 = '+' if sec['+']['E'][0] <= sec['-']['E'][0] else '-'
    gap_q = sec[s0]['E'][1] - sec[s0]['E'][0]
    rows.append(dict(n=n, gap=float(gap), gap_sector=float(gap_q),
                     ratio=float(gap_q / gap)))
    print(f"n={n:2d}  global gap={gap:.3e}  sector gap={gap_q:.4f}  ratio={gap_q/gap:.3g}")

ns = np.array([r['n'] for r in rows], float)
slope = np.polyfit(ns, np.log([r['ratio'] for r in rows]), 1)[0]
print(f"fit: ratio grows as exp({slope:.3f} n)")
json.dump(dict(rows=rows, slope=float(slope)), open('expB2_sector_scaling.json', 'w'), indent=1)
