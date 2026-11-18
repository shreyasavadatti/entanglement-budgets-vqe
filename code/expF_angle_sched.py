"""
Experiment F.  Remaining validations and the angle-resolved certificate.

F1  Angle-resolved certificate on the n=4 Heisenberg singlet (tight single-entangler bound).
F2  Angle sweep on the critical TFIM with a parity-preserving ansatz.
F3  Scheduling test: identical g_C, different schedule -> identical certificate.
    Also the gate-count-matched staggered vs all-NN comparison.
F4  XXZ: does the gap-weighted certificate order the regimes better than entropy?
F5  Core tables: single-cut floors, and trained VQE vs the floor at n=4.
"""
import numpy as np, json
import certlib as C

out = {}

# F1
print("F1  angle-resolved certificate, n=4 Heisenberg singlet")
n = 4
H = (C.kron_op2(C.X, C.X, 1, 2, n).real + C.kron_op2(C.Y, C.Y, 1, 2, n).real
     + C.kron_op2(C.Z, C.Z, 1, 2, n).real) - 3 * (C.kron_op(C.Z, 0, n).real
                                                  + C.kron_op(C.Z, 3, n).real)
w, v = np.linalg.eigh(H)
E0, gap = w[0], w[1] - w[0]
sv = C.schmidt_vals(v[:, 0], 2, n)
L1, L2 = C.Lam(sv, 1), C.Lam(sv, 2)
print(f"  E0={E0:.4f}  gap={gap:.4f}  Lam(1)={L1:.4f}  Lam(2)={L2:.4f}")


def tight_floor(theta, L1, L2, gap):
    s = min(abs(np.sin(theta)), abs(np.cos(theta)))
    bstar = np.sqrt(max(L2 - L1, 0) / L2) if L2 > 0 else 0.0
    b = min(s, bstar)
    O = np.sqrt(1 - b ** 2) * np.sqrt(L1) + b * np.sqrt(max(L2 - L1, 0))
    return gap * (1 - min(O, 1.0) ** 2), s


def triangle_floor(theta, L1, gap):
    e = abs(np.sin(theta))
    x = np.sqrt(L1) + e
    return gap * (1 - min(x, 1.0) ** 2)


rowsF1 = []
for th in [0.05, 0.15, 0.30, 0.50, np.pi / 4]:
    A = C.Ansatz(n, [0, 1, 0], L=1, kind='hea_zz')      # one ZZ across the central cut
    e = C.train(H, A, restarts=20, seed=11, maxiter=600,
                fixed=np.array([th])) - E0
    tf, s = tight_floor(th, L1, L2, gap)
    tr = triangle_floor(th, L1, gap)
    rowsF1.append(dict(theta=float(th), s=float(s), err=float(e),
                       exact_rank=float(gap * (1 - L2)), tight=float(tf), tri=float(tr)))
    print(f"  th={th:.4f} s={s:.4f}  trained={e:.4f}  exact-rank={gap*(1-L2):.1e}"
          f"  tight={tf:.4f}  triangle={tr:.4f}   saturation={e/max(tf,1e-12):.3f}")
out['F1'] = dict(E0=float(E0), gap=float(gap), Lam1=float(L1), Lam2=float(L2), rows=rowsF1)

# F2
print("\nF2  one-sweep all-bond ZZ circuit, critical TFIM n=6 (open)")
print("  The tight single-entangler bound applies at every cut: for cut k the")
print("  other ZZ gates and all rotations are cut-local.")
n = 6
H = C.build_TFIM(n, h=1.0, periodic=False)
w, v = np.linalg.eigh(H)
E0, gap = w[0], w[1] - w[0]
svs = C.cut_data(v[:, 0], n)
print(f"  E0={E0:.4f}  gap={gap:.4f}")
rowsF2 = []
for th in [0.02, 0.05, 0.15, 0.30, 0.50, np.pi / 4]:
    A = C.Ansatz(n, [1] * (n - 1), L=1, kind='hea_zz')
    e = C.train(H, A, restarts=20, seed=21, maxiter=600,
                fixed=np.full(n - 1, th)) - E0
    ex = max(gap * (1 - C.Lam(svs[k], 2)) for k in range(n - 1))
    ang = max(tight_floor(th, C.Lam(svs[k], 1), C.Lam(svs[k], 2), gap)[0]
              for k in range(n - 1))
    tri = max(triangle_floor(th, C.Lam(svs[k], 1), gap) for k in range(n - 1))
    rowsF2.append(dict(theta=float(th), err=float(e), exact_rank=float(ex),
                       angle=float(ang), tri=float(tri)))
    print(f"  th={th:.4f}  trained={e:.4f}  exact-rank={ex:.2e}  "
          f"angle-resolved={ang:.4f}  triangle={tri:.4f}")
out['F2'] = dict(E0=float(E0), gap=float(gap), rows=rowsF2)

# F3
print("\nF3  scheduling: identical g_C, different schedule")
n = 8
H = C.build_TFIM(n, h=1.0, periodic=True)
w, v = np.linalg.eigh(H)
E0, gap = w[0], w[1] - w[0]
svs = C.cut_data(v[:, 0], n)
scheds = {
    'all-NN, 2 layers      (14 gates, depth 2)': [list(range(7))] * 2,
    'staggered, 2 layers   ( 7 gates, depth 2)': [[0, 2, 4, 6], [1, 3, 5]],
    'staggered, 4 layers   (14 gates, depth 4)': [[0, 2, 4, 6], [1, 3, 5]] * 2,
}
rowsF3 = []
for name, sch in scheds.items():
    A = C.Ansatz(n, schedule=sch, kind='hea')
    fl = C.certificate(svs, gap, A.alloc)
    e = C.train(H, A, restarts=12, seed=31, maxiter=400) - E0
    rowsF3.append(dict(name=name, alloc=A.alloc, ngates=int(sum(A.alloc)),
                       depth=A.L, floor=float(fl), err=float(e)))
    print(f"  {name}: g_C={A.alloc}  floor={fl:.2e}  trained err={e:.4f}")
out['F3'] = dict(E0=float(E0), gap=float(gap), rows=rowsF3)

# F4
print("\nF4  XXZ n=8 periodic: entropy vs certificate vs trained error")
rowsF4 = []
for Jz in [0.0, 0.5, 1.0, 2.0, 4.0]:
    H = C.build_XXZ(8, 1.0, Jz, periodic=True)
    w, v = np.linalg.eigh(H)
    E0, gap = w[0], w[1] - w[0]
    sv = C.schmidt_vals(v[:, 0], 4, 8)
    p = np.sort(sv ** 2)[::-1]
    S = float(-(p * np.log2(p)).sum())
    fl = gap * (1 - C.Lam(sv, 2))
    A = C.Ansatz(8, [1] * 7, L=1, kind='hea')
    e = C.train(H, A, restarts=12, seed=41, maxiter=400) - E0
    rowsF4.append(dict(Jz=Jz, S_bits=S, gap=float(gap), Lam2=float(C.Lam(sv, 2)),
                       floor=float(fl), err=float(e)))
    print(f"  Jz={Jz:4.1f}  S={S:.3f} bits  gap={gap:.4f}  Lam(2)={C.Lam(sv,2):.4f}  "
          f"floor={fl:.3e}  trained err={e:.4f}")
o = rowsF4
from scipy.stats import spearmanr
print("  rank corr with trained error:  entropy rho=%.3f   certificate rho=%.3f" % (
    spearmanr([r['S_bits'] for r in o], [r['err'] for r in o]).statistic,
    spearmanr([r['floor'] for r in o], [r['err'] for r in o]).statistic))
out['F4'] = rowsF4

# F5
print("\nF5  core tables")
H = C.build_TFIM(6, h=1.0, periodic=True)
w, v = np.linalg.eigh(H)
E0, gap = w[0], w[1] - w[0]
sv = C.schmidt_vals(v[:, 0], 3, 6)
t5 = [dict(g=g, R=2 ** g, Lam=float(C.Lam(sv, 2 ** g)),
           floor=float(gap * (1 - C.Lam(sv, 2 ** g)))) for g in range(4)]
print(f"  TFIM n=6 periodic critical: E0={E0:.4f} gap={gap:.4f}")
for r in t5:
    print(f"    g={r['g']} R={r['R']:2d} Lam={r['Lam']:.4f} floor={r['floor']:.2e}")

H4 = C.build_TFIM(4, h=1.0, periodic=True)
w4, v4 = np.linalg.eigh(H4)
E04, gap4 = w4[0], w4[1] - w4[0]
sv4 = C.schmidt_vals(v4[:, 0], 2, 4)
t6 = []
for g in range(3):
    fl = gap4 * (1 - C.Lam(sv4, 2 ** g))
    A = C.Ansatz(4, [g, g, g], L=3, kind='hea')
    e = C.train(H4, A, restarts=24, seed=51, maxiter=800)
    t6.append(dict(g=g, floor=float(E04 + fl), evqe=float(e), margin=float(e - E04 - fl)))
    print(f"    n=4 g={g}: Efloor={E04+fl:.4f}  E_VQE={e:.4f}  margin={e-E04-fl:.2e}")
out['F5'] = dict(single_cut=dict(E0=float(E0), gap=float(gap), rows=t5),
                 n4=dict(E0=float(E04), gap=float(gap4), rows=t6))

json.dump(out, open('expF.json', 'w'), indent=1)
print("\nwrote expF.json")
