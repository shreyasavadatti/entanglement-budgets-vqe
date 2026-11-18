"""
Experiment I.  The angle-budget certificate with several entanglers per cut.

I1  Random circuits: a product state across a cut, g gates exp(-i theta Z x Z)
    across it, Haar-random unitaries on each side between gates. Checks the
    distance lemma Lambda_psi(K) >= cos^2(sum of angle costs of the paid gates).
I2  Tightness: n=4 singlet target, p entanglers of fixed angle on the central
    bond, all rotations trained. Aligned gates can reach the bound.
I3  Critical TFIM, n=8, p layers of exp(-i theta ZZ) on every bond at fixed
    theta, all rotations trained. Compares the angle-budget floor with the rank
    floor, the earlier triangle floor, and the trained error.
"""
import json
import numpy as np
from scipy.stats import unitary_group
import certlib as C

out = {}

print("I1  distance lemma on random circuits")
rng = np.random.default_rng(3)


def zz_phase(n, cut):
    i = np.arange(2 ** n)
    same = ((i >> (n - cut)) & 1) == ((i >> (n - cut - 1)) & 1)
    return np.where(same, 1.0, -1.0)


checks, worst, tightest = 0, -np.inf, np.inf
for t in range(600):
    n = 6
    cut = int(rng.integers(1, n))
    g = int(rng.integers(1, 5))
    th = rng.uniform(-0.6, 0.6, g) if t % 2 else rng.uniform(-np.pi, np.pi, g)
    a = rng.normal(size=2 ** cut) + 1j * rng.normal(size=2 ** cut)
    b = rng.normal(size=2 ** (n - cut)) + 1j * rng.normal(size=2 ** (n - cut))
    psi = np.kron(a / np.linalg.norm(a), b / np.linalg.norm(b))
    z = zz_phase(n, cut)
    for x in th:
        UA = unitary_group.rvs(2 ** cut, random_state=rng)
        UB = unitary_group.rvs(2 ** (n - cut), random_state=rng)
        psi = np.kron(UA, UB) @ psi
        psi = np.exp(-1j * x * z) * psi
    sv = C.schmidt_vals(psi, cut, n)
    cost = sorted(C.angle_cost(x) for x in th)
    for m in range(g + 1):
        Th = sum(cost[:m])
        if Th < np.pi / 2:
            gap_ = C.Lam(sv, 2 ** (g - m)) - np.cos(Th) ** 2
            worst = max(worst, -gap_)
            tightest = min(tightest, gap_)
            checks += 1
print(f"  {checks} checks; largest violation of Lambda_psi(K) >= cos^2(Theta): {worst:.1e}")
out['I1'] = dict(checks=checks, max_violation=float(worst))

print("\nI2  singlet target, p entanglers on the central bond")
n = 4
H = (C.kron_op2(C.X, C.X, 1, 2, n).real + C.kron_op2(C.Y, C.Y, 1, 2, n).real
     + C.kron_op2(C.Z, C.Z, 1, 2, n).real) - 3 * (C.kron_op(C.Z, 0, n).real
                                                  + C.kron_op(C.Z, 3, n).real)
w, v = np.linalg.eigh(H)
E0, gap = w[0], w[1] - w[0]
sv = C.schmidt_vals(v[:, 0], 2, n)
rows = []
for p in (1, 2, 3):
    for th in (0.05, 0.10, 0.15, 0.25):
        A = C.Ansatz(n, [0, p, 0], L=p, kind='hea_zz')
        e = C.train(H, A, restarts=20, seed=11, maxiter=800, fixed=np.full(p, th)) - E0
        thetas = [th] * p
        nb, m = C.budget_floor(sv, gap, thetas, return_m=True)
        tri = C.triangle_floor_multi(sv, gap, thetas)
        rk = gap * (1 - C.Lam(sv, 2 ** p))
        rows.append(dict(p=p, theta=th, err=float(e), budget=float(nb), m=m,
                         triangle=float(tri), rank=float(rk)))
        print(f"  p={p} th={th:.2f}  trained={e:.4f}  budget={nb:.4f} (m={m})  "
              f"triangle={tri:.4f}  rank={rk:.1e}  ratio={e / max(nb, 1e-12):.4f}")
out['I2'] = dict(E0=float(E0), gap=float(gap), rows=rows)

print("\nI3  critical TFIM n=8, p layers of exp(-i theta ZZ) on every bond")
n = 8
H = C.build_TFIM(n, h=1.0, periodic=False)
w, v = np.linalg.eigh(H)
E0, gap = w[0], w[1] - w[0]
svs = C.cut_data(v[:, 0], n)
print(f"  E0={E0:.4f}  gap={gap:.4f}")
rows = []
for p in (1, 2, 3, 4):
    for th in (0.02, 0.05, 0.10, 0.20, np.pi / 4):
        A = C.Ansatz(n, [p] * (n - 1), L=p, kind='hea_zz')
        e = C.train(H, A, restarts=10, seed=21, maxiter=800,
                    fixed=np.full(p * (n - 1), th)) - E0
        thetas = [th] * p
        nb = max(C.budget_floor(svs[k], gap, thetas) for k in range(n - 1))
        tri = max(C.triangle_floor_multi(svs[k], gap, thetas) for k in range(n - 1))
        rk = max(gap * (1 - C.Lam(svs[k], 2 ** p)) for k in range(n - 1))
        rows.append(dict(p=p, theta=float(th), err=float(e), budget=float(nb),
                         triangle=float(tri), rank=float(rk)))
        print(f"  p={p} th={th:.3f}  trained={e:.4f}  budget={nb:.2e}  "
              f"triangle={tri:.2e}  rank={rk:.2e}")
out['I3'] = dict(E0=float(E0), gap=float(gap), rows=rows)

json.dump(out, open('expI_angle_budget.json', 'w'), indent=1)
print("\nwrote expI_angle_budget.json")
