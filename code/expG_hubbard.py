"""Experiment G. 1D Fermi-Hubbard (open chain, L=6, U/t=4), ordering-dependent
bottleneck. Built directly under Jordan-Wigner with numpy/scipy only and
diagonalised in the full 12-qubit space."""
import numpy as np, json
from scipy.sparse import identity, kron, csr_matrix
from scipy.sparse.linalg import eigsh
import certlib as C

L, U, t = 6, 4.0, 1.0
nq = 2 * L
I2 = identity(2, format='csr')
Zs = csr_matrix(np.diag([1., -1.]))
Sm = csr_matrix(np.array([[0., 1.], [0., 0.]]))   # annihilation on a qubit


def jw_c(p, nq):
    """JW annihilation operator on mode p."""
    ops = [Zs] * p + [Sm] + [I2] * (nq - p - 1)
    M = ops[0]
    for o in ops[1:]:
        M = kron(M, o, format='csr')
    return M


def hubbard(order):
    c = [jw_c(order(i, s), nq) for i in range(L) for s in (0, 1)]
    idx = {(i, s): k for k, (i, s) in enumerate([(i, s) for i in range(L) for s in (0, 1)])}
    H = csr_matrix((2 ** nq, 2 ** nq))
    for i in range(L - 1):
        for s in (0, 1):
            a, b = c[idx[(i, s)]], c[idx[(i + 1, s)]]
            H = H - t * (a.getH() @ b + b.getH() @ a)
    for i in range(L):
        nu = c[idx[(i, 0)]].getH() @ c[idx[(i, 0)]]
        nd = c[idx[(i, 1)]].getH() @ c[idx[(i, 1)]]
        H = H + U * (nu @ nd)
    return H


res = {}
for name, order in [('interleaved', lambda i, s: 2 * i + s),
                    ('spin-sector', lambda i, s: i + s * L)]:
    H = hubbard(order)
    w, v = eigsh(H, k=2, which='SA', tol=1e-12, v0=np.ones(2 ** nq))
    o = np.argsort(w)
    E0, gap = w[o][0], w[o][1] - w[o][0]
    psi = np.asarray(v[:, o[0]]).ravel()
    svs = C.cut_data(psi, nq)
    fl = [float(gap * (1 - C.Lam(svs[k], 2))) for k in range(nq - 1)]
    S = []
    for k in range(nq - 1):
        p = np.sort(svs[k] ** 2)[::-1]
        p = p[p > 1e-15]
        S.append(float(-(p * np.log2(p)).sum()))
    # the open chain is reflection symmetric, so cuts k and nq-k are equivalent;
    # report the one nearer the left end
    k = int(np.argmax(fl)) + 1
    peak_cut = min(k, nq - k)
    res[name] = dict(E0=float(E0), gap=float(gap), floors=fl, entropy=S,
                     peak=float(max(fl)), peak_cut=peak_cut)
    print(f"{name:12s} E0={E0:.4f} gap={gap:.4f} peak floor={max(fl):.4f} at cut {peak_cut}")

r = res['interleaved']['peak'] / res['spin-sector']['peak']
print(f"\nbottleneck ratio interleaved/spin-sector = {r:.2f}x")
print(f"E0 identical to {abs(res['interleaved']['E0']-res['spin-sector']['E0']):.2e} (relabelling)")
# Particle-number structure. H has no chemical potential, so its global ground
# state need not be half filled. Check <N> on the lowest states and the lowest
# energy in the N = L (half-filled) block.
H = hubbard(lambda i, s: 2 * i + s)
N = np.array([bin(k).count('1') for k in range(2 ** nq)])
w, v = eigsh(H, k=4, which='SA', tol=1e-12, v0=np.ones(2 ** nq))
o = np.argsort(w)
levels = [dict(E=float(w[j]), N=float(np.sum(N * np.abs(v[:, j]) ** 2))) for j in o[:3]]
half = np.where(N == L)[0]
E_half = float(eigsh(H[half][:, half], k=1, which='SA', tol=1e-12)[0][0])
for lv in levels:
    print(f"E={lv['E']:.4f}  <N>={lv['N']:.4f}")
print(f"lowest half-filled (N={L}) energy: {E_half:.4f}")
res['number_check'] = dict(levels=levels, E_half_filled=E_half)

json.dump(res, open('expG_hubbard.json', 'w'), indent=1)
