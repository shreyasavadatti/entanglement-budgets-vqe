"""
certlib.py -- core routines for entanglement-capacity certificates.

Pure numpy/scipy. Contains:
  * Hamiltonian builders (TFIM open/periodic, disordered, XXZ, Fermi-Hubbard)
  * Schmidt spectra and the capacity function Lambda_C(R)
  * fast statevector VQE with adjoint (analytic) gradients
  * symmetry-sector (Z2 parity) resolved spectra
  * exact free-fermion Schmidt spectra for the open TFIM (any n)
  * nonuniform-bond-dimension MPS fitting (joint-capacity diagnostic)
"""
import numpy as np
from scipy.linalg import svd, schur
from scipy.optimize import minimize
import heapq

# Pauli / Hamiltonian builders (dense; n <= 12)
I2 = np.eye(2)
X = np.array([[0., 1.], [1., 0.]])
Y = np.array([[0., -1j], [1j, 0.]])
Z = np.diag([1., -1.])


def kron_op(P, i, n):
    M = np.eye(1)
    for k in range(n):
        M = np.kron(M, P if k == i else I2)
    return M


def kron_op2(P, Q, i, j, n):
    M = np.eye(1)
    for k in range(n):
        M = np.kron(M, P if k == i else (Q if k == j else I2))
    return M


def build_TFIM(n, J=1.0, h=1.0, periodic=False, hs=None):
    """H = -J sum ZZ - sum_i h_i X_i.  hs overrides uniform h."""
    H = np.zeros((2 ** n, 2 ** n))
    bonds = list(range(n - 1)) + ([n - 1] if periodic else [])
    for i in bonds:
        H -= J * kron_op2(Z, Z, i, (i + 1) % n, n).real
    fields = np.full(n, h) if hs is None else np.asarray(hs, float)
    for i in range(n):
        H -= fields[i] * kron_op(X, i, n).real
    return H


def build_XXZ(n, Jxy=1.0, Jz=0.5, periodic=True):
    H = np.zeros((2 ** n, 2 ** n))
    bonds = list(range(n - 1)) + ([n - 1] if periodic else [])
    for i in bonds:
        j = (i + 1) % n
        H += 0.5 * Jxy * (kron_op2(X, X, i, j, n).real + kron_op2(Y, Y, i, j, n).real)
        H += Jz * kron_op2(Z, Z, i, j, n).real
    return H


# Schmidt spectra and capacity
def schmidt_vals(psi, cut, n, tol=1e-13):
    """Singular values of psi across the contiguous cut |A|=cut."""
    s = svd(psi.reshape(2 ** cut, 2 ** (n - cut)), compute_uv=False)
    return s[s > tol]


def Lam(svals, R):
    """Lambda_C(R): largest squared fidelity of any rank-R state with the target."""
    p = np.sort(svals ** 2)[::-1]
    return float(p[:min(R, len(p))].sum())


def R_needed(svals, eps):
    """R_C(eps) = min{R : 1 - Lambda_C(R) <= eps}   (triviality threshold input)."""
    p = np.sort(svals ** 2)[::-1]
    c = np.cumsum(p)
    idx = np.searchsorted(c, 1.0 - eps)
    return int(min(idx + 1, len(p)))


def cut_data(psi, n):
    return [schmidt_vals(psi, c, n) for c in range(1, n)]


def per_cut_floors(svs, gap, alloc):
    """phi_k(g_k) = gap * (1 - Lambda_k(2^{g_k})) for each cut."""
    return np.array([gap * (1.0 - Lam(svs[k], 2 ** alloc[k])) for k in range(len(svs))])


def certificate(svs, gap, alloc):
    return float(per_cut_floors(svs, gap, alloc).max())


# Allocation strategies
def waterfill(svs, gap, G, nb, floor_start=None):
    """Greedy min-max water-filling of the per-cut floors."""
    a = list(floor_start) if floor_start is not None else [0] * nb
    for _ in range(G - sum(a)):
        vals = [gap * (1.0 - Lam(svs[k], 2 ** a[k])) for k in range(nb)]
        a[int(np.argmax(vals))] += 1
    return a


def layer_filled(G, nb):
    """Standard hardware-efficient partial-layer fill, left to right."""
    base, rem = divmod(G, nb)
    return [base + (1 if k < rem else 0) for k in range(nb)]


def connected_waterfill(svs, gap, G, nb):
    """g_C >= 1 on every bond, then water-fill the residual budget."""
    assert G >= nb, "connected allocation needs G >= n-1"
    return waterfill(svs, gap, G, nb, floor_start=[1] * nb)


def connected_random(G, nb, rng):
    """Control arm: connectivity enforced, residual budget placed at random."""
    a = [1] * nb
    for _ in range(G - nb):
        a[int(rng.integers(nb))] += 1
    return a


# Fast statevector VQE with adjoint (analytic) gradients
def _ry_(st, t, q, n, inv=False):
    th = -t if inv else t
    c, s = np.cos(th / 2), np.sin(th / 2)
    v = st.reshape(2 ** q, 2, 2 ** (n - 1 - q))
    a, b = v[:, 0, :].copy(), v[:, 1, :]
    v[:, 0, :] = c * a - s * b
    v[:, 1, :] = s * a + c * b
    return st


def _rz_(st, t, q, n, inv=False):
    th = -t if inv else t
    e = np.exp(-0.5j * th)
    v = st.reshape(2 ** q, 2, 2 ** (n - 1 - q))
    v[:, 0, :] *= e
    v[:, 1, :] *= np.conj(e)
    return st


def _rx_(st, t, q, n, inv=False):
    th = -t if inv else t
    c, s = np.cos(th / 2), -1j * np.sin(th / 2)
    v = st.reshape(2 ** q, 2, 2 ** (n - 1 - q))
    a, b = v[:, 0, :].copy(), v[:, 1, :]
    v[:, 0, :] = c * a + s * b
    v[:, 1, :] = s * a + c * b
    return st


def _cx_(st, k, n):
    """CNOT on adjacent (k, k+1); self-inverse."""
    v = st.reshape(2 ** k, 4, 2 ** (n - 2 - k))
    v[:, [2, 3], :] = v[:, [3, 2], :]
    return st


def _zz_(st, t, k, n, inv=False):
    """exp(-i t Z_k Z_{k+1}); diagonal, self-inverse structure."""
    th = -t if inv else t
    v = st.reshape(2 ** k, 4, 2 ** (n - 2 - k))
    a, b = np.exp(-1j * th), np.exp(1j * th)
    v[:, 0, :] *= a
    v[:, 3, :] *= a
    v[:, 1, :] *= b
    v[:, 2, :] *= b
    return st


def _pauli_expect(lam, psi, P, q, n):
    """<lam| P_q |psi> without allocating P|psi>."""
    L = lam.reshape(2 ** q, 2, 2 ** (n - 1 - q))
    S = psi.reshape(2 ** q, 2, 2 ** (n - 1 - q))
    if P == 'Z':
        return np.vdot(L[:, 0, :], S[:, 0, :]) - np.vdot(L[:, 1, :], S[:, 1, :])
    if P == 'Y':
        return -1j * np.vdot(L[:, 0, :], S[:, 1, :]) + 1j * np.vdot(L[:, 1, :], S[:, 0, :])
    if P == 'X':
        return np.vdot(L[:, 0, :], S[:, 1, :]) + np.vdot(L[:, 1, :], S[:, 0, :])
    raise ValueError(P)


def _zz_expect(lam, psi, k, n):
    L = lam.reshape(2 ** k, 4, 2 ** (n - 2 - k))
    S = psi.reshape(2 ** k, 4, 2 ** (n - 2 - k))
    return (np.vdot(L[:, 0, :], S[:, 0, :]) + np.vdot(L[:, 3, :], S[:, 3, :])
            - np.vdot(L[:, 1, :], S[:, 1, :]) - np.vdot(L[:, 2, :], S[:, 2, :]))


class Ansatz:
    """
    Layered ansatz.  kind='hea'   : Ry,Rz on every qubit + CNOTs on scheduled bonds
                     kind='parity': Rx on every qubit + exp(-i t ZZ) on scheduled bonds
                                    (commutes with P = prod_i X_i; init |+...+>)
    alloc[k] = number of entanglers on bond k; entangler present in layer l iff l < alloc[k].
    L = number of entangling layers (>= max(alloc)); fixed across arms for fair comparison.
    """

    def __init__(self, n, alloc=None, L=None, kind='hea', schedule=None):
        self.schedule = [list(x) for x in schedule] if schedule is not None else None
        if self.schedule is not None:
            alloc = [sum(1 for lay in self.schedule if k in lay) for k in range(n - 1)]
            L = len(self.schedule)
        self.n, self.alloc, self.kind = n, list(alloc), kind
        self.L = int(max(max(alloc), 1) if L is None else L)
        self.rot = 1 if kind == 'parity' else 2
        self.n_rot = self.rot * n * (self.L + 1)
        self.n_ent = (sum(min(a, self.L) for a in alloc)
                      if kind in ('parity', 'hea_zz') else 0)
        self.npar = self.n_rot + self.n_ent

    def _init_state(self):
        st = np.zeros(2 ** self.n, complex)
        if self.kind == 'parity':
            st[:] = 2 ** (-self.n / 2)          # |+>^n : parity P=+1 eigenstate
        else:
            st[0] = 1.0
        return st

    def _tape(self, p):
        """List of (type, param_index_or_None, qubit/bond) in application order."""
        ops, i, e = [], 0, self.n_rot
        for l in range(self.L):
            for q in range(self.n):
                if self.kind == 'parity':
                    ops.append(('rx', i, q)); i += 1
                else:
                    ops.append(('ry', i, q)); i += 1
                    ops.append(('rz', i, q)); i += 1
            active = self.schedule[l] if self.schedule is not None else \
                     [k for k in range(self.n - 1) if l < self.alloc[k]]
            for k in active:
                if True:
                    if self.kind in ('parity', 'hea_zz'):
                        ops.append(('zz', e, k)); e += 1
                    else:
                        ops.append(('cx', None, k))
        for q in range(self.n):
            if self.kind == 'parity':
                ops.append(('rx', i, q)); i += 1
            else:
                ops.append(('ry', i, q)); i += 1
                ops.append(('rz', i, q)); i += 1
        return ops

    def state(self, p):
        st = self._init_state()
        n = self.n
        for typ, ip, q in self._tape(p):
            if typ == 'ry': _ry_(st, p[ip], q, n)
            elif typ == 'rz': _rz_(st, p[ip], q, n)
            elif typ == 'rx': _rx_(st, p[ip], q, n)
            elif typ == 'cx': _cx_(st, q, n)
            elif typ == 'zz': _zz_(st, p[ip], q, n)
        return st

    def energy_and_grad(self, p, H):
        """Adjoint differentiation: all npar gradients in ~2 circuit passes."""
        n = self.n
        tape = self._tape(p)
        psi = self.state(p)
        E = float(np.vdot(psi, H @ psi).real)
        lam = H @ psi
        g = np.zeros(self.npar)
        for typ, ip, q in reversed(tape):
            if typ == 'ry':
                g[ip] = np.imag(_pauli_expect(lam, psi, 'Y', q, n))
                _ry_(psi, p[ip], q, n, inv=True); _ry_(lam, p[ip], q, n, inv=True)
            elif typ == 'rz':
                g[ip] = np.imag(_pauli_expect(lam, psi, 'Z', q, n))
                _rz_(psi, p[ip], q, n, inv=True); _rz_(lam, p[ip], q, n, inv=True)
            elif typ == 'rx':
                g[ip] = np.imag(_pauli_expect(lam, psi, 'X', q, n))
                _rx_(psi, p[ip], q, n, inv=True); _rx_(lam, p[ip], q, n, inv=True)
            elif typ == 'cx':
                _cx_(psi, q, n); _cx_(lam, q, n)
            elif typ == 'zz':
                # exp(-i t ZZ):  dU/dt = -i ZZ U  ->  grad = 2*Re(-i <lam|ZZ|psi>)
                g[ip] = 2.0 * np.imag(_zz_expect(lam, psi, q, n))
                _zz_(psi, p[ip], q, n, inv=True); _zz_(lam, p[ip], q, n, inv=True)
        return E, g


def train(H, ansatz, restarts=8, seed=0, maxiter=300, fixed=None):
    """Best energy over random restarts. `fixed` pins entangling params (angle sweeps)."""
    rng = np.random.default_rng(seed)
    best = np.inf
    Hc = H.astype(complex)
    for _ in range(restarts):
        x0 = rng.uniform(-np.pi, np.pi, ansatz.npar)
        if fixed is not None:
            x0[ansatz.n_rot:] = fixed
            mask = np.zeros(ansatz.npar, bool); mask[ansatz.n_rot:] = True

            def f(pr):
                p = np.empty(ansatz.npar); p[~mask] = pr; p[mask] = fixed
                E, g = ansatz.energy_and_grad(p, Hc)
                return E, g[~mask]
            res = minimize(f, x0[~mask], jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-10})
        else:
            res = minimize(lambda p: ansatz.energy_and_grad(p, Hc), x0, jac=True,
                           method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-10})
        best = min(best, float(res.fun))
    return best


# Z2 parity sector resolution (TFIM):  P = prod_i X_i
def parity_projectors(n):
    """Return (idx_plus, idx_minus) index sets in the X-product eigenbasis."""
    # In the computational basis P = prod X flips every bit.  Work in the
    # Hadamard-rotated basis where P is diagonal with sign (-1)^{popcount}.
    Hm = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
    U = np.eye(1)
    for _ in range(n):
        U = np.kron(U, Hm)
    par = np.array([bin(i).count('1') % 2 for i in range(2 ** n)])
    return U, np.where(par == 0)[0], np.where(par == 1)[0]


def sector_spectrum(H, n, k=3):
    """Eigen-decompose H within each Z2 parity sector.  Returns dict per sector."""
    U, ip, im = parity_projectors(n)
    Hr = U.T @ H @ U                       # real orthogonal similarity
    out = {}
    for tag, idx in (('+', ip), ('-', im)):
        Hs = Hr[np.ix_(idx, idx)]
        w, v = np.linalg.eigh(Hs)
        vecs = []
        for j in range(min(k, len(w))):
            full = np.zeros(2 ** n)
            full[idx] = v[:, j]
            vecs.append(U @ full)          # back to computational basis
        out[tag] = dict(E=w[:k], vecs=vecs)
    return out


# Free-fermion Schmidt spectra for the open TFIM (any n)
def tfim_majorana_Omega(n, J=1.0, h=1.0):
    """H = (i/4) sum Omega_{mn} a_m a_n for H = -J sum ZZ - h sum X (open)."""
    Om = np.zeros((2 * n, 2 * n))
    for j in range(n):                      # -h X_j = -h i a_{2j} a_{2j+1}
        Om[2 * j, 2 * j + 1] = -2 * h
        Om[2 * j + 1, 2 * j] = 2 * h
    for j in range(n - 1):                  # -J Z_jZ_{j+1} = -J i a_{2j+1} a_{2j+2}
        Om[2 * j + 1, 2 * j + 2] = -2 * J
        Om[2 * j + 2, 2 * j + 1] = 2 * J
    return Om


def tfim_covariance(n, J=1.0, h=1.0):
    """Ground-state Majorana covariance Gamma and single-particle energies eps."""
    Om = tfim_majorana_Omega(n, J, h)
    T, O = schur(Om, output='real')         # O.T @ Om @ O = T ; O orthogonal
    eps = np.array([T[2 * k, 2 * k + 1] for k in range(n)])
    G = np.zeros_like(T)
    for k in range(n):
        s = np.sign(eps[k]) if eps[k] != 0 else 1.0
        G[2 * k, 2 * k + 1] = s
        G[2 * k + 1, 2 * k] = -s
    Gamma = O @ G @ O.T
    return Gamma, np.abs(eps)


def ff_block_nu(n, cut, J=1.0, h=1.0, Gamma=None):
    """Entanglement-spectrum parameters nu_j in [0,1] for the left block."""
    if Gamma is None:
        Gamma, _ = tfim_covariance(n, J, h)
    GA = Gamma[:2 * cut, :2 * cut]
    w = np.linalg.eigvals(GA)               # +- i nu
    nu = np.sort(np.abs(w.imag))[::-1][::2][:cut]   # +-i nu pairs: take distinct
    return np.clip(nu, 0.0, 1.0)


def top_probs_from_nu(nu, R):
    """Largest R eigenvalues of rho_A from the single-mode parameters nu."""
    p0 = float(np.prod((1.0 + nu) / 2.0))
    ratios = np.sort((1.0 - nu) / (1.0 + nu))[::-1]
    m = len(ratios)
    out, seen = [], set()
    heap = [(-p0, 0, ())]                   # (-prob, next index, flipped set)
    while heap and len(out) < R:
        negp, i, flips = heapq.heappop(heap)
        if flips in seen:
            continue
        seen.add(flips)
        out.append(-negp)
        for j in range(i, m):
            if ratios[j] <= 0:
                break
            nf = tuple(sorted(flips + (j,)))
            if nf not in seen:
                heapq.heappush(heap, (negp * ratios[j], j + 1, nf))
    return np.array(out)


def ff_entropy(nu):
    """von Neumann entropy (nats) from the single-mode parameters nu."""
    p = np.clip((1.0 + np.clip(nu, 0.0, 1.0)) / 2.0, 0.0, 1.0)
    t = np.zeros_like(p)
    for q in (p, 1.0 - p):
        m = q > 0
        t[m] -= q[m] * np.log(q[m])
    return float(t.sum())


# Nonuniform-bond MPS fit, used as a numerical diagnostic for Lambda_joint(R).
#
# States with rank_{C_k}(phi) <= R_k at every contiguous cut are exactly the MPS
# with bond dimensions R_k. Maximising |<psi_0|phi>|^2 over that set gives a lower
# bound on Lambda_joint (the fit may be sub-optimal), hence an upper bound on the
# joint floor. It measures how loose the independent-cut bound might be; it is
# not a stronger certificate.
def mps_to_state(T):
    v = T[0]
    for t in T[1:]:
        v = np.tensordot(v, t, axes=([-1], [0]))
    return v.reshape(-1)


def mps_from_state(psi, n, dims):
    """Sequential-SVD MPS, bond dim on cut k capped by dims[k]."""
    T, resid, Dl = [], np.asarray(psi, complex).reshape(1, -1), 1
    for k in range(n - 1):
        U, s, Vh = svd(resid.reshape(Dl * 2, -1), full_matrices=False)
        D = int(min(dims[k], len(s)))
        T.append(U[:, :D].reshape(Dl, 2, D))
        resid = np.diag(s[:D]).astype(complex) @ Vh[:D, :]
        Dl = D
    T.append(resid.reshape(Dl, 2, 1))
    return T


def _center_right(T, k):
    """Make site k left-canonical, push the remainder into site k+1."""
    Dl, d, Dr = T[k].shape
    Q, R = np.linalg.qr(T[k].reshape(Dl * d, Dr))
    T[k] = Q.reshape(Dl, d, -1)
    T[k + 1] = np.tensordot(R, T[k + 1], axes=([1], [0]))


def _center_left(T, k):
    """Make site k right-canonical, push the remainder into site k-1."""
    Dl, d, Dr = T[k].shape
    M = T[k].reshape(Dl, d * Dr)
    Q, R = np.linalg.qr(M.conj().T)            # M = R^H Q^H, Q^H rows orthonormal
    T[k] = (Q.conj().T).reshape(-1, d, Dr)
    T[k - 1] = np.tensordot(T[k - 1], R.conj().T, axes=([2], [0]))


def _blocks(T, k, n):
    BL = np.ones((1, 1), complex)
    for j in range(k):
        BL = np.tensordot(BL, T[j], axes=([1], [0])).reshape(-1, T[j].shape[2])
    BR = np.ones((1, 1), complex)
    for j in range(n - 1, k, -1):
        BR = np.tensordot(T[j], BR, axes=([2], [0])).reshape(T[j].shape[0], -1)
    return BL, BR


def mps_fit_overlap(psi, n, dims, sweeps=30, seed=0, n_random=2, tol=1e-13):
    """Best |<psi|phi>|^2 over MPS with bond dims <= dims, alternating exact updates."""
    psi = np.asarray(psi, complex)
    psi = psi / np.linalg.norm(psi)
    rng = np.random.default_rng(seed)
    T0 = mps_from_state(psi, n, dims)
    starts = [T0] + [[rng.normal(size=t.shape) + 1j * rng.normal(size=t.shape)
                      for t in T0] for _ in range(n_random)]
    best = 0.0
    for T in starts:
        T = [t.astype(complex).copy() for t in T]
        for k in range(n - 1, 0, -1):          # right-canonical, centre at 0
            _center_left(T, k)
        T[0] /= max(np.linalg.norm(T[0]), 1e-300)
        prev, f = -1.0, 0.0
        for _ in range(sweeps):
            for k in range(n):                 # sweep right
                BL, BR = _blocks(T, k, n)
                env = np.einsum('la,lsr,br->asb', BL,
                                psi.reshape(BL.shape[0], 2, BR.shape[1]).conj(),
                                BR, optimize=True)
                ne = np.linalg.norm(env)
                if ne > 1e-15:
                    T[k] = env.conj() / ne
                    f = float(ne ** 2)
                if k < n - 1:
                    _center_right(T, k)
            for k in range(n - 1, -1, -1):     # sweep left
                BL, BR = _blocks(T, k, n)
                env = np.einsum('la,lsr,br->asb', BL,
                                psi.reshape(BL.shape[0], 2, BR.shape[1]).conj(),
                                BR, optimize=True)
                ne = np.linalg.norm(env)
                if ne > 1e-15:
                    T[k] = env.conj() / ne
                    f = float(ne ** 2)
                if k > 0:
                    _center_left(T, k)
            if abs(f - prev) < tol:
                break
            prev = f
        best = max(best, f)
    return best


# Multi-block bound on the joint capacity.
#
# If phi has Schmidt rank at most R_k across every contiguous cut, then for any
# contiguous block I = [a, b) its reduced state is supported on a subspace of
# dimension at most R_a * R_b (one factor for each cut that bounds I inside the
# chain). Projectors on disjoint blocks commute, so phi lies in the range of
# their product Pi, and |<psi0|phi>|^2 <= <psi0|Pi|psi0>, which is at most the
# sum of the K largest eigenvalues of psi0's reduced state on the union of the
# blocks, K = prod_I rank(I). Taking a single end block recovers the
# independent-cut bound, so the minimum over block choices is never weaker.

def _runs(mask, n):
    """Maximal runs [a, b) of sites with mask True."""
    runs, a = [], None
    for i in range(n + 1):
        on = i < n and mask[i]
        if on and a is None:
            a = i
        elif not on and a is not None:
            runs.append((a, i)); a = None
    return runs


def _block_rank(a, b, n, R):
    """Rank allowed on block [a, b); R[k] is the rank at the cut after site k."""
    r = 1
    if a > 0:
        r *= R[a - 1]
    if b < n:
        r *= R[b - 1]
    return r


def multiblock_capacity(psi, n, R, return_config=False):
    """Upper bound on Lambda_joint(R) for a state vector psi on n qubits.

    R has length n-1. Every split of the chain into projected blocks and traced
    gaps is tried; a block whose allowed rank reaches its full dimension adds no
    constraint and is skipped, since tracing it out instead is never worse.
    """
    psi = np.asarray(psi).reshape([2] * n)
    best, best_cfg = 1.0, None
    for code in range(1, 2 ** n):
        mask = [(code >> (n - 1 - i)) & 1 == 1 for i in range(n)]
        runs = _runs(mask, n)
        K, ok = 1, True
        for a, b in runs:
            r = _block_rank(a, b, n, R)
            if r >= 2 ** (b - a):
                ok = False; break
            K *= r
        if not ok:
            continue
        U = [i for i in range(n) if mask[i]]
        G = [i for i in range(n) if not mask[i]]
        M = np.transpose(psi, U + G).reshape(2 ** len(U), 2 ** len(G))
        p = np.sort(np.linalg.svd(M, compute_uv=False) ** 2)[::-1]
        val = float(min(1.0, p[:K].sum()))
        if val < best:
            best, best_cfg = val, runs
    return (best, best_cfg) if return_config else best


def ff_union_top_probs(Gamma, sites, K):
    """Largest K eigenvalues of a Gaussian state's reduced state on any set of sites."""
    idx = [m for i in sorted(sites) for m in (2 * i, 2 * i + 1)]
    GU = Gamma[np.ix_(idx, idx)]
    w = np.linalg.eigvals(GU)
    nu = np.sort(np.abs(w.imag))[::-1][::2][:len(sites)]
    return top_probs_from_nu(np.clip(nu, 0.0, 1.0), K)


# Angle budget.
#
# A gate exp(-i theta P x Q), with P and Q Hermitian involutions on the two
# sides of a cut, moves any state at most angle_cost(theta) further (in
# Fubini-Study distance) from the set of states of Schmidt rank <= K across that
# cut, for every K; cut-local unitaries do not move it at all. Gates not paid
# for in angle can instead be paid for in rank, each at most doubling it.

def angle_cost(theta):
    """Fubini-Study cost of exp(-i theta P x Q): min(|theta|, pi/2 - |theta|) mod pi/2."""
    t = abs(float(theta)) % (np.pi / 2)
    return min(t, np.pi / 2 - t)


def budget_overlap(svals, R, K, Theta):
    """Largest |<psi0|psi>| over psi of rank <= R within angle Theta of rank K."""
    LK, LR = Lam(svals, K), Lam(svals, R)
    A, B = np.sqrt(LK), np.sqrt(max(LR - LK, 0.0))
    if Theta >= np.pi / 2 or A + B == 0:
        return float(np.sqrt(LR))
    bstar = B / np.sqrt(A * A + B * B)
    b = min(np.sin(Theta), bstar)
    return float(np.sqrt(1 - b * b) * A + b * B)


def budget_floor(svals, gap, thetas, return_m=False):
    """Certified floor at one cut crossed by entanglers with the given angles.

    The m weakest gates pay in angle and the rest in rank; the best m is taken.
    m = 0 is the rank certificate. The floor is written as
    gap * [(1 - Lambda_R) + Lambda_R * sin^2([gamma - Theta]_+)] so that the
    rank part is not lost to cancellation when Lambda_R is close to one.
    """
    c = sorted(angle_cost(t) for t in thetas)
    g = len(c)
    LR = Lam(svals, 2 ** g)
    best, best_m = -1.0, 0
    for m in range(g + 1):
        LK = Lam(svals, 2 ** (g - m))
        gam = float(np.arccos(np.sqrt(min(1.0, LK / LR)))) if LR > 0 else 0.0
        extra = np.sin(max(gam - sum(c[:m]), 0.0)) ** 2
        f = gap * ((1.0 - LR) + LR * extra)
        if f > best + 1e-300:
            best, best_m = f, m
    return (best, best_m) if return_m else best


def triangle_floor_multi(svals, gap, thetas):
    """The earlier triangle bound, best over which gates are truncated."""
    c = sorted(np.sin(angle_cost(t)) for t in thetas)
    g = len(c)
    best = 1.0
    for m in range(g + 1):
        best = min(best, np.sqrt(Lam(svals, 2 ** (g - m))) + sum(c[:m]))
    return gap * (1 - min(best, 1.0) ** 2)


def dist_to_rank(psi, cut, n, K):
    """Fubini-Study distance from psi to the states of Schmidt rank <= K across cut."""
    return float(np.arccos(np.sqrt(min(1.0, Lam(schmidt_vals(psi, cut, n), K)))))
