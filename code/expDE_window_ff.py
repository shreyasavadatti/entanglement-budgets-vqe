"""
Experiments D and E.

D. Informative window / triviality threshold.
   The certificate is nonzero at cut C iff 2^{g_C} < d_C, and d_C <= 2^{min(|A|,|B|)},
   so g_C >= min(|A|,|B|) forces a zero floor by dimension counting alone.
   The useful (tolerance-resolved) statement uses
        R_C(eps) = min{R : 1 - Lambda_C(R) <= eps},
   giving g_C >= ceil(log2 R_C(eps))  ==>  floor <= Delta*eps.

E. Free-fermion Schmidt spectra for the open TFIM at any n.
   The entanglement spectrum follows from the Majorana covariance matrix, so the
   certificate inputs are available without many-body truncation far beyond
   exact diagonalisation. Values are computed in floating point and are not
   interval-certified. The sector gap comes from the same representation.
"""
import numpy as np, json
import certlib as C

# validation
print("free-fermion vs exact diagonalisation (open critical TFIM)")
val = []
for n in (8, 10, 12):
    H = C.build_TFIM(n, h=1.0, periodic=False)
    w, v = np.linalg.eigh(H)
    psi0 = v[:, 0]
    sv = C.schmidt_vals(psi0, n // 2, n)
    p_ed = np.sort(sv ** 2)[::-1]
    S_ed = float(-(p_ed * np.log(p_ed)).sum())

    Gam, eps = C.tfim_covariance(n, 1.0, 1.0)
    nu = C.ff_block_nu(n, n // 2, Gamma=Gam)
    p_ff = C.top_probs_from_nu(nu, 8)
    S_ff = C.ff_entropy(nu)
    e_s = np.sort(eps)
    sec = C.sector_spectrum(H, n, k=3)
    s0 = '+' if sec['+']['E'][0] <= sec['-']['E'][0] else '-'
    gsec = float(sec[s0]['E'][1] - sec[s0]['E'][0])
    val.append(dict(n=n, S_ed=S_ed, S_ff=S_ff,
                    gap_ed=float(w[1] - w[0]), gap_ff=float(e_s[0]),
                    gapsec_ed=gsec, gapsec_ff=float(e_s[0] + e_s[1]),
                    dLam=float(np.abs(p_ed[:8] - p_ff[:8]).max())))
    print(f" n={n:2d}  S: {S_ed:.9f} vs {S_ff:.9f}   gap: {w[1]-w[0]:.9f} vs {e_s[0]:.9f}"
          f"   sector gap: {gsec:.9f} vs {e_s[0]+e_s[1]:.9f}   max|dp|={val[-1]['dLam']:.2e}")

# scaling
print("\nexact free-fermion certificate inputs, critical TFIM (open)")
NS = [12, 16, 24, 32, 48, 64, 96, 128]
scal = []
for n in NS:
    Gam, eps = C.tfim_covariance(n, 1.0, 1.0)
    nu = C.ff_block_nu(n, n // 2, Gamma=Gam)
    p = C.top_probs_from_nu(nu, 256)
    cum = np.cumsum(p)
    e_s = np.sort(eps)
    gap, gap_sec = float(e_s[0]), float(e_s[0] + e_s[1])
    row = dict(n=n, S=C.ff_entropy(nu), gap=gap, gap_sector=gap_sec)
    for g in (1, 2, 3, 4):
        L = float(cum[min(2 ** g, len(cum)) - 1])
        row[f"Lam{g}"] = L
        row[f"floor{g}"] = gap * (1 - L)
        row[f"floorsec{g}"] = gap_sec * (1 - L)
    for eps_t in (1e-2, 1e-3, 1e-4):
        R = int(np.searchsorted(cum, 1 - eps_t) + 1)
        row[f"gstar_{eps_t:g}"] = int(np.ceil(np.log2(max(R, 1))))
    scal.append(row)
    print(f" n={n:3d} S={row['S']:.4f} gap={gap:.3e} gap_sec={gap_sec:.3e} "
          f"Lam(2)={row['Lam1']:.6f} floor_g1={row['floor1']:.2e} "
          f"g*(1e-2)={row['gstar_0.01']} g*(1e-3)={row['gstar_0.001']} g*(1e-4)={row['gstar_0.0001']}")

# Calabrese-Cardy fit, open boundaries: S = (c/6) ln(2n/pi) + const
ns = np.array([r['n'] for r in scal], float)
Ss = np.array([r['S'] for r in scal])
m = ns >= 32
A_ = np.vstack([np.log(2 * ns[m] / np.pi), np.ones(m.sum())]).T
sl, _ = np.linalg.lstsq(A_, Ss[m], rcond=None)[0]
print(f"\n Calabrese-Cardy fit over n>=32:  c = {6*sl:.4f}   (exact Ising c = 0.5)")

# window map
print("\ntriviality threshold across the phase diagram (n=64)")
win = []
for h in [0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7, 2.0, 2.5]:
    Gam, eps = C.tfim_covariance(64, 1.0, h)
    nu = C.ff_block_nu(64, 32, Gamma=Gam)
    p = C.top_probs_from_nu(nu, 256)
    cum = np.cumsum(p)
    e_s = np.sort(eps)
    row = dict(h=h, S=C.ff_entropy(nu), gap=float(e_s[0]), gap_sector=float(e_s[0] + e_s[1]))
    for eps_t in (1e-2, 1e-3, 1e-4):
        R = int(np.searchsorted(cum, 1 - eps_t) + 1)
        row[f"gstar_{eps_t:g}"] = int(np.ceil(np.log2(max(R, 1))))
    win.append(row)
    print(f" h/J={h:4.1f}  S={row['S']:.4f}  g*: "
          + " ".join(f"{row[f'gstar_{e:g}']}" for e in (1e-2, 1e-3, 1e-4)))

json.dump(dict(validation=val, scaling=scal, window=win, c_fit=float(6 * sl)),
          open('expDE.json', 'w'), indent=1)
print("\nwrote expDE.json")
