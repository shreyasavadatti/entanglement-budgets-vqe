"""
Experiment B.  Symmetry-sector-resolved certificate.

For the TFIM, P = prod_i X_i commutes with H.  An ansatz built from Rx rotations
and exp(-i t ZZ) entanglers commutes with P, and |+>^n is a P=+1 eigenstate, so
the whole variational manifold lies in the P=+1 sector.  The binding gap is then
the in-sector gap Delta_+ = E_1^+ - E_0^+, not the global gap, which in the
ordered phase is exponentially small purely because of the Z2 partner in the
other sector.

In the ordered phase the global-gap floor is of order 1e-3, while the in-sector
floor is of order 1.

The ordinary Ry/Rz + CNOT ansatz does not preserve P, so the sector certificate
does not apply to it, and it can reach the nearly product symmetry-broken state.
The comparison shows that the symmetry hypothesis matters.
"""
import numpy as np, json
import certlib as C

n = 8
out = {}

for tag, h in [("ordered", 0.5), ("critical", 1.0), ("paramagnetic", 2.0)]:
    H = C.build_TFIM(n, h=h, periodic=True)
    w = np.linalg.eigvalsh(H)
    E0g, gap_global = w[0], w[1] - w[0]

    sec = C.sector_spectrum(H, n, k=3)
    # which sector holds the global ground state?
    s0 = '+' if sec['+']['E'][0] <= sec['-']['E'][0] else '-'
    E0s = sec[s0]['E'][0]
    gap_sec = sec[s0]['E'][1] - sec[s0]['E'][0]
    psi_s = sec[s0]['vecs'][0]
    svs = C.cut_data(psi_s, n)
    cc = n // 2 - 1                                   # central cut index

    rec = dict(h=h, E0=float(E0g), gap_global=float(gap_global),
               sector=s0, E0_sector=float(E0s), gap_sector=float(gap_sec),
               Lam=[float(C.Lam(svs[cc], 2 ** g)) for g in range(4)],
               floor_global=[float(gap_global * (1 - C.Lam(svs[cc], 2 ** g))) for g in range(4)],
               floor_sector=[float(gap_sec * (1 - C.Lam(svs[cc], 2 ** g))) for g in range(4)])

    # trained parity-preserving VQE (Rx + ZZ, init |+>^n) at g gates per bond
    trained = []
    for g in range(3):
        alloc = [g] * (n - 1)
        A = C.Ansatz(n, alloc, L=max(g, 1), kind='parity')
        e = C.train(H, A, restarts=10, seed=400 + g, maxiter=400)
        trained.append(float(e - E0s))
    rec['trained_parity'] = trained

    # contrast: symmetry-breaking hardware-efficient ansatz, same budget
    trained_hea = []
    for g in range(3):
        A = C.Ansatz(n, [g] * (n - 1), L=max(g, 1), kind='hea')
        e = C.train(H, A, restarts=10, seed=900 + g, maxiter=400)
        trained_hea.append(float(e - E0g))
    rec['trained_hea'] = trained_hea

    out[tag] = rec
    print(f"\n{tag}  h/J={h}")
    print(f"  E0={E0g:.6f}  global gap={gap_global:.3e}   sector {s0}: gap={gap_sec:.4f}")
    print(f"  Lam_C(2^g)      : " + "  ".join(f"{x:.4f}" for x in rec['Lam']))
    print(f"  floor (global)  : " + "  ".join(f"{x:.2e}" for x in rec['floor_global']))
    print(f"  floor (sector)  : " + "  ".join(f"{x:.2e}" for x in rec['floor_sector']))
    print(f"  trained parity  : " + "  ".join(f"{x:.4f}" for x in trained))
    print(f"  trained HEA     : " + "  ".join(f"{x:.4f}" for x in trained_hea))

json.dump(out, open('expB_sector.json', 'w'), indent=1)
print("\nwrote expB_sector.json")
