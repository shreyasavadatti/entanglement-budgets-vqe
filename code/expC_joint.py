"""
Experiment C.  Joint multi-cut capacity, numerically.

States with rank_{C_k} <= R_k at every contiguous cut are exactly the MPS with
bond dimensions R_k.  Fitting the best such MPS to the target gives

      F_fit(R)  <=  Lambda_joint(R)  <=  min_k Lambda_{C_k}(R_k),

a numerical bracket.  The right side is what the independent-cut certificate
uses (max_C of the floor == min_k of the capacity).  The gap between the two
sides measures simultaneous multi-cut incompatibility.

We evaluate the bracket on the disordered ensemble at G=5 for the layer-filled
and water-filled allocations, and on the critical chain at uniform budgets.
Because F_fit is only a lower bound on Lambda_joint, a gap between the two sides
is consistent with multi-cut incompatibility but does not certify it.
"""
import numpy as np, json
import certlib as C

n, NB, G = 8, 7, 5
NINST = 16
DIMS_CAP = 16                                   # 2^{|A|} cap handled per cut

rng = np.random.default_rng(7)
inst_fields = [np.exp(rng.uniform(np.log(0.3), np.log(3.0), n)) for _ in range(NINST)]

rows = []
for inst, hs in enumerate(inst_fields):
    H = C.build_TFIM(n, hs=hs, periodic=False)
    w, v = np.linalg.eigh(H)
    E0, psi0, gap = w[0], v[:, 0], w[1] - w[0]
    svs = C.cut_data(psi0, n)

    arms = {'layer': C.layer_filled(G, NB), 'wf_unc': C.waterfill(svs, gap, G, NB)}
    rec = dict(inst=inst, gap=float(gap))
    for name, a in arms.items():
        dims = [min(2 ** a[k], 2 ** min(k + 1, n - k - 1)) for k in range(NB)]
        indep = min(C.Lam(svs[k], 2 ** a[k]) for k in range(NB))
        joint = C.mps_fit_overlap(psi0, n, dims, sweeps=30, seed=inst, n_random=3)
        rec[name] = dict(alloc=list(a), dims=dims,
                         indep_cap=float(indep), joint_fit=float(joint),
                         floor_indep=float(gap * (1 - indep)),
                         floor_joint_ub=float(gap * (1 - joint)))
    rows.append(rec)
    print(f"[{inst:2d}] layer: indep={rec['layer']['indep_cap']:.4f} joint={rec['layer']['joint_fit']:.4f}"
          f" | wf: indep={rec['wf_unc']['indep_cap']:.4f} joint={rec['wf_unc']['joint_fit']:.4f}")

# Critical TFIM at uniform budgets: how loose is the independent-cut bound?
crit = []
for nn in (8, 10, 12):
    H = C.build_TFIM(nn, h=1.0, periodic=False)
    w, v = np.linalg.eigh(H)
    psi0, gap = v[:, 0], w[1] - w[0]
    svs = C.cut_data(psi0, nn)
    for g in (1, 2):
        dims = [min(2 ** g, 2 ** min(k + 1, nn - k - 1)) for k in range(nn - 1)]
        indep = min(C.Lam(svs[k], 2 ** g) for k in range(nn - 1))
        joint = C.mps_fit_overlap(psi0, nn, dims, sweeps=30, seed=g, n_random=2)
        crit.append(dict(n=nn, g=g, gap=float(gap), indep=float(indep), joint=float(joint),
                         floor_indep=float(gap * (1 - indep)),
                         floor_joint_ub=float(gap * (1 - joint))))
        print(f"critical n={nn} g={g}: indep_cap={indep:.6f} joint_fit={joint:.6f}"
              f"  floors {gap*(1-indep):.2e} -> {gap*(1-joint):.2e}")

json.dump(dict(ensemble=rows, critical=crit), open('expC_joint.json', 'w'), indent=1)
print("\nwrote expC_joint.json")
