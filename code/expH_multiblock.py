"""
Experiment H.  Two-sided bounds on the joint multi-cut capacity.

For a rank profile R (R_k at the cut after site k), with per-cut tails
t_k = 1 - Lambda_k(R_k), the joint capacity obeys

    1 - sum_k t_k  <=  Lambda_joint(R)  <=  Lambda_MB(R)  <=  min_k Lambda_k(R_k).

The left side comes from sequential SVD truncation (TT-SVD), the middle from
the multi-block Ky Fan bound in certlib.multiblock_capacity. In floor terms, the
best floor that any certificate built from the per-cut ranks alone can give lies
between the multi-block floor and Delta * min(1, sum_k t_k).

Part 1: the disordered ensemble of Experiment A, every budget and arm, compared
        with the trained errors stored in expA_rows.json.
Part 2: the critical chain, exact for n <= 12 (with the multi-block bound) and
        free-fermion for n up to 128 (independent floor and the sum bound).
"""
import json
import numpy as np
import certlib as C

n, NB, NINST = 8, 7, 16
rng = np.random.default_rng(7)                  # same fields as Experiment A
inst_fields = [np.exp(rng.uniform(np.log(0.3), np.log(3.0), n)) for _ in range(NINST)]
rows_A = json.load(open('expA_rows.json'))

ens = []
for inst, hs in enumerate(inst_fields):
    H = C.build_TFIM(n, hs=hs, periodic=False)
    w, v = np.linalg.eigh(H)
    psi0, gap = v[:, 0], w[1] - w[0]
    svs = C.cut_data(psi0, n)
    for r in (x for x in rows_A if x['inst'] == inst):
        R = [2 ** g for g in r['alloc']]
        t = [1.0 - C.Lam(svs[k], R[k]) for k in range(NB)]
        mb, cfg = C.multiblock_capacity(psi0, n, R, return_config=True)
        ens.append(dict(inst=inst, G=r['G'], arm=r['arm'], alloc=r['alloc'], gap=float(gap),
                        err=r['err'], floor_indep=float(gap * max(t)),
                        floor_mb=float(gap * (1 - mb)),
                        floor_max=float(gap * min(1.0, sum(t))),
                        mb_blocks=[list(b) for b in cfg] if cfg else []))

# At G=5 the MPS fits of Experiment C tighten the upper end further, since
# F_fit <= Lambda_joint as well.
fits = {(r['inst'], arm): r[arm]['joint_fit']
        for r in json.load(open('expC_joint.json'))['ensemble'] for arm in ('layer', 'wf_unc')}
for e in ens:
    if (e['inst'], e['arm']) in fits and e['G'] == 5:
        e['floor_max'] = min(e['floor_max'], e['gap'] * (1 - fits[(e['inst'], e['arm'])]))

print("Ensemble: medians over instances, and the worst case of error / largest possible floor")
for G in sorted({e['G'] for e in ens}):
    for arm in ('layer', 'wf_unc', 'wf_con', 'rand_con'):
        sel = [e for e in ens if e['G'] == G and e['arm'] == arm]
        if not sel:
            continue
        gain = [e['floor_mb'] / e['floor_indep'] for e in sel]
        slack = [e['err'] / e['floor_max'] for e in sel]
        print(f"G={G:2d} {arm:8s} indep={np.median([e['floor_indep'] for e in sel]):.2e} "
              f"mb={np.median([e['floor_mb'] for e in sel]):.2e} "
              f"max={np.median([e['floor_max'] for e in sel]):.2e} "
              f"err={np.median([e['err'] for e in sel]):.2e} | mb gain >1% in "
              f"{sum(x > 1.01 for x in gain)}/{len(sel)} (max x{max(gain):.2f}) | "
              f"err/max-floor min {min(slack):.1f}")

# Each arm's best rank-only floor lies in [floor_mb, floor_max]. Disjoint
# intervals mean every rank-only certificate orders the two arms the same way.
print("\nG=5, per instance: interval for the best rank-only floor of each arm")
tally = dict(layer=0, wf=0, overlap=0, same=0)
for inst in range(NINST):
    a = {e['arm']: e for e in ens if e['inst'] == inst and e['G'] == 5}
    L, W = a['layer'], a['wf_unc']
    if L['alloc'] == W['alloc']:
        tally['same'] += 1
        continue
    if L['floor_mb'] > W['floor_max']:
        verdict = 'layer'
    elif W['floor_mb'] > L['floor_max']:
        verdict = 'wf'
    else:
        verdict = 'overlap'
    tally[verdict] += 1
    print(f"[{inst:2d}] layer [{L['floor_mb']:.2e}, {L['floor_max']:.2e}] err {L['err']:.3f} | "
          f"wf [{W['floor_mb']:.2e}, {W['floor_max']:.2e}] err {W['err']:.3f} | higher floor: {verdict}")
print(f"identical allocations {tally['same']}, layer floor certainly higher {tally['layer']}, "
      f"wf floor certainly higher {tally['wf']}, undecided {tally['overlap']}")

crit = []
print("\nCritical chain, exact")
for nn in (8, 10, 12):
    H = C.build_TFIM(nn, h=1.0, periodic=False)
    w, v = np.linalg.eigh(H)
    psi0, gap = v[:, 0], w[1] - w[0]
    svs = C.cut_data(psi0, nn)
    for g in (1, 2):
        R = [2 ** g] * (nn - 1)
        t = [1.0 - C.Lam(svs[k], R[k]) for k in range(nn - 1)]
        mb = C.multiblock_capacity(psi0, nn, R)
        crit.append(dict(n=nn, g=g, method='exact', gap=float(gap),
                         floor_indep=float(gap * max(t)), floor_mb=float(gap * (1 - mb)),
                         floor_max=float(gap * min(1.0, sum(t)))))
        c = crit[-1]
        print(f"n={nn:3d} g={g} indep={c['floor_indep']:.3e} mb={c['floor_mb']:.3e} "
              f"sum={c['floor_max']:.3e} ratio sum/indep={c['floor_max']/c['floor_indep']:.1f}")

print("\nCritical chain, free fermion")
for nn in (16, 32, 64, 128):
    Gam, eps = C.tfim_covariance(nn, 1.0, 1.0)
    gap = float(np.min(eps))                    # gap of the full spectrum, as in Experiment D
    for g in (1, 2):
        t = []
        for k in range(1, nn):
            nu = C.ff_block_nu(nn, k, 1.0, 1.0, Gam)
            t.append(1.0 - float(np.sum(C.top_probs_from_nu(nu, 2 ** g))))
        crit.append(dict(n=nn, g=g, method='free_fermion', gap=gap,
                         floor_indep=float(gap * max(t)),
                         floor_max=float(gap * min(1.0, sum(t)))))
        c = crit[-1]
        print(f"n={nn:3d} g={g} indep={c['floor_indep']:.3e} sum={c['floor_max']:.3e} "
              f"ratio={c['floor_max']/c['floor_indep']:.1f}")

json.dump(dict(ensemble=ens, critical=crit), open('expH_multiblock.json', 'w'), indent=1)
print("\nwrote expH_multiblock.json")
