"""
Two follow-ups to Experiment A.

(1) Within-instance test: does choosing the lower certified floor pick the better
    layout?  Cross-instance correlation is confounded by instance difficulty
    (hard instances have both a higher floor and a higher error), so the only
    meaningful test compares arms inside a fixed instance.

(2) Fragmentation test at G=5 (< n-1, so every allocation is disconnected):
    is trained error governed by where the zero-gate bonds fall rather than by
    the certified floor?
"""
import numpy as np, json, sys
from scipy.stats import wilcoxon, binomtest
import certlib as C

rows = json.load(open('expA_rows.json'))
by = {(r['inst'], r['G'], r['arm']): r for r in rows}
insts = sorted({r['inst'] for r in rows})
ARMS = ['layer', 'wf_unc', 'wf_con', 'rand_con']

print("(1) Within instance: does the lower certified floor identify the better layout?")
for G in (8, 9, 11):
    agree = tot = 0
    dd = []
    lo_arms, hi_arms = [], []
    for i in insts:
        # arms with identical allocations are counted once; ties in the floor
        # are broken by the fixed arm order in ARMS
        seen, arms = set(), []
        for a in ARMS:
            if (i, G, a) not in by:
                continue
            r = by[(i, G, a)]
            if tuple(r['alloc']) in seen:
                continue
            seen.add(tuple(r['alloc']))
            arms.append((a, r['floor'], r['err']))
        best_floor = min(arms, key=lambda t: t[1])
        worst_floor = max(arms, key=lambda t: t[1])
        if best_floor[1] == worst_floor[1]:
            continue
        tot += 1
        lo_arms.append(best_floor[0]); hi_arms.append(worst_floor[0])
        agree += int(best_floor[2] < worst_floor[2])
        dd.append(best_floor[2] - worst_floor[2])
    dd = np.array(dd)
    p = wilcoxon(dd).pvalue if len(dd) > 3 and np.any(dd != 0) else np.nan
    p_sign = binomtest(agree, tot, 0.5).pvalue
    print(f"  G={G:2d}: lowest-floor arm also had lowest error in {agree}/{tot} instances "
          f"({100*agree/max(tot,1):.0f}%);  sign test p={p_sign:.2g};  "
          f"median err(best floor) - err(worst floor) = {np.median(dd):+.4f} (Wilcoxon p={p:.3f})")
    print(f"        lowest-floor arm: {sorted(set(lo_arms))}   "
          f"highest-floor arm counts: { {a: hi_arms.count(a) for a in set(hi_arms)} }")
print("\n  50% agreement would mean the certified floor carries no information about")
print("  which layout trains better at fixed budget.")

# (2)
print("(2) Fragmentation at G=5 (budget < n-1, so every allocation is disconnected)")
n, NB, G = 8, 7, 5
rng = np.random.default_rng(7)
fields = [np.exp(rng.uniform(np.log(0.3), np.log(3.0), n)) for _ in range(16)]

CANDS = {'edge':   [1, 1, 1, 1, 1, 0, 0],     # zeros at the boundary  -> blocks 6,1,1
         'offset': [0, 1, 1, 1, 1, 1, 0],     # zeros at both ends     -> blocks 1,6,1
         'centre': [1, 1, 0, 1, 0, 1, 1]}     # zeros in the bulk      -> blocks 3,2,3


def fragments(a):
    b, cur = [], 1
    for x in a:
        if x == 0:
            b.append(cur); cur = 1
        else:
            cur += 1
    b.append(cur)
    return b


res = []
for inst, hs in enumerate(fields):
    H = C.build_TFIM(n, hs=hs, periodic=False)
    w, v = np.linalg.eigh(H)
    E0, psi0, gap = w[0], v[:, 0], w[1] - w[0]
    svs = C.cut_data(psi0, n)
    rec = dict(inst=inst)
    for name, a in CANDS.items():
        A = C.Ansatz(n, a, L=1, kind='hea')
        e = C.train(H, A, restarts=8, seed=10_000 + 97 * inst, maxiter=250) - E0
        rec[name] = dict(err=float(e), floor=float(C.certificate(svs, gap, a)),
                         maxblock=int(max(fragments(a))))
    wf = C.waterfill(svs, gap, G, NB)
    A = C.Ansatz(n, wf, L=max(max(wf), 1), kind='hea')
    e = C.train(H, A, restarts=8, seed=10_000 + 97 * inst, maxiter=250) - E0
    rec['waterfill'] = dict(err=float(e), floor=float(C.certificate(svs, gap, wf)),
                            maxblock=int(max(fragments(wf))), alloc=list(wf))
    res.append(rec)
    print(f"[{inst:2d}] " + "  ".join(
        f"{k}(blk{rec[k]['maxblock']}):{rec[k]['err']:.3f}/fl{rec[k]['floor']:.1e}"
        for k in ('edge', 'offset', 'centre', 'waterfill')))
    sys.stdout.flush()

print("\n  medians:")
for k in ('edge', 'offset', 'centre', 'waterfill'):
    e = np.array([r[k]['err'] for r in res])
    f = np.array([r[k]['floor'] for r in res])
    b = np.array([r[k]['maxblock'] for r in res])
    print(f"    {k:10s} largest block={b.mean():.1f}  median floor={np.median(f):.2e}  "
          f"median trained err={np.median(e):.4f}")

ee = np.array([r['edge']['err'] for r in res])
cc = np.array([r['centre']['err'] for r in res])
print(f"\n  edge-zeros vs centre-zeros: median delta = {np.median(cc-ee):+.4f}, "
      f"p = {wilcoxon(cc, ee).pvalue:.4f}, centre worse in {100*(cc>ee).mean():.0f}% of instances")
ff = np.array([r['edge']['floor'] for r in res]), np.array([r['centre']['floor'] for r in res])
print(f"  but the certified floor prefers centre-zeros in "
      f"{100*(ff[1] < ff[0]).mean():.0f}% of instances")

json.dump(res, open('expA_fragment.json', 'w'), indent=1)
print("\nwrote expA_fragment.json")
