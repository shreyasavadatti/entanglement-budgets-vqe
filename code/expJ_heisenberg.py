"""
Experiment J. Connected-budget control on a disordered open Heisenberg chain,
n = 8 (7 bonds), log-uniform antiferromagnetic couplings J_k in [0.3, 3.0].

By Lieb-Mattis the ground state is a unique (non-degenerate) total-spin singlet,
and singlet correlations across strong bonds keep the rank certificate active at
connected budgets (G >= n - 1 = 7), reducing the floor-to-error ratio by more
than an order of magnitude compared with the disordered TFIM ensemble.
"""
import json
import os
import sys
import time
import numpy as np
from scipy.stats import wilcoxon, binomtest
import certlib as C

n, NB = 8, 7
NINST = 16
RESTARTS = 8
MAXITER = 250
BUDGETS = [7, 8, 9, 11]
ARMS = ['layer', 'wf_unc', 'wf_con', 'rand_con']


def build_disordered_heisenberg(n, Js):
    H = np.zeros((2 ** n, 2 ** n))
    for i, J in enumerate(Js):
        H += J * (C.kron_op2(C.X, C.X, i, i + 1, n).real
                  + C.kron_op2(C.Y, C.Y, i, i + 1, n).real
                  + C.kron_op2(C.Z, C.Z, i, i + 1, n).real)
    return H


def boot_median(x, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return np.percentile(m, [2.5, 97.5])


t0 = time.time()
if '--analyse-only' in sys.argv and os.path.exists('expJ_heisenberg.json'):
    rows = json.load(open('expJ_heisenberg.json'))['rows']
else:
    rng = np.random.default_rng(7)
    inst_Js = [np.exp(rng.uniform(np.log(0.3), np.log(3.0), NB)) for _ in range(NINST)]
    rows = []
    for inst, Js in enumerate(inst_Js):
        H = build_disordered_heisenberg(n, Js)
        w, v = np.linalg.eigh(H)
        E0, psi0, gap = w[0], v[:, 0], w[1] - w[0]
        assert gap > 1e-8, f"unexpected degeneracy at inst {inst}: gap={gap}"
        svs = C.cut_data(psi0, n)
        arng = np.random.default_rng(5000 + inst)

        for G in BUDGETS:
            arms = {
                'layer': C.layer_filled(G, NB),
                'wf_unc': C.waterfill(svs, gap, G, NB),
                'wf_con': C.connected_waterfill(svs, gap, G, NB),
                'rand_con': C.connected_random(G, NB, arng),
            }
            L = max(max(a) for a in arms.values())
            cache = {}
            for name, a in arms.items():
                key = tuple(a)
                if key not in cache:
                    ans = C.Ansatz(n, a, L=L, kind='hea')
                    cache[key] = C.train(H, ans, restarts=RESTARTS,
                                         seed=10_000 + 97 * inst, maxiter=MAXITER)
                rows.append(dict(
                    inst=inst, G=G, arm=name, alloc=list(a), L=L,
                    gap=float(gap),
                    floor=float(C.certificate(svs, gap, a)),
                    err=float(cache[key] - E0),
                    nzero=int(sum(1 for x in a if x == 0)),
                ))
        print(f"[inst {inst:2d}] gap={gap:.4f}  ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

by = {(r['inst'], r['G'], r['arm']): r for r in rows}
summary = {'by_G': {}}

print("\nConnected disordered-Heisenberg control (n=8, 16 instances):")
for G in BUDGETS:
    fl_all = [by[(i, G, arm)]['floor'] for i in range(NINST) for arm in ARMS]
    er_all = [by[(i, G, arm)]['err'] for i in range(NINST) for arm in ARMS]
    ratio = float(np.median(er_all) / np.median(fl_all))
    print(f"\nBudget G = {G}: median floor={np.median(fl_all):.3e}  "
          f"median err={np.median(er_all):.3e}  err/floor={ratio:.1f}x")
    g_info = {
        'med_floor': float(np.median(fl_all)),
        'med_err': float(np.median(er_all)),
        'ratio': ratio,
        'arms': {},
    }
    base_err = np.array([by[(i, G, 'layer')]['err'] for i in range(NINST)])
    base_fl = np.array([by[(i, G, 'layer')]['floor'] for i in range(NINST)])
    for arm in ARMS:
        ae = np.array([by[(i, G, arm)]['err'] for i in range(NINST)])
        af = np.array([by[(i, G, arm)]['floor'] for i in range(NINST)])
        d = ae - base_err
        ci_lo, ci_hi = [float(v) for v in boot_median(d)]
        if arm == 'layer':
            pval = float('nan')
        else:
            pval = float(wilcoxon(ae, base_err).pvalue) if np.any(ae != base_err) else 1.0
        win = float(np.mean(ae < base_err) * 100.0)
        fl_gain = float(np.median(base_fl / np.maximum(af, 1e-300)))
        nzero_mean = float(np.mean([by[(i, G, arm)]['nzero'] for i in range(NINST)]))
        print(f"  {arm:9s} med.err={np.median(ae):.4f}  dmed={np.median(d):+.4f} "
              f"[{ci_lo:+.4f},{ci_hi:+.4f}]  win={win:4.0f}%  p={pval:.3f}  "
              f"floor_gain={fl_gain:.2f}x  mean_zeros={nzero_mean:.2f}")
        g_info['arms'][arm] = dict(
            med_err=float(np.median(ae)), dmed=float(np.median(d)),
            ci=[ci_lo, ci_hi], win=win, p=pval, fl_gain=fl_gain,
            mean_zeros=nzero_mean,
        )
    a_cnt = t_cnt = 0
    for i in range(NINST):
        seen, cand = set(), []
        for x in ARMS:
            al = tuple(by[(i, G, x)]['alloc'])
            if al not in seen:
                seen.add(al)
                cand.append((by[(i, G, x)]['floor'], by[(i, G, x)]['err']))
        b = min(cand, key=lambda r: r[0])
        wo = max(cand, key=lambda r: r[0])
        if b[0] == wo[0]:
            continue
        t_cnt += 1
        a_cnt += int(b[1] < wo[1])
    if t_cnt > 0:
        sp = float(binomtest(a_cnt, t_cnt, 0.5).pvalue)
        print(f"  within-instance: lowest-floor wins {a_cnt}/{t_cnt} ({100*a_cnt/t_cnt:.0f}%), sign p={sp:.3g}")
        g_info['within'] = dict(wins=a_cnt, total=t_cnt, pct=100 * a_cnt / t_cnt, sign_p=sp)
    summary['by_G'][str(G)] = g_info

json.dump({'rows': rows, 'summary': summary}, open('expJ_heisenberg.json', 'w'), indent=1)
print(f"\nwrote expJ_heisenberg.json")
