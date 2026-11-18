import numpy as np, json
from scipy.stats import wilcoxon, spearmanr

rows = json.load(open('expA_rows.json'))
BUD = sorted({r['G'] for r in rows})
ARMS = ['layer', 'wf_unc', 'wf_con', 'rand_con']
by = {(r['inst'], r['G'], r['arm']): r for r in rows}
insts = sorted({r['inst'] for r in rows})


def boot_median(x, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return np.percentile(m, [2.5, 97.5])


print("Paired comparison with the layer-filled baseline (delta = err_arm - err_layer; negative favours the arm)")
summary = {}
for G in BUD:
    print(f"\nBudget G = {G}  ({'connected arms feasible' if G >= 7 else 'connected arms not feasible: G < n-1 = 7'})")
    base = np.array([by[(i, G, 'layer')]['err'] for i in insts])
    for arm in ARMS:
        if (insts[0], G, arm) not in by:
            continue
        e = np.array([by[(i, G, arm)]['err'] for i in insts])
        f = np.array([by[(i, G, arm)]['floor'] for i in insts])
        fb = np.array([by[(i, G, 'layer')]['floor'] for i in insts])
        d = e - base
        lo, hi = boot_median(d)
        if arm == 'layer':
            p = np.nan
        else:
            p = wilcoxon(e, base).pvalue if np.any(e != base) else 1.0
        summary[(G, arm)] = dict(med=float(np.median(d)), lo=float(lo), hi=float(hi),
                                 win=float((e < base).mean()), p=float(p),
                                 med_err=float(np.median(e)),
                                 floor_ratio=float(np.median(fb / np.maximum(f, 1e-300))))
        print(f"  {arm:9s} med.err={np.median(e):.4f}  "
              f"delta med={np.median(d):+.4f} [95% {lo:+.4f},{hi:+.4f}]  "
              f"win={100*(e<base).mean():4.0f}%  p={p:.3f}  "
              f"floor better than baseline by x{np.median(fb/np.maximum(f,1e-300)):.1f}")

print("Does the certified floor predict trained error? (Spearman across all arms, per budget)")
for G in BUD:
    fl, er, mb = [], [], []
    for i in insts:
        for arm in ARMS:
            if (i, G, arm) in by:
                r = by[(i, G, arm)]
                fl.append(r['floor']); er.append(r['err']); mb.append(r['maxblock'])
    rs, ps = spearmanr(fl, er)
    line = f"  G={G:2d}:  rho(floor, err) = {rs:+.3f} (p={ps:.3f})"
    if len(set(mb)) > 1:
        rb, pb = spearmanr(mb, er)
        line += f"    rho(max connected block, err) = {rb:+.3f} (p={pb:.3f})"
    else:
        line += "    (every layout is connected, block size constant)"
    print(line)

print("Floor height vs trained error")
for G in BUD:
    f = np.array([by[(i, G, a)]['floor'] for i in insts for a in ARMS if (i, G, a) in by])
    e = np.array([by[(i, G, a)]['err'] for i in insts for a in ARMS if (i, G, a) in by])
    print(f"  G={G:2d}: median floor={np.median(f):.2e}  median trained err={np.median(e):.2e}"
          f"   err/floor = {np.median(e)/np.median(f):.0f}x")

print("Zero-gate bonds per arm")
for G in BUD:
    for arm in ARMS:
        if (insts[0], G, arm) not in by:
            continue
        z = np.array([by[(i, G, arm)]['nzero'] for i in insts])
        mb = np.array([by[(i, G, arm)]['maxblock'] for i in insts])
        print(f"  G={G:2d} {arm:9s}: mean #zero-gate bonds={z.mean():.2f}  "
              f"mean largest connected block={mb.mean():.2f}/8")

json.dump({f"{k[0]}_{k[1]}": v for k, v in summary.items()},
          open('expA_summary.json', 'w'), indent=1)
print("\nwrote expA_summary.json")
