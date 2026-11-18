"""
Experiment A. Disordered open TFIM ensemble, n=8 (7 bonds).

Arms at fixed two-qubit budget G:
  layer    : standard hardware-efficient partial-layer fill (baseline)
  wf_unc   : unconstrained min-max water-filling  (Thm: minimises certified floor)
  wf_con   : connected water-filling, g_C >= 1 on every bond, residual water-filled
  rand_con : connected random  (control: isolates connectivity from water-filling)

Connected arms require G >= n-1 = 7, so G=5 (the manuscript's original budget)
admits only the two disconnected arms.

All arms at a given (instance, G) share the same number of rotation layers L,
hence the same parameter count -- placement is the only difference.
"""
import numpy as np, json, sys, time, os
import certlib as C

n, NB = 8, 7
NINST = 16
RESTARTS = 8
MAXITER = 250
BUDGETS = [5, 7, 8, 9, 11]


def fragments(alloc):
    """Connectivity blocks induced by zero-gate bonds."""
    blocks, cur = [], 1
    for a in alloc:
        if a == 0:
            blocks.append(cur); cur = 1
        else:
            cur += 1
    blocks.append(cur)
    return blocks


rng = np.random.default_rng(7)          # same stream as the original benchmark
inst_fields = [np.exp(rng.uniform(np.log(0.3), np.log(3.0), n)) for _ in range(NINST)]

LO, HI = int(sys.argv[1]), int(sys.argv[2])
rows = json.load(open('expA_rows.json')) if os.path.exists('expA_rows.json') else []
rows = [r for r in rows if not (LO <= r['inst'] < HI)]
t0 = time.time()
for inst, hs in enumerate(inst_fields):
    if not (LO <= inst < HI):
        continue
    H = C.build_TFIM(n, hs=hs, periodic=False)
    w, v = np.linalg.eigh(H)
    E0, psi0, gap = w[0], v[:, 0], w[1] - w[0]
    svs = C.cut_data(psi0, n)
    arng = np.random.default_rng(5000 + inst)

    for G in BUDGETS:
        arms = {'layer': C.layer_filled(G, NB),
                'wf_unc': C.waterfill(svs, gap, G, NB)}
        if G >= NB:
            arms['wf_con'] = C.connected_waterfill(svs, gap, G, NB)
            arms['rand_con'] = C.connected_random(G, NB, arng)

        L = max(max(a) for a in arms.values())
        cache = {}
        for name, a in arms.items():
            key = tuple(a)
            if key not in cache:
                A = C.Ansatz(n, a, L=L, kind='hea')
                cache[key] = C.train(H, A, restarts=RESTARTS,
                                     seed=10_000 + 97 * inst, maxiter=MAXITER)
            blocks = fragments(a)
            rows.append(dict(inst=inst, G=G, arm=name, alloc=list(a), L=L,
                             tq_depth=int(max(a)), gap=float(gap),
                             floor=float(C.certificate(svs, gap, a)),
                             err=float(cache[key] - E0),
                             nzero=int(sum(1 for x in a if x == 0)),
                             maxblock=int(max(blocks)), nblocks=len(blocks)))
        el = time.time() - t0
        print(f"[inst {inst:2d} G={G:2d}] " + "  ".join(
            f"{r['arm']}:err={r['err']:.4f},fl={r['floor']:.1e}"
            for r in rows[-len(arms):]) + f"   ({el:.0f}s)")
        sys.stdout.flush()

json.dump(rows, open('expA_rows.json', 'w'), indent=1)
print("wrote expA_rows.json (%d rows) in %.1f min" % (len(rows), (time.time()-t0)/60))
