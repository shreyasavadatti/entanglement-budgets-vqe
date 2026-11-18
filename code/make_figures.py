"""make_figures.py -- regenerates every figure in the manuscript."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import certlib as C

os.makedirs('figures', exist_ok=True)

# Publication-grade styling (colorblind-safe Okabe-Ito inspired palette)
PAL = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 8.5,
    'axes.titlesize': 9.2,
    'axes.titlepad': 6,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.8,
    'ytick.labelsize': 7.8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 3.2,
    'ytick.major.size': 3.2,
    'axes.grid': True,
    'grid.alpha': 0.24,
    'grid.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.06,
    'lines.linewidth': 1.5,
})

DE = json.load(open('expDE.json'))
B = json.load(open('expB_sector.json'))
Cj = json.load(open('expC_joint.json'))
F = json.load(open('expF.json'))
A = json.load(open('expA_rows.json'))
FR = json.load(open('expA_fragment.json'))


def clean_legend(ax, **kwargs):
    defaults = dict(frameon=True, facecolor='white', edgecolor='none',
                    framealpha=0.88, borderpad=0.3, handlelength=1.8)
    defaults.update(kwargs)
    return ax.legend(**defaults)


# fig 0: conceptual schematic
fig, ax = plt.subplots(1, 3, figsize=(11.2, 2.95), layout='constrained')
fig.get_layout_engine().set(w_pad=0.10, wspace=0.06)

a0 = ax[0]
a0.set_xlim(0, 10.6)
a0.set_ylim(-1.35, 5.85)
a0.axis('off')
a0.set_title('(a) cut register & read cross-cut budgets')

a0.add_patch(FancyBboxPatch((0.25, 2.75), 9.25, 2.82,
                            boxstyle='round,pad=0.08,rounding_size=0.18',
                            fc='#0072B2', alpha=0.07, ec='none', zorder=0))
a0.add_patch(FancyBboxPatch((0.25, -0.32), 9.25, 2.82,
                            boxstyle='round,pad=0.08,rounding_size=0.18',
                            fc='#009E73', alpha=0.07, ec='none', zorder=0))
a0.text(0.45, 5.45, 'subsystem $A$', fontsize=7.6, fontweight='bold',
        color=PAL[0], va='top', ha='left')
a0.text(0.45, -0.20, 'subsystem $B$', fontsize=7.6, fontweight='bold',
        color=PAL[2], va='bottom', ha='left')

ys = [0.45, 1.25, 2.05, 3.15, 3.95, 4.75]
for y in ys:
    a0.plot([1.25, 9.35], [y, y], color='#555555', lw=1.0, zorder=1)
    a0.text(1.08, y, r'$|0\rangle$', fontsize=7.4, ha='right', va='center', color='#333333')

a0.plot([0.2, 9.55], [2.6, 2.6], ls='--', color='#B22222', lw=1.25, zorder=2)
a0.text(9.62, 2.6, 'cut $C$', fontsize=8.0, fontweight='bold', color='#B22222',
        va='center', ha='left')


def _gate_box(ax_, x, y0, y1, label='', fc='#EFEFEF', ec='#666666', tc='#333333', w=0.82):
    box = FancyBboxPatch((x - w / 2, y0 - 0.23), w, (y1 - y0) + 0.46,
                         boxstyle='round,pad=0.02,rounding_size=0.12',
                         fc=fc, ec=ec, lw=0.95, zorder=3)
    ax_.add_patch(box)
    if label:
        ax_.text(x, 0.5 * (y0 + y1), label, fontsize=7.2, ha='center', va='center',
                 color=tc, zorder=4)


_gate_box(a0, 2.25, 3.95, 4.75, '$U_A$')
_gate_box(a0, 2.25, 0.45, 1.25, '$U_B$')
_gate_box(a0, 3.95, 2.05, 3.15, r'$\theta_1$', fc='#DCEAF7', ec=PAL[0], tc=PAL[0], w=1.05)
_gate_box(a0, 5.65, 3.15, 4.75, r"$U_A'$")
_gate_box(a0, 5.65, 0.45, 2.05, r"$U_B'$")
_gate_box(a0, 7.35, 2.05, 3.15, r'$\theta_2$', fc='#DCEAF7', ec=PAL[0], tc=PAL[0], w=1.05)
for y in ys:
    _gate_box(a0, 8.75, y, y, '$R$', w=0.52)

a0.text(4.9, -0.92,
        r'rank budget: $g_C = 2 \;\Rightarrow\; R \leq 2^{g_C} = 4$' + '\n' +
        r'angle budget: $\Theta_C = \tilde{\theta}_1 + \tilde{\theta}_2$',
        fontsize=7.4, ha='center', va='center', color='#222222',
        bbox=dict(boxstyle='round,pad=0.28', fc='#F8F9FA', ec='#CCCCCC', lw=0.7))

a1 = ax[1]
a1.grid(False, axis='x')
idx = np.arange(1, 9)
weights = np.array([0.42, 0.22, 0.13, 0.08, 0.06, 0.042, 0.028, 0.020])
R_cap = 4
a1.bar(idx[:R_cap], weights[:R_cap], width=0.62, color=PAL[0],
       label=r'captured $\Lambda_C(R)$')
a1.bar(idx[R_cap:], weights[R_cap:], width=0.62, color=PAL[1],
       label=r'uncaptured tail $1-\Lambda_C(R)$')
a1.axvline(R_cap + 0.5, color='#222222', ls='--', lw=1.1)
a1.text(R_cap + 0.35, 0.30, r'$R = 2^{g_C}$', fontsize=7.6,
        color='#222222', va='top', ha='right')
a1.annotate('unavoidable\nfidelity deficit', xy=(5.4, weights[4] + 0.008),
            xytext=(5.5, 0.20), fontsize=7.3, color=PAL[1],
            arrowprops=dict(arrowstyle='->', color=PAL[1], lw=0.9))
a1.set_xticks(idx)
a1.set_xlabel(r'Schmidt index $i$ across cut $C$')
a1.set_ylabel(r'target weight $(\lambda_i^{(C)})^2$')
a1.set_ylim(0, 0.48)
a1.set_title('(b) cap ground-state overlap')
clean_legend(a1, fontsize=6.9, loc='upper right')

a2 = ax[2]
a2.set_xlim(0, 10)
a2.set_ylim(-0.35, 5.2)
a2.spines['bottom'].set_visible(False)
a2.grid(False)
a2.set_xticks([])
a2.set_yticks([0.3, 1.85, 3.65])
a2.set_yticklabels([r'$E_0$', r'$E_{\mathrm{floor}}$', r'$E_1$'], fontsize=8.2)
a2.set_ylabel('energy')
a2.set_title('(c) convert deficit into energy floor')

a2.plot([0.6, 3.6], [0.3, 0.3], color='#222222', lw=1.8)
a2.text(2.1, 0.0, 'ground state', fontsize=7.0, ha='center', va='top', color='#444444')
a2.plot([0.6, 3.6], [3.65, 3.65], color='#555555', lw=1.4)
a2.text(2.1, 3.80, '1st excited', fontsize=7.0, ha='center', va='bottom', color='#444444')
a2.plot([0.6, 3.6], [4.65, 4.65], color=PAL[2], lw=1.3, ls='--')
a2.text(2.1, 4.78, r'in-sector $E_{1,q}$', fontsize=7.0, ha='center', va='bottom', color=PAL[2])

a2.annotate('', xy=(1.15, 3.62), xytext=(1.15, 0.33),
            arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.0))
a2.text(1.32, 1.98, r'$\Delta$', fontsize=8.2, va='center', color='#222222')
a2.annotate('', xy=(3.15, 4.62), xytext=(3.15, 0.33),
            arrowprops=dict(arrowstyle='<->', color='#009E73', lw=1.0))
a2.text(3.30, 2.55, r'$\Delta_q$', fontsize=8.2, va='center', color=PAL[2])

a2.fill_between([4.8, 9.5], 0.3, 1.85, color=PAL[1], alpha=0.15)
a2.plot([4.8, 9.5], [0.3, 0.3], color='#222222', lw=1.0, ls=':')
a2.plot([4.8, 9.5], [1.85, 1.85], color=PAL[1], lw=1.8)
a2.text(7.15, 1.05, 'forbidden for any $\\theta$\n' + r'$\Delta[1-\Lambda_C(2^{g_C})]$',
        fontsize=7.3, ha='center', va='center', color='#A63A00')

a2.fill_between([4.8, 9.5], 1.85, 4.55, color=PAL[0], alpha=0.08)
a2.plot([5.4, 8.9], [2.85, 2.85], color=PAL[0], lw=1.4, ls='-.')
a2.text(7.15, 3.05, r'optimum $E_{\mathcal{A}}^\star \geq E_{\mathrm{floor}}$',
        fontsize=7.3, ha='center', va='bottom', color=PAL[0])
a2.annotate('', xy=(7.15, 1.88), xytext=(7.15, 2.82),
            arrowprops=dict(arrowstyle='<-', color=PAL[0], lw=1.0))
fig.savefig('figures/schematic.png')
plt.close(fig)


# fig 1: window
fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.1), layout='constrained')
fig.get_layout_engine().set(w_pad=0.08, wspace=0.06)

ns = [r['n'] for r in DE['scaling']]
for idx, (eps, mk) in enumerate([(1e-2, 'o-'), (1e-3, 's-'), (1e-4, '^-')]):
    ax[0].plot(ns, [r[f'gstar_{eps:g}'] for r in DE['scaling']], mk,
               color=PAL[idx], ms=4.2,
               label=rf'$\varepsilon=10^{{{int(np.log10(eps))}}}$')
ax[0].plot(ns, [n // 2 for n in ns], '--', color='#222222', lw=1.1,
           label=r'dimension cap $\min(|A|,|B|)$')
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xticks([12, 16, 24, 32, 64, 128])
ax[0].set_xticklabels(['12', '16', '24', '32', '64', '128'])
ax[0].minorticks_off()
ax[0].set_xlabel('$n$')
ax[0].set_ylabel(r'$g^\star_C=\lceil\log_2 R_C(\varepsilon)\rceil$')
ax[0].set_title('(a) active window, critical TFIM')
ax[0].set_ylim(0.7, 300)
clean_legend(ax[0], fontsize=6.8, loc='upper left')

hs = [r['h'] for r in DE['window']]
for idx, (eps, mk) in enumerate([(1e-2, 'o-'), (1e-3, 's-'), (1e-4, '^-')]):
    ax[1].plot(hs, [r[f'gstar_{eps:g}'] for r in DE['window']], mk,
               color=PAL[idx], ms=4.2,
               label=rf'$\varepsilon=10^{{{int(np.log10(eps))}}}$')
ax[1].axvline(1.0, color='#333333', ls=':', lw=1.0)
ax[1].set_xlabel('$h/J$')
ax[1].set_ylabel(r'$g^\star_C$')
ax[1].set_title('(b) window vs. field ($n=64$)')
ax[1].set_ylim(0.85, 3.35)
ax[1].set_yticks([1, 1.5, 2, 2.5, 3])
clean_legend(ax[1], fontsize=6.8, loc='upper right')

for idx, n_ in enumerate((16, 32, 64, 128)):
    r = [x for x in DE['scaling'] if x['n'] == n_][0]
    gs = [1, 2, 3, 4]
    ax[2].semilogy(gs, [max(r[f'floor{g}'], 1e-18) for g in gs], 'o-',
                   color=PAL[idx], ms=4.2, label=f'$n={n_}$')
ax[2].set_xlabel('$g_C$')
ax[2].set_ylabel(r'central-cut floor $\Delta(1-\Lambda_C(2^{g_C}))$')
ax[2].set_title('(c) floor collapse in $g_C$')
ax[2].set_xticks([1, 2, 3, 4])
clean_legend(ax[2], fontsize=6.8, loc='upper right')
fig.savefig('figures/window.png')
plt.close(fig)


# fig 2: sector
fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.1), layout='constrained')
fig.get_layout_engine().set(w_pad=0.08, wspace=0.06)

tags = ['ordered', 'critical', 'paramagnetic']
lab_short = ['ordered\n$h/J=0.5$', 'critical\n$h/J=1$', 'paramag.\n$h/J=2$']
lab_leg = [r'ordered $h/J=0.5$', r'critical $h/J=1$', r'param. $h/J=2$']
x = np.arange(3)
wd = 0.36
ax[0].bar(x - wd / 2, [B[t]['gap_global'] for t in tags], wd,
          color=PAL[0], label=r'global gap $\Delta$')
ax[0].bar(x + wd / 2, [B[t]['gap_sector'] for t in tags], wd,
          color=PAL[1], label=r'sector gap $\Delta_q$')
ax[0].set_yscale('log')
ax[0].set_xticks(x)
ax[0].set_xticklabels(lab_short, fontsize=7.5)
ax[0].set_ylabel('gap')
ax[0].set_title('(a) the binding gap')
ax[0].set_ylim(1e-3, 6e2)
clean_legend(ax[0], fontsize=6.8, loc='upper right')
for i, t in enumerate(tags):
    ratio = B[t]['gap_sector'] / B[t]['gap_global']
    txt = rf'$\times${ratio:.0f}' if ratio >= 10 else rf'$\times${ratio:.1f}'
    ax[0].text(i, B[t]['gap_sector'] * 1.65, txt, ha='center', va='bottom',
               fontsize=7.2, color='#222222')

gg = np.arange(4)
for idx, (t, m) in enumerate(zip(tags, ['o-', 's-', '^-'])):
    ax[1].semilogy(gg, np.maximum(B[t]['floor_global'], 1e-12), m, ms=4.0,
                   alpha=0.38, color=PAL[idx], lw=1.2)
    ax[1].semilogy(gg, np.maximum(B[t]['floor_sector'], 1e-12), m, ms=4.2,
                   color=PAL[idx], label=lab_leg[idx])
ax[1].set_xlabel('$g_C$')
ax[1].set_ylabel('certified floor')
ax[1].set_title('(b) sector (solid) vs. global (faded)')
ax[1].set_xticks(gg)
ax[1].set_ylim(3e-11, 4.0)
clean_legend(ax[1], fontsize=6.6, loc='upper right')

t = 'ordered'
ax[2].semilogy(gg[:3], np.maximum(B[t]['trained_parity'], 1e-12), 'o-',
               color=PAL[0], ms=4.8, label='trained, parity-preserving')
ax[2].semilogy(gg[:3], np.maximum(B[t]['trained_hea'], 1e-12), 's-',
               color=PAL[1], ms=4.8, label='trained, symmetry-breaking')
ax[2].semilogy(gg[:3], np.maximum(B[t]['floor_sector'][:3], 1e-12), '--',
               color='#222222', lw=1.3, label='sector floor (parity-preserving only)')
ax[2].semilogy(gg[:3], np.maximum(B[t]['floor_global'][:3], 1e-12), ':',
               color='#222222', lw=1.3, label='global floor')
ax[2].set_xlabel('$g_C$')
ax[2].set_ylabel('$E - E_0$')
ax[2].set_title('(c) ordered phase, $n=8$')
ax[2].set_xticks(gg[:3])
ax[2].set_ylim(1e-8, 30)
clean_legend(ax[2], fontsize=6.3, loc='lower left')
fig.savefig('figures/sector.png')
plt.close(fig)


# fig 3: allocation
by = {(r['inst'], r['G'], r['arm']): r for r in A}
insts = sorted({r['inst'] for r in A})
BUD = sorted({r['G'] for r in A})
ARMS = ['layer', 'wf_unc', 'wf_con', 'rand_con']
NAME = {'layer': 'layer-filled (baseline)', 'wf_unc': 'water-filling',
        'wf_con': 'connected water-filling', 'rand_con': 'connected random'}

fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.1), layout='constrained')
fig.get_layout_engine().set(w_pad=0.08, wspace=0.06)

for idx, (arm, m) in enumerate(zip(ARMS, ['o-', 's-', 'd--', '^-'])):
    xs, ys = [], []
    for G in BUD:
        if (insts[0], G, arm) in by:
            xs.append(G)
            ys.append(np.median([by[(i, G, arm)]['err'] for i in insts]))
    ax[0].plot(xs, ys, m, color=PAL[idx], ms=4.2, label=NAME[arm])
ax[0].axvline(7, color='#333333', ls=':', lw=1.0)
ax[0].set_ylim(0.035, 0.255)
ax[0].text(7.15, 0.244, '$G=n-1$', fontsize=7.0, va='top', color='#333333')
ax[0].set_xlabel('two-qubit budget $G$')
ax[0].set_ylabel('median trained error')
ax[0].set_title('(a) trained error at fixed budget')
ax[0].set_xticks([5, 6, 7, 8, 9, 10, 11])
clean_legend(ax[0], fontsize=6.4, loc='upper right')

agree = []
for G in (8, 9, 11):
    a = t = 0
    for i in insts:
        seen, arms = set(), []
        for x in ARMS:
            if (i, G, x) in by and tuple(by[(i, G, x)]['alloc']) not in seen:
                seen.add(tuple(by[(i, G, x)]['alloc']))
                arms.append((by[(i, G, x)]['floor'], by[(i, G, x)]['err']))
        b = min(arms, key=lambda r: r[0])
        wo = max(arms, key=lambda r: r[0])
        if b[0] == wo[0]:
            continue
        t += 1
        a += int(b[1] < wo[1])
    agree.append(100 * a / max(t, 1))
bars = ax[1].bar(range(3), agree, width=0.56, color=[PAL[0], PAL[1], PAL[3]])
ax[1].axhline(50, color='#333333', ls='--', lw=1.0)
ax[1].text(2.42, 52.5, 'chance (50%)', fontsize=6.8, ha='right', color='#333333')
for rect, val in zip(bars, agree):
    ax[1].text(rect.get_x() + rect.get_width() / 2, val + 2.5, f'{val:.0f}%',
               ha='center', va='bottom', fontsize=7.2, color='#222222')
ax[1].set_xticks(range(3))
ax[1].set_xticklabels(['$G=8$', '$G=9$', '$G=11$'])
ax[1].set_ylabel('% of instances')
ax[1].set_ylim(0, 100)
ax[1].set_title('(b) lowest floor = lowest error?')

fl = np.array([by[k]['floor'] for k in by])
er = np.array([by[k]['err'] for k in by])
gg2 = np.array([by[k]['G'] for k in by])
for idx, (G, mk) in enumerate(zip(BUD, ['o', 's', '^', 'd', 'v'])):
    m = gg2 == G
    ax[2].loglog(fl[m], er[m], mk, color=PAL[idx], ms=3.8, alpha=0.72,
                 label=f'$G={G}$', zorder=3)
lo = np.array([1e-7, 1e-1])
ax[2].loglog(lo, lo, '--', color='#222222', lw=1.1, label='error = floor', zorder=2)
ax[2].set_xlabel('certified floor')
ax[2].set_ylabel('trained error')
ax[2].set_title(r'(c) floor $\ll$ trained error')
clean_legend(ax[2], fontsize=6.5, loc='lower right')
fig.savefig('figures/allocation.png')
plt.close(fig)


# fig 4: joint
Hm = json.load(open('expH_multiblock.json'))
fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.25), layout='constrained')
fig.get_layout_engine().set(w_pad=0.08, wspace=0.06)

e5 = [e for e in Hm['ensemble'] if e['G'] == 5]
for arm, off, c, name in [('layer', -0.17, PAL[0], 'layer-filled'),
                          ('wf_unc', 0.17, PAL[1], 'water-filled')]:
    sel = sorted((e for e in e5 if e['arm'] == arm), key=lambda e: e['inst'])
    x = np.array([e['inst'] for e in sel]) + off
    ax[0].vlines(x, [e['floor_mb'] for e in sel], [e['floor_max'] for e in sel],
                 color=c, lw=2.3, label=f'{name}: joint rank bracket')
    ax[0].plot(x, [e['floor_indep'] for e in sel], '_', color=c, ms=6.5, mew=1.2)
    ax[0].plot(x, [e['err'] for e in sel], 'x', color=c, ms=4.2, mew=1.1, alpha=0.85)
ax[0].plot([], [], '_', color='#222222', ms=6.5, mew=1.2, label='independent-cut floor')
ax[0].plot([], [], 'x', color='#222222', ms=4.2, mew=1.1, label='trained error')
ax[0].set_yscale('log')
ax[0].set_xlabel('instance')
ax[0].set_ylabel('energy')
ax[0].set_xticks(range(0, 16, 3))
ax[0].set_title('(a) disordered ensemble, $G=5$')
ax[0].set_ylim(2.5e-4, 0.7)
clean_legend(ax[0], fontsize=6.1, loc='lower center', ncol=2)

fit = {(r['n'], r['g']): r['floor_joint_ub'] for r in Cj['critical']}
for g, c in [(1, PAL[0]), (2, PAL[1])]:
    rows = sorted((r for r in Hm['critical'] if r['g'] == g), key=lambda r: r['n'])
    ns_ = [r['n'] for r in rows]
    ax[1].plot(ns_, [r['floor_indep'] for r in rows], 'o-', color=c, ms=3.8,
               label=f'$g={g}$: certified floor')
    ax[1].plot(ns_, [r['floor_max'] for r in rows], 's--', color=c, ms=3.4,
               mfc='white', label=f'$g={g}$: sum bound (upper)')
    fn = [n_ for n_ in ns_ if (n_, g) in fit]
    ax[1].plot(fn, [fit[(n_, g)] for n_ in fn], '^', color=c, ms=5.2,
               mfc='white', mew=1.1, label=f'$g={g}$: MPS fit (upper)')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xticks([8, 16, 32, 64, 128])
ax[1].set_xticklabels(['8', '16', '32', '64', '128'])
ax[1].minorticks_off()
ax[1].set_xlabel('$n$')
ax[1].set_ylabel('floor')
ax[1].set_title('(b) critical chain, uniform $g_C=g$')
ax[1].set_ylim(2e-8, 2.0)
clean_legend(ax[1], fontsize=6.0, ncol=2, loc='upper left')
fig.savefig('figures/joint.png')
plt.close(fig)


# fig 5: scaling
fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.1), layout='constrained')
fig.get_layout_engine().set(w_pad=0.08, wspace=0.06)

ns = np.array([r['n'] for r in DE['scaling']], float)
Ss = np.array([r['S'] for r in DE['scaling']])
ax[0].plot(np.log(2 * ns / np.pi), Ss, 'o', color=PAL[0], ms=4.8, label='free-fermion')
m = ns >= 32
A_ = np.vstack([np.log(2 * ns[m] / np.pi), np.ones(m.sum())]).T
sl, ic = np.linalg.lstsq(A_, Ss[m], rcond=None)[0]
xx = np.linspace(np.log(2 * 12 / np.pi), np.log(2 * 128 / np.pi), 10)
ax[0].plot(xx, sl * xx + ic, '--', color='#222222', lw=1.1,
           label=rf'fit $n\geq 32$: $c={6*sl:.3f}$')
ax[0].set_xlabel(r'$\ln(2n/\pi)$')
ax[0].set_ylabel('$S_{n/2}$ (nats)')
ax[0].set_title('(a) Calabrese–Cardy fit')
clean_legend(ax[0], fontsize=6.8, loc='upper left')

# (b) gap shrinks, deficit grows, floor stays flat (Lam1 = Lambda_C(2^1))
gap = np.array([r['gap'] for r in DE['scaling']])
deficit = np.array([1 - r['Lam1'] for r in DE['scaling']])
floor1 = np.array([r['floor1'] for r in DE['scaling']])
assert np.allclose(gap * deficit, floor1, rtol=1e-6)
ax[1].loglog(ns, gap, 's-', color=PAL[0], ms=4.0, label=r'gap $\Delta$')
ax[1].loglog(ns, deficit, '^-', color=PAL[1], ms=4.2, label=r'deficit $1-\Lambda_C(2)$')
ax[1].loglog(ns, floor1, 'o-', color=PAL[2], ms=4.2,
             label=r'floor $\Delta[1-\Lambda_C(2)]$')
ax[1].set_xticks([12, 16, 24, 32, 64, 128])
ax[1].set_xticklabels(['12', '16', '24', '32', '64', '128'])
ax[1].minorticks_off()
ax[1].set_xlabel('$n$')
ax[1].set_ylabel('central cut, $g_C=1$')
ax[1].set_title('(b) gap–deficit cancellation')
ax[1].set_ylim(1e-4, 1.0)
clean_legend(ax[1], fontsize=6.6, loc='upper right')

nn = 128
Gam, _ = C.tfim_covariance(nn, 1.0, 1.0)
cuts = np.arange(4, nn, 4)
for idx, (g, m2) in enumerate([(1, 'o-'), (2, 's-')]):
    d = []
    for c in cuts:
        nu = C.ff_block_nu(nn, int(c), Gamma=Gam)
        p = C.top_probs_from_nu(nu, 2 ** g)
        d.append(max(1 - p.sum(), 1e-18))
    ax[2].semilogy(cuts, d, m2, color=PAL[idx], ms=3.2, label=f'$g_C={g}$')
ax[2].set_xlabel('cut position $|A|$')
ax[2].set_ylabel(r'deficit $1-\Lambda_C(2^{g_C})$')
ax[2].set_title(f'(c) per-cut deficit, $n={nn}$')
clean_legend(ax[2], fontsize=6.8, loc='lower center')
fig.savefig('figures/scaling.png')
plt.close(fig)


# fig 6: angle
I = json.load(open('expI_angle_budget.json'))
fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.2), layout='constrained')
fig.get_layout_engine().set(w_pad=0.08, wspace=0.06)

gapS = I['I2']['gap']
sv = np.array([1.0, 1.0]) / np.sqrt(2)
th = np.linspace(0.0, 0.40, 200)
for p, c in [(1, PAL[0]), (2, PAL[1]), (3, PAL[2])]:
    ax[0].plot(th, [C.budget_floor(sv, gapS, [t] * p) for t in th], '-',
               color=c, lw=1.5, label=f'$p={p}$')
    ax[0].plot(th, [C.triangle_floor_multi(sv, gapS, [t] * p) for t in th], ':',
               color=c, lw=1.1)
    rr = [r for r in I['I2']['rows'] if r['p'] == p]
    ax[0].plot([r['theta'] for r in rr], [r['err'] for r in rr], 'o',
               color=c, ms=5.0, mfc='none', mew=1.2)
ax[0].plot([], [], '-', color='#222222', label='angle budget')
ax[0].plot([], [], ':', color='#222222', label='triangle bound')
ax[0].plot([], [], 'o', color='#222222', mfc='none', mew=1.2, label='trained')
ax[0].set_xlabel(r'$\theta$ per gate')
ax[0].set_ylabel(r'$E - E_0$')
ax[0].set_title(r'(a) singlet, $p$ gates across the cut')
clean_legend(ax[0], fontsize=6.4, ncol=2, loc='upper right')

for p, c in [(1, PAL[0]), (2, PAL[1]), (3, PAL[2]), (4, PAL[3])]:
    rr = sorted((r for r in I['I3']['rows'] if r['p'] == p and r['theta'] < 0.5),
                key=lambda r: r['theta'])
    t_ = [r['theta'] for r in rr]
    ax[1].plot(t_, [max(r['budget'], 1e-17) for r in rr], 'o-', color=c,
               ms=3.6, label=f'$p={p}$')
    ax[1].plot(t_, [max(r['rank'], 1e-17) for r in rr], '--', color=c, lw=1.0)
    ax[1].plot(t_, [r['err'] for r in rr], 'x', color=c, ms=5.0, mew=1.2)
ax[1].plot([], [], '-', color='#222222', label='angle budget')
ax[1].plot([], [], '--', color='#222222', label='rank floor')
ax[1].plot([], [], 'x', color='#222222', mew=1.2, label='trained')
ax[1].set_yscale('log')
ax[1].set_ylim(1e-17, 3.5)
ax[1].set_xlabel(r'$\theta$ per gate')
ax[1].set_ylabel('energy')
ax[1].set_title(r'(b) critical TFIM, $n=8$, $p$ layers')
clean_legend(ax[1], fontsize=6.1, ncol=2, loc='lower left')
fig.savefig('figures/angle.png')
plt.close(fig)


# fig 7: per-cut
n = 8
H = C.build_TFIM(n, h=1.0, periodic=True)
w, v = np.linalg.eigh(H)
gap = w[1] - w[0]
svs = C.cut_data(v[:, 0], n)
fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.1), layout='constrained')
fig.get_layout_engine().set(w_pad=0.08, wspace=0.06)

for idx, (g, m) in enumerate([(2, 'o-'), (1, 's-')]):
    ax[0].semilogy(range(1, n),
                   [max(gap * (1 - C.Lam(svs[k], 2 ** g)), 1e-17) for k in range(n - 1)],
                   m, color=PAL[idx], ms=4.2,
                   label=f'$g_C={g}$ ' + ('(all-NN, 2 layers)' if g == 2 else '(staggered, 2 layers)'))
ax[0].set_xlabel('cut $|A|$')
ax[0].set_ylabel('per-cut floor')
ax[0].set_title('(a) bottleneck is the central cut')
ax[0].set_xticks(range(1, n))
ax[0].set_ylim(2e-17, 3e-2)
clean_legend(ax[0], fontsize=6.6, loc='lower center')

rows = F['F3']['rows']
xx = np.arange(len(rows))
a2 = ax[1].twinx()
a2.spines['top'].set_visible(False)
b1 = ax[1].bar(xx - 0.19, [r['floor'] for r in rows], 0.36, color=PAL[0],
               label='certified floor (left)')
ax[1].set_yscale('log')
ax[1].set_ylabel('certified floor', color=PAL[0])
ax[1].tick_params(axis='y', colors=PAL[0])
ax[1].set_ylim(3e-5, 1.2e-2)
b2 = a2.bar(xx + 0.19, [r['err'] for r in rows], 0.36, color=PAL[1],
            label='trained error (right)')
a2.set_ylabel('trained error', color=PAL[1])
a2.tick_params(axis='y', colors=PAL[1])
a2.set_ylim(0, 0.19)
a2.grid(False)
ax[1].set_xticks(xx)
ax[1].set_xticklabels(['all-NN\n2 lay.\n(14 gates)', 'stagg.\n2 lay.\n(7 gates)',
                       'stagg.\n4 lay.\n(14 gates)'], fontsize=7.0)
ax[1].set_title(r'(b) $59\times$ in floor, $1.25\times$ in error')
ax[1].legend([b1, b2], ['certified floor (left)', 'trained error (right)'],
             fontsize=6.4, loc='upper right', frameon=True, facecolor='white',
             edgecolor='none', framealpha=0.88)
fig.savefig('figures/percut.png')
plt.close(fig)


# fig 8: hubbard
Hb = json.load(open('expG_hubbard.json'))
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0), layout='constrained')
fig.get_layout_engine().set(w_pad=0.08, wspace=0.06)

for idx, (k, m) in enumerate([('interleaved', 'o-'), ('spin-sector', 's-')]):
    cuts = range(1, len(Hb[k]['floors']) + 1)
    ax[0].plot(cuts, Hb[k]['floors'], m, color=PAL[idx], ms=4.2, label=k)
    ax[1].plot(cuts, Hb[k]['entropy'], m, color=PAL[idx], ms=4.2, label=k)
ax[0].set_xlabel('qubit cut $|A|$')
ax[0].set_ylabel(r'per-cut floor, $g_C=1$')
ax[0].set_title('(a) ordering sets the bottleneck')
clean_legend(ax[0], fontsize=6.8, loc='lower center')
ax[0].annotate(rf"$\times${Hb['interleaved']['peak']/Hb['spin-sector']['peak']:.2f}",
               xy=(4, Hb['interleaved']['peak']),
               xytext=(5.5, Hb['interleaved']['peak'] * 0.90),
               fontsize=7.5, color='#222222',
               arrowprops=dict(arrowstyle='->', lw=0.8, color='#222222'))
ax[1].set_xlabel('qubit cut $|A|$')
ax[1].set_ylabel('entropy (bits)')
ax[1].set_title('(b) entropy profile')
clean_legend(ax[1], fontsize=6.8, loc='lower center')
fig.savefig('figures/hubbard.png')
plt.close(fig)

print("figures written:", sorted(os.listdir('figures')))
