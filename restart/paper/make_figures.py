# Figures for restart/paper/paper.tex, built only from stored result files (read-only).
#   python restart/paper/make_figures.py
# Writes vector PDFs to restart/paper/figs/ and prints every plotted number for traceability.
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, 'figs')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42})
SEEDS = range(200, 220)
J = lambda p: json.load(open(p))


# ── Fig. 2: round sensitivity a_t, normalised by the baseline score ──────────────────────────
def pulse_profile(analysis, base_glob=None, key=None):
    from scipy import stats
    a = J(analysis)
    cell = a['cells']['2'] if 'cells' in a else a
    d = np.array(cell['drop_mean'])
    if 'drop_ci' in cell:                                   # list of [lo, hi] per round
        ci = np.array([(hi - lo) / 2 for lo, hi in cell['drop_ci']])
    else:                                                   # per-seed drops, seeds x rounds
        P = np.array(cell['drop_per_seed'])
        ci = stats.t.ppf(0.975, P.shape[0] - 1) * P.std(0, ddof=1) / np.sqrt(P.shape[0])
    base = cell['baseline_map'] if 'baseline_map' in cell else np.mean([J(f)[key] for f in glob.glob(base_glob)])
    return d, ci, base

prof = {'Anti-UAV drone': pulse_profile(os.path.join(RES, 'phase3', 'analysis_V1.json')),
        'VisDrone car': pulse_profile(os.path.join(RES, 'phase4f', 'analysis_pulse.json'),
                                      os.path.join(RES, 'phase4f', 'pulse_K9_B64_base_seed*.json'), 'score'),
        'CIFAR-10': pulse_profile(os.path.join(RES, 'phase4c', 'cifar10', 'analysis_pulse.json'),
                                  os.path.join(RES, 'phase4c', 'cifar10', 'pulse_K9_B64_base_seed*.json'), 'score')}
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for (lab, (d, ci, base)), c in zip(prof.items(), ('tab:red', 'tab:orange', 'tab:blue')):
    x = np.arange(1, len(d) + 1)
    ax.errorbar(x, d / base, yerr=ci / base, fmt='o-', ms=3, lw=1, capsize=2, color=c, label='%s, baseline %.3f' % (lab, base))
    print('Fig2 %-22s baseline %.4f  drop/base: %s' % (lab, base, ' '.join('%+.3f' % v for v in d / base)))
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('round t receiving the noise pulse (σ doubled)')
ax.set_ylabel('drop in final score / baseline score')
ax.set_xticks(range(1, 13)); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_pulse.pdf'))


# ── Fig. 4: the noise-dominance mechanism across privacy levels (Phase 5b table, no attack) ──
sig = [0, 0.5, 1, 2, 4]
eps = ['∞', '30.16', '5.73', '1.86', '0.79']
cos = [0.086, 0.027, 0.015, 0.0075, 0.0033]             # phase5b/RESULTS.md, Mechanism table
ratio = [1.01, 0.54, 0.30, 0.26, 0.20]                   # FLTrust / FedAvg, clean
shrink = [1.02, 1 / 10.9, 1 / 20.4, 1 / 38.6, 1 / 73.7]  # root norm / honest norm
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
x = np.arange(len(sig))
ax[0].semilogy(x, cos, 'o-', color='k', label='honest cosine with root update')
ax[0].semilogy(x, shrink, 's--', color='tab:gray', label='FLTrust shrink factor (root / honest norm)')
ax[0].set_xticks(x); ax[0].set_xticklabels(['σ %g\nε %s' % (s, e) for s, e in zip(sig, eps)])
ax[0].set_ylabel('value (log scale)'); ax[0].legend(frameon=False); ax[0].set_title('(a) honest DP updates are mostly noise')
ax[1].plot(x, ratio, 'o-', color='tab:red', label='FLTrust / FedAvg, no attack')
ax[1].axhline(1, color='k', lw=0.6)
vn = {0: 0.253 / 0.250, 1: 0.187 / 0.179, 2: 0.147 / 0.132, 3: 0.0711 / 0.0592, 4: 0.018 / 0.014}   # phase5c Table W
ax[1].plot(list(vn), list(vn.values()), 's-', color='tab:blue', label='Variant N / FedAvg, no attack')
ax[1].set_xticks(x); ax[1].set_xticklabels(['σ %g\nε %s' % (s, e) for s, e in zip(sig, eps)])
ax[1].set_ylabel('ratio of mean test mAP@0.5'); ax[1].set_ylim(0, 1.4); ax[1].legend(frameon=False)
ax[1].set_title('(b) FLTrust loses at every DP level; Variant N does not')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_mechanism.pdf'))
print('Fig4 cos', cos, 'shrink', [round(v, 4) for v in shrink], 'ratio', ratio, 'VarN/FedAvg', {k: round(v, 3) for k, v in vn.items()})


# ── Fig. 7: per-seed paired differences for the headline comparisons (n = 20 each) ────────────
def series(fmt):
    return {s: J(fmt % s)['test_map50'] for s in SEEDS}

P4, P4E, P4H = (os.path.join(RES, p) for p in ('phase4', 'phase4e', 'phase4h'))
P5, P5B, P5C, P5D = (os.path.join(RES, p) for p in ('phase5', 'phase5b', 'phase5c', 'phase5d'))
cmp = [
    ('A2 − uniform (P)', series(P4 + '/eval_P_A2_seed%d.json'), series(P4 + '/eval_P_A1_seed%d.json')),
    ('A3 − uniform (P)', series(P4 + '/eval_P_A3_seed%d.json'), series(P4 + '/eval_P_A1_seed%d.json')),
    ('A5 DRL − uniform (P)', series(P4 + '/eval_P_A5_seed%d.json'), series(P4 + '/eval_P_A1_seed%d.json')),
    ('A7 BO − A3 (P)', series(P4E + '/eval_P_A7_seed%d.json'), series(P4 + '/eval_P_A3_seed%d.json')),
    ('A5-relative − A3 (P)', series(P4H + '/eval_P_A5r_seed%d.json'), series(P4 + '/eval_P_A3_seed%d.json')),
    ('FLTrust − FedAvg, no attack', series(P5 + '/main_A1_fltrust_none_seed%d.json'), series(P5 + '/main_A1_fedavg_none_seed%d.json')),
    ('Variant N − FedAvg, no attack', series(P5B + '/variant_s2_fltrust_N_none_seed%d.json'), series(P5 + '/main_A1_fedavg_none_seed%d.json')),
    ('Variant N − FedAvg, trust-capture', series(P5C + '/adaptive_s2_fltrust_N_adaptive_seed%d.json'), series(P5C + '/adaptive_s2_fedavg_adaptive_seed%d.json')),
    ('cap 1/K − no cap, trust-capture', series(P5D + '/grid_m1_adaptive_seed%d.json'), series(P5C + '/adaptive_s2_fltrust_N_adaptive_seed%d.json')),
]
fig, ax = plt.subplots(figsize=(7.0, 3.0))
rng = np.random.default_rng(0)
for i, (lab, a, b) in enumerate(cmp):
    d = np.array([a[s] - b[s] for s in SEEDS])
    ax.scatter(d, i + rng.uniform(-0.15, 0.15, len(d)), s=8, color='tab:gray', alpha=0.8)
    ax.plot([d.mean()] * 2, [i - 0.3, i + 0.3], color='tab:red', lw=2)
    print('Fig7 %-36s mean %+.4f  wins %2d losses %2d  min %+.4f max %+.4f' % (lab, d.mean(), (d > 0).sum(), (d < 0).sum(), d.min(), d.max()))
ax.axvline(0, color='k', lw=0.6)
ax.set_yticks(range(len(cmp))); ax.set_yticklabels([c[0] for c in cmp]); ax.invert_yaxis()
ax.set_xlabel('paired difference in test mAP@0.5 (one dot per seed, n = 20; red = mean)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_perseed.pdf'))
