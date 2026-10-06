# Phase 2 analysis (restart/phase2/PREREG.md): gate G2, the 2x2 contrasts with
# paired-t 95% CIs and win counts, per-environment mAP, epsilon, and a figure.
import os, sys, json
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

O = os.path.join(C.RESULTS, 'phase2')
sel = dict(kv.split('=') for kv in open(os.path.join(O, 'selected.txt')).read().split())
LR, MU, DLR = sel['LR'], sel['MU'], sel['DLR']
SEEDS = range(10)
CELLS = {'FL+DP': 'fl_dp_B256_s2_lr%s_mu%s_R12_seed%%d' % (DLR, MU),
         'FL no-DP': 'fl_nodp_B256_s0_lr%s_mu%s_R12_seed%%d' % (LR, MU),
         'DP no-FL': 'central_dp_B1024_s2_lr0.03_mu0_R12_seed%d',
         'neither': 'central_nodp_B1024_s0_lr0.03_mu0_R12_seed%d'}
R = {k: [json.load(open(os.path.join(O, (v % s) + '.json'))) for s in SEEDS] for k, v in CELLS.items()}
M = {k: np.array([r['val_map50'] for r in v]) for k, v in R.items()}
triv = json.load(open(os.path.join(C.RESULTS, 'phase1', 'trivial.json')))['val_map50']


def ci(d):
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    return m, m - h, m + h, int((d > 0).sum()), stats.ttest_1samp(d, 0).pvalue


g2 = all(M[k].mean() >= triv + 0.10 for k in M)
print('=' * 76)
print('GATE G2: %s   (selected: FL lr %s, FedProx mu %s, FL+DP lr %s)' % ('PASS' if g2 else 'FAIL', LR, MU, DLR))
for k in CELLS:
    print('  %-9s mean %.4f  sd %.4f  (>= %.2f %s)' % (k, M[k].mean(), M[k].std(ddof=1), triv + 0.10,
                                                    'ok' if M[k].mean() >= triv + 0.10 else 'FAIL'))
print('=' * 76)
con = {'cost of FL (no DP)       FL no-DP - neither': M['FL no-DP'] - M['neither'],
       'cost of DP (no FL)       DP no-FL - neither': M['DP no-FL'] - M['neither'],
       'cost of DP inside FL     FL+DP - FL no-DP': M['FL+DP'] - M['FL no-DP'],
       'interaction  (FL+DP-FL) - (DPnoFL-neither)': (M['FL+DP'] - M['FL no-DP']) - (M['DP no-FL'] - M['neither'])}
print('%-44s %8s %20s %6s %8s' % ('contrast (n=10 paired seeds)', 'mean', '95% CI', 'wins', 'p'))
for k, d in con.items():
    m, lo, hi, w, p = ci(d)
    print('%-44s %+8.4f  [%+.4f, %+.4f] %3d/10 %8.4f' % (k, m, lo, hi, w, p))
print('\nper-environment val mAP@0.5 (mean over seeds):')
envs = sorted(R['neither'][0]['val_map50_by_env'])
print('  %-9s %s' % ('cell', ''.join('%10s' % e for e in envs)))
for k in CELLS:
    print('  %-9s %s' % (k, ''.join('%10.3f' % np.mean([r['val_map50_by_env'][e] for r in R[k]]) for e in envs)))
fe = R['FL+DP'][0]; ce = R['DP no-FL'][0]
print('\nepsilon at delta=1e-5, sigma=2:  FL clients max %.2f / median %.2f  (per client: %s);  no-FL %.2f'
      % (fe['epsilon_max'], fe['epsilon_median'], {c: round(v, 2) for c, v in fe['epsilon_delta1e-5'].items()},
         ce['epsilon_max']))
print('FedProx selection (FL no-DP, seed 0): %s' % {mu: round(json.load(open(os.path.join(
    O, 'fl_nodp_B256_s0_lr%s_mu%s_R12_seed0.json' % (LR, mu))))['val_map50'], 4) for mu in ('0', '0.01', '0.1', '1')})

fig, ax = plt.subplots(figsize=(6.4, 4))
cols = {'FL+DP': '#2a78d6', 'FL no-DP': '#eb6834', 'DP no-FL': '#1baf7a', 'neither': '#52514e'}
for k in CELLS:
    h = np.mean([[x['val_map50'] for x in r['history']] for r in R[k]], 0)
    ax.plot(range(1, len(h) + 1), h, lw=2, color=cols[k], label='%s (final %.3f)' % (k, M[k].mean()))
ax.set_xlabel('round (= epoch over the data)'); ax.set_ylabel('val mAP@0.5 (mean of 10 seeds)')
ax.set_ylim(0, 0.7); ax.grid(axis='y', color='#e4e3df')
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.set_title('Phase 2: FL x DP ablation (sigma=2, 8 clients)', loc='left', fontsize=10)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(C.ROOT, 'restart', 'figures', 'phase2_ablation.%s' % ext), dpi=200)
