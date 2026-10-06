# Control (a): adaptive (per-round proxy p50) vs calibrated FIXED C* (proxy p50 after
# 2 proxy rounds, set once). Rule (Phase 2 Amendment 4): adaptive better only if the
# 95% CI of (adaptive - calibrated) is entirely > 0 at >= 2 sigma levels; reverse reported.
import os, sys, json
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

O = os.path.join(C.RESULTS, 'phase2')
SIG = ['0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4']
SUF = {'fixed45.8': '', 'calibrated': '_clipCal', 'adaptive': '_clipA50'}
load = lambda s, arm: [json.load(open(os.path.join(O, 'fl_dp_B64_s%s_lr0.03_mu0_R12_it20%s_seed%d.json'
                                                   % (s, SUF[arm], k)))) for k in range(10)]
R = {s: {arm: load(s, arm) for arm in SUF} for s in SIG}
M = {s: {arm: np.array([r['val_map50'] for r in R[s][arm]]) for arm in SUF} for s in SIG}

def ci(d):
    if d.std() == 0:
        return d.mean(), d.mean(), d.mean(), float('nan')
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h, stats.ttest_1samp(d, 0).pvalue

pos = neg = 0
print('%-5s %9s %10s %9s %27s %6s %8s %13s' % ('sigma', 'fixed45.8', 'calibrated', 'adaptive',
      'adaptive-calibrated [95%CI]', 'wins', 'p', 'eps max/med'))
for s in SIG:
    d = M[s]['adaptive'] - M[s]['calibrated']
    m, lo, hi, p = ci(d)
    pos += lo > 0; neg += hi < 0
    e = R[s]['calibrated'][0]
    for arm in SUF:
        assert s == '0' or abs(R[s][arm][0]['epsilon_max'] - e['epsilon_max']) < 1e-9
    print('%-5s %9.4f %10.4f %9.4f   %+.4f [%+.4f, %+.4f]%s %3d/10 %8.4f %13s' % (
        s, M[s]['fixed45.8'].mean(), M[s]['calibrated'].mean(), M[s]['adaptive'].mean(), m, lo, hi,
        ' +' if lo > 0 else ' -' if hi < 0 else '  ', int((d > 0).sum()), p,
        'inf' if s == '0' else '%.2f/%.2f' % (e['epsilon_max'], e['epsilon_median'])))
print('\nDECISION: adaptive beats calibrated-fixed at %d sigma levels (CI > 0); calibrated-fixed beats '
      'adaptive at %d (CI < 0).' % (pos, neg))
print('  -> %s' % ('ADAPTIVE STILL WINS: evidence of useful per-round adaptation' if pos >= 2 else
                   'ADVANTAGE DISAPPEARS: the earlier gain came from C = 45.8 being mis-calibrated'))
cs = [r['C_star'] for r in R['1']['calibrated']]
print('\nC* per seed: %s  (mean %.2f, range %.2f-%.2f)' % (' '.join('%.2f' % c for c in cs), np.mean(cs), min(cs), max(cs)))
for s in ('1', '2', '4'):
    Cr = np.mean([r['clip_per_round'] for r in R[s]['adaptive']], 0)
    print('adaptive C per round, sigma %s: %s' % (s, ' '.join('%5.1f' % c for c in Cr)))

fig, ax = plt.subplots(figsize=(7, 4.2))
xs = [float(s) for s in SIG]
for arm, col, mk, lbl in (('fixed45.8', '#2a78d6', 'o', 'fixed C = 45.8 (init median)'),
                          ('calibrated', '#1baf7a', '^', 'fixed C* (proxy, after 2 rounds) ≈ %.1f' % np.mean(cs)),
                          ('adaptive', '#eb6834', 's', 'adaptive C per round (proxy p50)')):
    m = [M[s][arm].mean() for s in SIG]
    h = [stats.t.ppf(0.975, 9) * M[s][arm].std(ddof=1) / np.sqrt(10) for s in SIG]
    ax.errorbar(xs, m, yerr=h, color=col, marker=mk, ms=5, lw=2, capsize=2, label=lbl)
ax.set_xlabel('noise multiplier σ'); ax.set_ylabel('val mAP@0.5 (mean, 95% CI, 10 seeds)')
ax.set_title('Clipping control: FL, 8 clients, B=64, 20 iters/round', loc='left', fontsize=10)
ax.grid(axis='y', color='#e4e3df'); ax.legend(frameon=False, fontsize=8)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(C.ROOT, 'restart', 'figures', 'clip_control.%s' % ext), dpi=200)
