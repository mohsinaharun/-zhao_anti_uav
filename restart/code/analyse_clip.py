# Adaptive (proxy-percentile, per round) vs fixed (C=45.8) clipping, paired over 10
# seeds. Decision rule (Phase 2 Amendment 3): adaptive helps only if the 95% CI of
# (adaptive - fixed) lies entirely above 0 at >= 2 sigma levels.
import os, sys, json
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

O = os.path.join(C.RESULTS, 'phase2')
SIG = ['0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4']
f = lambda s, k, ad: os.path.join(O, 'fl_dp_B64_s%s_lr0.03_mu0_R12_it20%s_seed%d.json' % (s, '_clipA50' if ad else '', k))
R = {s: {ad: [json.load(open(f(s, k, ad))) for k in range(10)] for ad in (False, True)} for s in SIG}

rows, helps = [], 0
print('%-5s %8s %8s %24s %6s %8s %14s' % ('sigma', 'fixed', 'adaptive', 'adaptive-fixed [95% CI]', 'wins', 'p', 'eps max/med'))
for s in SIG:
    fx = np.array([r['val_map50'] for r in R[s][False]]); ad = np.array([r['val_map50'] for r in R[s][True]])
    d = ad - fx
    h = stats.t.ppf(0.975, 9) * d.std(ddof=1) / np.sqrt(10) if d.std() > 0 else 0.0
    p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else float('nan')
    sig_pos = d.mean() - h > 0
    helps += sig_pos
    e = R[s][True][0]
    eps = 'inf' if s == '0' else '%.2f/%.2f' % (e['epsilon_max'], e['epsilon_median'])
    assert s == '0' or abs(R[s][False][0]['epsilon_max'] - e['epsilon_max']) < 1e-9, 'epsilon differs between arms'
    rows.append((float(s), fx, ad, d, h))
    print('%-5s %8.4f %8.4f   %+.4f [%+.4f, %+.4f]%s %3d/10 %8.4f %14s' % (s, fx.mean(), ad.mean(), d.mean(),
          d.mean() - h, d.mean() + h, ' *' if sig_pos else '  ', int((d > 0).sum()), p, eps))
print('\nDECISION (CI entirely > 0 at >= 2 sigma levels): %s  (%d sigma levels qualify)'
      % ('ADAPTIVE CLIPPING HELPS' if helps >= 2 else 'NOT SHOWN TO HELP', helps))

print('\nC chosen per round (mean over 10 seeds) vs fixed 45.8:')
for s in ('0', '1', '2', '4'):
    Cr = np.mean([r['clip_per_round'] for r in R[s][True]], 0)
    print('  sigma %-3s %s' % (s, ' '.join('%5.1f' % c for c in Cr)))
print('\nmean fraction of per-sample gradients clipped (all rounds, clients, seeds):')
for s in ('0.5', '1', '2', '4'):
    fr = {ad: np.mean([v['frac_clipped'] for r in R[s][ad] for rr in r['client_grad_norms'] for v in rr.values()])
          for ad in (False, True)}
    md = {ad: np.mean([v['median'] for r in R[s][ad] for rr in r['client_grad_norms'] for v in rr.values()])
          for ad in (False, True)}
    print('  sigma %-3s fixed %.1f%% clipped (median pre-clip norm %.1f) | adaptive %.1f%% clipped (median %.1f)'
          % (s, 100 * fr[False], md[False], 100 * fr[True], md[True]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
xs = [r[0] for r in rows]
for k, (lbl, col, mk) in enumerate((('fixed C = 45.8', '#2a78d6', 'o'), ('adaptive C (proxy p50)', '#eb6834', 's'))):
    m = [r[1 + k].mean() for r in rows]
    ci = [stats.t.ppf(0.975, 9) * r[1 + k].std(ddof=1) / np.sqrt(10) for r in rows]
    axes[0].errorbar(xs, m, yerr=ci, color=col, marker=mk, ms=5, lw=2, capsize=2, label=lbl)
axes[0].set_xlabel('noise multiplier σ'); axes[0].set_ylabel('val mAP@0.5 (mean, 95% CI)')
axes[0].set_title('(a) FL, B=64: fixed vs adaptive clipping', loc='left', fontsize=10)
axes[0].legend(frameon=False, fontsize=8)
for s, col in (('0', '#52514e'), ('1', '#eb6834'), ('2', '#1baf7a'), ('4', '#4a3aa7')):
    Cr = np.mean([r['clip_per_round'] for r in R[s][True]], 0)
    axes[1].plot(range(1, len(Cr) + 1), Cr, color=col, lw=2, marker='o', ms=4, label='σ=%s' % s)
axes[1].axhline(45.8, color='#2a78d6', ls='--', lw=1.2, label='fixed C = 45.8')
axes[1].set_xlabel('round'); axes[1].set_ylabel('clip bound C used'); axes[1].set_ylim(0, None)
axes[1].set_title('(b) adaptive C per round (mean of 10 seeds)', loc='left', fontsize=10)
axes[1].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.grid(axis='y', color='#e4e3df')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(C.ROOT, 'restart', 'figures', 'clip_adaptive_vs_fixed.%s' % ext), dpi=200)
