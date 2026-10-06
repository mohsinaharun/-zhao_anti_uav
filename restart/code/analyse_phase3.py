# Phase 3 analysis (restart/phase3/PREREG.md), one variant at a time:
#   python analyse_phase3.py V1     (R=12, K=9)   V2 (R=25, K=9)   V3 (R=12, K=7)
# Per sigma: a_t profile (mean drop per pulse round, 95% CI), slope test (per-seed OLS
# slope of drop on round), shape test (first third - last third), informativeness
# (pooled drop > 0), gradient-norm trend (baseline runs; all rounds and rounds 3..R).
import os, sys, json
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

VAR = {'V1': (12, 9), 'V2': (25, 9), 'V3': (12, 7)}
v = sys.argv[1] if len(sys.argv) > 1 else 'V1'
R, K = VAR[v]
O = os.path.join(C.RESULTS, 'phase3')
SIG, SEEDS = ['1', '2', '4'], range(20)


def path(s, seed, k):
    return os.path.join(O, 'fl_dp_B64_s%s_lr0.03_mu0_R%d_it20_clipCal%s%s_seed%d.json'
                        % (s, R, '_K%d' % K if K != 9 else '', '_pk%d' % k if k else '', seed))


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std() == 0:
        return x.mean(), x.mean(), x.mean(), float('nan')
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h, stats.ttest_1samp(x, 0).pvalue


third = R // 3
out, verdicts = {}, {}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
cols = {'1': '#2a78d6', '2': '#eb6834', '4': '#1baf7a'}
print('PHASE 3 %s  (R=%d rounds, K=%d)  pulse = 2x sigma in one round' % (v, R, K))
for s in SIG:
    seeds = [sd for sd in SEEDS if all(os.path.exists(path(s, sd, k)) for k in range(R + 1))]
    if len(seeds) < 2:
        print('\nsigma %s: only %d complete seeds, skipped' % (s, len(seeds))); continue
    base = {sd: json.load(open(path(s, sd, 0))) for sd in seeds}
    D = np.array([[base[sd]['val_map50'] - json.load(open(path(s, sd, k)))['val_map50']
                   for k in range(1, R + 1)] for sd in seeds])            # seeds x rounds
    ks = np.arange(1, R + 1)
    slopes = [np.polyfit(ks, D[i], 1)[0] for i in range(len(seeds))]
    shape = D[:, :third].mean(1) - D[:, -third:].mean(1)
    pooled = D.mean(1)
    sl, sh, po = ci(slopes), ci(shape), ci(pooled)
    informative = po[1] > 0
    trend = not (sl[1] <= 0 <= sl[2]); varies = not (sh[1] <= 0 <= sh[2])
    verdict = 'NOT IDENTIFIABLE' if not informative else ('VARIES' if (trend or varies) else 'FLAT')
    verdicts[s] = verdict
    # gradient norm (baseline runs): clients' mean pre-clip median per round
    G = np.array([[np.mean([c['median'] for c in rr.values()]) for rr in base[sd]['client_grad_norms']] for sd in seeds])
    gs_all = ci([np.polyfit(ks, G[i], 1)[0] for i in range(len(seeds))])
    gs_late = ci([np.polyfit(ks[2:], G[i, 2:], 1)[0] for i in range(len(seeds))])
    bm = np.mean([base[sd]['val_map50'] for sd in seeds])
    e = base[seeds[0]]
    print('\n== sigma %s  (n = %d seeds)  baseline mAP %.4f  eps max %.2f / median %.2f  C* mean %.2f'
          % (s, len(seeds), bm, e['epsilon_max'], e['epsilon_median'], np.mean([base[sd]['C_star'] for sd in seeds])))
    print('   round: ' + ' '.join('%7d' % k for k in ks))
    print('   drop : ' + ' '.join('%+7.4f' % m for m in D.mean(0)))
    print('   95%CI: ' + ' '.join('%7.4f' % (stats.t.ppf(0.975, len(seeds) - 1) * D[:, j].std(ddof=1) / np.sqrt(len(seeds))) for j in range(R)))
    print('   pooled drop (informativeness): %+.4f [%+.4f, %+.4f]  -> %s' % (po[0], po[1], po[2], 'informative' if informative else 'NOT informative'))
    print('   slope of drop per round      : %+.5f [%+.5f, %+.5f] p=%.3f  -> %s' % (sl[0], sl[1], sl[2], sl[3], 'TREND' if trend else 'no trend'))
    print('   first third - last third     : %+.4f [%+.4f, %+.4f] p=%.3f  -> %s' % (sh[0], sh[1], sh[2], sh[3], 'VARIES' if varies else 'no shape effect'))
    print('   grad-norm slope, all rounds  : %+.4f [%+.4f, %+.4f]   rounds 3..%d: %+.4f [%+.4f, %+.4f]'
          % (gs_all[0], gs_all[1], gs_all[2], R, gs_late[0], gs_late[1], gs_late[2]))
    print('   grad norm per round (mean)   : ' + ' '.join('%.1f' % g for g in G.mean(0)))
    print('   VERDICT for this cell: %s' % verdict)
    out[s] = {'n': len(seeds), 'drop_mean': D.mean(0).tolist(), 'drop_per_seed': D.tolist(), 'slope': sl[:3],
              'shape': sh[:3], 'pooled': po[:3], 'verdict': verdict, 'gradnorm_mean': G.mean(0).tolist(),
              'gradnorm_slope_all': gs_all[:3], 'gradnorm_slope_late': gs_late[:3], 'baseline_map': bm}
    h = [stats.t.ppf(0.975, len(seeds) - 1) * D[:, j].std(ddof=1) / np.sqrt(len(seeds)) for j in range(R)]
    axes[0].errorbar(ks, D.mean(0), yerr=h, color=cols[s], marker='o', ms=4, lw=2, capsize=2,
                     label='σ=%s (n=%d): %s' % (s, len(seeds), verdict))
    axes[1].plot(ks, G.mean(0), color=cols[s], marker='o', ms=4, lw=2, label='σ=%s' % s)
axes[0].axhline(0, color='#52514e', lw=0.8)
axes[0].set_xlabel('pulse round k'); axes[0].set_ylabel('accuracy drop from a 2σ pulse in round k')
axes[0].set_title('(a) a_t profile, %s (R=%d, K=%d)' % (v, R, K), loc='left', fontsize=10)
axes[1].set_xlabel('round'); axes[1].set_ylabel('clients\' median pre-clip gradient norm')
axes[1].set_title('(b) gradient norm by round (baseline runs)', loc='left', fontsize=10)
for ax in axes:
    ax.grid(axis='y', color='#e4e3df'); ax.legend(frameon=False, fontsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(C.ROOT, 'restart', 'figures', 'phase3_%s.%s' % (v, ext)), dpi=200)
json.dump({'variant': v, 'R': R, 'K': K, 'cells': out}, open(os.path.join(O, 'analysis_%s.json' % v), 'w'), indent=1)
print('\n%s cell verdicts: %s' % (v, verdicts))
