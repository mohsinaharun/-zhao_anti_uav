# =============================================================================
# Phase 4 analysis (restart/phase4/PREREG.md §6-§8). TEST pool, seeds 200-219, paired.
#   Primary family at P (Holm, alpha 0.05): A5-A1, A5-A2, A5-A3, A4-A1.
#   WIN/LOSS if Holm rejects (sign of mean), else TIE (+ EQUIVALENT if 95% CI in [-0.02, 0.02]).
#   Secondary (unadjusted): A2-A1, A3-A1 at P; all pairs at S1/S2.
# Writes results/phase4/analysis.json and plots/phase4_schedules.png. Missing arms are skipped.
# =============================================================================
import os, sys, json, glob
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(C.RESULTS, 'phase4')
ARMS = ['A1', 'A2', 'A3', 'A4', 'A5']
SEEDS = list(range(200, 220))
PRIMARY = [('A5', 'A1'), ('A5', 'A2'), ('A5', 'A3'), ('A4', 'A1')]
EQ = 0.02


def load(op, arm):
    recs = {}
    for s in SEEDS:
        f = os.path.join(OUT, 'eval_%s_%s_seed%d.json' % (op, arm, s))
        if os.path.exists(f):
            recs[s] = json.load(open(f))
    return recs


def compare(a, b):
    common = sorted(set(a) & set(b))
    d = np.array([a[s]['test_map50'] - b[s]['test_map50'] for s in common])
    n = len(d)
    if n < 2:
        return None
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p = float(stats.ttest_1samp(d, 0).pvalue) if se > 0 else 1.0
    return {'n': n, 'diff': float(m), 'ci95': [float(m - h), float(m + h)], 'p': p,
            'wins': int((d > 0).sum()), 'losses': int((d < 0).sum())}


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj


def verdict(c, p_adj):
    if p_adj < 0.05:
        return 'WIN' if c['diff'] > 0 else 'LOSS'
    return 'TIE (EQUIVALENT)' if -EQ <= c['ci95'][0] and c['ci95'][1] <= EQ else 'TIE'


def agent_cost(arm, op):
    f = os.path.join(OUT, 'agent_%s_%s_log.json' % (arm, op))
    if not os.path.exists(f):
        return None
    l = json.load(open(f)); e = l['episodes']
    return {'episodes': len(e), 'gpu_hours': e[-1]['seconds'] / 3600 if e else 0.0,
            'G4b': l.get('G4b'), 'selected_checkpoint': l.get('selected_checkpoint')}


res = {'ops': {}}
for op in ('P', 'S2', 'S1'):
    data = {arm: load(op, arm) for arm in ARMS}
    r = {'arms': {}, 'pairs': {}}
    print('\n=== %s ===' % op)
    print('%-4s %3s  %-8s %-8s  %-9s  %s' % ('arm', 'n', 'mAP', 'sd', 'eps max', 'mean sigma_t'))
    for arm, d in data.items():
        if not d:
            continue
        m = np.array([x['test_map50'] for x in d.values()])
        eps = max(x['eps_max'] for x in d.values()); tgt = next(iter(d.values()))['target_eps']
        sig = np.mean([x['sigmas'] for x in d.values()], axis=0)
        r['arms'][arm] = {'n': len(d), 'mean': float(m.mean()), 'sd': float(m.std(ddof=1)) if len(m) > 1 else 0.0,
                          'eps_max': eps, 'target_eps': tgt, 'mean_sigmas': sig.tolist(),
                          'cost': agent_cost(arm, op) if arm in ('A4', 'A5') else {'episodes': 0, 'gpu_hours': 0.0}}
        print('%-4s %3d  %.4f   %.4f   %.4f    %s' % (arm, len(d), m.mean(), r['arms'][arm]['sd'], eps,
                                                    ' '.join('%.2f' % s for s in sig)))
    pairs = PRIMARY + [('A2', 'A1'), ('A3', 'A1')] if op == 'P' else \
        [(x, y) for i, x in enumerate(ARMS) for y in ARMS[:i]]
    for x, y in pairs:
        if data[x] and data[y]:
            c = compare(data[x], data[y])
            if c:
                r['pairs']['%s-%s' % (x, y)] = c
    if op == 'P':
        fam = [('%s-%s' % p) for p in PRIMARY if '%s-%s' % p in r['pairs']]
        if fam:
            adj = holm(np.array([r['pairs'][k]['p'] for k in fam]))
            for k, pa in zip(fam, adj):
                r['pairs'][k].update({'family': 'primary', 'p_holm': float(pa), 'verdict': verdict(r['pairs'][k], pa)})
            r['primary_complete'] = len(fam) == len(PRIMARY)
    print('%-7s %3s  %-8s %-19s %-5s %-7s %-7s %s' % ('pair', 'n', 'diff', '95% CI', 'W/L', 'p', 'p_holm', 'verdict'))
    for k, c in r['pairs'].items():
        print('%-7s %3d  %+.4f  [%+.4f, %+.4f]  %2d/%-2d %.4f  %s  %s' % (
            k, c['n'], c['diff'], c['ci95'][0], c['ci95'][1], c['wins'], c['losses'], c['p'],
            '%.4f ' % c['p_holm'] if 'p_holm' in c else '  -    ', c.get('verdict', '(secondary)')))
    res['ops'][op] = r

if 'P' in res['ops'] and res['ops']['P'].get('primary_complete'):
    pr = res['ops']['P']['pairs']; a = res['ops']['P']['arms']
    best_fixed = max(('A2', 'A3'), key=lambda k: a[k]['mean'])
    res['learned_allocation_helps'] = pr['A5-A1']['verdict'] == 'WIN' and pr['A5-%s' % best_fixed]['verdict'] == 'WIN'
    res['best_fixed_formula'] = best_fixed
    print('\nbest of A2/A3 at P: %s;  "learned allocation helps": %s' % (best_fixed, res['learned_allocation_helps']))
res['meta'] = C.meta()
C.save_json(res, os.path.join(OUT, 'analysis.json'))

# learned vs theory schedules (§8)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cols = {'A1': '#7f7f7f', 'A2': '#1f77b4', 'A3': '#2ca02c', 'A4': '#ff7f0e', 'A5': '#d62728'}
names = {'A1': 'A1 uniform', 'A2': 'A2 tuned exp.', 'A3': 'A3 theory sqrt(a_t)', 'A4': 'A4 blind DRL', 'A5': 'A5 feedback DRL'}
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, op in zip(axs, ('P', 'S2', 'S1')):
    for arm, v in res['ops'].get(op, {}).get('arms', {}).items():
        s = v['mean_sigmas']
        ax.plot(range(1, len(s) + 1), s, '-o', ms=3, color=cols[arm],
                label='%s  (mAP %.3f)' % (names[arm], v['mean']))
    ax.set_title('%s  (eps = %.2f)' % (op, next(iter(res['ops'][op]['arms'].values()))['target_eps'])
                 if res['ops'].get(op, {}).get('arms') else op)
    ax.set_xlabel('round'); ax.set_ylabel('sigma_t (mean over test seeds)'); ax.set_yscale('log')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout()
os.makedirs(os.path.join(C.ROOT, 'plots'), exist_ok=True)
fig.savefig(os.path.join(C.ROOT, 'plots', 'phase4_schedules.png'), dpi=150)
print('saved plots/phase4_schedules.png')
