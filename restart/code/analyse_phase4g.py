# =============================================================================
# Phase 4g analysis (restart/phase4g/PREREG.md §6-7). F11: A2 between consecutive round counts
# (Holm over 3); gap closure G(R) = A2 / no-DP at the same R; verdict; predictions.
# R = 12 references are Phase 4 (A1, A2) and Phase 5 (no-DP FedAvg, no attack), read-only.
# =============================================================================
import os, sys, json, glob
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from phase4c import compare, holm, verdict
from phase4g import OUT, SEEDS

RS = (12, 25, 50, 100)
ARMS = ('A2', 'A1', 'noDP')


def load():
    S, recs = {}, {}
    def add(arm, R, r):
        S.setdefault((arm, R), {})[r['seed']] = r['test_map50']; recs.setdefault((arm, R), {})[r['seed']] = r
    for f in glob.glob(os.path.join(OUT, 'grid_*_seed*.json')):
        r = json.load(open(f)); add(r['arm'], r['R'], r)
    for arm in ('A1', 'A2'):
        for s in SEEDS:
            add(arm, 12, json.load(open(os.path.join(C.RESULTS, 'phase4', 'eval_P_%s_seed%d.json' % (arm, s)))))
    for s in SEEDS:
        r = json.load(open(os.path.join(C.RESULTS, 'phase5', 'nodp_noDP_fedavg_none_seed%d.json' % s)))
        r['eps_max'] = float('inf'); add('noDP', 12, r)
    return S, recs


def ci(v):
    v = np.array(list(v.values())); m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = stats.t.ppf(0.975, len(v) - 1) * se
    return float(m), float(v.std(ddof=1)), [float(m - h), float(m + h)]


def fam(pairs, label):
    res = {n: compare(a, b) for n, a, b in pairs}
    adj = holm(np.array([res[n]['p'] for n, _, _ in pairs]))
    for (n, _, _), pa in zip(pairs, adj):
        res[n].update({'p_holm': float(pa), 'verdict': verdict(res[n], pa)})
    print('\n%s' % label)
    for n, c in res.items():
        print('  %-22s diff %+.4f [%+.4f, %+.4f]  W/L %2d/%-2d  p %.4f  p_holm %.4f  %s' % (
            n, c['diff'], c['ci95'][0], c['ci95'][1], c['wins'], c['losses'], c['p'], c['p_holm'], c['verdict']))
    return res


def unadj(n, a, b):
    c = compare(a, b)
    print('  %-26s diff %+.4f [%+.4f, %+.4f]  W/L %2d/%-2d  p %.4f' % (n, c['diff'], c['ci95'][0], c['ci95'][1], c['wins'], c['losses'], c['p']))
    return c


def main():
    S, recs = load()
    res = {'counts': {'%s|R%d' % k: len(v) for k, v in S.items()}}
    print('reproduction: %s' % json.load(open(os.path.join(OUT, 'repro_ok.json')))['repro'])
    print('counts: %s' % res['counts'])
    res['table'] = {}
    print('\nTest mAP@0.5 by round count (20 seeds): mean, sd, 95% CI; worst-client eps; sigma range')
    for arm in ARMS:
        for R in RS:
            m, sd, c = ci(S[(arm, R)])
            sig = [r['sigmas'] for r in recs[(arm, R)].values()][0]
            eps = float(np.mean([r['eps_max'] for r in recs[(arm, R)].values()]))
            res['table']['%s|%d' % (arm, R)] = {'mean': m, 'sd': sd, 'ci95': c, 'eps': eps, 'sigma_first': sig[0], 'sigma_last': sig[-1]}
            print('  %-5s R %3d  %.4f  sd %.4f  [%.4f, %.4f]  eps %s  sigma %.2f..%.2f' % (
                arm, R, m, sd, c[0], c[1], '%.4f' % eps if np.isfinite(eps) else 'inf', sig[0], sig[-1]))
    ceil = {R: res['table']['noDP|%d' % R]['mean'] for R in RS}
    G = {R: res['table']['A2|%d' % R]['mean'] / ceil[R] for R in RS}
    G250 = {R: res['table']['A2|%d' % R]['mean'] / 0.250 for R in RS}
    res.update({'ceiling': ceil, 'G': G, 'G_vs_0.250': G250})
    print('\nceiling(R) = no-DP at R: %s' % {R: round(v, 4) for R, v in ceil.items()})
    print('G(R) = A2 / ceiling(R): %s' % {R: round(v, 3) for R, v in G.items()})
    print('A2 / 0.250 (12-round ceiling): %s' % {R: round(v, 3) for R, v in G250.items()})

    a2 = lambda R: S[('A2', R)]
    F11 = fam([('A2 R25 - R12', a2(25), a2(12)), ('A2 R50 - R25', a2(50), a2(25)), ('A2 R100 - R50', a2(100), a2(50))],
              'F11 (Holm over 3): A2 between consecutive round counts')
    res['F11'] = F11
    last = F11['A2 R100 - R50']['verdict']
    if last == 'WIN' or G[100] >= 0.8:
        v = 'CONVERGENCE-LIMITED'
    elif max(G.values()) < 0.5:
        v = 'DECLINING (noise ceiling)' if last == 'LOSS' else 'NOISE CEILING / PLATEAU'
    else:
        v = 'PARTIAL'
    res['verdict'] = v
    print('\nVERDICT Phase 4g: %s  (last step %s; G(100) %.3f; best G %.3f)' % (v, last, G[100], max(G.values())))

    print('\nSecondary (unadjusted)')
    sec = {}
    for R in RS:
        sec['A2-A1 R%d' % R] = unadj('A2 - A1 at R %d' % R, S[('A2', R)], S[('A1', R)])
    for arm in ('A1', 'noDP'):
        for r0, r1 in ((12, 25), (25, 50), (50, 100)):
            sec['%s R%d-R%d' % (arm, r1, r0)] = unadj('%s R%d - R%d' % (arm, r1, r0), S[(arm, r1)], S[(arm, r0)])
    bestR = max(RS, key=lambda R: res['table']['A2|%d' % R]['mean'])
    if bestR != 100:
        sec['A2 R100 - best'] = unadj('A2 R100 - R%d (best)' % bestR, a2(100), a2(bestR))
    for R in RS:   # per-seed gap closure, for a CI on G
        g = np.array([S[('A2', R)][s] / S[('noDP', R)][s] for s in SEEDS])
        h = stats.t.ppf(0.975, len(g) - 1) * g.std(ddof=1) / np.sqrt(len(g))
        sec['perseed_G R%d' % R] = {'mean': float(g.mean()), 'ci95': [float(g.mean() - h), float(g.mean() + h)]}
        print('  per-seed A2/noDP at R %3d: %.3f [%.3f, %.3f]' % (R, g.mean(), g.mean() - h, g.mean() + h))
    res['secondary'] = sec

    # proxy-loss trajectories (server-side public proxy session; logged for free in observe())
    traj = {}
    for arm in ARMS:
        for R in RS[1:]:
            L = np.array([[o['proxy_loss'] for o in r['obs']] for r in recs[(arm, R)].values()])
            traj['%s|%d' % (arm, R)] = L.mean(0).tolist()
    res['proxy_loss'] = traj
    for R in RS[1:]:
        print('  proxy loss R %3d, final: %s' % (R, '  '.join('%s %.3f' % (a, traj['%s|%d' % (a, R)][-1]) for a in ARMS)))

    m25 = res['table']['A2|25']['mean'] - res['table']['A2|12']['mean']
    preds = {
        'P1_plateau_last_step_not_WIN_and_max_A2_below_0.125': bool(last != 'WIN' and max(res['table']['A2|%d' % R]['mean'] for R in RS) < 0.125),
        'P2_rise_12_to_25_small (<0.02)': bool(abs(m25) < 0.02),
        'P3a_noDP_R100_above_0.250': bool(ceil[100] > 0.250),
        'P3b_G_falls_with_R': bool(all(G[a] > G[b] for a, b in zip(RS, RS[1:]))),
    }
    res['predictions'] = preds
    print('\nPredictions:')
    for k, v in preds.items():
        print('  %-55s %s' % (k, v))
    res['meta'] = C.meta()
    C.save_json(res, os.path.join(OUT, 'analysis.json'))

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    col = {'A2': 'tab:red', 'A1': 'tab:blue', 'noDP': 'k'}
    lab = {'A2': 'A2 (k 0.5), eps 1.86', 'A1': 'uniform, eps 1.86', 'noDP': 'no DP (ceiling)'}
    for arm in ARMS:
        mm = [res['table']['%s|%d' % (arm, R)]['mean'] for R in RS]
        lo = [res['table']['%s|%d' % (arm, R)]['ci95'][0] for R in RS]; hi = [res['table']['%s|%d' % (arm, R)]['ci95'][1] for R in RS]
        ax[0].plot(RS, mm, 'o-', color=col[arm], label=lab[arm]); ax[0].fill_between(RS, lo, hi, color=col[arm], alpha=0.15)
    ax[0].axhline(0.250, ls=':', color='gray', label='0.250 (no DP, 12 rounds)')
    ax[0].set_xscale('log'); ax[0].set_xticks(RS); ax[0].set_xticklabels(RS); ax[0].set_xlabel('communication rounds R')
    ax[0].set_ylabel('test mAP@0.5 (20 seeds, 95% CI)'); ax[0].set_title('mAP vs rounds at fixed eps 1.86'); ax[0].legend(fontsize=8)
    ax[1].plot(RS, [G[R] for R in RS], 'o-', color='tab:red', label='A2 / no-DP at same R')
    ax[1].plot(RS, [G250[R] for R in RS], 's--', color='tab:orange', label='A2 / 0.250')
    ax[1].axhline(0.5, ls=':', color='gray'); ax[1].axhline(0.8, ls=':', color='gray')
    ax[1].set_xscale('log'); ax[1].set_xticks(RS); ax[1].set_xticklabels(RS); ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('communication rounds R'); ax[1].set_title('gap closure G(R)'); ax[1].legend(fontsize=8)
    for R, ls in zip(RS[1:], ('-', '--', ':')):
        for arm in ('A2', 'noDP'):
            y = traj['%s|%d' % (arm, R)]
            ax[2].plot(np.linspace(0, 1, len(y)), y, ls, color=col[arm], label='%s R %d' % (arm, R))
    ax[2].set_xlabel('fraction of training (round / R)'); ax[2].set_ylabel('proxy loss (public session)')
    ax[2].set_title('server-side proxy loss during training'); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, 'plots', 'phase4g_rounds.png'), dpi=150)


if __name__ == '__main__':
    main()
