# =============================================================================
# Phase 4h analysis (restart/phase4h/PREREG.md §4-6). Per operating point, one Holm family of 4:
# A5r vs A1, A2, A3, A5 (Phase 4 runs, read-only). Genuine improvement iff WIN vs A1 AND vs the
# better of A2/A3 (fixed: A3 at P, A2 at S1/S2). Adjustment-pattern classification (§5).
# =============================================================================
import os, sys, json, glob
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from phase4c import compare, holm, verdict
from phase4h import OUT, SEEDS

OPS = ('P', 'S1', 'S2')
BETTER = {'P': 'A3', 'S1': 'A2', 'S2': 'A2'}
P4OUT = os.path.join(C.RESULTS, 'phase4')


def load():
    S, recs = {}, {}
    for op in OPS:
        for arm in ('A1', 'A2', 'A3', 'A5'):
            for s in SEEDS:
                r = json.load(open(os.path.join(P4OUT, 'eval_%s_%s_seed%d.json' % (op, arm, s))))
                S.setdefault((op, arm), {})[s] = r['test_map50']; recs.setdefault((op, arm), {})[s] = r
        for f in glob.glob(os.path.join(OUT, 'eval_%s_A5r_seed*.json' % op)):
            r = json.load(open(f))
            S.setdefault((op, 'A5r'), {})[r['seed']] = r['test_map50']; recs.setdefault((op, 'A5r'), {})[r['seed']] = r
    return S, recs


def fam(pairs, label):
    res = {n: compare(a, b) for n, a, b in pairs}
    adj = holm(np.array([res[n]['p'] for n, _, _ in pairs]))
    for (n, _, _), pa in zip(pairs, adj):
        res[n].update({'p_holm': float(pa), 'verdict': verdict(res[n], pa)})
    print('\n%s' % label)
    for n, c in res.items():
        print('  %-10s diff %+.4f [%+.4f, %+.4f]  W/L %2d/%-2d  p %.4f  p_holm %.4f  %s' % (
            n, c['diff'], c['ci95'][0], c['ci95'][1], c['wins'], c['losses'], c['p'], c['p_holm'], c['verdict']))
    return res


def pattern(recs):
    D = np.array([r['deltas'] for r in recs.values()])           # seeds x (R-1)
    n = D.shape[0]
    m = D.mean(0); sd = D.std(0, ddof=1); h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    zero = float((np.abs(D) < 0.02).mean()); sat = float((np.abs(D) >= 0.19).mean())
    cls = 'ZERO-COLLAPSE' if zero >= 0.9 else 'SATURATED' if sat >= 0.9 else 'STRUCTURED'
    excl = float(((m - h > 0) | (m + h < 0)).mean())
    fb = float(sd.mean())
    ratio = np.array([np.array(r['sigmas']) / np.array(r['a3']) for r in recs.values()])
    lam = np.array([r['lams'] for r in recs.values()])
    return {'class': cls, 'frac_abs_below_0.02': zero, 'frac_abs_at_bound': sat, 'mean_delta': m.tolist(),
            'ci_half': h.tolist(), 'frac_rounds_ci_excl_0': excl, 'consistent': bool(excl >= 1 / 3),
            'between_seed_sd': fb, 'uses_feedback': bool(fb >= 0.02),
            'sigma_over_A3_mean': ratio.mean(0).tolist(), 'sigma_over_A3_range': [float(ratio.min()), float(ratio.max())],
            'lambda_range': [float(lam.min()), float(lam.max())], 'mean_abs_delta': float(np.abs(D).mean())}


def main():
    S, recs = load()
    res = {'counts': {'%s|%s' % k: len(v) for k, v in S.items()}}
    g = json.load(open(os.path.join(OUT, 'gate_G4h.json')))
    res['gate'] = g
    print('gate G4h ok: %s' % g['ok'])
    for op in OPS:
        print('  %s: zero-delta vs A3 %.1e; random budget err %.1e' % (op, g[op]['zero_reproduces_A3_max_rel'], g[op]['random_200_max_rel_err_vs_A3_eps']))
    res['ops'] = {}
    for op in OPS:
        o = {}
        o['means'] = {a: float(np.mean(list(S[(op, a)].values()))) for a in ('A1', 'A2', 'A3', 'A5', 'A5r')}
        o['eps'] = {a: float(np.mean([r['eps_max'] for r in recs[(op, a)].values()])) for a in ('A1', 'A2', 'A3', 'A5', 'A5r')}
        print('\n=== %s: mean test mAP %s' % (op, '  '.join('%s %.4f' % kv for kv in o['means'].items())))
        print('    eps %s' % '  '.join('%s %.4f' % kv for kv in o['eps'].items()))
        F = fam([('A5r - %s' % a, S[(op, 'A5r')], S[(op, a)]) for a in ('A1', 'A2', 'A3', 'A5')],
                'F10-%s (Holm over 4)' % op)
        o['F10'] = F
        gen = F['A5r - A1']['verdict'] == 'WIN' and F['A5r - %s' % BETTER[op]]['verdict'] == 'WIN'
        o['genuine_improvement'] = bool(gen)
        print('  -> genuine improvement at %s (WIN vs A1 and vs %s): %s' % (op, BETTER[op], gen))
        p = pattern(recs[(op, 'A5r')])
        o['pattern'] = p
        print('  pattern: %s | |delta|<0.02 %.0f%%, at bound %.0f%%, mean |delta| %.3f | rounds with CI excl 0: %.0f%% (consistent %s) | between-seed sd %.4f (uses feedback %s)'
              % (p['class'], 100 * p['frac_abs_below_0.02'], 100 * p['frac_abs_at_bound'], p['mean_abs_delta'],
                 100 * p['frac_rounds_ci_excl_0'], p['consistent'], p['between_seed_sd'], p['uses_feedback']))
        print('  mean delta by round: %s' % ' '.join('%+.3f' % x for x in p['mean_delta']))
        print('  realised sigma / A3 by round: %s  (lambda %.3f..%.3f)' % (' '.join('%.2f' % x for x in p['sigma_over_A3_mean']), *p['lambda_range']))
        print('  realised sigma (seed 200): %s' % ' '.join('%.2f' % x for x in recs[(op, 'A5r')][200]['sigmas']))
        print('  A3 sigma:                  %s' % ' '.join('%.2f' % x for x in recs[(op, 'A5r')][200]['a3']))
        L = json.load(open(os.path.join(OUT, 'agent_A5r_%s_log.json' % op)))
        o['training'] = {'G4b': L['G4b'], 'selected_checkpoint': L['selected_checkpoint'], 'hours': L.get('hours'),
                         'evals': [(e['ep'], e['mean_reward']) for e in L['evals']]}
        print('  training: G4b learned %s (%.4f -> %.4f), deployed checkpoint ep %d, %.1f h' % (
            L['G4b']['learned'], L['G4b']['first25'], L['G4b']['last25'], L['selected_checkpoint'], L.get('hours') or float('nan')))
        res['ops'][op] = o
    res['headline_genuine_at_P'] = res['ops']['P']['genuine_improvement']
    print('\nHEADLINE: A5-relative genuine improvement at P: %s' % res['headline_genuine_at_P'])

    P, S1, S2 = (res['ops'][o]['F10'] for o in OPS)
    preds = {
        'P1_not_genuine_at_P_and_ties_A3': bool(not res['ops']['P']['genuine_improvement'] and P['A5r - A3']['verdict'].startswith('TIE')),
        'P2_beats_A5_at_P': P['A5r - A5']['verdict'] == 'WIN',
        'P3a_pattern_not_ZERO_at_P': res['ops']['P']['pattern']['class'] != 'ZERO-COLLAPSE',
        'P3b_no_feedback_use_at_P (sd < 0.02)': not res['ops']['P']['pattern']['uses_feedback'],
        'P4a_S1_does_not_beat_A1': S1['A5r - A1']['verdict'] != 'WIN',
        'P4b_S2_ties_A2': S2['A5r - A2']['verdict'].startswith('TIE'),
    }
    res['predictions'] = preds
    print('\nPredictions:')
    for k, v in preds.items():
        print('  %-42s %s' % (k, v))
    res['meta'] = C.meta()
    C.save_json(res, os.path.join(OUT, 'analysis.json'))

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    for i, op in enumerate(OPS):
        a = ax[i, 0]
        for arm, c in (('A1', 'tab:blue'), ('A2', 'tab:orange'), ('A3', 'tab:green'), ('A5', 'tab:purple'), ('A5r', 'tab:red')):
            sg = np.array([r['sigmas'] for r in recs[(op, arm)].values()]).mean(0)
            a.plot(range(1, len(sg) + 1), sg, 'o-' if arm == 'A5r' else '.-', color=c, label=arm, lw=2.2 if arm == 'A5r' else 1)
        a.set_title('%s: realised sigma schedule (mean over test seeds)' % op); a.set_xlabel('round'); a.legend(fontsize=7)
        p = res['ops'][op]['pattern']; m, h = np.array(p['mean_delta']), np.array(p['ci_half'])
        a = ax[i, 1]
        for r in recs[(op, 'A5r')].values():
            a.plot(range(1, len(m) + 1), r['deltas'], color='gray', alpha=0.25, lw=0.8)
        a.errorbar(range(1, len(m) + 1), m, yerr=h, fmt='o-', color='tab:red', capsize=3)
        a.axhline(0, color='k', lw=0.8); a.axhline(0.2, ls=':', color='gray'); a.axhline(-0.2, ls=':', color='gray')
        a.set_ylim(-0.25, 0.25); a.set_xlabel('round'); a.set_title('%s: adjustment delta_t (%s)' % (op, p['class']))
        a = ax[i, 2]
        L = json.load(open(os.path.join(OUT, 'agent_A5r_%s_log.json' % op)))
        rw = np.array([e['reward'] for e in L['episodes']])
        a.plot(np.convolve(rw, np.ones(25) / 25, 'valid'), color='tab:red', label='A5r train (25-ep mean)')
        a.plot([e['ep'] for e in L['evals']], [e['mean_reward'] for e in L['evals']], 'ko', ms=3, label='A5r eval')
        LA = os.path.join(P4OUT, 'agent_A5_%s_log.json' % op)
        if os.path.exists(LA):
            la = json.load(open(LA)); rw5 = np.array([e['reward'] for e in la['episodes']])
            a.plot(np.convolve(rw5, np.ones(25) / 25, 'valid'), color='tab:purple', alpha=0.7, label='A5 train (Phase 4)')
        a.set_xlabel('episode'); a.set_title('%s: simulator reward' % op); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, 'plots', 'phase4h_relative_agent.png'), dpi=130)


if __name__ == '__main__':
    main()
