# =============================================================================
# Phase 5c analysis (restart/phase5c/PREREG.md §3-5). Part 1: F8 per level (Variant N vs FedAvg, Holm
# over 4). Part 2: F9 (3 tests, Holm). Phase 5 / 5b runs are read-only references.
# =============================================================================
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from phase4c import compare, holm, verdict
from phase5c import OUT, SEEDS, MAL, ATTACKS
import phase5 as P5, phase5b as P5B

LV = ('0', '0.5', '1', '2', '4')


def load():
    R, recs = {}, {}
    def add(key, r):
        R.setdefault(key, {})[r['seed']] = r['test_map50']; recs.setdefault(key, {})[r['seed']] = r
    for f in glob.glob(os.path.join(OUT, '*_seed*.json')):
        r = json.load(open(f))
        if r['block'] in ('levels', 'adaptive'):
            add((r['level'], r['agg'], r['attack']), r)
    for f in glob.glob(os.path.join(P5B.OUT, '*_seed*.json')):
        r = json.load(open(f))
        if r['block'] == 'sweep' and r['agg'] in ('fedavg', 'fltrust') or r['block'] == 'variant' and r['agg'] == 'fltrust_N':
            add((r['level'], r['agg'], r['attack']), r)
    for f in glob.glob(os.path.join(P5.OUT, '*_seed*.json')):
        r = json.load(open(f))
        if r['block'] == 'main' and r['schedule'] == 'A1':
            add(('2', r['agg'], r['attack']), r)
        elif r['block'] == 'nodp':
            add(('0', r['agg'], r['attack']), r)
    return R, recs


def fam(pairs, label):
    res = {n: compare(a, b) for n, a, b in pairs}
    adj = holm(np.array([res[n]['p'] for n, _, _ in pairs]))
    for (n, _, _), pa in zip(pairs, adj):
        res[n].update({'p_holm': float(pa), 'verdict': verdict(res[n], pa)})
    print('\n%s' % label)
    for n, c in res.items():
        print('  %-34s diff %+.4f [%+.4f, %+.4f]  W/L %2d/%-2d  p %.4f  p_holm %.4f  %s' % (
            n, c['diff'], c['ci95'][0], c['ci95'][1], c['wins'], c['losses'], c['p'], c['p_holm'], c['verdict']))
    return res


def share(recs, key):
    return float(np.mean([sum(x['trust'][c] for c in MAL) / max(sum(x['trust'].values()), 1e-30)
                          for r in recs[key].values() for x in r['rounds']]))


def main():
    R, recs = load()
    res = {'counts': {'%s|%s|%s' % k: len(v) for k, v in R.items()}}
    m = lambda l, a, t: R[(l, a, t)]
    print('reproduction: %s' % json.load(open(os.path.join(OUT, 'repro_ok.json')))['repro'])
    eps = {l: float(np.mean([r['eps_max'] for r in recs[(l, 'fltrust_N', 'none')].values()])) for l in LV}
    res['eps'] = eps; print('eps: %s' % eps)

    # ── Part 1 ──────────────────────────────────────────────────────────────
    print('\nMean test mAP (20 seeds)')
    tab = {}
    for l in LV:
        for a in ('fedavg', 'fltrust', 'fltrust_N'):
            v = [float(np.mean(list(m(l, a, t).values()))) for t in ATTACKS]
            tab['sigma %s %s' % (l, a)] = dict(zip(ATTACKS, v))
            print('  %-20s' % ('sigma %s %s' % (l, a)) + ''.join('%9.4f' % x for x in v))
    res['mean_map'] = tab
    same0 = {t: int(sum(abs(m('0', 'fltrust_N', t)[s] - m('0', 'fltrust', t)[s]) < 1e-4 for s in SEEDS)) for t in ATTACKS}
    exact0 = {t: int(sum(m('0', 'fltrust_N', t)[s] == m('0', 'fltrust', t)[s] for s in SEEDS)) for t in ATTACKS}
    res['sigma0_vs_phase5_fltrust'] = {'within_1e-4': same0, 'identical': exact0}
    print('\nsigma 0: Variant N vs Phase 5 no-DP FLTrust, seeds within 1e-4: %s; identical: %s' % (same0, exact0))

    res['F8'] = {}
    for l in LV:
        F = fam([('N-FedAvg | %s' % t, m(l, 'fltrust_N', t), m(l, 'fedavg', t)) for t in ATTACKS],
                'F8 sigma %s (eps %.2f): Variant N vs FedAvg%s' % (l, eps[l], ' [= Phase 5b F6-N]' if l == '2' else ''))
        rec_ = all(c['verdict'] != 'LOSS' for c in F.values())
        print('  -> recovers at sigma %s: %s' % (l, rec_))
        sec = {t: compare(m(l, 'fltrust_N', t), m(l, 'fltrust', t)) for t in ATTACKS}
        print('  secondary N - FLTrust: ' + '  '.join('%s %+.4f %d/%d' % (t, c['diff'], c['wins'], c['losses']) for t, c in sec.items()))
        sh = {t: share(recs, (l, 'fltrust_N', t)) for t in ATTACKS[1:]}
        print('  attacker weight share under N: %s' % {t: round(v, 3) for t, v in sh.items()})
        res['F8'][l] = {'tests': F, 'recovers': rec_, 'N_minus_FLTrust': sec, 'attacker_share': sh}
    res['recovers_every_level'] = all(res['F8'][l]['recovers'] for l in LV)
    print('\nHEADLINE Part 1: Variant N recovers at every level: %s; levels failing: %s' % (
        res['recovers_every_level'], [l for l in LV if not res['F8'][l]['recovers']]))

    # ── Part 2 ──────────────────────────────────────────────────────────────
    print('\nPart 2: adaptive trust-capture attack, sigma 2 (eps %.4f)' % eps['2'])
    for a in ('fltrust_N', 'fedavg', 'fltrust'):
        print('  %-10s none %.4f  ALIE %.4f  adaptive %.4f' % (a, np.mean(list(m('2', a, 'none').values())),
              np.mean(list(m('2', a, 'ALIE').values())), np.mean(list(m('2', a, 'adaptive').values()))))
    res['adaptive_means'] = {a: float(np.mean(list(m('2', a, 'adaptive').values()))) for a in ('fltrust_N', 'fedavg', 'fltrust')}
    F9 = fam([('damage N (none - adaptive)', m('2', 'fltrust_N', 'none'), m('2', 'fltrust_N', 'adaptive')),
              ('N - FedAvg | adaptive', m('2', 'fltrust_N', 'adaptive'), m('2', 'fedavg', 'adaptive')),
              ('N adaptive - N ALIE', m('2', 'fltrust_N', 'adaptive'), m('2', 'fltrust_N', 'ALIE'))],
             'F9 (Holm over 3)')
    res['F9'] = F9
    v2 = F9['N - FedAvg | adaptive']['verdict']
    res['part2_verdict'] = 'BROKEN' if v2 == 'LOSS' else 'SURVIVES'
    print('  -> Variant N under the adaptive attack: %s' % res['part2_verdict'])
    print('\nSecondary (unadjusted): damage to FedAvg and FLTrust')
    res['part2_secondary'] = {}
    for a in ('fedavg', 'fltrust'):
        c = compare(m('2', a, 'none'), m('2', a, 'adaptive')); res['part2_secondary']['damage ' + a] = c
        print('  damage %-8s %+.4f [%+.4f, %+.4f] %d/%d p %.4f' % (a, c['diff'], *c['ci95'], c['wins'], c['losses'], c['p']))
    for a in ('fltrust',):
        c = compare(m('2', 'fltrust_N', 'adaptive'), m('2', a, 'adaptive')); res['part2_secondary']['N - FLTrust | adaptive'] = c
        print('  N - FLTrust | adaptive %+.4f [%+.4f, %+.4f] %d/%d p %.4f' % (c['diff'], *c['ci95'], c['wins'], c['losses'], c['p']))
    rs = list(recs[('2', 'fltrust_N', 'adaptive')].values())
    cs = np.array([[x['adv_c'] for x in r['rounds']] for r in rs])
    shr = np.array([[sum(x['trust'][c] for c in MAL) / max(sum(x['trust'].values()), 1e-30) for x in r['rounds']] for r in rs])
    res['adaptive_c_by_round'] = cs.mean(0).tolist(); res['adaptive_share_by_round'] = shr.mean(0).tolist()
    res['adaptive_share_ge_0.85'] = float((shr >= 0.85).mean())
    print('  chosen c by round: %s' % ' '.join('%.2f' % x for x in cs.mean(0)))
    print('  attacker weight share by round: %s; share of rounds >= 0.85: %.2f' % (' '.join('%.2f' % x for x in shr.mean(0)), (shr >= 0.85).mean()))

    pr = {
        'P1_recovers_at_0_0.5_1_and_ties_all_at_4': all(res['F8'][l]['recovers'] for l in ('0', '0.5', '1'))
            and all(c['verdict'].startswith('TIE') for c in res['F8']['4']['tests'].values()),
        'P2_sigma0_matches_FLTrust_>=18_of_20_each': all(v >= 18 for v in same0.values()),
        'P3_capture_>=0.85_in_most_rounds': res['adaptive_share_ge_0.85'] > 0.5,
        'P4_N_broken_mAP_below_0.02_and_LOSS': res['adaptive_means']['fltrust_N'] < 0.02 and v2 == 'LOSS',
        'P4b_FedAvg_hurt_much_less': res['part2_secondary']['damage fedavg']['diff'] < 0.5 * F9['damage N (none - adaptive)']['diff'],
        'P5_FLTrust_hurt_little (damage < 0.005)': res['part2_secondary']['damage fltrust']['diff'] < 0.005,
    }
    res['predictions'] = pr
    print('\nPredictions:'); [print('  %-48s %s' % (k, v)) for k, v in pr.items()]
    res['meta'] = C.meta()
    C.save_json(res, os.path.join(OUT, 'analysis.json'))
    plot(tab, res)


def plot(tab, res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    x = np.arange(len(LV))
    for t, col in zip(ATTACKS, ('#7f7f7f', '#1f77b4', '#2ca02c', '#d62728')):
        ax[0].plot(x, [tab['sigma %s fltrust_N' % l][t] / tab['sigma %s fedavg' % l][t] for l in LV], 'o-', color=col, label='Variant N, %s' % t)
        ax[0].plot(x, [tab['sigma %s fltrust' % l][t] / tab['sigma %s fedavg' % l][t] for l in LV], 'x:', color=col, alpha=0.6)
    ax[0].axhline(1, color='k', lw=0.6); ax[0].set_xticks(x); ax[0].set_xticklabels(['σ %s' % l for l in LV])
    ax[0].set_ylabel('mean mAP / FedAvg (dotted: plain FLTrust)'); ax[0].set_title('Variant N across privacy levels'); ax[0].legend(fontsize=7)
    arms = [('fedavg', 'FedAvg'), ('fltrust', 'FLTrust'), ('fltrust_N', 'Variant N')]
    for i, t in enumerate(('none', 'ALIE', 'adaptive')):
        vals = [tab['sigma 2 %s' % a][t] if t != 'adaptive' else res['adaptive_means'][a] for a, _ in arms]
        ax[1].bar(np.arange(3) + (i - 1) * 0.27, vals, 0.27, label=t)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([n for _, n in arms]); ax[1].set_ylabel('test mAP@0.5')
    ax[1].set_title('Adaptive trust-capture attack, eps 1.86'); ax[1].legend(fontsize=8)
    r = np.arange(1, 13)
    ax[2].plot(r, res['adaptive_share_by_round'], 'o-', label='attackers\' weight share (Variant N)')
    ax[2].plot(r, res['adaptive_c_by_round'], 's-', label='chosen cosine c')
    ax[2].axhline(2 / 8, color='k', ls=':', lw=0.8, label='2/8 (equal share)')
    ax[2].set_xlabel('round'); ax[2].set_ylim(0, 1); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    ax[2].set_title('How the attack captures Variant N')
    fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, 'plots', 'phase5c_variantN.png'), dpi=150)


if __name__ == '__main__':
    main()
