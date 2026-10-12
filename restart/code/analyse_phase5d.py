# =============================================================================
# Phase 5d analysis (restart/phase5d/PREREG.md §5-6). Per margin m: family F12-m (Holm over 6):
# (a1) capped N vs uncapped N under trust-capture, (a2) capped N vs FedAvg under trust-capture,
# (b1-b4) capped N vs uncapped N under none / ALIE / IPM / Fang. (+A) F12A-m with the cap-aware attack.
# Comparators (read-only): Phase 5b variant_s2_fltrust_N_*, Phase 5c adaptive_s2_*, Phase 5 main_A1_fedavg_*.
# =============================================================================
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from phase4c import compare, holm, verdict
from phase5d import OUT, SEEDS, MAL, ATTACKS, MARGINS
import phase5 as P5, phase5b as P5B, phase5c as P5C

STD = ATTACKS                       # none, ALIE, IPM, Fang


def load():
    R, recs = {}, {}
    def add(key, r):
        R.setdefault(key, {})[r['seed']] = r['test_map50']; recs.setdefault(key, {})[r['seed']] = r
    for f in glob.glob(os.path.join(OUT, 'grid_*_seed*.json')):
        r = json.load(open(f)); add(('cap' + r['m'], r['attack']), r)
    for t in STD:
        for s in SEEDS:
            add(('N', t), json.load(open(os.path.join(P5B.OUT, 'variant_s2_fltrust_N_%s_seed%d.json' % (t, s)))))
            add(('fedavg', t), json.load(open(os.path.join(P5.OUT, 'main_A1_fedavg_%s_seed%d.json' % (t, s)))))
    for agg, key in (('fltrust_N', 'N'), ('fedavg', 'fedavg')):
        for s in SEEDS:
            add((key, 'adaptive'), json.load(open(os.path.join(P5C.OUT, 'adaptive_s2_%s_adaptive_seed%d.json' % (agg, s)))))
    return R, recs


def fam(pairs, label):
    res = {n: compare(a, b) for n, a, b in pairs}
    adj = holm(np.array([res[n]['p'] for n, _, _ in pairs]))
    for (n, _, _), pa in zip(pairs, adj):
        res[n].update({'p_holm': float(pa), 'verdict': verdict(res[n], pa)})
    print('\n%s' % label)
    for n, c in res.items():
        print('  %-40s diff %+.4f [%+.4f, %+.4f]  W/L %2d/%-2d  p %.4f  p_holm %.4f  %s' % (
            n, c['diff'], c['ci95'][0], c['ci95'][1], c['wins'], c['losses'], c['p'], c['p_holm'], c['verdict']))
    return res


def rstats(recs):
    rr = [x for r in recs.values() for x in r['rounds']]
    g = lambda k: [x[k] for x in rr if k in x]
    return {'cap_bound': float(np.mean(g('cap_bound'))), 'w_sum_mean': float(np.mean(g('w_sum'))), 'w_sum_min': float(np.min(g('w_sum'))),
            'n_pos_mean': float(np.mean(g('n_pos'))), 'mal_share': float(np.mean(g('mal_share'))),
            'mal_share_uncapped': float(np.mean(g('mal_share_uncapped'))),
            'mal_share_by_round': [float(np.mean([r['rounds'][i].get('mal_share', np.nan) for r in recs.values()])) for i in range(12)],
            'adv_c_mean': float(np.mean(g('adv_c'))) if g('adv_c') else None}


def main():
    R, recs = load()
    res = {'counts': {'%s|%s' % k: len(v) for k, v in R.items()}}
    print('reproduction: %s' % json.load(open(os.path.join(OUT, 'repro_ok.json')))['repro'])
    mean = lambda k: float(np.mean(list(R[k].values())))
    zeros = lambda k: int(sum(v < 0.0005 for v in R[k].values()))
    conds = STD + ('adaptive', 'adaptive_cap')
    rows = ['fedavg', 'N'] + ['cap' + m for m in ('2', '1.5', '1')]
    res['means'] = {}
    print('\nMean test mAP@0.5 (20 seeds), eps 1.86; [seeds at 0.000]')
    print('  %-8s ' % '' + ' '.join('%14s' % c for c in conds))
    for row in rows:
        cells = []
        for c in conds:
            if (row, c) in R:
                res['means']['%s|%s' % (row, c)] = mean((row, c))
                cells.append('%8.4f [%2d]' % (mean((row, c)), zeros((row, c))))
            else:
                cells.append('%14s' % '-')
        print('  %-8s ' % row + ' '.join('%14s' % x for x in cells))
    eps = [r['eps_max'] for k, v in recs.items() if k[0].startswith('cap') for r in v.values()]
    res['eps_range'] = [float(min(eps)), float(max(eps))]
    print('  eps (capped runs): %.4f..%.4f' % tuple(res['eps_range']))

    res['F12'], res['F12A'], res['stats'] = {}, {}, {}
    for m in MARGINS:
        k = 'cap' + m
        for c in conds:
            if (k, c) in recs:
                res['stats']['%s|%s' % (k, c)] = rstats(recs[(k, c)])
        def family(att, tag):
            pairs = [('(a1) cap%s - N | %s' % (m, att), R[(k, att)], R[('N', 'adaptive')]),
                     ('(a2) cap%s - FedAvg | %s' % (m, att), R[(k, att)], R[('fedavg', 'adaptive')])]
            pairs += [('(b) cap%s - N | %s' % (m, t), R[(k, t)], R[('N', t)]) for t in STD]
            F = fam(pairs, '%s, m = %s (kappa %.4f), Holm over 6' % (tag, m, float(m) / 8))
            names = list(F)
            a1, a2, bs = F[names[0]]['verdict'], F[names[1]]['verdict'], [F[n]['verdict'] for n in names[2:]]
            ok = a1 == 'WIN' and a2 != 'LOSS' and 'LOSS' not in bs
            print('  -> genuine fix at m = %s (%s): %s   [a1 %s; a2 %s; b %s]' % (m, att, ok, a1, a2, ', '.join(bs)))
            return {'tests': F, 'genuine_fix': bool(ok), 'all_b_equivalent': all(b == 'TIE (EQUIVALENT)' for b in bs)}
        res['F12'][m] = family('adaptive', 'F12')
        if (k, 'adaptive_cap') in R:
            res['F12A'][m] = family('adaptive_cap', 'F12A (+A cap-aware attacker)')

    print('\nRound statistics (mean over 20 seeds x 12 rounds)')
    for key, st in res['stats'].items():
        print('  %-20s cap binds %3.0f%%  sum w %.2f (min %.2f)  n_pos %.1f  attackers share %.3f (uncapped %.3f)%s' % (
            key, 100 * st['cap_bound'], st['w_sum_mean'], st['w_sum_min'], st['n_pos_mean'], st['mal_share'], st['mal_share_uncapped'],
            '  c %.2f' % st['adv_c_mean'] if st['adv_c_mean'] is not None else ''))

    print('\nSecondary (unadjusted)')
    sec = {}
    nref = np.mean([mean(('N', t)) for t in STD])
    for m in MARGINS:
        k = 'cap' + m
        for att in ('adaptive', 'adaptive_cap'):
            if (k, att) in R:
                sec['frac_of_N_%s_%s' % (m, att)] = mean((k, att)) / nref
                c = compare(R[(k, att)], R[(k, 'ALIE')]); sec['cap%s %s - cap%s ALIE' % (m, att, m)] = c
                print('  cap%s under %-12s = %.4f = %.2f of N\'s mean over none/ALIE/IPM/Fang (%.4f); vs own ALIE %+.4f (%d/%d, p %.4f)'
                      % (m, att, mean((k, att)), mean((k, att)) / nref, nref, c['diff'], c['wins'], c['losses'], c['p']))
        for t in STD:
            c = compare(R[(k, t)], R[('fedavg', t)]); sec['cap%s - FedAvg | %s' % (m, t)] = c
            print('  cap%s - FedAvg | %-5s %+.4f [%+.4f, %+.4f] %2d/%-2d p %.4f' % (m, t, c['diff'], c['ci95'][0], c['ci95'][1], c['wins'], c['losses'], c['p']))
        if (k, 'adaptive_cap') in R:
            c = compare(R[(k, 'adaptive_cap')], R[(k, 'adaptive')]); sec['cap%s capaware - 5c attack' % m] = c
            print('  cap%s cap-aware - Phase 5c attack: %+.4f [%+.4f, %+.4f] %d/%d p %.4f' % (m, c['diff'], c['ci95'][0], c['ci95'][1], c['wins'], c['losses'], c['p']))
    res['secondary'] = sec

    S = res['stats']; F = res['F12']; FA = res['F12A']
    t = lambda m, n: F[m]['tests'][n]['verdict']
    preds = {
        'P1_binds_ge90pct_none_m1.5': S['cap1.5|none']['cap_bound'] >= 0.9,
        'P1b_binds_50-70pct_none_m2': 0.5 <= S['cap2|none']['cap_bound'] <= 0.7,
        'P2_a1_WIN_both_m (mean>0.01, <5 zeros)': all(t(m, '(a1) cap%s - N | adaptive' % m) == 'WIN' and mean(('cap' + m, 'adaptive')) > 0.01
                                                    and zeros(('cap' + m, 'adaptive')) < 5 for m in ('1.5', '2')),
        'P3_attacker_share_near_2kappa_above_0.171': all(S['cap%s|adaptive' % m]['mal_share'] > 0.171 for m in ('1.5', '2')),
        'P4_m2_not_genuine_a2_LOSS': (not F['2']['genuine_fix']) and t('2', '(a2) cap2 - FedAvg | adaptive') == 'LOSS',
        'P5_m1.5_genuine': F['1.5']['genuine_fix'],
        'P6_below_half_of_N_none (<0.036)': all(mean(('cap' + m, 'adaptive')) < 0.5 * mean(('N', 'none')) for m in ('1.5', '2')),
        'P7_capaware_within_0.01': all(abs(sec['cap%s capaware - 5c attack' % m]['diff']) <= 0.01 for m in ('1.5', '2')),
    }
    res['predictions'] = {k: bool(v) for k, v in preds.items()}
    print('\nPredictions:')
    for k, v in preds.items():
        print('  %-46s %s' % (k, bool(v)))
    res['meta'] = C.meta()
    C.save_json(res, os.path.join(OUT, 'analysis.json'))

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    lab = {'fedavg': 'FedAvg', 'N': 'Variant N (no cap)', 'cap2': 'N + cap 2/K', 'cap1.5': 'N + cap 1.5/K', 'cap1': 'N + cap 1/K'}
    col = {'fedavg': 'tab:gray', 'N': 'tab:blue', 'cap2': 'tab:orange', 'cap1.5': 'tab:red', 'cap1': 'tab:purple'}
    xs = np.arange(len(conds)); wd = 0.16
    for i, row in enumerate(rows):
        ys, lo, hi = [], [], []
        for c in conds:
            if (row, c) in R:
                v = np.array(list(R[(row, c)].values())); h = 2.093 * v.std(ddof=1) / np.sqrt(len(v))
                ys.append(v.mean()); lo.append(h); hi.append(h)
            else:
                ys.append(np.nan); lo.append(0); hi.append(0)
        ax[0].bar(xs + (i - 2) * wd, ys, wd, yerr=[lo, hi], color=col[row], label=lab[row], capsize=2)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(['no attack', 'ALIE', 'IPM', 'Fang', 'trust-capture\n(Phase 5c)', 'trust-capture\ncap-aware (+A)'])
    ax[0].set_ylabel('test mAP@0.5 (20 seeds, 95% CI)'); ax[0].set_title('Trust-weight cap on Variant N, eps 1.86'); ax[0].legend(fontsize=8)
    for key, c, ls in (('cap2|adaptive', 'tab:orange', '-'), ('cap1.5|adaptive', 'tab:red', '-'), ('cap1|adaptive', 'tab:purple', '-'),
                       ('cap2|adaptive_cap', 'tab:orange', '--'), ('cap1.5|adaptive_cap', 'tab:red', '--')):
        if key in S:
            ax[1].plot(range(1, 13), S[key]['mal_share_by_round'], ls, color=c, marker='o', ms=3, label=key.replace('|', ', '))
    un = [float(np.mean([r['rounds'][i]['trust']['c3'] + r['rounds'][i]['trust']['c6'] for r in recs[('N', 'adaptive')].values()]) /
                np.mean([sum(r['rounds'][i]['trust'].values()) for r in recs[('N', 'adaptive')].values()])) for i in range(12)]
    ax[1].plot(range(1, 13), un, 'k-', marker='s', ms=3, label='no cap (Phase 5c)')
    ax[1].axhline(0.171, ls=':', color='gray', label='attackers\' FedAvg share 0.171')
    ax[1].set_xlabel('round'); ax[1].set_ylabel('attackers\' share of aggregation weight'); ax[1].set_ylim(0, 1)
    ax[1].set_title('Attackers\' weight share under trust-capture'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, 'plots', 'phase5d_trust_cap.png'), dpi=140)


if __name__ == '__main__':
    main()
