# =============================================================================
# Phase 5 analysis (restart/phase5/PREREG.md §7-8): decision families F1-F4 (Holm within each),
# secondary tables, prediction scoring, figure. Reads results/phase5/ (and Phase 4 read-only).
# =============================================================================
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from phase4c import compare, holm, verdict
from phase5 import OUT, P4OUT, SEEDS, MAL, SCHED, ATTACKS

ATT = ATTACKS[1:]
ENV = {k: v['env'] for k, v in json.load(open(os.path.join(C.ROOT, 'restart', 'data', 'clients.json')))['clients'].items()}


def load():
    R, recs = {}, {}
    for f in glob.glob(os.path.join(OUT, '*_seed*.json')):
        r = json.load(open(f))
        key = (r['block'], r['schedule'], r['agg'], r['attack'])
        R.setdefault(key, {})[r['seed']] = r['test_map50']
        recs.setdefault(key, {})[r['seed']] = r
    return R, recs


def damage(R, key_none, key_att):
    return {s: R[key_none][s] - R[key_att][s] for s in SEEDS if s in R[key_none] and s in R[key_att]}


def fam(pairs, label):
    """pairs: [(name, dict_a, dict_b)] -> compare a - b, Holm within the family."""
    res = {n: compare(a, b) for n, a, b in pairs}
    adj = holm(np.array([res[n]['p'] for n, _, _ in pairs]))
    for (n, _, _), pa in zip(pairs, adj):
        res[n].update({'p_holm': float(pa), 'verdict': verdict(res[n], pa)})
    print('\n%s' % label)
    for n, c in res.items():
        print('  %-34s diff %+.4f [%+.4f, %+.4f]  W/L %2d/%-2d  p %.4f  p_holm %.4f  %s' % (
            n, c['diff'], c['ci95'][0], c['ci95'][1], c['wins'], c['losses'], c['p'], c['p_holm'], c['verdict']))
    return res


def sec(n, a, b):
    c = compare(a, b)
    print('  %-34s diff %+.4f [%+.4f, %+.4f]  W/L %2d/%-2d  p %.4f (unadjusted)' % (n, c['diff'], c['ci95'][0], c['ci95'][1], c['wins'], c['losses'], c['p']))
    return c


def trust_stats(recs, key):
    """Mean FLTrust trust per round: honest, malicious, honest per environment (over seeds)."""
    if key not in recs:
        return None
    rs = list(recs[key].values())
    Rn = len(rs[0]['rounds'])
    h = np.array([[np.mean([x['trust'][c] for c in x['trust'] if c not in MAL]) for x in r['rounds']] for r in rs])
    m = np.array([[np.mean([x['trust'][c] for c in MAL]) for x in r['rounds']] for r in rs])
    env = {}
    for e in sorted(set(ENV.values())):
        cs = [c for c in ENV if ENV[c] == e and c not in MAL]
        env[e] = float(np.mean([[x['trust'][c] for c in cs for x in r['rounds']] for r in rs]))
    zero = float(np.mean([[sum(x['trust'].values()) == 0 for x in r['rounds']] for r in rs]))
    return {'honest_by_round': h.mean(0).tolist(), 'malicious_by_round': m.mean(0).tolist(),
            'honest_mean': float(h.mean()), 'malicious_mean': float(m.mean()), 'honest_by_env': env,
            'rounds_all_zero_trust': zero, 'n_rounds': Rn}


def main():
    R, recs = load()
    res = {'counts': {'%s|%s|%s|%s' % k: len(v) for k, v in R.items()}}
    mk = lambda s, a, t: ('main', s, a, t)

    # ── reproduction: FedAvg / no attack vs Phase 4 stored test runs ───────────
    rep = {}
    for s in SCHED:
        same = [R[mk(s, 'fedavg', 'none')][sd] == json.load(open(os.path.join(P4OUT, 'eval_P_%s_seed%d.json' % (s, sd))))['test_map50']
                for sd in SEEDS if sd in R.get(mk(s, 'fedavg', 'none'), {})]
        rep[s] = {'identical': int(sum(same)), 'n': len(same)}
    res['reproduction_vs_phase4'] = rep
    print('Reproduction (FedAvg, no attack) vs Phase 4 stored test runs: %s' % rep)

    # ── mAP table ────────────────────────────────────────────────────────────
    print('\nMean test mAP@0.5 (20 seeds; eps 1.86 except no-DP)')
    tab = {}
    rows = [(mk(s, a, '%s'), '%s %s' % (s, a)) for a in ('fedavg', 'fltrust') for s in SCHED] + \
           [(('ref', 'A1', a, '%s'), 'A1 %s' % a) for a in ('median', 'krum')] + \
           [(('nodp', 'noDP', a, '%s'), 'no-DP %s' % a) for a in ('fedavg', 'fltrust')]
    print('  %-16s' % '' + ''.join('%10s' % t for t in ATTACKS))
    for (b, s, a, _), lab in rows:
        v = [float(np.mean(list(R[(b, s, a, t)].values()))) if (b, s, a, t) in R else float('nan') for t in ATTACKS]
        tab[lab] = dict(zip(ATTACKS, v))
        print('  %-16s' % lab + ''.join('%10.4f' % x for x in v))
    res['mean_map'] = tab
    eps = {('%s' % s): float(np.mean([r['eps_max'] for r in recs[mk(s, 'fedavg', 'none')].values()])) for s in SCHED}
    res['eps'] = eps
    print('  eps (worst client): %s' % {k: round(v, 4) for k, v in eps.items()})

    # ── F1: defence ──────────────────────────────────────────────────────────
    res['F1'] = fam([('FLTrust-FedAvg | %s' % t, R[mk('A1', 'fltrust', t)], R[mk('A1', 'fedavg', t)]) for t in ATT],
                    'F1 (defence, A1): FLTrust vs FedAvg under attack')
    # ── F2: clean cost ───────────────────────────────────────────────────────
    res['F2'] = fam([('FLTrust-FedAvg | none', R[mk('A1', 'fltrust', 'none')], R[mk('A1', 'fedavg', 'none')]),
                     ('median-FedAvg | none', R[('ref', 'A1', 'median', 'none')], R[mk('A1', 'fedavg', 'none')]),
                     ('Krum-FedAvg | none', R[('ref', 'A1', 'krum', 'none')], R[mk('A1', 'fedavg', 'none')])],
                    'F2 (clean cost, A1, no attack): each defence vs FedAvg')
    print('  secondary:')
    f2s = {'FLTrust-median | none': sec('FLTrust-median | none', R[mk('A1', 'fltrust', 'none')], R[('ref', 'A1', 'median', 'none')]),
           'FLTrust-Krum | none': sec('FLTrust-Krum | none', R[mk('A1', 'fltrust', 'none')], R[('ref', 'A1', 'krum', 'none')])}
    res['F2_secondary'] = f2s
    keeps = all(c['ci95'][0] > 0 for c in f2s.values())
    wins = sum(res['F1'][k]['verdict'] == 'WIN' for k in res['F1'])
    effective = wins >= 2 and res['F2']['FLTrust-FedAvg | none']['verdict'] != 'LOSS'
    res['headline'] = {'FLTrust_wins_under_attacks': wins, 'FLTrust_clean_verdict': res['F2']['FLTrust-FedAvg | none']['verdict'],
                       'FLTrust_effective_defence_at_P': effective, 'FLTrust_keeps_more_averaging': keeps}
    print('\nHEADLINE: FLTrust WINS under %d/3 attacks; clean vs FedAvg: %s -> effective defence at P: %s; '
          'keeps more averaging than median AND Krum: %s' % (wins, res['F2']['FLTrust-FedAvg | none']['verdict'], effective, keeps))

    # ── F3: allocation as a security parameter (FLTrust, damage) ─────────────
    dmg = {(s, a, t): damage(R, mk(s, a, 'none'), mk(s, a, t)) for s in SCHED for a in ('fedavg', 'fltrust') for t in ATT}
    res['F3'] = fam([('damage %s - damage A1 | %s' % (s, t), dmg[(s, 'fltrust', t)], dmg[('A1', 'fltrust', t)])
                     for s in ('A3', 'A5') for t in ATT],
                    'F3 (FLTrust): does the schedule change attack damage? (positive = MORE damage than under uniform)')
    res['F3_changes_robustness'] = any(c['p_holm'] < 0.05 for c in res['F3'].values())
    # ── F4: learned vs theory under attack ───────────────────────────────────
    res['F4'] = fam([('A5-A3 | %s' % t, R[mk('A5', 'fltrust', t)], R[mk('A3', 'fltrust', t)]) for t in ATT],
                    'F4 (FLTrust): learned A5 vs theory A3 under attack')

    # ── secondary ────────────────────────────────────────────────────────────
    print('\nSecondary: mean damage (no attack - attack), 20 seeds')
    dt = {}
    for a in ('fedavg', 'fltrust'):
        for s in SCHED:
            dt['%s %s' % (s, a)] = {t: float(np.mean(list(dmg[(s, a, t)].values()))) for t in ATT}
    for a in ('median', 'krum'):
        dt['A1 %s' % a] = {t: float(np.mean(list(damage(R, ('ref', 'A1', a, 'none'), ('ref', 'A1', a, t)).values()))) for t in ATT}
    for a in ('fedavg', 'fltrust'):
        dt['no-DP %s' % a] = {t: float(np.mean(list(damage(R, ('nodp', 'noDP', a, 'none'), ('nodp', 'noDP', a, t)).values()))) for t in ATT}
    for k, v in dt.items():
        print('  %-16s' % k + ''.join('  %s %+.4f' % (t, x) for t, x in v.items()))
    res['mean_damage'] = dt
    print('\nSecondary: attack damage significance, FedAvg A1 and no-DP (unadjusted)')
    res['damage_tests'] = {}
    for lab, kn in (('A1 fedavg', mk('A1', 'fedavg', '%s')), ('A1 fltrust', mk('A1', 'fltrust', '%s')),
                    ('no-DP fedavg', ('nodp', 'noDP', 'fedavg', '%s')), ('no-DP fltrust', ('nodp', 'noDP', 'fltrust', '%s'))):
        for t in ATT:
            a = R[tuple(list(kn[:3]) + ['none'])]; b = R[tuple(list(kn[:3]) + [t])]
            res['damage_tests']['%s | %s' % (lab, t)] = sec('%s: none - %s' % (lab, t), a, b)
    print('\nSecondary: A5 realised schedule (mean sigma per round), FLTrust / FedAvg')
    res['A5_schedules'] = {}
    for a in ('fedavg', 'fltrust'):
        for t in ATTACKS:
            sg = np.array([r['sigmas'] for r in recs[mk('A5', a, t)].values()]).mean(0)
            res['A5_schedules']['%s %s' % (a, t)] = sg.tolist()
            print('  %-7s %-5s %s' % (a, t, ' '.join('%.2f' % x for x in sg)))
    print('\nSecondary: Krum picks a malicious update (share of rounds)')
    res['krum_malicious_pick'] = {}
    for t in ATTACKS:
        picks = [x['krum_pick'] for r in recs[('ref', 'A1', 'krum', t)].values() for x in r['rounds']]
        res['krum_malicious_pick'][t] = float(np.mean([p in MAL for p in picks]))
        print('  %-5s %.2f' % (t, res['krum_malicious_pick'][t]))

    # ── trust scores + predictions ───────────────────────────────────────────
    print('\nFLTrust trust scores (mean over rounds and seeds)')
    ts = {}
    for lab, key in [('A1 %s' % t, mk('A1', 'fltrust', t)) for t in ATTACKS] + [('no-DP %s' % t, ('nodp', 'noDP', 'fltrust', t)) for t in ATTACKS]:
        ts[lab] = trust_stats(recs, key)
        print('  %-12s honest %.4f  malicious %.4f  all-zero rounds %.2f  honest by env %s' % (
            lab, ts[lab]['honest_mean'], ts[lab]['malicious_mean'], ts[lab]['rounds_all_zero_trust'],
            {k: round(v, 4) for k, v in ts[lab]['honest_by_env'].items()}))
    res['trust'] = ts
    pr = {
        'P1_honest_trust_below_0.2_under_DP_and_higher_without': ts['A1 none']['honest_mean'] < 0.2 and ts['no-DP none']['honest_mean'] > ts['A1 none']['honest_mean'],
        'P2_ALIE_trust_above_honest_under_DP': ts['A1 ALIE']['malicious_mean'] > ts['A1 ALIE']['honest_mean'],
        'P3_IPM_trust_near_0_and_neutralised': ts['A1 IPM']['malicious_mean'] < 0.01 and res['damage_tests']['A1 fltrust | IPM']['ci95'][0] <= 0,
        'P4_median_and_Krum_LOSE_clean': res['F2']['median-FedAvg | none']['verdict'] == 'LOSS' and res['F2']['Krum-FedAvg | none']['verdict'] == 'LOSS',
        'P5_damage_at_P_smaller_than_without_DP (FedAvg, mean over attacks)': float(np.mean(list(dt['A1 fedavg'].values()))) < float(np.mean(list(dt['no-DP fedavg'].values()))),
    }
    res['predictions'] = pr
    print('\nPredictions (PREREG §8):'); [print('  %-70s %s' % (k, v)) for k, v in pr.items()]
    res['meta'] = C.meta()
    C.save_json(res, os.path.join(OUT, 'analysis.json'))
    plot(R, ts, dt)


def plot(R, ts, dt):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    groups = [('main', 'A1', 'fedavg', 'FedAvg'), ('main', 'A1', 'fltrust', 'FLTrust'), ('ref', 'A1', 'median', 'Median'),
              ('ref', 'A1', 'krum', 'Krum')]
    w = 0.2
    for i, t in enumerate(ATTACKS):
        ax[0].bar(np.arange(len(groups)) + (i - 1.5) * w, [np.mean(list(R[(b, s, a, t)].values())) for b, s, a, _ in groups], w, label=t)
    ax[0].set_xticks(range(len(groups))); ax[0].set_xticklabels([g[3] for g in groups])
    ax[0].set_ylabel('test mAP@0.5'); ax[0].set_title('Uniform schedule, eps 1.86, 2/8 attackers'); ax[0].legend(fontsize=8)
    for t, col in zip(ATTACKS, ('#7f7f7f', '#1f77b4', '#2ca02c', '#d62728')):
        ax[1].plot(range(1, 13), ts['A1 %s' % t]['honest_by_round'], color=col, ls='-', label='%s honest' % t)
        if t != 'none':
            ax[1].plot(range(1, 13), ts['A1 %s' % t]['malicious_by_round'], color=col, ls='--', label='%s attacker' % t)
    ax[1].set_xlabel('round'); ax[1].set_ylabel('FLTrust trust score'); ax[1].set_title('FLTrust trust under DP (A1)')
    ax[1].legend(fontsize=7, ncol=2); ax[1].grid(alpha=0.3)
    labs = ['A1 fltrust', 'A2 fltrust', 'A3 fltrust', 'A5 fltrust', 'A1 fedavg', 'no-DP fedavg', 'no-DP fltrust']
    for i, t in enumerate(ATT):
        ax[2].bar(np.arange(len(labs)) + (i - 1) * 0.27, [dt[l][t] for l in labs], 0.27, label=t)
    ax[2].axhline(0, color='k', lw=0.6); ax[2].set_xticks(range(len(labs))); ax[2].set_xticklabels(labs, rotation=30, ha='right', fontsize=8)
    ax[2].set_ylabel('damage = mAP(no attack) - mAP(attack)'); ax[2].set_title('Attack damage by schedule / aggregator'); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, 'plots', 'phase5_security.png'), dpi=150)


if __name__ == '__main__':
    main()
