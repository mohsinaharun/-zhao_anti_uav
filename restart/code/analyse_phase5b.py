# =============================================================================
# Phase 5b analysis (restart/phase5b/PREREG.md §3-5). Part 1: per-level F1/F2, direction, ratio family
# F5 (paired bootstrap, Holm), mechanism Spearman over sigma 0/0.5/1/2/4. Part 2: F6/F7 for variants
# L and N. Phase 5 results are read-only references (sigma 2 and no-DP).
# =============================================================================
import os, sys, json, glob
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from phase4c import compare, holm, verdict
from phase5b import OUT, P5OUT, LEVELS, AGGS, VARIANTS, SEEDS, MAL, ATTACKS

ATT = ATTACKS[1:]
EPS = {}


def load():
    """R[(level, agg, attack)] = {seed: mAP}; recs likewise. Phase 5 cells mapped to levels '2' and '0'."""
    R, recs = {}, {}
    def add(key, r):
        R.setdefault(key, {})[r['seed']] = r['test_map50']; recs.setdefault(key, {})[r['seed']] = r
    for f in glob.glob(os.path.join(OUT, '*_seed*.json')):
        r = json.load(open(f))
        if r['block'] in ('sweep', 'variant'):
            add((r['level'], r['agg'], r['attack']), r)
    for f in glob.glob(os.path.join(P5OUT, '*_seed*.json')):
        r = json.load(open(f))
        if r['block'] == 'main' and r['schedule'] == 'A1' or r['block'] == 'ref':
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
        print('  %-30s diff %+.4f [%+.4f, %+.4f]  W/L %2d/%-2d  p %.4f  p_holm %.4f  %s' % (
            n, c['diff'], c['ci95'][0], c['ci95'][1], c['wins'], c['losses'], c['p'], c['p_holm'], c['verdict']))
    return res


def mech(recs, key):
    """Mean honest cosine and shrink factor (root norm / mean honest norm) over rounds and seeds."""
    cs, sh = [], []
    for r in recs[key].values():
        for x in r['rounds']:
            hon = [c for c in x['cos'] if c not in MAL]
            cs.append(np.mean([x['cos'][c] for c in hon]))
            sh.append(x['root_norm'] / np.mean([x['norm'][c] for c in hon]))
    return float(np.mean(cs)), float(np.mean(sh))


def ratio_test(R, lv, att, nboot=10000):
    """Ratio of means FLTrust/FedAvg at level lv minus at sigma 2; paired bootstrap over seeds."""
    S = list(SEEDS)
    f = lambda l, a: np.array([R[(l, a, att)][s] for s in S])
    a1, b1, a2, b2 = f(lv, 'fltrust'), f(lv, 'fedavg'), f('2', 'fltrust'), f('2', 'fedavg')
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(S), (nboot, len(S)))
    with np.errstate(divide='ignore', invalid='ignore'):
        d = a1[idx].mean(1) / b1[idx].mean(1) - a2[idx].mean(1) / b2[idx].mean(1)
    d = d[np.isfinite(d)]
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {'ratio_level': float(a1.mean() / b1.mean()), 'ratio_sigma2': float(a2.mean() / b2.mean()),
            'diff': float(a1.mean() / b1.mean() - a2.mean() / b2.mean()),
            'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'p': float(p)}


def main():
    R, recs = load()
    res = {'counts': {'%s|%s|%s' % k: len(v) for k, v in R.items()}}
    m = lambda l, a, t: R[(l, a, t)]
    eps = {l: float(np.mean([r['eps_max'] for r in recs[(l, 'fedavg', 'none')].values()])) for l in ('0.5', '1', '2', '4')}
    res['eps'] = eps
    print('eps (worst client): %s' % {k: round(v, 4) for k, v in eps.items()})
    rep = json.load(open(os.path.join(OUT, 'repro_ok.json')))
    print('reproduction checks: %s' % rep['repro'])

    # ── mean table ──────────────────────────────────────────────────────────
    print('\nMean test mAP (20 seeds)')
    print('  %-22s' % '' + ''.join('%9s' % t for t in ATTACKS))
    tab = {}
    rows = [(l, a) for l in ('0.5', '1', '2', '4') for a in AGGS] + [('0', 'fedavg'), ('0', 'fltrust')] + [('2', v) for v in VARIANTS]
    for l, a in rows:
        v = [float(np.mean(list(m(l, a, t).values()))) for t in ATTACKS]
        tab['sigma %s %s' % (l, a)] = dict(zip(ATTACKS, v))
        print('  %-22s' % ('sigma %s %s' % (l, a)) + ''.join('%9.4f' % x for x in v))
    res['mean_map'] = tab

    # ── Part 1 ──────────────────────────────────────────────────────────────
    res['part1'] = {}
    for l in LEVELS:
        P = {}
        P['F1'] = fam([('FLTrust-FedAvg | %s' % t, m(l, 'fltrust', t), m(l, 'fedavg', t)) for t in ATT],
                      'F1 sigma %s (eps %.2f): FLTrust vs FedAvg under attack' % (l, eps[l]))
        P['F2'] = fam([('%s-FedAvg | none' % a, m(l, a, 'none'), m(l, 'fedavg', 'none')) for a in ('fltrust', 'median', 'krum')],
                      'F2 sigma %s: clean cost vs FedAvg' % l)
        wins = sum(c['verdict'] == 'WIN' for c in P['F1'].values())
        P['effective_defence'] = wins >= 2 and P['F2']['fltrust-FedAvg | none']['verdict'] != 'LOSS'
        allv = [P['F2']['fltrust-FedAvg | none']['verdict']] + [P['F1'][k]['verdict'] for k in P['F1']]
        P['direction_changes'] = any(v != 'LOSS' for v in allv)
        print('  sigma %s: FLTrust wins %d/3, effective defence %s, direction changes (any non-LOSS) %s %s'
              % (l, wins, P['effective_defence'], P['direction_changes'], allv))
        print('  secondary (unadjusted) median / Krum vs FedAvg under attack:')
        P['secondary'] = {}
        for a in ('median', 'krum'):
            for t in ATT:
                c = compare(m(l, a, t), m(l, 'fedavg', t)); P['secondary']['%s-FedAvg | %s' % (a, t)] = c
                print('    %-24s %+.4f [%+.4f, %+.4f] %d/%d p %.4f' % ('%s-FedAvg | %s' % (a, t), c['diff'], *c['ci95'], c['wins'], c['losses'], c['p']))
        res['part1'][l] = P

    print('\nF5: ratio FLTrust/FedAvg (means) vs the same ratio at sigma 2; paired bootstrap, Holm over 12')
    F5 = {('%s | %s' % (l, t)): ratio_test(R, l, t) for l in LEVELS for t in ATTACKS}
    adj = holm(np.array([c['p'] for c in F5.values()]))
    for (k, c), pa in zip(F5.items(), adj):
        c['p_holm'] = float(pa); c['rejects'] = bool(pa < 0.05)
        print('  sigma %-14s ratio %.3f vs %.3f  diff %+.3f [%+.3f, %+.3f]  p %.4f  p_holm %.4f  %s' % (
            k, c['ratio_level'], c['ratio_sigma2'], c['diff'], *c['ci95'], c['p'], c['p_holm'], 'CHANGES' if c['rejects'] else '-'))
    res['F5'] = F5; res['F5_size_changes'] = any(c['rejects'] for c in F5.values())

    print('\nMechanism (no attack, FLTrust runs): honest cosine, shrink factor, clean ratio FLTrust/FedAvg')
    ms = {}
    for l in ('0', '0.5', '1', '2', '4'):
        cs, sh = mech(recs, (l, 'fltrust', 'none'))
        ms[l] = {'honest_cos': cs, 'shrink': sh,
                 'ratio_clean': float(np.mean(list(m(l, 'fltrust', 'none').values())) / np.mean(list(m(l, 'fedavg', 'none').values())))}
        print('  sigma %-4s cos %.4f  shrink %.4f (1/%.1f)  ratio %.3f' % (l, cs, sh, 1 / sh, ms[l]['ratio_clean']))
    rr = [ms[l]['ratio_clean'] for l in ms]
    for k in ('honest_cos', 'shrink'):
        rho = stats.spearmanr([ms[l][k] for l in ms], rr).correlation
        ms['spearman_' + k] = float(rho)
        print('  Spearman(%s, ratio) = %+.2f -> %s' % (k, rho, 'TRACKS' if rho == 1 else 'does not track fully'))
    res['mechanism'] = ms

    # ── Part 2 ──────────────────────────────────────────────────────────────
    res['part2'] = {}
    for v in VARIANTS:
        P = {}
        P['F6'] = fam([('%s-FedAvg | %s' % (v, t), m('2', v, t), m('2', 'fedavg', t)) for t in ATTACKS],
                      'F6 (%s vs FedAvg, sigma 2 / eps 1.86)' % v)
        P['F7'] = fam([('%s-FLTrust | %s' % (v, t), m('2', v, t), m('2', 'fltrust', t)) for t in ATTACKS],
                      'F7 (%s vs Phase 5 FLTrust)' % v)
        full = all(c['verdict'] != 'LOSS' for c in P['F6'].values())
        better = any(c['verdict'] == 'WIN' for c in P['F7'].values())
        P['verdict'] = 'RECOVERS' if full else ('PARTIAL' if better else 'STILL LOSES')
        print('  %s: %s' % (v, P['verdict']))
        sec = {}
        for t in ATTACKS:
            rs = list(recs[('2', v, t)].values())
            share = np.mean([sum(x['trust'][c] for c in MAL) / max(sum(x['trust'].values()), 1e-30) for r in rs for x in r['rounds']])
            un = np.mean([x['upd_norm'] for r in rs for x in r['rounds']])
            sec[t] = {'attacker_weight_share': float(share), 'mean_update_norm': float(un)}
        P['secondary'] = sec
        print('    attacker weight share / mean update norm: %s' % {t: (round(s['attacker_weight_share'], 3), round(s['mean_update_norm'], 2)) for t, s in sec.items()})
        res['part2'][v] = P
    fa = [x['upd_norm'] for r in recs[('2', 'fedavg', 'none')].values() for x in r['rounds'] if 'upd_norm' in x]
    print('  (FedAvg mean update norm at sigma 2 not logged in Phase 5: %s)' % ('n/a' if not fa else np.mean(fa)))

    # predictions
    p1 = [ms[l]['ratio_clean'] for l in ('4', '2', '1', '0.5', '0')]
    pr = {
        'P1_ratio_monotone_and_both_track': all(x < y for x, y in zip(p1, p1[1:])) and ms['spearman_honest_cos'] == 1 and ms['spearman_shrink'] == 1,
        'P2_FLTrust_LOSES_clean_at_sigma_1_and_0.5': all(res['part1'][l]['F2']['fltrust-FedAvg | none']['verdict'] == 'LOSS' for l in ('1', '0.5')),
        'P3_median_Krum_LOSE_clean_every_DP_level': all(res['part1'][l]['F2']['%s-FedAvg | none' % a]['verdict'] == 'LOSS' for l in LEVELS for a in ('median', 'krum')),
        'P4_L_LOSES_clean_and_worse_than_FLTrust_under_ALIE': res['part2']['fltrust_L']['F6']['fltrust_L-FedAvg | none']['verdict'] == 'LOSS'
            and res['part2']['fltrust_L']['F7']['fltrust_L-FLTrust | ALIE']['diff'] < 0,
        'P5_N_ties_clean_wins_Fang_loses_IPM_ALIE_partial': [res['part2']['fltrust_N']['F6']['fltrust_N-FedAvg | %s' % t]['verdict'] for t in ATTACKS] == ['TIE', 'LOSS', 'LOSS', 'WIN'],
    }
    res['predictions'] = pr
    print('\nPredictions:'); [print('  %-55s %s' % (k, v)) for k, v in pr.items()]
    res['meta'] = C.meta()
    C.save_json(res, os.path.join(OUT, 'analysis.json'))
    plot(R, ms, tab)


def plot(R, ms, tab):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    lv = ['0', '0.5', '1', '2', '4']; x = np.arange(len(lv))
    for t, col in zip(ATTACKS, ('#7f7f7f', '#1f77b4', '#2ca02c', '#d62728')):
        ax[0].plot(x, [tab['sigma %s fltrust' % l][t] / tab['sigma %s fedavg' % l][t] for l in lv], 'o-', color=col, label=t)
    ax[0].axhline(1, color='k', lw=0.6); ax[0].set_xticks(x); ax[0].set_xticklabels(['σ %s' % l for l in lv])
    ax[0].set_ylabel('mean mAP FLTrust / FedAvg'); ax[0].set_title('FLTrust relative to FedAvg vs privacy'); ax[0].legend(fontsize=8)
    ax[1].plot(x, [ms[l]['honest_cos'] for l in lv], 'o-', label='honest cosine')
    ax[1].plot(x, [ms[l]['shrink'] for l in lv], 's-', label='shrink factor (root / honest norm)')
    ax[1].set_yscale('log'); ax[1].set_xticks(x); ax[1].set_xticklabels(['σ %s' % l for l in lv]); ax[1].legend(fontsize=8)
    ax[1].set_title('Mechanism (no attack)'); ax[1].grid(alpha=0.3)
    arms = [('fedavg', 'FedAvg'), ('fltrust', 'FLTrust'), ('fltrust_L', 'Variant L'), ('fltrust_N', 'Variant N')]
    for i, t in enumerate(ATTACKS):
        ax[2].bar(np.arange(len(arms)) + (i - 1.5) * 0.2, [tab['sigma 2 %s' % a][t] for a, _ in arms], 0.2, label=t)
    ax[2].set_xticks(range(len(arms))); ax[2].set_xticklabels([n for _, n in arms]); ax[2].set_ylabel('test mAP@0.5')
    ax[2].set_title('DP-aware FLTrust at eps 1.86'); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, 'plots', 'phase5b_fltrust.png'), dpi=150)


if __name__ == '__main__':
    main()
