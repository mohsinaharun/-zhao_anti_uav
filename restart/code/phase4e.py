# =============================================================================
# Phase 4e (restart/phase4e/PREREG.md, approved 2026-10-11): A7 = Bayesian-optimisation search
# over complete noise schedules in the proxy simulator (zero client privacy cost), then a fixed-
# schedule test evaluation paired with Phase 4's stored A1/A2/A3/A5 runs (read only).
#   python phase4e.py gate                     (G4e-a synthetic test of the optimiser)
#   python phase4e.py run --op P               (search -> selection -> repro check -> test eval)
#   python phase4e.py analyse
# =============================================================================
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, budget, bo, phase4 as P4

OUT = os.path.join(C.RESULTS, 'phase4e')
P4OUT = os.path.join(C.RESULTS, 'phase4')
D, LO, HI = 4, -2.0, 2.0
KNOTS = np.array([0, 0.25, 0.5, 0.75, 1.0])
N_INIT_SOBOL, N_TOTAL = 15, 100
SEARCH_SEEDS, SELECT_SEEDS, TEST_SEEDS = (30000, 30001, 30002), tuple(range(900, 906)), range(200, 220)
BO_SEED = 20261012


def to_x(u):                                    # [0,1]^4 -> knot values in [-2, 2]
    return LO + (HI - LO) * np.asarray(u, float)


def schedule(x, R, tgt, qs):
    """Knots (0, x1..x4) -> log-sigma piecewise linear -> scaled to the exact budget; shrink x0.9
    toward 0 until feasible (PREREG §2)."""
    x = np.asarray(x, float)
    for k in range(200):
        xs = x * 0.9 ** k
        shape = np.exp(np.interp(np.arange(R) / (R - 1), KNOTS, np.r_[0.0, xs]))
        sig, eps = budget.scale_to_budget(shape, tgt, qs)
        if sig is not None:
            return [float(s) for s in sig], float(0.9 ** k)
    raise RuntimeError('no feasible schedule')


# ── G4e-a: synthetic gate ────────────────────────────────────────────────────
def gate():
    xs = np.array([-0.6, -1.2, -1.5, -1.9])
    f = lambda x: 0.30 - 0.01 * np.sum((np.asarray(x) - xs) ** 2)
    noise_sd = 0.02                             # sd of a 3-seed mean (simulator reward scale)
    rng = np.random.default_rng(1)
    U = list(bo.initial_design(D, N_INIT_SOBOL, BO_SEED)); Y = [f(to_x(u)) + rng.normal(0, noise_sd) for u in U]
    while len(U) < N_TOTAL:
        gp = bo.GP(np.array(U), np.array(Y), seed=len(U))
        u = bo.propose(gp, D, BO_SEED + len(U)); U.append(u); Y.append(f(to_x(u)) + rng.normal(0, noise_sd))
    gp = bo.GP(np.array(U), np.array(Y), seed=999)
    top = np.argsort(-gp.predict(np.array(U))[0])[:5]
    sel = max(top, key=lambda i: np.mean([f(to_x(U[i])) + rng.normal(0, noise_sd * np.sqrt(3 / 6)) for _ in range(1)]))
    regret, span = f(xs) - f(to_x(U[sel])), f(xs) - f(np.zeros(D))
    ok = regret <= 0.1 * span
    rec = {'optimum': xs.tolist(), 'selected': to_x(U[sel]).tolist(), 'regret': float(regret), 'span': float(span),
           'criterion': 'regret <= 10% of (f(opt) - f(uniform))', 'passed': bool(ok), 'noise_sd': noise_sd}
    C.save_json(rec, os.path.join(OUT, 'gate_G4e_a.json'))
    print('G4e-a: selected %s (optimum %s); regret %.4f vs 10%% of span %.4f -> %s'
          % (np.round(to_x(U[sel]), 2).tolist(), xs.tolist(), regret, 0.1 * span, 'PASS' if ok else 'FAIL'), flush=True)
    return ok


# ── search + selection in the simulator ──────────────────────────────────────
def sim_run(data, seed, sig):
    import fl_env
    env = fl_env.FLEnv(data, seed); env.reset()
    for s in sig:
        env.step(s)
    return env.evaluate()


def search(op):
    import fl_env
    R = P4.OPS[op][0]
    data = fl_env.build_data('sim')
    tgt = P4.target_eps(op, data['qs'])
    f = os.path.join(OUT, 'search_%s.json' % op)
    log = json.load(open(f)) if os.path.exists(f) else {'op': op, 'target_eps': tgt, 'trials': [], 'meta': C.meta()}
    init = bo.initial_design(D, N_INIT_SOBOL, BO_SEED)
    t0 = time.time()
    while len(log['trials']) < N_TOTAL:
        i = len(log['trials'])
        if i < len(init):
            u = init[i]
        else:
            U = np.array([t['u'] for t in log['trials']]); Y = np.array([t['mean'] for t in log['trials']])
            u = bo.propose(bo.GP(U, Y, seed=i), D, BO_SEED + i)
        x = to_x(u)
        sig, shrink = schedule(x, R, tgt, data['qs'])
        t1 = time.time()
        rw = [sim_run(data, s, sig) for s in SEARCH_SEEDS]
        log['trials'].append({'i': i, 'u': list(map(float, u)), 'x': list(map(float, x)), 'shrink': shrink,
                              'sigmas': sig, 'rewards': rw, 'mean': float(np.mean(rw)), 'dur': time.time() - t1})
        C.save_json(log, f)
        best = max(log['trials'], key=lambda t: t['mean'])
        print('[%s] trial %3d  reward %.4f  (best so far %.4f, uniform %.4f)  sigma %s  [%.0f min]' % (
            op, i, log['trials'][-1]['mean'], best['mean'], log['trials'][0]['mean'],
            ' '.join('%.2f' % s for s in sig), (time.time() - t0) / 60), flush=True)
    if 'selection' not in log:
        U = np.array([t['u'] for t in log['trials']]); Y = np.array([t['mean'] for t in log['trials']])
        gp = bo.GP(U, Y, seed=12345)
        pm = gp.predict(U)[0]
        top = [int(i) for i in np.argsort(-pm)[:5]]
        sel, t2 = [], time.time()
        for i in top:
            rw = [sim_run(data, s, log['trials'][i]['sigmas']) for s in SELECT_SEEDS]
            sel.append({'trial': i, 'posterior_mean': float(pm[i]), 'rewards': rw, 'mean': float(np.mean(rw))})
            print('[%s] selection: trial %d  posterior %.4f  6-seed mean %.4f' % (op, i, pm[i], sel[-1]['mean']), flush=True)
        sel_dur = time.time() - t2
        uni = [sim_run(data, s, log['trials'][0]['sigmas']) for s in SELECT_SEEDS]   # reference only (not in budget)
        chosen = max(sel, key=lambda s: s['mean'])
        log['selection'] = {'candidates': sel, 'chosen_trial': chosen['trial'], 'chosen_mean': chosen['mean'],
                            'uniform_same_seeds': uni, 'uniform_mean': float(np.mean(uni)),
                            'sim_gain_over_uniform': chosen['mean'] - float(np.mean(uni)),
                            'deployed_sigmas': log['trials'][chosen['trial']]['sigmas'],
                            'runs_in_budget': len(log['trials']) * len(SEARCH_SEEDS) + 5 * len(SELECT_SEEDS),
                            'gpu_hours': (sum(t['dur'] for t in log['trials']) + sel_dur) / 3600}
        C.save_json(log, f)
        print('[%s] deployed A7 = trial %d: sim %.4f vs uniform %.4f (6 selection seeds)' % (
            op, chosen['trial'], chosen['mean'], log['selection']['uniform_mean']), flush=True)
    return log['selection']['deployed_sigmas']


# ── test evaluation (real clients, TEST pool) ────────────────────────────────
def evaluate(op, sig):
    import fl_env
    d = fl_env.build_data('real', 'test')
    if op == 'P':                                # reproduction check of the pairing (PREREG §4)
        for s in (200, 201, 202):
            g = os.path.join(OUT, 'repro_P_A1_seed%d.json' % s)
            if os.path.exists(g):
                continue
            env = P4.run_fixed(d, s, [P4.OPS['P'][1]] * P4.OPS['P'][0])
            v = env.evaluate(); old = json.load(open(os.path.join(P4OUT, 'eval_P_A1_seed%d.json' % s)))['test_map50']
            C.save_json({'seed': s, 'test_map50': v, 'phase4_stored': old, 'identical': v == old}, g)
            print('[P] repro A1 seed %d: %.6f vs stored %.6f -> %s' % (s, v, old, v == old), flush=True)
            if v != old:
                raise SystemExit('pairing check FAILED: stop')
    for s in TEST_SEEDS:
        g = os.path.join(OUT, 'eval_%s_A7_seed%d.json' % (op, s))
        if os.path.exists(g):
            continue
        t0 = time.time()
        env = P4.run_fixed(d, s, sig)
        eps = env.epsilon()
        rec = {'op': op, 'arm': 'A7', 'seed': s, 'sigmas': env.sigmas, 'test_map50': env.evaluate(), 'eps': eps,
               'eps_max': max(eps.values()), 'target_eps': P4.target_eps(op, d['qs']), 'C_star': env.C_star,
               'seconds': time.time() - t0, 'meta': C.meta()}
        C.save_json(rec, g)
        print('[%s] eval A7 seed %d: test mAP %.4f  eps %.4f (target %.4f)' % (op, s, rec['test_map50'], rec['eps_max'], rec['target_eps']), flush=True)


# ── analysis (Phase 4 rule; Phase 4 runs read only) ──────────────────────────
def analyse():
    from scipy import stats
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    def load(op, arm):
        base = OUT if arm == 'A7' else P4OUT
        return {s: json.load(open(os.path.join(base, 'eval_%s_%s_seed%d.json' % (op, arm, s)))) for s in TEST_SEEDS
                if os.path.exists(os.path.join(base, 'eval_%s_%s_seed%d.json' % (op, arm, s)))}

    def compare(a, b):
        ks = sorted(set(a) & set(b)); dd = np.array([a[k]['test_map50'] - b[k]['test_map50'] for k in ks]); n = len(dd)
        m, se = dd.mean(), dd.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
        return {'n': n, 'diff': float(m), 'ci95': [float(m - h), float(m + h)],
                'p': float(stats.ttest_1samp(dd, 0).pvalue) if se > 0 else 1.0,
                'wins': int((dd > 0).sum()), 'losses': int((dd < 0).sum())}

    def holm(ps):
        o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
        for i, k in enumerate(o):
            run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
        return adj

    res = {'ops': {}}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {'A1': '#7f7f7f', 'A2': '#1f77b4', 'A3': '#2ca02c', 'A5': '#d62728', 'A7': '#9467bd'}
    for ax, op in zip(axs, ('P', 'S2', 'S1')):
        data = {a: load(op, a) for a in ('A1', 'A2', 'A3', 'A5', 'A7')}
        if len(data['A7']) < 20:
            continue
        r = {'arms': {}, 'pairs': {}}
        print('\n=== Phase 4e %s (eps target %.4f) ===' % (op, next(iter(data['A7'].values()))['target_eps']))
        for a, dd in data.items():
            m = np.array([x['test_map50'] for x in dd.values()])
            sg = np.mean([x['sigmas'] for x in dd.values()], axis=0)
            r['arms'][a] = {'n': len(m), 'mean': float(m.mean()), 'sd': float(m.std(ddof=1)),
                            'eps_max': max(x['eps_max'] for x in dd.values()), 'mean_sigmas': sg.tolist()}
            print('%-3s n=%d  mAP %.4f  sd %.4f  eps %.4f  sigma %s' % (a, len(m), m.mean(), m.std(ddof=1),
                  r['arms'][a]['eps_max'], ' '.join('%.2f' % s for s in sg)))
            ax.plot(range(1, len(sg) + 1), sg, '-o', ms=3, color=cols[a], lw=2.2 if a == 'A7' else 1.2,
                    label='%s (mAP %.3f)' % (a, m.mean()))
        fam = [('A7', 'A1'), ('A7', 'A2'), ('A7', 'A3')]
        for x, y in fam + [('A7', 'A5')]:
            r['pairs']['%s-%s' % (x, y)] = compare(data[x], data[y])
        adj = holm(np.array([r['pairs']['%s-%s' % p]['p'] for p in fam]))
        for p, pa in zip(fam, adj):
            c = r['pairs']['%s-%s' % p]
            c.update({'p_holm': float(pa), 'verdict': ('WIN' if c['diff'] > 0 else 'LOSS') if pa < 0.05 else
                      ('TIE (EQUIVALENT)' if -0.02 <= c['ci95'][0] and c['ci95'][1] <= 0.02 else 'TIE')})
        for k, c in r['pairs'].items():
            print('%-6s diff %+.4f [%+.4f, %+.4f]  W/L %2d/%-2d  p %.4f  %s' % (k, c['diff'], c['ci95'][0], c['ci95'][1],
                  c['wins'], c['losses'], c['p'], 'p_holm %.4f %s' % (c['p_holm'], c['verdict']) if 'p_holm' in c else '(secondary)'))
        bf = max(('A2', 'A3'), key=lambda a: r['arms'][a]['mean'])
        r['better_fixed'] = bf
        r['genuine_improvement'] = r['pairs']['A7-A1']['verdict'] == 'WIN' and r['pairs']['A7-%s' % bf]['verdict'] == 'WIN'
        s = json.load(open(os.path.join(OUT, 'search_%s.json' % op)))['selection']
        r['search'] = {k: s[k] for k in ('chosen_trial', 'chosen_mean', 'uniform_mean', 'sim_gain_over_uniform', 'runs_in_budget', 'gpu_hours')}
        print('better of A2/A3: %s;  A7 genuine improvement: %s;  sim gain over uniform %+.4f' % (bf, r['genuine_improvement'], s['sim_gain_over_uniform']))
        res['ops'][op] = r
        ax.set_title(op); ax.set_xlabel('round'); ax.set_ylabel('sigma_t (mean over test seeds)'); ax.set_yscale('log')
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, 'plots', 'phase4e_schedules.png'), dpi=150)
    if 'P' in res['ops']:
        res['verdict_P'] = 'A7 is a genuine improvement' if res['ops']['P']['genuine_improvement'] else 'A7 is NOT a genuine improvement'
        print('\nPRIMARY (P): %s' % res['verdict_P'])
    res['meta'] = C.meta()
    C.save_json(res, os.path.join(OUT, 'analysis.json'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['gate', 'run', 'analyse'])
    ap.add_argument('--op', choices=list(P4.OPS))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'gate':
        sys.exit(0 if gate() else 1)
    if a.mode == 'analyse':
        analyse(); sys.exit(0)
    g = os.path.join(OUT, 'gate_G4e_a.json')
    if not (os.path.exists(g) and json.load(open(g))['passed']):
        raise SystemExit('gate G4e-a has not passed: stop')
    sig = search(a.op)
    evaluate(a.op, sig)
    print('[%s] PHASE 4e DONE' % a.op, flush=True)
