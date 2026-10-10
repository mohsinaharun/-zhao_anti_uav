# =============================================================================
# Phase 4b Part 2 (restart/phase4b/PREREG.md): A5-warmstart ("A5w") at P.
# A5 exactly (phase4_agents.TD3GRU, seed 7, simulator, 300 episodes, same reward/exploration/
# checkpoint rule/G4b) except: (1) actor initialised by behaviour cloning to A3's schedule
# expressed as online actions; (2) warm-up episodes centred on that policy; (3) actor updates
# held until the end of warm-up.
#   python phase4b_agent.py train
#   python phase4b_agent.py eval   (TEST, seeds 200-219)
#   python phase4b_agent.py analyse
# =============================================================================
import os, sys, json, time, copy, shutil, argparse
import numpy as np, torch
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, budget, fl_env, phase4 as P4, phase4_agents as AG

OUT = os.path.join(C.RESULTS, 'phase4b')
OP, ARM = 'P', 'A5w'
BC_SEEDS = range(20000, 20005)


def a3_actions(qs):
    """A3's schedule at P as actions of the online mapping, clipped to [0, 1]."""
    R = P4.OPS[OP][0]
    tgt = P4.target_eps(OP, qs)
    a3 = P4.schedule_for('A3', OP, qs)[0]
    qw, ai, total = budget._unit(tgt, qs, R)
    rdp = lambda s: float(budget._rdp_fast(qw, s, budget.STEPS)[ai])
    h, acts = [], []
    for t in range(R):
        rem = total - sum(rdp(x) for x in h)
        a = float(np.clip(np.log(rdp(a3[t]) / (rem / (R - t)) / 0.25) / np.log(16), 0, 1))
        acts.append(a)
        h.append(budget.sigma_for_round(h, 0.25 * 16 ** a, R - t, tgt, qs))
    return acts, h, list(a3)


class FixedActions:
    def __init__(self, acts):
        self.acts = acts

    def reset(self):
        pass

    def act(self, obs, t, R, spent, explore_std=0.0):
        return self.acts[t]


class TD3GRUWarm(AG.TD3GRU):
    actor_frozen = True

    def update(self, n, **kw):
        if not self.actor_frozen:
            return super().update(n, **kw)
        oa = self.oa                                   # critic-only updates: skip the actor step
        class _NoStep:
            def zero_grad(self): pass
            def step(self): pass
        self.oa = _NoStep()
        saved = [p.detach().clone() for p in self.actor.parameters()] + [p.detach().clone() for p in self.ta.parameters()]
        super().update(n, **kw)
        for p, s in zip(list(self.actor.parameters()) + list(self.ta.parameters()), saved):
            p.data.copy_(s)                            # undo the soft target update of the actor
        self.oa = oa


def episode(data, seed, pol, explore_std=0.0, warm_center=None):
    R, _ = P4.OPS[OP]
    tgt = P4.target_eps(OP, data['qs'])
    env = fl_env.FLEnv(data, seed); obs = [env.reset()]
    pol.reset(); X, A = [], []
    for t in range(R):
        spent = budget.eps_worst(env.sigmas, data['qs']) / tgt if env.sigmas else 0.0
        X.append(AG.features(obs, t, R, spent))
        if warm_center is not None:
            pol.act(obs, t, R, spent)
            a = float(np.clip(np.random.normal(warm_center[t], AG.STD0), 0, 1))
        else:
            a = pol.act(obs, t, R, spent, explore_std)
        A.append(a)
        obs.append(env.step(budget.sigma_for_round(env.sigmas, 0.25 * 16 ** a, R - t, tgt, data['qs'])))
    X.append(AG.features(obs, R, R, budget.eps_worst(env.sigmas, data['qs']) / tgt))
    return {'X': X, 'A': A, 'reward': env.evaluate(), 'sigmas': env.sigmas, 'seed': seed}


def train(episodes=300):
    tag = 'agent_%s_%s' % (ARM, OP)
    logf, ckdir = os.path.join(OUT, tag + '_log.json'), os.path.join(C.CACHE, 'phase4b_ckpt', tag)
    os.makedirs(ckdir, exist_ok=True)
    np.random.seed(7); torch.manual_seed(7)
    agent = TD3GRUWarm(7)
    data = fl_env.build_data('sim')
    R = P4.OPS[OP][0]
    acts, sig_repr, sig_a3 = a3_actions(data['qs'])
    # behaviour cloning on roll-outs that follow A3 (not stored in the replay buffer)
    rolls = [episode(data, s, FixedActions(acts)) for s in BC_SEEDS]
    X = torch.tensor([r['X'][:R] for r in rolls], dtype=torch.float32, device=AG.DEV)
    Y = torch.tensor([acts] * len(rolls), dtype=torch.float32, device=AG.DEV).unsqueeze(2)
    opt = torch.optim.Adam(agent.actor.parameters(), 1e-3)
    for it in range(5000):
        loss = ((agent.actor(X) - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < 1e-5:
            break
    agent.ta = copy.deepcopy(agent.actor)
    with torch.no_grad():
        fit = agent.actor(X)[:, :, 0].mean(0).tolist()
    log = {'arm': ARM, 'op': OP, 'a3_actions': acts, 'a3_sigmas': sig_a3, 'a3_representable_sigmas': sig_repr,
           'bc': {'iters': it + 1, 'mse': float(loss.item()), 'fitted_actions': fit,
                  'bc_rewards': [r['reward'] for r in rolls]},
           'episodes': [], 'evals': [], 'meta': C.meta()}
    print('BC: %d iters, mse %.2e; target %s; fitted %s; A3-followed sim reward %.4f'
          % (it + 1, loss.item(), ' '.join('%.3f' % a for a in acts), ' '.join('%.3f' % a for a in fit),
             np.mean(log['bc']['bc_rewards'])), flush=True)
    t0 = time.time()
    for e in range(1, episodes + 1):
        agent.actor_frozen = e <= AG.WARMUP
        std = AG.STD0 + (AG.STD1 - AG.STD0) * min(1.0, (e - 1) / max(episodes - 1, 1))
        ep = episode(data, 10000 + e, agent.policy(), explore_std=std, warm_center=acts if e <= AG.WARMUP else None)
        agent.store(ep)
        agent.update(2 * R)
        log['episodes'].append({'ep': e, 'reward': ep['reward'], 'actions': ep['A'], 'sigmas': ep['sigmas'],
                                'std': std, 'seconds': time.time() - t0})
        if e % AG.EVAL_EVERY == 0:
            pol = agent.policy()
            evs = [episode(data, s, pol) for s in AG.EVAL_SEEDS]
            torch.save(agent.actor.state_dict(), os.path.join(ckdir, 'ep%03d.pt' % e))
            log['evals'].append({'ep': e, 'mean_reward': float(np.mean([x['reward'] for x in evs])),
                                 'rewards': [x['reward'] for x in evs], 'sigmas': [x['sigmas'] for x in evs]})
            print('%s ep %3d  train reward (last 25) %.4f  eval reward %.4f  sigma %s  [%.0f min]'
                  % (tag, e, np.mean([x['reward'] for x in log['episodes'][-25:]]), log['evals'][-1]['mean_reward'],
                     ' '.join('%.2f' % s for s in evs[0]['sigmas']), (time.time() - t0) / 60), flush=True)
        C.save_json(log, logf)
    best = max(log['evals'], key=lambda x: x['mean_reward'])
    shutil.copy(os.path.join(ckdir, 'ep%03d.pt' % best['ep']), os.path.join(OUT, tag + '_best.pt'))
    log['selected_checkpoint'] = best['ep']
    first = [x['reward'] for x in log['episodes'][:25]]; last = [x['reward'] for x in log['episodes'][-25:]]
    diff = np.mean(last) - np.mean(first)
    se = np.sqrt(np.var(first, ddof=1) / 25 + np.var(last, ddof=1) / 25)
    log['G4b'] = {'first25': float(np.mean(first)), 'last25': float(np.mean(last)), 'diff': float(diff),
                  'ci95': [float(diff - 1.96 * se), float(diff + 1.96 * se)], 'learned': bool(diff - 1.96 * se > 0)}
    C.save_json(log, logf)
    print('%s done: selected checkpoint ep %d (eval %.4f); G4b %s' % (tag, best['ep'], best['mean_reward'], log['G4b']), flush=True)


def evaluate():
    d = fl_env.build_data('real', 'test')
    pol = AG.load_policy(os.path.join(OUT, 'agent_%s_%s_best.pt' % (ARM, OP)), 'A5')
    for seed in range(200, 220):
        name = 'eval_%s_%s_seed%d' % (OP, ARM, seed)
        f = os.path.join(OUT, name + '.json')
        if os.path.exists(f):
            continue
        t0 = time.time()
        env = P4.run_policy(d, seed, pol, OP)
        eps = env.epsilon()
        rec = {'op': OP, 'arm': ARM, 'seed': seed, 'sigmas': env.sigmas, 'test_map50': env.evaluate(),
               'eps': eps, 'eps_max': max(eps.values()), 'target_eps': P4.target_eps(OP, d['qs']),
               'C_star': env.C_star, 'seconds': time.time() - t0, 'meta': C.meta()}
        C.save_json(rec, f)
        print('%s: test mAP %.4f  eps max %.4f (target %.4f)' % (name, rec['test_map50'], rec['eps_max'], rec['target_eps']), flush=True)


# paired comparison / Holm / verdict: copied verbatim from analyse_phase4.py (which runs on import)
EQ = 0.02


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



class AP:
    OUT = os.path.join(C.RESULTS, 'phase4')
    SEEDS = list(range(200, 220))
    compare, holm, verdict = staticmethod(compare), staticmethod(holm), staticmethod(verdict)


def analyse():
    def load(arm):
        base = OUT if arm == ARM else AP.OUT
        return {s: json.load(open(os.path.join(base, 'eval_%s_%s_seed%d.json' % (OP, arm, s)))) for s in AP.SEEDS
                if os.path.exists(os.path.join(base, 'eval_%s_%s_seed%d.json' % (OP, arm, s)))}
    data = {a: load(a) for a in ('A1', 'A2', 'A3', 'A5', ARM)}
    res = {'arms': {}, 'pairs': {}}
    print('\n=== Phase 4b, P (eps target %.4f) ===' % next(iter(data['A1'].values()))['target_eps'])
    for a, dd in data.items():
        m = np.array([x['test_map50'] for x in dd.values()])
        res['arms'][a] = {'n': len(m), 'mean': float(m.mean()), 'sd': float(m.std(ddof=1)),
                          'eps_max': max(x['eps_max'] for x in dd.values()),
                          'mean_sigmas': np.mean([x['sigmas'] for x in dd.values()], axis=0).tolist()}
        print('%-4s n=%d  mAP %.4f  sd %.4f  eps %.4f  sigma %s' % (a, len(m), m.mean(), res['arms'][a]['sd'],
              res['arms'][a]['eps_max'], ' '.join('%.2f' % s for s in res['arms'][a]['mean_sigmas'])))
    fam = [(ARM, 'A1'), (ARM, 'A2'), (ARM, 'A3')]
    for x, y in fam + [(ARM, 'A5')]:
        res['pairs']['%s-%s' % (x, y)] = AP.compare(data[x], data[y])
    adj = AP.holm(np.array([res['pairs']['%s-%s' % p]['p'] for p in fam]))
    for p, pa in zip(fam, adj):
        k = '%s-%s' % p
        res['pairs'][k].update({'family': 'primary', 'p_holm': float(pa), 'verdict': AP.verdict(res['pairs'][k], pa)})
    for k, c in res['pairs'].items():
        print('%-8s diff %+.4f [%+.4f, %+.4f]  W/L %d/%d  p %.4f  %s  %s' % (k, c['diff'], c['ci95'][0], c['ci95'][1],
              c['wins'], c['losses'], c['p'], 'p_holm %.4f' % c['p_holm'] if 'p_holm' in c else '(secondary)', c.get('verdict', '')))
    best_fixed = max(('A2', 'A3'), key=lambda k: res['arms'][k]['mean'])
    res['warmstart_learning_helps'] = res['pairs']['%s-A1' % ARM]['verdict'] == 'WIN' and res['pairs']['%s-%s' % (ARM, best_fixed)]['verdict'] == 'WIN'
    res['phase5_learned_schedule'] = ARM if res['arms'][ARM]['mean'] > res['arms']['A5']['mean'] else 'A5'
    res['meta'] = C.meta()
    C.save_json(res, os.path.join(OUT, 'analysis.json'))
    print('"warm-started learning helps": %s;  Phase 5 uses: %s' % (res['warmstart_learning_helps'], res['phase5_learned_schedule']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['train', 'eval', 'analyse']); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {'train': train, 'eval': evaluate, 'analyse': analyse}[a.mode]()
