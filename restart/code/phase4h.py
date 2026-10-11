# =============================================================================
# Phase 4h (restart/phase4h/PREREG.md, approved 2026-10-12): A5-relative agent.
# Same learner, state, training protocol and checkpoint rule as A5 (phase4_agents.py, imported
# read-only); only the action changes: a in [0,1] -> delta = 0.4a - 0.2 (+-20%) on A3's sigma for
# the round, with A3's remaining rounds rescaled by one factor lambda_t so the budget is exact.
#   python phase4h.py go --op P|S1|S2        (gate [P worker only], train, evaluate; resumable per stage)
#   python phase4h.py analyse
# =============================================================================
import os, sys, json, time, shutil, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, budget, fl_env, phase4 as P4, phase4_agents as AG

OUT = os.path.join(C.RESULTS, 'phase4h')
P4OUT = os.path.join(C.RESULTS, 'phase4')
SEEDS = range(200, 220)


def delta_of(a):
    return 0.4 * float(a) - 0.2


class Anchor:
    """A3's schedule for an operating point, re-planned online (PREREG §3)."""

    def __init__(self, op, qs):
        self.s = np.asarray(P4.schedule_for('A3', op, qs)[0], float)
        self.R, self.qs = len(self.s), qs
        self.tgt = budget.eps_worst(self.s, qs)                 # A3's own budget (same fast accountant)

    def lam(self, hist):
        t = len(hist)
        if list(hist) == self.s[:t].tolist():
            return 1.0                                         # no deviation so far: the plan is A3
        rest = self.s[t:]
        f = lambda l: budget.eps_worst(list(hist) + np.clip(l * rest, budget.SIG_MIN, budget.SIG_MAX).tolist(), self.qs)
        lo, hi = budget.SIG_MIN / rest.max(), budget.SIG_MAX / rest.min()
        if f(hi) > self.tgt:
            return hi
        if f(lo) <= self.tgt:
            return lo
        for _ in range(50):                                    # eps decreases as lambda grows
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if f(mid) > self.tgt else (lo, mid)
        return hi                                              # the side within budget

    def sigma(self, hist, delta):
        t = len(hist)
        lam = self.lam(hist)
        d = 0.0 if t == self.R - 1 else delta                  # last round: exact remainder, action ignored
        return float(np.clip((1 + d) * lam * self.s[t], budget.SIG_MIN, budget.SIG_MAX)), lam


def episode(data, seed, pol, op, explore_std=0.0, random_actions=False):
    """= phase4_agents.episode with the relative action mapping."""
    R, _ = P4.OPS[op]
    tgt = P4.target_eps(op, data['qs'])
    anc = Anchor(op, data['qs'])
    env = fl_env.FLEnv(data, seed); obs = [env.reset()]
    pol.reset(); X, A, L = [], [], []
    for t in range(R):
        spent = budget.eps_worst(env.sigmas, data['qs']) / tgt if env.sigmas else 0.0
        X.append(AG.features(obs, t, R, spent))
        a = float(np.clip(np.random.normal(0.5, AG.STD0), 0, 1)) if random_actions else pol.act(obs, t, R, spent, explore_std)
        if random_actions:
            pol.act(obs, t, R, spent)                          # keep the policy's sequence in step
        A.append(a)
        sig, lam = anc.sigma(env.sigmas, delta_of(a))
        L.append(lam)
        obs.append(env.step(sig))
    X.append(AG.features(obs, R, R, budget.eps_worst(env.sigmas, data['qs']) / tgt))
    return {'X': X, 'A': A, 'deltas': [delta_of(a) for a in A[:-1]], 'lams': L, 'reward': None, 'env': env,
            'sigmas': env.sigmas, 'a3': anc.s.tolist(), 'seed': seed, 'obs': obs}


def sim_episode(data, seed, pol, op, **kw):
    ep = episode(data, seed, pol, op, **kw)
    ep['reward'] = ep.pop('env').evaluate(); ep.pop('obs')
    return ep


class ZeroPolicy:                                              # delta = 0 every round
    def reset(self):
        pass

    def act(self, obs, t, R, spent, explore_std=0.0):
        return 0.5


# ── Gate G4h (PREREG §3) ─────────────────────────────────────────────────────
def gate():
    res, ok = {}, True
    d = fl_env.build_data('real', 'test')
    rng = np.random.default_rng(0)
    for op in P4.OPS:
        anc = Anchor(op, d['qs'])
        h = []
        for t in range(anc.R):
            h.append(anc.sigma(h, 0.0)[0])
        rel = float(np.max(np.abs(np.array(h) / anc.s - 1)))
        a3e = budget.eps_worst(anc.s, d['qs'], exact=True)
        worst = 0.0
        for _ in range(200):
            h = []
            for t in range(anc.R):
                h.append(anc.sigma(h, float(rng.uniform(-0.2, 0.2)))[0])
            worst = max(worst, abs(budget.eps_worst(h, d['qs'], exact=True) - a3e) / a3e)
        res[op] = {'zero_reproduces_A3_max_rel': rel, 'A3_eps_exact': a3e,
                   'target_eps_phase4': P4.target_eps(op, d['qs']), 'random_200_max_rel_err_vs_A3_eps': worst}
        ok &= rel < 1e-4 and worst < 1e-3
        print('G4h %s: zero-delta vs A3 max rel %.2e; 200 random sequences max eps rel err %.2e (A3 eps %.4f)'
              % (op, rel, worst, a3e), flush=True)
    res['test_P'] = {}
    for seed in (200, 201, 202):
        ep = episode(d, seed, ZeroPolicy(), 'P')
        v = ep['env'].evaluate()
        ref = json.load(open(os.path.join(P4OUT, 'eval_P_A3_seed%d.json' % seed)))['test_map50']
        res['test_P'][seed] = [v, ref]
        ok &= v == ref
        print('G4h test P seed %d: zero policy %.6f vs Phase 4 A3 %.6f %s' % (seed, v, ref, 'OK' if v == ref else 'MISMATCH'), flush=True)
    res['ok'] = bool(ok); res['meta'] = C.meta()
    C.save_json(res, os.path.join(OUT, 'gate_G4h.json'))
    return ok


# ── training (= phase4_agents __main__, A5 settings) ─────────────────────────
def train(op, episodes=300):
    tag = 'agent_A5r_%s' % op
    best_pt = os.path.join(OUT, tag + '_best.pt')
    if os.path.exists(best_pt):
        return best_pt
    logf, ckdir = os.path.join(OUT, tag + '_log.json'), os.path.join(C.CACHE, 'phase4h_ckpt', tag)
    os.makedirs(ckdir, exist_ok=True)
    np.random.seed(7); torch.manual_seed(7)
    agent = AG.TD3GRU(7)
    data = fl_env.build_data('sim')
    R = P4.OPS[op][0]
    log = {'arm': 'A5r', 'op': op, 'episodes': [], 'evals': [], 'meta': C.meta()}
    t0 = time.time()
    for e in range(1, episodes + 1):
        std = AG.STD0 + (AG.STD1 - AG.STD0) * min(1.0, (e - 1) / max(episodes - 1, 1))
        ep = sim_episode(data, 10000 + e, agent.policy(), op, explore_std=std, random_actions=e <= AG.WARMUP)
        agent.store(ep)
        agent.update(2 * R)
        log['episodes'].append({'ep': e, 'reward': ep['reward'], 'actions': ep['A'], 'deltas': ep['deltas'],
                                'sigmas': ep['sigmas'], 'lams': ep['lams'], 'std': std, 'seconds': time.time() - t0})
        if e % AG.EVAL_EVERY == 0:
            pol = agent.policy()
            evs = [sim_episode(data, s, pol, op) for s in AG.EVAL_SEEDS]
            torch.save(agent.actor.state_dict(), os.path.join(ckdir, 'ep%03d.pt' % e))
            log['evals'].append({'ep': e, 'mean_reward': float(np.mean([x['reward'] for x in evs])),
                                 'rewards': [x['reward'] for x in evs], 'sigmas': [x['sigmas'] for x in evs],
                                 'deltas': [x['deltas'] for x in evs]})
            print('%s ep %3d  train reward (last 25) %.4f  eval reward %.4f  delta %s  [%.0f min]'
                  % (tag, e, np.mean([x['reward'] for x in log['episodes'][-25:]]), log['evals'][-1]['mean_reward'],
                     ' '.join('%+.2f' % x for x in evs[0]['deltas']), (time.time() - t0) / 60), flush=True)
        C.save_json(log, logf)
    best = max(log['evals'], key=lambda x: x['mean_reward'])
    shutil.copy(os.path.join(ckdir, 'ep%03d.pt' % best['ep']), best_pt)
    log['selected_checkpoint'] = best['ep']
    first = [x['reward'] for x in log['episodes'][:25]]; last = [x['reward'] for x in log['episodes'][-25:]]
    diff = np.mean(last) - np.mean(first)
    se = np.sqrt(np.var(first, ddof=1) / 25 + np.var(last, ddof=1) / 25)
    log['G4b'] = {'first25': float(np.mean(first)), 'last25': float(np.mean(last)), 'diff': float(diff),
                  'ci95': [float(diff - 1.96 * se), float(diff + 1.96 * se)], 'learned': bool(diff - 1.96 * se > 0)}
    log['hours'] = (time.time() - t0) / 3600
    C.save_json(log, logf)
    print('%s done: selected checkpoint ep %d (eval %.4f); G4b %s' % (tag, best['ep'], best['mean_reward'], log['G4b']), flush=True)
    return best_pt


def evaluate(op, best_pt):
    d = fl_env.build_data('real', 'test')
    for seed in SEEDS:
        name = 'eval_%s_A5r_seed%d' % (op, seed)
        f = os.path.join(OUT, name + '.json')
        if os.path.exists(f):
            continue
        t0 = time.time()
        pol = AG.load_policy(best_pt, 'A5')
        ep = episode(d, seed, pol, op)
        env = ep['env']
        eps = env.epsilon()
        rec = {'op': op, 'arm': 'A5r', 'seed': seed, 'policy': best_pt, 'sigmas': env.sigmas, 'a3': ep['a3'],
               'actions': ep['A'], 'deltas': ep['deltas'], 'lams': ep['lams'],
               'test_map50': env.evaluate(), 'eps': eps, 'eps_max': max(eps.values()),
               'target_eps': P4.target_eps(op, d['qs']), 'C_star': env.C_star,
               'seconds': time.time() - t0, 'meta': C.meta()}
        C.save_json(rec, f)
        print('%s: test mAP %.4f  eps max %.4f (target %.4f)' % (name, rec['test_map50'], rec['eps_max'], rec['target_eps']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['go', 'gate', 'analyse'])
    ap.add_argument('--op', choices=list(P4.OPS), default='P')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'analyse':
        import analyse_phase4h; analyse_phase4h.main(); sys.exit(0)
    gate_f, stop_f = os.path.join(OUT, 'gate_ok.json'), os.path.join(OUT, 'STOPPED.json')
    if a.mode == 'gate' or (a.op == 'P' and not os.path.exists(gate_f)):
        if not gate():
            C.save_json({'gate': 'FAILED'}, stop_f); sys.exit(1)
        C.save_json({'gate': 'OK', 'meta': C.meta()}, gate_f)
        if a.mode == 'gate':
            sys.exit(0)
    while not os.path.exists(gate_f):
        if os.path.exists(stop_f):
            sys.exit(1)
        time.sleep(10)
    evaluate(a.op, train(a.op))
    print('A5r %s DONE' % a.op, flush=True)
