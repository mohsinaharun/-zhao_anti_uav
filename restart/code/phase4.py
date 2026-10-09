# =============================================================================
# Phase 4 runner (restart/phase4/PREREG.md + Amendment 1). Arms:
#   A1 uniform | A2 tuned exponential | A3 theory (rho_t ~ sqrt(a_t)) | A4/A5 DRL policies
# Operating points: P (R=12, budget = uniform sigma 2), S1 (R=25, sigma 2), S2 (R=12, sigma 4).
# Budget = worst-client epsilon of the uniform schedule (real clients' sampling rates;
# the simulator's pseudo-clients have identical sizes, so identical targets).
#   python phase4.py tune  --op P            (A2 k on VAL, tuning seeds 100-104)
#   python phase4.py eval  --op P --arm A1 --seeds 200-219 [--policy file]  (TEST pool)
# =============================================================================
import os, sys, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, budget, fl_env

OPS = {'P': (12, 2.0), 'S1': (25, 2.0), 'S2': (12, 4.0)}
KS = [0, 0.25, 0.5, 1, 1.5, 2, 3]
OUT = os.path.join(C.RESULTS, 'phase4')


def target_eps(op, qs):
    R, s = OPS[op]
    return budget.eps_worst([s] * R, qs, exact=True)


def shape_exp(R, k):
    t = np.arange(R)
    return np.exp(-k * t / max(R - 1, 1))                          # k > 0: less noise late


def shape_theory(op):
    """A3: rho_t ~ sqrt(a_t), a_t = Phase 3 mean drop per pulse round (V1 for P/S2, V2 for S1,
    at the operating point's sigma); negatives floored at 0.1 x the median; sigma_t ~ a_t^(-1/4)."""
    R, s = OPS[op]
    v = 'V2' if op == 'S1' else 'V1'
    a = np.array(json.load(open(os.path.join(C.RESULTS, 'phase3', 'analysis_%s.json' % v)))
                 ['cells']['%g' % s]['drop_mean'], float)
    floor = 0.1 * np.median(a)
    a = np.maximum(a, floor if floor > 0 else 1e-4)
    return a ** -0.25, a.tolist()


def schedule_for(arm, op, qs, k=None):
    R, s = OPS[op]
    tgt = target_eps(op, qs)
    if arm == 'A1':
        return [s] * R, tgt
    shape = shape_exp(R, k) if arm == 'A2' else shape_theory(op)[0]
    sig, _ = budget.scale_to_budget(shape, tgt, qs)
    return sig, tgt


def run_fixed(data, seed, sigmas):
    env = fl_env.FLEnv(data, seed); env.reset()
    for s in sigmas:
        env.step(s)
    return env


def run_policy(data, seed, policy, op):
    """Online arm: policy(obs_history, t, R, spent_frac) -> action in [0, 1]."""
    R, _ = OPS[op]
    tgt = target_eps(op, data['qs'])
    env = fl_env.FLEnv(data, seed); obs = [env.reset()]
    policy.reset()
    for t in range(R):
        spent = budget.eps_worst(env.sigmas, data['qs']) / tgt if env.sigmas else 0.0
        a = float(policy.act(obs, t, R, spent))
        share = 0.25 * 16 ** a
        sig = budget.sigma_for_round(env.sigmas, share, R - t, tgt, data['qs'])
        obs.append(env.step(sig))
    return env


def save(rec, name):
    os.makedirs(OUT, exist_ok=True)
    C.save_json(rec, os.path.join(OUT, name + '.json'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['tune', 'eval', 'schedules'])
    ap.add_argument('--op', choices=list(OPS), required=True)
    ap.add_argument('--arm', default='A1')
    ap.add_argument('--seeds', default='200-219')
    ap.add_argument('--policy', default='')
    a = ap.parse_args()
    R, s_uni = OPS[a.op]

    if a.mode == 'schedules':
        d = fl_env.build_data('real', 'val')
        print('target eps %.4f' % target_eps(a.op, d['qs']))
        best = json.load(open(os.path.join(OUT, 'tune_%s.json' % a.op)))['best_k'] if os.path.exists(os.path.join(OUT, 'tune_%s.json' % a.op)) else 1
        for arm, k in (('A1', None), ('A2', best), ('A3', None)):
            sig, tgt = schedule_for(arm, a.op, d['qs'], k)
            print('%s%s: %s  eps %.4f' % (arm, ' k=%g' % k if k is not None else '', ' '.join('%.2f' % x for x in sig),
                                          budget.eps_worst(sig, d['qs'], exact=True)))
        sys.exit(0)

    if a.mode == 'tune':                                             # A2 k on VAL
        d = fl_env.build_data('real', 'val')
        res = {}
        for k in KS:
            sig, tgt = schedule_for('A2', a.op, d['qs'], k)
            if sig is None:
                res[str(k)] = None; print('k=%g infeasible' % k); continue
            m = []
            for seed in range(100, 105):
                name = 'tune_%s_k%g_seed%d' % (a.op, k, seed)
                f = os.path.join(OUT, name + '.json')
                if os.path.exists(f):
                    m.append(json.load(open(f))['val_map50']); continue
                env = run_fixed(d, seed, sig)
                v = env.evaluate(); m.append(v)
                save({'op': a.op, 'k': k, 'seed': seed, 'sigmas': sig, 'val_map50': v,
                      'eps': env.epsilon(), 'C_star': env.C_star, 'meta': C.meta()}, name)
            res[str(k)] = float(np.mean(m))
            print('op %s k=%-4g sigma %.2f..%.2f  val mAP %.4f (5 tuning seeds)' % (a.op, k, sig[0], sig[-1], res[str(k)]), flush=True)
        best = max((k for k in res if res[k] is not None), key=lambda k: res[k])
        save({'op': a.op, 'grid': res, 'best_k': float(best), 'meta': C.meta()}, 'tune_%s' % a.op)
        print('best k for %s: %s' % (a.op, best))
        sys.exit(0)

    # eval on TEST
    d = fl_env.build_data('real', 'test')
    lo, hi = map(int, a.seeds.split('-'))
    pol = None
    if a.arm in ('A4', 'A5'):
        import phase4_agents as AG
        pol = AG.load_policy(a.policy, a.arm)
    k = json.load(open(os.path.join(OUT, 'tune_%s.json' % a.op)))['best_k'] if a.arm == 'A2' else None
    for seed in range(lo, hi + 1):
        name = 'eval_%s_%s_seed%d' % (a.op, a.arm, seed)
        if os.path.exists(os.path.join(OUT, name + '.json')):
            continue
        t0 = time.time()
        if pol is None:
            sig, _ = schedule_for(a.arm, a.op, d['qs'], k)
            env = run_fixed(d, seed, sig)
        else:
            env = run_policy(d, seed, pol, a.op)
        eps = env.epsilon()
        rec = {'op': a.op, 'arm': a.arm, 'seed': seed, 'k': k, 'policy': a.policy, 'sigmas': env.sigmas,
               'test_map50': env.evaluate(), 'eps': eps, 'eps_max': max(eps.values()),
               'target_eps': target_eps(a.op, d['qs']), 'C_star': env.C_star,
               'seconds': time.time() - t0, 'meta': C.meta()}
        save(rec, name)
        print('%s: test mAP %.4f  eps max %.4f (target %.4f)' % (name, rec['test_map50'], rec['eps_max'], rec['target_eps']), flush=True)
