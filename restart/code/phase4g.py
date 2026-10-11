# =============================================================================
# Phase 4g (restart/phase4g/PREREG.md, approved 2026-10-12 with +A and +B): round-count check.
# A2 (k = 0.5), A1 uniform and no-DP (sigma 0, clipping kept) at R = 25, 50, 100 rounds,
# DP arms at worst-client eps 1.86 (= Phase 4 P budget). R = 12 is reused from Phase 4 / 5.
# Phase 4 / 5 code is imported read-only.
#   python phase4g.py run [--worker i --workers n]   (resumable; reproduction check first)
#   python phase4g.py analyse
# =============================================================================
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, budget, fl_env, phase4 as P4

OUT = os.path.join(C.RESULTS, 'phase4g')
SEEDS = range(200, 220)
ROUNDS = (25, 50, 100)
ARMS = ('A2', 'A1', 'noDP')
K = 0.5
_D = {}


def data():
    if 'd' not in _D:
        _D['d'] = fl_env.build_data('real', 'test')
    return _D['d']


def schedule(arm, R, qs):
    if arm == 'noDP':
        return [0.0] * R
    tgt = P4.target_eps('P', qs)
    shape = P4.shape_exp(R, K) if arm == 'A2' else np.ones(R)
    sig, _ = budget.scale_to_budget(shape, tgt, qs)
    return [float(x) for x in sig]


def run(block, arm, R, seed):
    name = '%s_%s_R%d_seed%d' % (block, arm, R, seed)
    f = os.path.join(OUT, name + '.json')
    if os.path.exists(f):
        return json.load(open(f))
    d, t0 = data(), time.time()
    sig = schedule(arm, R, d['qs'])
    env = fl_env.FLEnv(d, seed)
    obs = [env.reset()]
    for s in sig:
        obs.append(env.step(s))
    eps = env.epsilon() if min(env.sigmas) > 0 else None
    rec = {'block': block, 'arm': arm, 'R': R, 'k': K if arm == 'A2' else None, 'seed': seed, 'sigmas': env.sigmas,
           'test_map50': env.evaluate(), 'eps': eps, 'eps_max': max(eps.values()) if eps else float('inf'),
           'target_eps': P4.target_eps('P', d['qs']), 'C_star': env.C_star, 'obs': obs,
           'seconds': time.time() - t0, 'meta': C.meta()}
    C.save_json(rec, f)
    print('%s: mAP %.4f eps %.4f (%.0f s)' % (name, rec['test_map50'], rec['eps_max'], rec['seconds']), flush=True)
    return rec


def jobs():
    J = [(arm, R, seed) for R in sorted(ROUNDS, reverse=True) for seed in SEEDS for arm in ARMS]
    return [('grid',) + j for j in J]


def repro_check():
    bad = []
    for seed in (200, 201, 202):
        v = run('check', 'A2', 12, seed)['test_map50']
        ref = json.load(open(os.path.join(C.RESULTS, 'phase4', 'eval_P_A2_seed%d.json' % seed)))['test_map50']
        print('repro A2 R12 seed %d: %.6f vs Phase 4 %.6f %s' % (seed, v, ref, 'OK' if v == ref else 'MISMATCH'), flush=True)
        bad += [('A2', seed)] if v != ref else []
        v = run('check', 'noDP', 12, seed)['test_map50']
        ref = json.load(open(os.path.join(C.RESULTS, 'phase5', 'nodp_noDP_fedavg_none_seed%d.json' % seed)))['test_map50']
        print('repro noDP R12 seed %d: %.6f vs Phase 5 %.6f %s' % (seed, v, ref, 'OK' if v == ref else 'MISMATCH'), flush=True)
        bad += [('noDP', seed)] if v != ref else []
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'analyse'])
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'analyse':
        import analyse_phase4g; analyse_phase4g.main(); sys.exit(0)
    if a.worker == 0:
        bad = repro_check()
        if bad:
            C.save_json({'repro': 'MISMATCH', 'bad': bad}, os.path.join(OUT, 'STOPPED.json')); sys.exit(1)
        C.save_json({'repro': 'OK', 'meta': C.meta()}, os.path.join(OUT, 'repro_ok.json'))
    else:
        while not os.path.exists(os.path.join(OUT, 'repro_ok.json')):
            if os.path.exists(os.path.join(OUT, 'STOPPED.json')):
                sys.exit(1)
            time.sleep(10)
    for i, j in enumerate(jobs()):
        if i % a.workers == a.worker:
            run(*j)
    print('worker %d DONE' % a.worker, flush=True)
