# =============================================================================
# Phase 4b Part 1 (restart/phase4b/PREREG.md): per-round agent observations across the 20
# main-setting seeds at P under A1 / A3 / A5 schedules. Real clients; NO model evaluation.
#   python phase4b_variation.py            (resumable per arm/seed)
#   python phase4b_variation.py --analyse
# =============================================================================
import os, sys, json, argparse
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, budget, fl_env, phase4 as P4

OUT = os.path.join(C.RESULTS, 'phase4b')
SEEDS = range(200, 220)
ARMS = ('A1', 'A3', 'A5')
KEYS = ('proxy_loss', 'proxy_gnorm', 'proxy_dloss')


def run(d, arm, seed, pol, return_env=False):
    R, _ = P4.OPS['P']
    tgt = P4.target_eps('P', d['qs'])
    env = fl_env.FLEnv(d, seed); obs = [env.reset()]
    if pol is None:
        for s in P4.schedule_for(arm, 'P', d['qs'])[0]:
            obs.append(env.step(s))
    else:
        pol.reset()
        for t in range(R):
            spent = budget.eps_worst(env.sigmas, d['qs']) / tgt if env.sigmas else 0.0
            share = 0.25 * 16 ** float(pol.act(obs, t, R, spent))
            obs.append(env.step(budget.sigma_for_round(env.sigmas, share, R - t, tgt, d['qs'])))
    rec = {'arm': arm, 'seed': seed, 'sigmas': env.sigmas, 'C_star': env.C_star,
           'obs': [{k: float(o[k]) for k in KEYS} for o in obs], 'meta': C.meta()}
    return (rec, env) if return_env else rec


def analyse():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res, fig = {}, plt.figure(figsize=(15, 4.2))
    for i, arm in enumerate(ARMS):
        L = np.array([[o['proxy_loss'] for o in json.load(open(os.path.join(OUT, 'var_%s_seed%d.json' % (arm, s))))['obs']]
                      for s in SEEDS])                                     # seeds x (R+1)
        sd, mu = L.std(0, ddof=1), L.mean(0)
        prog = mu[0] - mu[-1]
        rho = [float(stats.spearmanr(L[:, t], L[:, -1])[0]) for t in range(L.shape[1])]
        rho_next = [float(stats.spearmanr(L[:, t], L[:, t + 1])[0]) for t in range(L.shape[1] - 1)]
        crit = [(t, sd[t] / prog, rho[t]) for t in range(1, 11) if sd[t] / prog >= 0.10 and rho[t] >= 0.5]
        res[arm] = {'mean': mu.tolist(), 'sd': sd.tolist(), 'min': L.min(0).tolist(), 'max': L.max(0).tolist(),
                    'progress': float(prog), 'sd_over_progress': (sd / prog).tolist(),
                    'spearman_vs_final': rho, 'spearman_vs_next': rho_next,
                    'criterion_rounds': crit, 'meaningful': bool(crit)}
        print('\n%s  total progress (round0 - round12 mean loss) %.4f' % (arm, prog))
        print(' t   mean     sd      sd/prog  min     max     rho(t,12)')
        for t in range(L.shape[1]):
            print('%2d  %.4f  %.4f  %5.1f%%  %.4f  %.4f  %+.2f' % (t, mu[t], sd[t], 100 * sd[t] / prog, L[:, t].min(), L[:, t].max(), rho[t]))
        print('criterion (sd/prog >= 10%% and rho >= 0.5, t in 1..10): %s' % ('MET at rounds %s' % [c[0] for c in crit] if crit else 'NOT met'))
        ax = fig.add_subplot(1, 3, i + 1)
        for row in L:
            ax.plot(range(L.shape[1]), row, color='#1f77b4', alpha=0.35, lw=1)
        ax.plot(range(L.shape[1]), mu, color='#d62728', lw=2, label='mean')
        ax.fill_between(range(L.shape[1]), mu - sd, mu + sd, color='#d62728', alpha=0.15, label='±1 sd')
        ax.set_title('%s at P: proxy loss, 20 seeds' % arm); ax.set_xlabel('round'); ax.set_ylabel('proxy loss')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.ROOT, 'plots', 'phase4b_variation.png'), dpi=150)
    res['decision'] = {'judged_on': 'A3', 'meaningful': res['A3']['meaningful']}
    res['meta'] = C.meta()
    C.save_json(res, os.path.join(OUT, 'variation_analysis.json'))
    print('\nDECISION (judged on A3): %s' % ('meaningful variation -> run Part 2' if res['A3']['meaningful'] else 'no exploitable variation -> Part 2 not run'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--analyse', action='store_true'); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.analyse:
        analyse(); sys.exit(0)
    d = fl_env.build_data('real', 'val')                # eval pool unused: nothing is evaluated
    import phase4_agents as AG
    for arm in ARMS:
        pol = AG.load_policy(os.path.join(C.RESULTS, 'phase4', 'agent_A5_P_best.pt'), 'A5') if arm == 'A5' else None
        for s in SEEDS:
            f = os.path.join(OUT, 'var_%s_seed%d.json' % (arm, s))
            if os.path.exists(f):
                continue
            rec = run(d, arm, s, pol)
            C.save_json(rec, f)
            print('%s seed %d: proxy loss %s' % (arm, s, ' '.join('%.3f' % o['proxy_loss'] for o in rec['obs'])), flush=True)
