# =============================================================================
# Phase 4d (restart/phase4d/PREREG.md): signal-redesign check. Replays the 20 main-setting
# seeds at P under A1 / A3 / A5, logging the server's proxy gradient norm g_t each round and
# the final model's VAL mAP@0.5 (no test data).
#   python phase4d_signal.py            (resumable)
#   python phase4d_signal.py --analyse
# =============================================================================
import os, sys, json, argparse
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, fl_env, phase4b_variation as V

OUT = os.path.join(C.RESULTS, 'phase4d')
SEEDS, ARMS, R = list(range(200, 220)), ('A1', 'A3', 'A5'), 12


def rolling(g, w=3):
    return np.array([np.mean(g[max(0, t - w + 1):t + 1]) for t in range(len(g))])


def detrend(x):                                 # x_t - (x_t-1 + x_t+1)/2 for t = 2..11 (1-based rounds)
    return np.array([x[t] - (x[t - 1] + x[t + 1]) / 2 for t in range(2, R)])


def lockstep(sig_rows, sigmas_rows):
    rs = []
    for g, sg in zip(sig_rows, sigmas_rows):
        dose = np.r_[np.nan, sg]                 # dose[t] = sigma of round t (t = 1..12); index 0 unused
        e, d = detrend(g), detrend(dose)
        rs.append(np.nan if np.std(d) < 1e-9 else float(np.corrcoef(e, d)[0, 1]))
    rs = np.array(rs)
    if np.isnan(rs).all():
        return {'mean_r': None, 'ci95': None, 'lockstep': None, 'note': 'no round-to-round dose variation'}
    m = float(np.nanmean(rs)); n = int(np.sum(~np.isnan(rs)))
    h = stats.t.ppf(0.975, n - 1) * np.nanstd(rs, ddof=1) / np.sqrt(n)
    lo, hi = m - h, m + h
    return {'mean_r': m, 'ci95': [float(lo), float(hi)], 'lockstep': bool((lo > 0 or hi < 0) and abs(m) >= 0.3)}


def analyse():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = {}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, arm in zip(axs, ARMS):
        recs = [json.load(open(os.path.join(OUT, 'sig_%s_seed%d.json' % (arm, s)))) for s in SEEDS]
        G = np.array([[o['proxy_gnorm'] for o in r['obs']] for r in recs])         # seeds x 13
        L = np.array([[o['proxy_loss'] for o in r['obs']] for r in recs])
        S = np.array([rolling(g) for g in G])
        val = np.array([r['val_map50'] for r in recs])
        sig = [r['sigmas'] for r in recs]
        prog = S[:, 0].mean() - S[:, -1].mean()
        sd = S.std(0, ddof=1)
        rho_out = [float(stats.spearmanr(S[:, t], val)[0]) for t in range(R + 1)]
        rho_fin = [float(stats.spearmanr(S[:, t], S[:, -1])[0]) for t in range(R + 1)]
        early = [rho_out[t] for t in range(1, 9)]
        pos = sum(r >= 0.5 for r in early); neg = sum(r <= -0.5 for r in early)
        a_ok = pos >= 2 or neg >= 2
        b_raw, b_roll, b_loss = lockstep(G, sig), lockstep(S, sig), lockstep(L, sig)
        res[arm] = {'progress': float(prog), 'sd_over_progress': (sd / prog).tolist(),
                    'max_spread_1_10': float(max(sd[1:11] / prog)), 'spearman_vs_final_val': rho_out,
                    'spearman_vs_final_signal': rho_fin, 'val_map_mean': float(val.mean()),
                    'a_met': bool(a_ok), 'a_rounds_pos': pos, 'a_rounds_neg': neg,
                    'lockstep_raw_gnorm': b_raw, 'lockstep_rolling': b_roll, 'lockstep_old_proxy_loss': b_loss}
        print('\n%s  (val mAP mean %.4f; progress s0-s12 = %.3f)' % (arm, val.mean(), prog))
        print(' t   mean s_t   sd     sd/prog  rho(s_t, final val)  rho(s_t, s_12)')
        for t in range(R + 1):
            print('%2d  %8.3f  %6.3f  %6.1f%%   %+.2f                %+.2f' % (t, S[:, t].mean(), sd[t], 100 * sd[t] / prog, rho_out[t], rho_fin[t]))
        print('(a) |rho|>=0.5 same sign in >=2 of rounds 1-8: %s (pos %d, neg %d)' % (a_ok, pos, neg))
        for k, b in (('raw g_t', b_raw), ('rolling s_t', b_roll), ('old proxy loss (control)', b_loss)):
            print('(b) lockstep, %-26s mean r %s  CI %s  -> %s' % (k, 'n/a' if b['mean_r'] is None else '%+.2f' % b['mean_r'],
                  b['ci95'] and np.round(b['ci95'], 2).tolist(), 'n/a' if b['lockstep'] is None else ('LOCKSTEP' if b['lockstep'] else 'decoupled')))
        for row in S:
            ax.plot(range(R + 1), row, color='#1f77b4', alpha=0.35, lw=1)
        ax.plot(range(R + 1), S.mean(0), color='#d62728', lw=2, label='mean')
        ax2 = ax.twinx(); ax2.step(range(1, R + 1), np.mean(sig, 0), where='mid', color='#2ca02c', alpha=0.6, label='sigma_t')
        ax2.set_ylabel('sigma_t', color='#2ca02c')
        ax.set_yscale('log'); ax.set_title('%s: rolling proxy grad-norm s_t, 20 seeds' % arm)
        ax.set_xlabel('round'); ax.set_ylabel('s_t (3-round mean)'); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, 'plots', 'phase4d_signal.png'), dpi=150)
    a3 = res['A3']
    b_ok = a3['lockstep_raw_gnorm']['lockstep'] is False
    res['decision'] = {'judged_on': 'A3', 'a': a3['a_met'], 'b': b_ok, 'train_A5v2': bool(a3['a_met'] and b_ok)}
    res['meta'] = C.meta()
    C.save_json(res, os.path.join(OUT, 'analysis.json'))
    print('\nDECISION (A3): (a) %s, (b) %s -> %s' % (a3['a_met'], b_ok,
          'train A5-v2 (after user review)' if res['decision']['train_A5v2'] else 'do NOT train A5-v2'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--analyse', action='store_true'); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.analyse:
        analyse(); sys.exit(0)
    d = fl_env.build_data('real', 'val')
    import phase4_agents as AG
    for arm in ARMS:
        pol = AG.load_policy(os.path.join(C.RESULTS, 'phase4', 'agent_A5_P_best.pt'), 'A5') if arm == 'A5' else None
        for s in SEEDS:
            f = os.path.join(OUT, 'sig_%s_seed%d.json' % (arm, s))
            if os.path.exists(f):
                continue
            rec, env = V.run(d, arm, s, pol, return_env=True)
            rec['val_map50'] = env.evaluate()
            old = json.load(open(os.path.join(C.RESULTS, 'phase4b', 'var_%s_seed%d.json' % (arm, s))))
            rec['identical_to_phase4b'] = old['sigmas'] == rec['sigmas'] and old['obs'] == rec['obs']
            C.save_json(rec, f)
            print('%s seed %d: val mAP %.4f  g_t %s  same-as-4b %s' % (arm, s, rec['val_map50'],
                  ' '.join('%.2f' % o['proxy_gnorm'] for o in rec['obs']), rec['identical_to_phase4b']), flush=True)
