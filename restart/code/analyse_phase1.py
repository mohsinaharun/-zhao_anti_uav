# Gate G1b (restart/phase1/PREREG.md), epsilon table and the noise-sweep figure.
import os, sys, glob, json
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

K, B, CC, LR = 9, 256, 45.8, 0.03
if len(sys.argv) > 1:
    K, B, CC, LR = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
R = os.path.join(C.RESULTS, 'phase1')
SIGMAS = [0, 2, 4, 8, 16, 32]
triv = json.load(open(os.path.join(R, 'trivial.json')))['val_map50']
nodp = [json.load(open(f))['val_map50'] for f in glob.glob(os.path.join(R, 'nodp_K%d_lr*_seed*.json' % K))]

rows = {}
for s in SIGMAS:
    fs = sorted(glob.glob(os.path.join(R, 'dp_K%d_B%d_C%g_s%g_lr%g_seed*.json' % (K, B, CC, s, LR))))
    rs = [json.load(open(f)) for f in fs]
    rows[s] = {'map': [r['val_map50'] for r in rs], 'eps': rs[0]['epsilon_delta1e-5'] if rs else None,
               'q': rs[0].get('sample_rate') if rs else None, 'steps': rs[0]['steps'] if rs else None}

means = [np.mean(rows[s]['map']) for s in SIGMAS]
c1 = means[0] - means[-1] >= 0.10
rho = spearmanr(SIGMAS, means).correlation
c2 = rho <= -0.80
c3 = means[1] >= triv + 0.10
print('=' * 70)
print('GATE G1b (K=%d, B=%d, C=%g, lr=%g): %s' % (K, B, CC, LR, 'PASS' if c1 and c2 and c3 else 'FAIL'))
print('  (1) mAP(0) - mAP(32) = %.4f  (>= 0.10)  %s' % (means[0] - means[-1], 'ok' if c1 else 'FAIL'))
print('  (2) Spearman(sigma, mAP) = %.3f (<= -0.80) %s' % (rho, 'ok' if c2 else 'FAIL'))
print('  (3) mAP(2) = %.4f vs trivial+0.10 = %.4f  %s' % (means[1], triv + 0.10, 'ok' if c3 else 'FAIL'))
print('=' * 70)
print('%6s %8s %8s %24s %10s' % ('sigma', 'mean', 'sd', 'seeds', 'epsilon'))
for s, m in zip(SIGMAS, means):
    e = rows[s]['eps']
    print('%6g %8.4f %8.4f %24s %10s' % (s, m, np.std(rows[s]['map'], ddof=1), ' '.join('%.3f' % v for v in rows[s]['map']),
                                         'inf' if e in (None, float('inf')) else '%.2f' % e))
print('non-DP reference (SGD, no clipping): %.4f   trivial: %.4f   q=%.5f steps=%s'
      % (np.mean(nodp), triv, rows[2]['q'], rows[2]['steps']))

fig, ax = plt.subplots(figsize=(6.6, 4))
xs = np.arange(len(SIGMAS))
for x, s in zip(xs, SIGMAS):
    ax.scatter([x] * len(rows[s]['map']), rows[s]['map'], color='#2a78d6', alpha=0.45, s=18, lw=0)
ax.plot(xs, means, color='#2a78d6', lw=2, marker='o', ms=7, label='DP-SGD, mean of 3 seeds')
ax.axhline(np.mean(nodp), color='#52514e', lw=1, ls='--', label='non-DP (no clipping)')
ax.axhline(triv, color='#52514e', lw=1, ls=':', label='trivial predictor')
ax.set_xticks(xs)
ax.set_xticklabels(['σ=%g\nε=%s' % (s, '∞' if rows[s]['eps'] in (None, float('inf')) else '%.2g' % rows[s]['eps'])
                    for s in SIGMAS])
ax.set_ylabel('val mAP@0.5'); ax.set_ylim(0, max(0.7, max(means) + 0.05))
ax.set_title('Phase 1 noise sweep (K=%d, B=%d, C=%g; ε at δ=1e-5)' % (K, B, CC), loc='left', fontsize=10)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(axis='y', color='#e4e3df'); ax.legend(frameon=False, loc='lower left', fontsize=8)
fig.tight_layout()
os.makedirs(os.path.join(C.ROOT, 'restart', 'figures'), exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(C.ROOT, 'restart', 'figures', 'phase1_noise_sweep_K%d_B%d.%s' % (K, B, ext)), dpi=200)
