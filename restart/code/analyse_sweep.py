# Granular sigma sweep table: mAP mean / sd / 95% CI over paired seeds, and epsilon.
import os, sys, glob, json, argparse
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--pattern', required=True, help="phase1 JSON name with {s} for sigma and {seed}")
ap.add_argument('--sigmas', nargs='+', type=float, default=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
ap.add_argument('--seeds', type=int, default=10)
ap.add_argument('--label', required=True)
ap.add_argument('--dir', default='phase1')
a = ap.parse_args()
R = os.path.join(C.RESULTS, a.dir)
rows = []
print('%-6s %8s %7s %18s %8s %4s' % ('sigma', 'mAP', 'sd', '95% CI', 'epsilon', 'n'))
for s in a.sigmas:
    rs = []
    for seed in range(a.seeds):
        f = os.path.join(R, a.pattern.format(s='%g' % s, seed=seed) + '.json')
        if os.path.exists(f):
            rs.append(json.load(open(f)))
    m = np.array([r['val_map50'] for r in rs])
    eps = rs[0].get('epsilon_delta1e-5', rs[0].get('epsilon_max')) if rs else None
    h = stats.t.ppf(0.975, len(m) - 1) * m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float('nan')
    rows.append((s, m.mean(), m.std(ddof=1) if len(m) > 1 else 0, h, eps, len(m), m))
    print('%-6g %8.4f %7.4f  [%6.4f, %6.4f] %8s %4d' % (s, m.mean(), rows[-1][2], m.mean() - h, m.mean() + h,
          'inf' if eps in (None, float('inf')) else '%.2f' % eps, len(m)))
json.dump([{'sigma': r[0], 'map_mean': r[1], 'map_sd': r[2], 'ci95_half': r[3], 'epsilon': r[4], 'n': r[5],
            'per_seed': r[6].tolist()} for r in rows],
          open(os.path.join(R, 'sweep_%s.json' % a.label), 'w'), indent=1)
fig, ax = plt.subplots(figsize=(7, 4))
xs = np.arange(len(rows))
for x, r in zip(xs, rows):
    ax.scatter([x] * len(r[6]), r[6], color='#2a78d6', alpha=0.35, s=12, lw=0)
ax.errorbar(xs, [r[1] for r in rows], yerr=[r[3] for r in rows], color='#2a78d6', lw=2, marker='o', ms=6, capsize=3,
            label='mean, 95% CI (%d seeds)' % a.seeds)
ax.set_xticks(xs); ax.set_xticklabels(['σ=%g\nε=%s' % (r[0], '∞' if r[4] in (None, float('inf')) else '%.2g' % r[4]) for r in rows], fontsize=8)
ax.set_ylabel('val mAP@0.5'); ax.set_ylim(0, 0.7); ax.grid(axis='y', color='#e4e3df')
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.set_title(a.label, loc='left', fontsize=10); ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(C.ROOT, 'restart', 'figures', 'sweep_%s.%s' % (a.label, ext)), dpi=200)
