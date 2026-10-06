# Step 4: FedProx mu sweep (new placement) vs no-FedProx baseline, 10 paired seeds.
import os, sys, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

O = os.path.join(C.RESULTS, 'phase2')
MUS = ['0', '0.01', '0.03', '0.1', '0.3', '1']
for s in ('0', '0.5'):
    R = {mu: [json.load(open(os.path.join(O, 'fl_dp_B64_s%s_lr0.03_mu%s_R12_it20_proxv2_seed%d.json' % (s, mu, k))))
              for k in range(10)] for mu in MUS}
    M = {mu: np.array([r['val_map50'] for r in R[mu]]) for mu in MUS}
    e = R['0'][0]
    eps = 'inf' if s == '0' else 'max %.2f / median %.2f' % (e['epsilon_max'], e['epsilon_median'])
    same_eps = all(R[mu][0].get('epsilon_max') == e.get('epsilon_max') for mu in MUS)
    print('\nsigma = %s   epsilon (delta=1e-5): %s   (identical for every mu: %s)' % (s, eps, same_eps))
    print('%-8s %8s %7s %26s %6s %8s' % ('mu', 'mAP', 'sd', 'diff vs mu=0 [95% CI]', 'wins', 'p'))
    for mu in MUS:
        m = M[mu]
        if mu == '0':
            print('%-8s %8.4f %7.4f %26s %6s %8s' % ('0 (base)', m.mean(), m.std(ddof=1), '-', '-', '-'))
            continue
        d = m - M['0']
        h = stats.t.ppf(0.975, 9) * d.std(ddof=1) / np.sqrt(10)
        p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else float('nan')
        print('%-8s %8.4f %7.4f   %+.4f [%+.4f, %+.4f] %3d/10 %8.4f' % (mu, m.mean(), m.std(ddof=1), d.mean(),
              d.mean() - h, d.mean() + h, int((d > 0).sum()), p))
