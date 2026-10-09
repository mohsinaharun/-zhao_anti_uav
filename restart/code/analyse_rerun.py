# Re-run analysis (restart/rerun/PREREG.md): R1 central sigma sweep with C*, R2 FL x DP
# 2x2 at the edge config (sigma 1, 2, 4), R3 FedProx mu sweep (sigma 1, 2). 10 paired seeds.
import os, sys, json
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RR, P1, P2 = (os.path.join(C.RESULTS, d) for d in ('rerun', 'phase1', 'phase2'))
SEEDS = range(10)
ld = lambda d, name: json.load(open(os.path.join(d, name + '.json')))
vals = lambda d, pat: np.array([ld(d, pat % k)['val_map50'] for k in SEEDS])


def ci(x):
    x = np.asarray(x, float)
    if x.std() == 0:
        return x.mean(), x.mean(), x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def eps_of(r):
    e = r.get('epsilon_max', r.get('epsilon_delta1e-5'))
    return 'inf' if e in (None, float('inf')) else '%.2f' % e


# ── R1 ────────────────────────────────────────────────────────────────────────
SIG = ['0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4']
print('R1 — central sigma sweep, B=64, calibrated C* (10 seeds)')
print('%-5s %9s %18s %8s %7s | %13s %15s' % ('sigma', 'mAP C*', '95% CI', 'eps', 'C*', 'old B64 C45.8', 'old B1024 C45.8'))
r1 = {}
for s in SIG:
    new = vals(RR, 'central_dp_B64_s%s_lr0.03_mu0_R12_clipCal_seed%%d' % s)
    r = ld(RR, 'central_dp_B64_s%s_lr0.03_mu0_R12_clipCal_seed0' % s)
    o64 = vals(P1, 'dp_K9_B64_C45.8_s%s_lr0.03_seed%%d' % s)
    o1024 = vals(P1, 'dp_K9_B1024_C45.8_s%s_lr0.03_seed%%d' % s)
    m, lo, hi = ci(new); r1[s] = new
    cs = np.mean([ld(RR, 'central_dp_B64_s%s_lr0.03_mu0_R12_clipCal_seed%d' % (s, k))['C_star'] for k in SEEDS])
    print('%-5s %9.4f  [%.4f, %.4f] %8s %7.2f | %13.4f %15.4f' % (s, m, lo, hi, eps_of(r), cs, o64.mean(), o1024.mean()))

# ── R2 ────────────────────────────────────────────────────────────────────────
print('\nR2 — FL x DP 2x2, edge config (B=64; FL: 8 clients, 20 local iters, mu=0), C* (10 seeds)')
nofl_nodp = vals(RR, 'central_nodp_B64_s0_lr0.03_mu0_R12_seed%d')
fl_nodp = vals(RR, 'fl_nodp_B64_s0_lr0.03_mu0_R12_it20_seed%d')
print('  no-DP cells: central %.4f, FL %.4f' % (nofl_nodp.mean(), fl_nodp.mean()))
for s in ('1', '2', '4'):
    fl_dp = vals(P2, 'fl_dp_B64_s%s_lr0.03_mu0_R12_it20_clipCal_seed%%d' % s)
    c_dp = r1[s]
    fe = ld(P2, 'fl_dp_B64_s%s_lr0.03_mu0_R12_it20_clipCal_seed0' % s)
    ce = ld(RR, 'central_dp_B64_s%s_lr0.03_mu0_R12_clipCal_seed0' % s)
    print('\n  sigma %s   central eps %s | FL client eps max %.2f / median %.2f' % (s, eps_of(ce), fe['epsilon_max'], fe['epsilon_median']))
    print('    %-26s %.4f' % ('neither (central, no DP)', nofl_nodp.mean()))
    print('    %-26s %.4f' % ('FL, no DP', fl_nodp.mean()))
    print('    %-26s %.4f' % ('DP, no FL (central)', c_dp.mean()))
    print('    %-26s %.4f' % ('FL + DP', fl_dp.mean()))
    for name, d in (('cost of FL (no DP)', fl_nodp - nofl_nodp), ('cost of DP (central)', c_dp - nofl_nodp),
                    ('cost of DP inside FL', fl_dp - fl_nodp),
                    ('interaction', (fl_dp - fl_nodp) - (c_dp - nofl_nodp))):
        m, lo, hi = ci(d)
        print('    %-26s %+.4f [%+.4f, %+.4f]  wins %d/10' % (name, m, lo, hi, int((d > 0).sum())))

# ── R3 ────────────────────────────────────────────────────────────────────────
print('\nR3 — FedProx mu sweep, FL+DP edge config, C* (10 seeds)')
for s in ('1', '2'):
    base = vals(P2, 'fl_dp_B64_s%s_lr0.03_mu0_R12_it20_clipCal_seed%%d' % s)
    e = ld(P2, 'fl_dp_B64_s%s_lr0.03_mu0_R12_it20_clipCal_seed0' % s)
    print('  sigma %s (eps max %.2f / median %.2f, identical for every mu)' % (s, e['epsilon_max'], e['epsilon_median']))
    print('    mu=0 (base)  %.4f' % base.mean())
    for mu in ('0.01', '0.03', '0.1', '0.3', '1'):
        v = vals(RR, 'fl_dp_B64_s%s_lr0.03_mu%s_R12_it20_clipCal_seed%%d' % (s, mu))
        m, lo, hi = ci(v - base)
        print('    mu=%-8s %.4f   diff %+.4f [%+.4f, %+.4f]  wins %d/10' % (mu, v.mean(), m, lo, hi, int((v - base > 0).sum())))

# figure: R1 comparison
fig, ax = plt.subplots(figsize=(7, 4.2))
xs = [float(s) for s in SIG]
for lab, col, mk, ser in (('central, B=64, calibrated C*', '#1baf7a', '^', [r1[s] for s in SIG]),
                          ('central, B=64, C = 45.8 (old)', '#eb6834', 's', [vals(P1, 'dp_K9_B64_C45.8_s%s_lr0.03_seed%%d' % s) for s in SIG]),
                          ('central, B=1,024, C = 45.8 (Step 1)', '#2a78d6', 'o', [vals(P1, 'dp_K9_B1024_C45.8_s%s_lr0.03_seed%%d' % s) for s in SIG])):
    ax.errorbar(xs, [v.mean() for v in ser], yerr=[ci(v)[2] - v.mean() for v in ser], color=col, marker=mk,
                ms=5, lw=2, capsize=2, label=lab)
ax.set_xlabel('noise multiplier σ'); ax.set_ylabel('val mAP@0.5 (mean, 95% CI, 10 seeds)')
ax.set_title('R1: central noise sweep with calibrated clipping', loc='left', fontsize=10)
ax.grid(axis='y', color='#e4e3df'); ax.legend(frameon=False, fontsize=8)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(C.ROOT, 'restart', 'figures', 'rerun_R1_central.%s' % ext), dpi=200)
