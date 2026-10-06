# Phase 1 Amendment 2 decision: single (S) vs causal 3-frame (T) input.
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R = os.path.join(C.RESULTS, 'phase1')
ld = lambda pat: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, pat)))]
best_T = max([0.003, 0.01, 0.03], key=lambda l: json.load(open(os.path.join(R, 'nodp_K9_lr%g_seed0_T.json' % l)))['val_map50'])
CT = round(json.load(open(os.path.join(R, 'measureC_K9_T.json')))['per_sample_grad_norm']['median'], 1)

S_nodp = [r['val_map50'] for r in ld('nodp_K9_lr0.03_seed[0-2].json')]
T_nodp = [r['val_map50'] for r in ld('nodp_K9_lr%g_seed[0-2]_T.json' % best_T)]
S_dp = [r['val_map50'] for r in ld('dp_K9_B1024_C45.8_s2_lr0.03_seed[0-2].json')]
T_dp = [r['val_map50'] for r in ld('dp_K9_B1024_C%g_s2_lr0.03_seed[0-2]_T.json' % CT)]
assert len(S_nodp) == len(T_nodp) == len(S_dp) == len(T_dp) == 3, (S_nodp, T_nodp, S_dp, T_dp)

d_nodp = np.mean(T_nodp) - np.mean(S_nodp)
d_dp = np.mean(T_dp) - np.mean(S_dp)
a_ok, b_ok = d_nodp >= 0.03, d_dp >= -0.02
print('=' * 72)
print('AMENDMENT 2 DECISION: %s' % ('ADOPT T (3-frame input)' if a_ok and b_ok else 'KEEP S (single frame)'))
print('  (a) non-DP  T - S = %+.4f (need >= +0.03)  %s' % (d_nodp, 'ok' if a_ok else 'FAIL'))
print('  (b) DP s=2  T - S = %+.4f (need >= -0.02)  %s' % (d_dp, 'ok' if b_ok else 'FAIL'))
print('=' * 72)
print('non-DP  S (lr 0.03): %s  mean %.4f' % (' '.join('%.3f' % v for v in S_nodp), np.mean(S_nodp)))
print('non-DP  T (lr %g):  %s  mean %.4f' % (best_T, ' '.join('%.3f' % v for v in T_nodp), np.mean(T_nodp)))
print('DP s=2  S (C 45.8):  %s  mean %.4f' % (' '.join('%.3f' % v for v in S_dp), np.mean(S_dp)))
print('DP s=2  T (C %g):  %s  mean %.4f' % (CT, ' '.join('%.3f' % v for v in T_dp), np.mean(T_dp)))
print('paired per-seed differences T-S: non-DP %s | DP %s'
      % (' '.join('%+.3f' % (t - s) for t, s in zip(T_nodp, S_nodp)),
         ' '.join('%+.3f' % (t - s) for t, s in zip(T_dp, S_dp))))
