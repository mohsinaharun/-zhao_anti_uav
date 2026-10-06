# =============================================================================
# Phase 0 §5: assign every recording SESSION to exactly one pool, stratified by
# the session's dominant environment. Written ONCE to restart/data/split.json and
# never regenerated after Phase 1 starts (refuses to overwrite).
#
# Within each environment, sessions are shuffled (seeded) and each goes to the
# pool currently furthest below its target share of TARGET FRAMES. Each
# environment is guaranteed >= 1 session in test, val and clients.
# Environment labels only steer stratification here; client grouping happens in
# Phase 2, so later label corrections move sessions between clients, never
# between train and test.
# =============================================================================
import os, csv, json, random, subprocess, collections as C

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'restart', 'data', 'split.json')
SEED = 20261005
SHARES = {'test': 0.15, 'val': 0.10, 'root': 0.05, 'proxy': 0.15, 'clients': 0.55}
MUST = ('clients', 'test', 'val')            # every environment appears in these

if os.path.exists(OUT):
    raise SystemExit('%s exists: the split is frozen. Delete it only with a dated '
                     'note in PHASE0.md.' % OUT)

inv = list(csv.DictReader(open(os.path.join(ROOT, 'restart', 'data', 'inventory_labelled.csv'))))
sess = C.defaultdict(lambda: {'folders': [], 'env': C.Counter(), 'tgt': 0})
for r in inv:
    s = sess[r['session']]
    s['folders'].append({'folder': r['folder'], 'official_split': r['official_split']})
    s['env'][r['environment']] += int(r['frames_with_target'])
    s['tgt'] += int(r['frames_with_target'])
dom = {k: v['env'].most_common(1)[0][0] for k, v in sess.items()}

rng = random.Random(SEED)
pools = {p: [] for p in SHARES}
for env in sorted(set(dom.values())):
    ss = sorted(k for k in sess if dom[k] == env)
    rng.shuffle(ss)
    tot = sum(sess[k]['tgt'] for k in ss)
    got = {p: 0 for p in SHARES}
    queue = list(MUST)                        # seed one session into each must-pool
    for k in ss:
        if queue:
            p = queue.pop(0)
        else:
            p = min(SHARES, key=lambda q: got[q] / tot - SHARES[q])
        pools[p].append(k); got[p] += sess[k]['tgt']

commit = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'],
                        capture_output=True, text=True).stdout.strip()
summary = {p: {env: sum(sess[k]['tgt'] for k in ks if dom[k] == env)
               for env in sorted(set(dom.values()))} for p, ks in pools.items()}
out = {'seed': SEED, 'shares_target': SHARES, 'code_commit': commit,
       'note': 'session-disjoint pools; stratified by dominant environment (proposed labels)',
       'pools': {p: {k: {'dominant_env': dom[k], 'target_frames': sess[k]['tgt'],
                         'folders': sess[k]['folders']} for k in sorted(ks)}
                 for p, ks in pools.items()},
       'target_frames_by_pool_env': summary}
json.dump(out, open(OUT, 'w'), indent=1)

all_t = sum(v['tgt'] for v in sess.values())
print('%-8s %8s %7s  %s' % ('pool', 'sessions', 'share', 'target frames by env'))
for p, ks in pools.items():
    t = sum(sess[k]['tgt'] for k in ks)
    print('%-8s %8d %6.1f%%  %s' % (p, len(ks), 100 * t / all_t, summary[p]))
seen = [k for ks in pools.values() for k in ks]
assert len(seen) == len(set(seen)) == len(sess), 'a session is in two pools or missing'
print('OK: %d sessions, each in exactly one pool -> %s' % (len(sess), OUT))
