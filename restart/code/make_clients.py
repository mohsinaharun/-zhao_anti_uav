# Phase 2: assign the `clients` pool's sessions to 8 clients, 2 per environment
# (user decision 2026-10-05). Within an environment, sessions go largest-first to
# the lighter of its two clients (by target frames); ties broken by a seeded
# shuffle. Written once to restart/data/clients.json; refuses to overwrite.
import os, json, random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'restart', 'data', 'clients.json')
SEED = 20261006
if os.path.exists(OUT):
    raise SystemExit('%s exists: client assignment is frozen.' % OUT)

pool = json.load(open(os.path.join(ROOT, 'restart', 'data', 'split.json')))['pools']['clients']
rng = random.Random(SEED)
clients, cid = {}, 0
for env in ('building', 'cloud', 'sky', 'terrain'):
    ss = [s for s in pool if pool[s]['dominant_env'] == env]
    rng.shuffle(ss)
    ss.sort(key=lambda s: -pool[s]['target_frames'])          # stable: shuffle breaks ties
    a, b = 'c%d' % cid, 'c%d' % (cid + 1); cid += 2
    clients[a] = {'env': env, 'sessions': []}; clients[b] = {'env': env, 'sessions': []}
    load = {a: 0, b: 0}
    for s in ss:
        k = min(load, key=lambda c: (load[c], c))
        clients[k]['sessions'].append(s); load[k] += pool[s]['target_frames']
    for k in (a, b):
        clients[k]['target_frames'] = load[k]
folder_to_client = {f['folder']: c for c, v in clients.items() for s in v['sessions']
                    for f in pool[s]['folders']}
assert sorted(s for v in clients.values() for s in v['sessions']) == sorted(pool), 'session lost/duplicated'
json.dump({'seed': SEED, 'clients': clients, 'folder_to_client': folder_to_client}, open(OUT, 'w'), indent=1)
for c, v in clients.items():
    print('%s %-9s %2d sessions  %6d target frames' % (c, v['env'], len(v['sessions']), v['target_frames']))
print('-> %s' % OUT)
