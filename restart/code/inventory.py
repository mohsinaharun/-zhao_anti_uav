# =============================================================================
# Phase 0 data census. Reads only folder names and IR_label.json files; trains
# nothing. Output: restart/data/inventory.csv (one row per folder) and
# restart/data/sessions.csv (one row per recording session).
#
# SESSION RULE (restart/phase0/PHASE0.md §3): folders that are segments of one
# continuous recording share a session id, and a session is never split.
#   YYYYMMDD_HHMMSS_a_b        -> YYYYMMDD_HHMMSS
#   3700000000002_HHMMSS_n     -> 3700000000002_HHMMSS        (camera _ time)
#   NN_VVVV_xxxx-yyyy          -> NN_VVVV                      (video + frame range)
#   wg2022_ir_NNN_split_XX     -> wg2022_ir_NNN
#   newNN_*                    -> newNN
# Anything else raises, so a new naming scheme cannot slip through silently.
# =============================================================================
import os, re, csv, json, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, '3rd_Anti-UAV_train_val')
OUT = os.path.join(ROOT, 'restart', 'data')

# Clear-sky folders, labelled by the user 2026-10-05.
CLEAR = set("""20190925_101846_1_3 20190925_111757_1_1 20190925_111757_1_2 20190925_111757_1_8
20190925_124612_1_1 20190925_124612_1_2 20190925_130434_1_3 20190925_130434_1_4
20190925_131530_1_7 20190925_134301_1_2 20190925_134301_1_7 20190925_140917_1_1
20190925_140917_1_8 20190925_141417_1_7 20190925_143900_1_1 20190925_143900_1_6
20190925_183946_1_2 20190925_183946_1_8 20190925_193610_1_1 20190925_193610_1_2
20190925_193610_1_3 20190925_205804_1_5 20190925_205804_1_6 20190926_095902_1_1
20190926_102042_1_6 20190926_111509_1_7 20190926_143632_1_3 20190926_143632_1_7
20190926_193515_1_8""".split())

RULES = [(re.compile(r'^(\d{8}_\d{6})_\d+_\d+$'), 1),
         (re.compile(r'^(3700000000\d{3}_\d{6})_\d+$'), 1),
         (re.compile(r'^(\d{2}_\d{4})_\d{4}[-_]\d{4}$'), 1),
         (re.compile(r'^(wg2022_ir_\d{3})_split_\d{2}$'), 1),
         (re.compile(r'^(new\d+)_.*$'), 1)]


def session_of(folder):
    for rx, g in RULES:
        m = rx.match(folder)
        if m:
            return m.group(g)
    raise ValueError('no session rule for folder %r' % folder)


rows = []
for split in ('train', 'validation'):
    for f in sorted(os.listdir(os.path.join(DATA, split))):
        d = os.path.join(DATA, split, f)
        if not os.path.isdir(d):
            continue
        n_img = sum(1 for x in os.listdir(d) if x.endswith('.jpg'))
        lp = os.path.join(d, 'IR_label.json')
        n_tgt, areas = 0, []
        if os.path.exists(lp):
            lab = json.load(open(lp))
            for gt, ex in zip(lab.get('gt_rect', []), lab.get('exist', [])):
                if ex and gt and len(gt) == 4 and gt[2] > 0 and gt[3] > 0:
                    n_tgt += 1
                    areas.append(gt[2] * gt[3])
        rows.append({'folder': f, 'official_split': split, 'session': session_of(f),
                     'frames': n_img, 'frames_with_target': n_tgt,
                     'median_box_px2': round(st.median(areas)) if areas else '',
                     'has_labels': os.path.exists(lp),
                     'environment': 'clear_sky' if f in CLEAR else 'unlabelled'})

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'inventory.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

sess = {}
for r in rows:
    s = sess.setdefault(r['session'], {'session': r['session'], 'folders': 0, 'frames': 0,
                                       'frames_with_target': 0, 'official_splits': set(),
                                       'environments': set()})
    s['folders'] += 1; s['frames'] += r['frames']; s['frames_with_target'] += r['frames_with_target']
    s['official_splits'].add(r['official_split']); s['environments'].add(r['environment'])
with open(os.path.join(OUT, 'sessions.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['session', 'folders', 'frames', 'frames_with_target',
                                       'official_splits', 'environments'])
    w.writeheader()
    for s in sess.values():
        w.writerow({**s, 'official_splits': '+'.join(sorted(s['official_splits'])),
                    'environments': '+'.join(sorted(s['environments']))})

leaky = [s for s in sess.values() if len(s['official_splits']) > 1]
mixed = [s for s in sess.values() if len(s['environments']) > 1]
print('folders %d  sessions %d  frames %d  frames-with-target %d'
      % (len(rows), len(sess), sum(r['frames'] for r in rows),
         sum(r['frames_with_target'] for r in rows)))
print('clear-sky: %d folders in %d sessions' % (
    sum(r['environment'] == 'clear_sky' for r in rows),
    sum('clear_sky' in s['environments'] for s in sess.values())))
print('sessions spanning BOTH official splits (official split leaks): %d -> %s'
      % (len(leaky), ', '.join(sorted(s['session'] for s in leaky))))
print('sessions with mixed environment labels: %d -> %s'
      % (len(mixed), ', '.join(sorted(s['session'] for s in mixed))))
print('folders without IR_label.json: %d' % sum(not r['has_labels'] for r in rows))
