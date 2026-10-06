# Error inspection for a saved Phase 1 model (diagnostic, not a thesis result).
# Classifies every val frame by its TOP-1 detection:
#   hit         target present, top-1 IoU >= 0.5
#   mislocated  target present, top-1 near it (centre within 2 cells) but IoU < 0.5
#   wrong_place target present, top-1 somewhere else (clutter / distractor)
#   false_alarm no target in the frame, top-1 score >= 0.3
# and draws examples (green = ground truth, red = top-1 with score).
import os, sys, json, argparse, random
import numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='diag_nodp_K9_lr0.03_seed0')
ap.add_argument('--K', type=int, default=9)
ap.add_argument('--input', default='single', choices=['single', 'temporal'])
a = ap.parse_args()

va = torch.load(os.path.join(C.CACHE, 'K%d_s5%s_val.pt' % (a.K, '_T' if a.input == 'temporal' else '')), weights_only=False)
net = C.Net(a.K).to(C.DEV)
net.load_state_dict(torch.load(os.path.join(C.CACHE, 'models', a.model + '.pt')))
net.eval()
X = va['feats']
bx, sc = [], []
with torch.no_grad():
    for s in range(0, X.shape[0], 512):
        b, p = C.decode(net(X[s:s + 512].to(C.DEV).float()), k=1)
        bx += b[:, 0].cpu().tolist(); sc += p[:, 0].cpu().tolist()


def iou(a_, b_):
    ax, ay, aw, ah = a_; bx_, by, bw, bh = b_
    iw = max(0, min(ax + aw, bx_ + bw) - max(ax, bx_)); ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    i = iw * ih
    return i / (aw * ah + bw * bh - i + 1e-9)


cat = []
for g, b, s in zip(va['boxes'], bx, sc):
    if g is None:
        cat.append('false_alarm' if s >= 0.3 else 'quiet_negative')
    else:
        d = np.hypot((g[0] + g[2] / 2) - (b[0] + b[2] / 2), (g[1] + g[3] / 2) - (b[1] + b[3] / 2))
        cat.append('hit' if iou(g, b) >= 0.5 else 'mislocated' if d <= 2 * C.STRIDE else 'wrong_place')
cnt = {k: cat.count(k) for k in ('hit', 'mislocated', 'wrong_place', 'false_alarm', 'quiet_negative')}
pos = sum(g is not None for g in va['boxes'])
print('val frames %d (%d with a drone, %d without)' % (len(cat), pos, len(cat) - pos))
for k in ('hit', 'mislocated', 'wrong_place'):
    print('  %-12s %5d  (%.1f%% of frames with a drone)' % (k, cnt[k], 100 * cnt[k] / pos))
print('  %-12s %5d  (%.1f%% of frames without a drone)' % ('false_alarm', cnt['false_alarm'],
      100 * cnt['false_alarm'] / max(1, len(cat) - pos)))
# by session environment
sess_env = {f['folder']: v['dominant_env'] for v in C.SPLIT['pools']['val'].values() for f in v['folders']}
by = {}
for p, c in zip(va['paths'], cat):
    e = sess_env[os.path.basename(os.path.dirname(p))]
    by.setdefault(e, []).append(c)
print('top-1 hit rate by environment (frames with a drone):')
for e, cs in sorted(by.items()):
    n = sum(c in ('hit', 'mislocated', 'wrong_place') for c in cs)
    print('  %-9s %5d frames  hit %.1f%%  mislocated %.1f%%  wrong place %.1f%%'
          % (e, n, 100 * cs.count('hit') / n, 100 * cs.count('mislocated') / n, 100 * cs.count('wrong_place') / n))

# example sheet: 8 of each error type + 8 hits, highest-confidence errors first
rng = random.Random(0)
TW, TH, COLS = 320, 256, 4
groups = []
for k, title in (('wrong_place', 'WRONG PLACE (clutter?)'), ('mislocated', 'MISLOCATED (box off)'),
                 ('false_alarm', 'FALSE ALARM (no drone)'), ('hit', 'HIT')):
    idx = [i for i, c in enumerate(cat) if c == k]
    idx = sorted(idx, key=lambda i: -sc[i])[:4] + rng.sample(idx, min(4, len(idx)))
    groups.append((title, list(dict.fromkeys(idx))[:8]))
rows = sum((len(g) + COLS - 1) // COLS for _, g in groups)
sheet = np.full((rows * (TH + 20) + len(groups) * 26, COLS * TW, 3), 255, np.uint8)
y = 0
for title, idx in groups:
    cv2.putText(sheet, title, (6, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA); y += 26
    for n, i in enumerate(idx):
        im = cv2.imread(va['paths'][i])
        g = va['boxes'][i]
        if g:
            cv2.rectangle(im, (int(g[0]) - 3, int(g[1]) - 3), (int(g[0] + g[2]) + 3, int(g[1] + g[3]) + 3), (0, 200, 0), 2)
        b = bx[i]
        cv2.rectangle(im, (int(b[0]) - 3, int(b[1]) - 3), (int(b[0] + b[2]) + 3, int(b[1] + b[3]) + 3), (0, 0, 255), 2)
        im = cv2.resize(im, (TW, TH))
        r, c = n // COLS, n % COLS
        yy = y + r * (TH + 20)
        sheet[yy + 20:yy + 20 + TH, c * TW:(c + 1) * TW] = im
        cv2.putText(sheet, '%s  score %.2f' % (os.path.basename(os.path.dirname(va['paths'][i]))[:24], sc[i]),
                    (c * TW + 4, yy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    y += ((len(idx) + COLS - 1) // COLS) * (TH + 20)
out = os.path.join(C.ROOT, 'restart', 'figures', 'phase1_error_examples%s.jpg' % ('_T' if a.input == 'temporal' else ''))
cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump({'counts': cnt, 'by_env': {e: {k: cs.count(k) for k in set(cs)} for e, cs in by.items()}},
          open(os.path.join(C.RESULTS, 'phase1', 'error_inspection_%s.json' % a.model), 'w'), indent=1)
print('sheet ->', out)
