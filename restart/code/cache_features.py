# Cache the frozen MobileNetV3-Small prefix (blocks 0..K-1) for a pool, so the
# training runs only touch the trainable part. Output (gitignored):
#   restart/cache/K{K}_s{step}_{pool}.pt  with feats (fp16), boxes, targets.
import os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from gpu_guard import GUARD

ap = argparse.ArgumentParser()
ap.add_argument('--K', type=int, default=9)
ap.add_argument('--step', type=int, default=5)
ap.add_argument('--pools', nargs='+', default=['clients', 'val'])
ap.add_argument('--temporal', action='store_true', help='causal 3-frame input (t-10, t-5, t)')
a = ap.parse_args()

pre = C.frozen_prefix(a.K).to(C.DEV)
os.makedirs(C.CACHE, exist_ok=True)
for pool in a.pools:
    path = os.path.join(C.CACHE, 'K%d_s%d%s_%s.pt' % (a.K, a.step, '_T' if a.temporal else '', pool))
    if os.path.exists(path):
        print('exists, skipping:', path); continue
    items, mismatch = C.frames(pool, a.step)
    t0, feats = time.time(), []
    with torch.no_grad():
        for s in range(0, len(items), 64):
            x = (C.load_batch_temporal if a.temporal else C.load_batch)([p for p, _ in items[s:s + 64]]).to(C.DEV)
            feats.append(pre(x).half().cpu())
            GUARD.tick()
    boxes = [b for _, b in items]
    obj, reg, cell, has = C.encode(boxes)
    torch.save({'feats': torch.cat(feats), 'boxes': boxes, 'obj': obj, 'reg': reg,
                'cell': cell, 'has': has, 'paths': [p for p, _ in items], 'K': a.K,
                'step': a.step, 'pool': pool, 'temporal': a.temporal, 'label_mismatch_frames': mismatch}, path)
    print('%s: %d frames (%d with target, %d img/label mismatches) feats %s in %.0fs -> %s'
          % (pool, len(items), int(has.sum()), mismatch, tuple(feats[0].shape[1:]),
             time.time() - t0, path), flush=True)
