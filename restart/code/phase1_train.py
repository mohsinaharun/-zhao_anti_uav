# =============================================================================
# Phase 1 runner (restart/phase1/PREREG.md). Modes:
#   --trivial            val mAP@0.5 of the median-box trivial predictor
#   --measure-C          median per-sample grad norm at init (fixes C for DP)
#   (default)            train: non-DP SGD, or DP-SGD with --dp --sigma --C
# Every run writes restart/results/phase1/<tag>.json (skipped if it exists).
# =============================================================================
import os, sys, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from gpu_guard import GUARD

ap = argparse.ArgumentParser()
ap.add_argument('--K', type=int, default=9)
ap.add_argument('--step', type=int, default=5)
ap.add_argument('--lr', type=float, default=0.01)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=12)
ap.add_argument('--batch', type=int, default=64, help='non-DP batch size')
ap.add_argument('--dp', action='store_true')
ap.add_argument('--sigma', type=float, default=0.0)
ap.add_argument('--C', type=float, default=None, help='clipping norm (DP)')
ap.add_argument('--B', type=int, default=256, help='DP expected (Poisson) batch size')
ap.add_argument('--trivial', action='store_true')
ap.add_argument('--measure-C', action='store_true')
ap.add_argument('--tag', default='')
ap.add_argument('--save-model', action='store_true', help='also save weights to restart/cache/models/<tag>.pt')
a = ap.parse_args()

OUT = os.path.join(C.RESULTS, 'phase1')
tag = a.tag or ('trivial' if a.trivial else 'measureC_K%d' % a.K if a.measure_C else
                ('dp_K%d_B%d_C%g_s%g_lr%g_seed%d' % (a.K, a.B, a.C or 0, a.sigma, a.lr, a.seed)
                 if a.dp else 'nodp_K%d_lr%g_seed%d' % (a.K, a.lr, a.seed)))
path = os.path.join(OUT, tag + '.json')
if os.path.exists(path):
    print('exists, skipping:', path); sys.exit(0)

tr = torch.load(os.path.join(C.CACHE, 'K%d_s%d_clients.pt' % (a.K, a.step)), weights_only=False)
va = torch.load(os.path.join(C.CACHE, 'K%d_s%d_val.pt' % (a.K, a.step)), weights_only=False)
rec = {'config': vars(a), 'meta': C.meta(), 'n_train': len(tr['boxes']), 'n_val': len(va['boxes'])}

# ── trivial predictor ────────────────────────────────────────────────────────
if a.trivial:
    b = np.array([x for x in tr['boxes'] if x is not None])
    cx, cy = np.median(b[:, 0] + b[:, 2] / 2), np.median(b[:, 1] + b[:, 3] / 2)
    w, h = np.median(b[:, 2]), np.median(b[:, 3])
    box = [cx - w / 2, cy - h / 2, w, h]
    n = len(va['boxes'])
    rec['trivial_box_xywh'] = [float(v) for v in box]
    rec['val_map50'] = C.map50(va['boxes'], [[box]] * n, [[1.0]] * n)
    C.save_json(rec, path); print('trivial val mAP@0.5 = %.4f  box %s' % (rec['val_map50'], box)); sys.exit(0)

torch.manual_seed(a.seed); np.random.seed(a.seed)
Xtr = tr['feats'].to(C.DEV); Ytr = [tr[k].to(C.DEV) for k in ('obj', 'reg', 'cell', 'has')]
Xva = va['feats'].to(C.DEV)
model = C.Net(a.K).to(C.DEV)
rec['trainable_params'] = C.n_trainable(model)


def evaluate(m):
    m.eval(); bx, sc = [], []
    with torch.no_grad():
        for s in range(0, Xva.shape[0], 512):
            b, p = C.decode(m(Xva[s:s + 512].float()))
            bx += b.cpu().tolist(); sc += p.cpu().tolist()
    m.train()
    return C.map50(va['boxes'], bx, sc)


# ── per-sample gradient norm at init -> C ────────────────────────────────────
if a.measure_C:
    from opacus import GradSampleModule
    gm = GradSampleModule(C.Net(a.K).to(C.DEV))
    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(Xtr.shape[0], generator=g)[:1024].to(C.DEV)
    norms = []
    for s in range(0, 1024, 128):
        j = idx[s:s + 128]
        gm.zero_grad()
        C.per_sample_loss(gm(Xtr[j].float()), *[y[j] for y in Ytr]).mean().backward()
        per = torch.cat([p.grad_sample.flatten(1) for p in gm.parameters() if p.requires_grad], 1)
        norms += per.norm(dim=1).tolist()   # Opacus grad_sample = exact per-sample grads (verified)
    rec['per_sample_grad_norm'] = {'median': float(np.median(norms)), 'p10': float(np.percentile(norms, 10)),
                                   'p90': float(np.percentile(norms, 90)), 'n': len(norms)}
    C.save_json(rec, path); print('per-sample grad norm:', rec['per_sample_grad_norm']); sys.exit(0)

# ── training ─────────────────────────────────────────────────────────────────
opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9)
N = Xtr.shape[0]
hist, t0 = [], time.time()
if a.dp:
    from opacus import PrivacyEngine
    if a.C is None:
        raise SystemExit('--dp needs --C (measure it with --measure-C)')
    ds = torch.utils.data.TensorDataset(torch.arange(N))
    dl = torch.utils.data.DataLoader(ds, batch_size=a.B,
                                     generator=torch.Generator().manual_seed(a.seed))
    pe = PrivacyEngine(accountant='rdp')
    model, opt, dl = pe.make_private(module=model, optimizer=opt, data_loader=dl,
                                     noise_multiplier=a.sigma, max_grad_norm=a.C,
                                     poisson_sampling=True)
    rec['sample_rate'] = a.B / N
steps = 0
for ep in range(a.epochs):
    if a.dp:
        batches = (b[0].to(C.DEV) for b in dl)
    else:
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(a.seed * 1000 + ep)).to(C.DEV)
        batches = (perm[s:s + a.batch] for s in range(0, N, a.batch))
    for j in batches:
        if len(j) == 0:
            continue
        opt.zero_grad()
        C.per_sample_loss(model(Xtr[j].float()), *[y[j] for y in Ytr]).mean().backward()
        opt.step(); steps += 1
        GUARD.tick()
    inner = model._module if hasattr(model, '_module') else model
    hist.append({'epoch': ep + 1, 'val_map50': evaluate(inner), 'seconds': time.time() - t0})
    print('  epoch %2d  val mAP@0.5 %.4f  (%.0fs)' % (ep + 1, hist[-1]['val_map50'], hist[-1]['seconds']), flush=True)

rec.update({'history': hist, 'val_map50': hist[-1]['val_map50'], 'steps': steps,
            'seconds': time.time() - t0})
if a.dp:
    rec['epsilon_delta1e-5'] = (float(pe.get_epsilon(1e-5)) if a.sigma > 0 else float('inf'))
if a.save_model:
    os.makedirs(os.path.join(C.CACHE, 'models'), exist_ok=True)
    inner = model._module if hasattr(model, '_module') else model
    torch.save(inner.state_dict(), os.path.join(C.CACHE, 'models', tag + '.pt'))
C.save_json(rec, path)
print('%s: final val mAP@0.5 %.4f%s' % (tag, rec['val_map50'],
      '  eps=%.2f' % rec['epsilon_delta1e-5'] if a.dp else ''))
