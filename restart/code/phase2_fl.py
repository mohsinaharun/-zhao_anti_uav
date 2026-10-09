# =============================================================================
# Phase 2 runner (restart/phase2/PREREG.md). One code path for all four cells:
#   --mode fl       8 clients from restart/data/clients.json, FedAvg (+ FedProx mu)
#   --mode central  the same loop with ONE client holding the whole clients pool
#   --dp            per-client DP-SGD (Opacus, Poisson sampling), noise --sigma
# R rounds x E local epochs; local SGD momentum reset each round. FedProx is a
# decoupled step w <- w - lr*mu*(w - w_global) after each local step: it is
# data-independent, so it is neither clipped nor charged to the privacy budget.
# Writes restart/results/phase2/<tag>.json (skipped if present).
# =============================================================================
import os, sys, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from gpu_guard import GUARD

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['fl', 'central'], required=True)
ap.add_argument('--dp', action='store_true')
ap.add_argument('--sigma', type=float, default=0.0)
ap.add_argument('--C', type=float, default=45.8)
ap.add_argument('--B', type=int, default=None, help='default 256 (fl) / 1024 (central)')
ap.add_argument('--lr', type=float, default=0.03)
ap.add_argument('--mu', type=float, default=0.0)
ap.add_argument('--rounds', type=int, default=12)
ap.add_argument('--local-epochs', type=int, default=1)
ap.add_argument('--local-iters', type=int, default=0,
                help='if > 0: exactly this many local steps per round (replaces --local-epochs)')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--clip', choices=['fixed', 'adaptive', 'calibrated'], default='fixed')
ap.add_argument('--clip-pct', type=float, default=50.0, help='adaptive: percentile of proxy per-sample norms')
ap.add_argument('--calib-n', type=int, default=512, help='adaptive: proxy frames per round')
ap.add_argument('--print-config', action='store_true')
ap.add_argument('--K', type=int, default=9, help='first trainable MobileNet block (9 = head only)')
ap.add_argument('--pulse-round', type=int, default=0, help='Phase 3: 1-based round whose sigma is multiplied')
ap.add_argument('--pulse-mult', type=float, default=2.0)
ap.add_argument('--outdir', default='phase2', help='results subfolder (phase3 for the pulse test)')
ap.add_argument('--tag', default='')
a = ap.parse_args()
B = a.B or (256 if a.mode == 'fl' else 1024)
# Per-sample gradient engine. K = 9 (head only): Opacus hooks (verified exact, Phase 1).
# K < 9: the MobileNet blocks reuse inputs (squeeze-excitation, residual), which Opacus
# hooks reject; expanded weights ('ew') is used instead, verified exact against
# single-sample backprop to 4.8e-7 (TF32 off) on 2026-10-06.
GS_MODE = 'hooks' if a.K == 9 else 'ew'
def GSM(m):
    if GS_MODE == 'hooks':
        from opacus import GradSampleModule
        return GradSampleModule(m)
    from opacus.grad_sample import GradSampleModuleExpandedWeights
    return GradSampleModuleExpandedWeights(m, batch_first=True, loss_reduction='mean')
tag = a.tag or '%s_%s_B%d_s%g_lr%g_mu%g_R%d%s_seed%d' % (
    a.mode, 'dp' if a.dp else 'nodp', B, a.sigma if a.dp else 0, a.lr, a.mu, a.rounds,
    ('_it%d' % a.local_iters if a.local_iters else '') + ('_clipA%g' % a.clip_pct if a.clip == 'adaptive' else '_clipCal' if a.clip == 'calibrated' else '')
    + ('_K%d' % a.K if a.K != 9 else '') + ('_pk%d' % a.pulse_round if a.pulse_round else ''), a.seed)
if a.print_config:
    print({**vars(a), 'B': B, 'tag': tag}); sys.exit(0)
path = os.path.join(C.RESULTS, a.outdir, tag + '.json')
if os.path.exists(path):
    print('exists, skipping:', path); sys.exit(0)

tr = torch.load(os.path.join(C.CACHE, 'K%d_s5_clients.pt' % a.K), weights_only=False)
va = torch.load(os.path.join(C.CACHE, 'K%d_s5_val.pt' % a.K), weights_only=False)
X = tr['feats'].to(C.DEV); Y = [tr[k].to(C.DEV) for k in ('obj', 'reg', 'cell', 'has')]
Xva = va['feats'].to(C.DEV)
if a.clip in ('adaptive', 'calibrated'):
    px = torch.load(os.path.join(C.CACHE, 'K%d_s5_proxy.pt' % a.K), weights_only=False)
    Xp = px['feats'].to(C.DEV); Yp = [px[k].to(C.DEV) for k in ('obj', 'reg', 'cell', 'has')]

if a.mode == 'fl':
    f2c = __import__('json').load(open(os.path.join(C.ROOT, 'restart', 'data', 'clients.json')))['folder_to_client']
    cl = [f2c[os.path.basename(os.path.dirname(p))] for p in tr['paths']]
    names = sorted(set(cl))
    clients = {c: torch.tensor([i for i, x in enumerate(cl) if x == c], device=C.DEV) for c in names}
else:
    clients = {'all': torch.arange(X.shape[0], device=C.DEV)}
sizes = {c: len(ix) for c, ix in clients.items()}
tot = sum(sizes.values())


def evaluate(m, idx=None):
    m.eval(); bx, sc = [], []
    Xe = Xva if idx is None else Xva[idx]
    with torch.no_grad():
        for s in range(0, Xe.shape[0], 512):
            b, p = C.decode(m(Xe[s:s + 512].float()))
            bx += b.cpu().tolist(); sc += p.cpu().tolist()
    gt = va['boxes'] if idx is None else [va['boxes'][i] for i in idx.tolist()]
    return C.map50(gt, bx, sc)


torch.manual_seed(a.seed)
glob = C.Net(a.K).to(C.DEV)
rec = {'config': {**vars(a), 'B': B}, 'meta': C.meta(), 'client_sizes': sizes, 'history': []}
C_star = None
if a.clip == 'calibrated':
    # Calibrated FIXED bound (control): from THIS seed's initial model, train 2 rounds' worth of
    # local steps (2 x local_iters, batch B, same SGD) on the PUBLIC proxy pool without noise,
    # then C* = proxy median per-sample grad norm. Set once; depends on the seed only (not on
    # sigma, not on any client data): zero privacy cost.
    cal = C.Net(a.K).to(C.DEV); cal.load_state_dict(glob.state_dict())
    copt = torch.optim.SGD(cal.parameters(), lr=a.lr, momentum=0.9)
    cg = torch.Generator().manual_seed(a.seed * 15485863)
    # '2 rounds worth' of local steps: local_iters per round (FL edge setup), else the
    # number of steps one local epoch takes on the (single) client (central mode).
    per_round = a.local_iters or a.local_epochs * -(-max(sizes.values()) // B)
    rec['C_star_calib_steps'] = 2 * per_round
    for _ in range(2 * per_round):
        jj = torch.randint(0, Xp.shape[0], (B,), generator=cg).to(C.DEV)
        copt.zero_grad(); C.per_sample_loss(cal(Xp[jj].float()), *[y[jj] for y in Yp]).mean().backward(); copt.step()
    gm = GSM(C.Net(a.K).to(C.DEV)); gm._module.load_state_dict(cal.state_dict())
    cj = torch.randperm(Xp.shape[0], generator=torch.Generator().manual_seed(a.seed * 104729 + 999))[:a.calib_n].to(C.DEV)
    norms = []
    for s0 in range(0, len(cj), 128):
        jj = cj[s0:s0 + 128]; gm.zero_grad()
        C.per_sample_loss(gm(Xp[jj].float()), *[y[jj] for y in Yp]).mean().backward()
        norms.append(torch.cat([q.grad_sample.flatten(1) for q in gm.parameters()], 1).norm(dim=1))
    C_star = float(torch.quantile(torch.cat(norms), a.clip_pct / 100.0))
    rec['C_star'] = C_star
    torch.manual_seed(a.seed)   # restore the global stream so client training matches the other arms

steps = {c: 0 for c in clients}
acct = {c: [] for c in clients}          # per-round (sigma, steps) for the RDP accountant
t0 = time.time()
for r in range(a.rounds):
    gstate = {k: v.detach().clone() for k, v in glob.state_dict().items()}
    gparams = [p.detach().clone() for p in glob.parameters()]
    new = {}
    C_r = C_star if C_star is not None else a.C
    sig_r = a.sigma * (a.pulse_mult if r + 1 == a.pulse_round else 1.0)
    rec.setdefault('sigma_per_round', []).append(sig_r)
    if a.clip == 'adaptive':
        # Server-side, on PUBLIC proxy frames (never client data): zero privacy cost.
        gm = GSM(C.Net(a.K).to(C.DEV)); gm._module.load_state_dict(gstate)
        cj = torch.randperm(Xp.shape[0], generator=torch.Generator().manual_seed(a.seed * 104729 + r))[:a.calib_n].to(C.DEV)
        norms = []
        for s0 in range(0, len(cj), 128):
            jj = cj[s0:s0 + 128]; gm.zero_grad()
            C.per_sample_loss(gm(Xp[jj].float()), *[y[jj] for y in Yp]).mean().backward()
            norms.append(torch.cat([q.grad_sample.flatten(1) for q in gm.parameters()], 1).norm(dim=1))
        C_r = float(torch.quantile(torch.cat(norms), a.clip_pct / 100.0))
    rec.setdefault('clip_per_round', []).append(C_r)
    rec.setdefault('client_grad_norms', []).append({})
    for ci, (c, idx) in enumerate(clients.items()):
        torch.manual_seed(a.seed * 1000003 + r * 1009 + ci)        # local stream: sampling + noise
        local = C.Net(a.K).to(C.DEV); local.load_state_dict(gstate)
        opt = torch.optim.SGD(local.parameters(), lr=a.lr, momentum=0.9)
        if a.dp:
            from opacus import PrivacyEngine
            dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(len(idx))), batch_size=B,
                                             generator=torch.Generator().manual_seed(a.seed * 7919 + r * 31 + ci))
            local, opt, dl = PrivacyEngine(accountant='rdp').make_private(
                module=local, optimizer=opt, data_loader=dl, noise_multiplier=sig_r,
                max_grad_norm=C_r, poisson_sampling=True, grad_sample_mode=GS_MODE)
        params = list((local._module if a.dp else local).parameters())
        def epoch_batches(e):
            if a.dp:
                return (idx[b[0].to(C.DEV)] for b in dl)
            perm = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(
                a.seed * 1000 + r * 10 + e + ci * 100000)).to(C.DEV)]
            return (perm[s:s + B] for s in range(0, len(idx), B))

        def all_batches():
            if not a.local_iters:                 # full local epochs
                for e in range(a.local_epochs):
                    yield from epoch_batches(e)
                return
            n, e = 0, 0                           # exactly local_iters steps, cycling epochs
            while True:
                for j in epoch_batches(e):
                    if n == a.local_iters:
                        return
                    yield j; n += 1
                e += 1

        _log = []
        if True:
            for j in all_batches():
                if len(j) == 0:
                    continue
                opt.zero_grad()
                # (a) task loss ONLY -> per-sample gradients (Opacus hooks record them)
                C.per_sample_loss(local(X[j].float()), *[y[j] for y in Y]).mean().backward()
                # (b) privatize: clip each per-sample gradient to C, sum, add N(0, sigma^2 C^2),
                #     divide by the expected batch size -> privatized gradient left in p.grad
                if a.dp:   # DIAGNOSTIC ONLY: pre-clip norms from private data, never transmitted
                    _n = torch.cat([q.grad_sample.flatten(1) for q in params], 1).norm(dim=1)
                    _log.append((float(_n.median()), float(torch.quantile(_n, 0.9)), float((_n > C_r).float().mean())))
                if a.dp and not opt.pre_step():
                    continue
                # (c) FedProx gradient mu*(w - w_global) added to the ALREADY-privatized
                #     gradient: data-independent, so never clipped, never noised, no privacy cost
                if a.mu > 0:
                    with torch.no_grad():
                        for p, g in zip(params, gparams):
                            p.grad.add_(a.mu * (p - g))
                # (d) one SGD(momentum) update on (privatized task gradient + prox gradient)
                (opt.original_optimizer if a.dp else opt).step(); steps[c] += 1
                GUARD.tick()
        if _log:
            rec['client_grad_norms'][-1][c] = {'median': float(np.mean([x[0] for x in _log])),
                                              'p90': float(np.mean([x[1] for x in _log])),
                                              'frac_clipped': float(np.mean([x[2] for x in _log]))}
        acct[c].append((sig_r, steps[c] - sum(n for _, n in acct[c])))
        new[c] = {k: v.detach() for k, v in (local._module if a.dp else local).state_dict().items()}
    glob.load_state_dict({k: sum(new[c][k].float() * sizes[c] for c in clients) / tot for k in gstate})
    rec['history'].append({'round': r + 1, 'val_map50': evaluate(glob), 'seconds': time.time() - t0})
    print('  round %2d  val mAP@0.5 %.4f  (%.0fs)' % (r + 1, rec['history'][-1]['val_map50'],
                                                     rec['history'][-1]['seconds']), flush=True)

rec['val_map50'] = rec['history'][-1]['val_map50']
env_of = {f['folder']: v['dominant_env'] for v in C.SPLIT['pools']['val'].values() for f in v['folders']}
envs = [env_of[os.path.basename(os.path.dirname(p))] for p in va['paths']]
rec['val_map50_by_env'] = {e: evaluate(glob, torch.tensor([i for i, x in enumerate(envs) if x == e], device=C.DEV))
                           for e in sorted(set(envs))}
rec['steps_per_client'] = steps
if a.dp:
    from opacus.accountants import RDPAccountant
    eps = {}
    for c in clients:
        acc = RDPAccountant()
        acc.history = [(sg, B / sizes[c], n) for sg, n in acct[c] if n > 0]
        eps[c] = float(acc.get_epsilon(1e-5)) if a.sigma > 0 else float('inf')
    rec['epsilon_delta1e-5'] = eps
    rec['epsilon_max'] = max(eps.values()); rec['epsilon_median'] = float(np.median(list(eps.values())))
rec['seconds'] = time.time() - t0
C.save_json(rec, path)
print('%s: final val mAP@0.5 %.4f  by env %s%s' % (tag, rec['val_map50'],
      {k: round(v, 3) for k, v in rec['val_map50_by_env'].items()},
      '  eps max %.2f median %.2f' % (rec['epsilon_max'], rec['epsilon_median']) if a.dp and a.sigma > 0 else ''))
