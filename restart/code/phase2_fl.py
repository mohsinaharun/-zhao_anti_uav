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
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default='')
a = ap.parse_args()
B = a.B or (256 if a.mode == 'fl' else 1024)
tag = a.tag or '%s_%s_B%d_s%g_lr%g_mu%g_R%d_seed%d' % (
    a.mode, 'dp' if a.dp else 'nodp', B, a.sigma if a.dp else 0, a.lr, a.mu, a.rounds, a.seed)
path = os.path.join(C.RESULTS, 'phase2', tag + '.json')
if os.path.exists(path):
    print('exists, skipping:', path); sys.exit(0)

tr = torch.load(os.path.join(C.CACHE, 'K9_s5_clients.pt'), weights_only=False)
va = torch.load(os.path.join(C.CACHE, 'K9_s5_val.pt'), weights_only=False)
X = tr['feats'].to(C.DEV); Y = [tr[k].to(C.DEV) for k in ('obj', 'reg', 'cell', 'has')]
Xva = va['feats'].to(C.DEV)

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
glob = C.Net(9).to(C.DEV)
rec = {'config': {**vars(a), 'B': B}, 'meta': C.meta(), 'client_sizes': sizes, 'history': []}
steps = {c: 0 for c in clients}
t0 = time.time()
for r in range(a.rounds):
    gstate = {k: v.detach().clone() for k, v in glob.state_dict().items()}
    gparams = [p.detach().clone() for p in glob.parameters()]
    new = {}
    for ci, (c, idx) in enumerate(clients.items()):
        torch.manual_seed(a.seed * 1000003 + r * 1009 + ci)        # local stream: sampling + noise
        local = C.Net(9).to(C.DEV); local.load_state_dict(gstate)
        opt = torch.optim.SGD(local.parameters(), lr=a.lr, momentum=0.9)
        if a.dp:
            from opacus import PrivacyEngine
            dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(len(idx))), batch_size=B,
                                             generator=torch.Generator().manual_seed(a.seed * 7919 + r * 31 + ci))
            local, opt, dl = PrivacyEngine(accountant='rdp').make_private(
                module=local, optimizer=opt, data_loader=dl, noise_multiplier=a.sigma,
                max_grad_norm=a.C, poisson_sampling=True)
        params = list((local._module if a.dp else local).parameters())
        for e in range(a.local_epochs):
            if a.dp:
                batches = (idx[b[0].to(C.DEV)] for b in dl)
            else:
                perm = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(
                    a.seed * 1000 + r * 10 + e + ci * 100000)).to(C.DEV)]
                batches = (perm[s:s + B] for s in range(0, len(idx), B))
            for j in batches:
                if len(j) == 0:
                    continue
                opt.zero_grad()
                C.per_sample_loss(local(X[j].float()), *[y[j] for y in Y]).mean().backward()
                opt.step(); steps[c] += 1
                if a.mu > 0:
                    with torch.no_grad():
                        for p, g in zip(params, gparams):
                            p.sub_(a.lr * a.mu * (p - g))
                GUARD.tick()
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
        acc.history = [(a.sigma, B / sizes[c], steps[c])]
        eps[c] = float(acc.get_epsilon(1e-5)) if a.sigma > 0 else float('inf')
    rec['epsilon_delta1e-5'] = eps
    rec['epsilon_max'] = max(eps.values()); rec['epsilon_median'] = float(np.median(list(eps.values())))
rec['seconds'] = time.time() - t0
C.save_json(rec, path)
print('%s: final val mAP@0.5 %.4f  by env %s%s' % (tag, rec['val_map50'],
      {k: round(v, 3) for k, v in rec['val_map50_by_env'].items()},
      '  eps max %.2f median %.2f' % (rec['epsilon_max'], rec['epsilon_median']) if a.dp and a.sigma > 0 else ''))
