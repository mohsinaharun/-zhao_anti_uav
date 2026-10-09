# =============================================================================
# Phase 4: one federated-training environment for every arm (uniform, fixed schedules,
# DRL), stepping ONE ROUND at a time with a caller-chosen sigma for that round.
# Replicates phase2_fl.py's FL path exactly (same seed formulas for init, C*
# calibration, per-client sampling and noise, Poisson DP-SGD, FedAvg); verified by
# reproducing a stored Phase 3 baseline bit-for-bit (gate G4a).
# Server-side observation signals are computed on PUBLIC proxy frames only.
# =============================================================================
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import budget
from gpu_guard import GUARD

FEATURE_SESSION = '01_2192'                                  # server feature set (sim AND real)
SIM_REWARD_SESSIONS = ('20190926_183400', '3700000000002_153918')   # proxy-val (reward), sim only


def _load(pool, K=9):
    return torch.load(os.path.join(C.CACHE, 'K%d_s5_%s.pt' % (K, pool)), weights_only=False)


def _session_of_paths(paths, pool):
    f2s = {f['folder']: s for s, v in C.SPLIT['pools'][pool].items() for f in v['folders']}
    return [f2s[os.path.basename(os.path.dirname(p))] for p in paths]


def _ys(d, idx=None):
    ys = [d[k] for k in ('obj', 'reg', 'cell', 'has')]
    return [y if idx is None else y[idx] for y in ys]


def build_data(world, eval_pool='val'):
    """world = 'real': the 8 real clients; eval on `eval_pool` (val or test).
       world = 'sim' : 8 pseudo-clients = 8 proxy sessions; eval (reward) on SIM_REWARD_SESSIONS.
       Both: C* calibrated on the full proxy pool; server features on FEATURE_SESSION."""
    px = _load('proxy')
    psess = _session_of_paths(px['paths'], 'proxy')
    feat_idx = torch.tensor([i for i, s in enumerate(psess) if s == FEATURE_SESSION])
    d = {'Xp': px['feats'].to(C.DEV), 'Yp': [y.to(C.DEV) for y in _ys(px)],
         'Xf': px['feats'][feat_idx].to(C.DEV), 'Yf': [y.to(C.DEV) for y in _ys(px, feat_idx)]}
    if world == 'real':
        tr = _load('clients')
        f2c = json.load(open(os.path.join(C.ROOT, 'restart', 'data', 'clients.json')))['folder_to_client']
        cl = [f2c[os.path.basename(os.path.dirname(p))] for p in tr['paths']]
        names = sorted(set(cl))
        d['clients'] = {c: torch.tensor([i for i, x in enumerate(cl) if x == c], device=C.DEV) for c in names}
        d['X'], d['Y'] = tr['feats'].to(C.DEV), [y.to(C.DEV) for y in _ys(tr)]
        ev = _load(eval_pool)
        d['Xe'], d['Be'] = ev['feats'].to(C.DEV), ev['boxes']
    else:
        excl = set(SIM_REWARD_SESSIONS) | {FEATURE_SESSION}
        train_sess = sorted(s for s in C.SPLIT['pools']['proxy'] if s not in excl)
        assert len(train_sess) == 8, train_sess
        d['clients'] = {'p%d' % k: torch.tensor([i for i, s in enumerate(psess) if s == ss], device=C.DEV)
                        for k, ss in enumerate(train_sess)}
        d['X'], d['Y'] = d['Xp'], d['Yp']
        ridx = [i for i, s in enumerate(psess) if s in SIM_REWARD_SESSIONS]
        d['Xe'], d['Be'] = px['feats'][torch.tensor(ridx)].to(C.DEV), [px['boxes'][i] for i in ridx]
    d['sizes'] = {c: len(ix) for c, ix in d['clients'].items()}
    d['qs'] = [64 / n for n in d['sizes'].values()]
    return d


class FLEnv:
    def __init__(self, data, seed, B=64, local_iters=20, lr=0.03, K=9, calib_n=512):
        self.d, self.seed, self.B, self.it, self.lr, self.K, self.calib_n = data, seed, B, local_iters, lr, K, calib_n

    # ── init + calibrated C* (identical to phase2_fl.py --clip calibrated) ────────
    def reset(self):
        from opacus import GradSampleModule
        d, s = self.d, self.seed
        torch.manual_seed(s)
        self.glob = C.Net(self.K).to(C.DEV)
        cal = C.Net(self.K).to(C.DEV); cal.load_state_dict(self.glob.state_dict())
        copt = torch.optim.SGD(cal.parameters(), lr=self.lr, momentum=0.9)
        cg = torch.Generator().manual_seed(s * 15485863)
        for _ in range(2 * self.it):
            jj = torch.randint(0, d['Xp'].shape[0], (self.B,), generator=cg).to(C.DEV)
            copt.zero_grad(); C.per_sample_loss(cal(d['Xp'][jj].float()), *[y[jj] for y in d['Yp']]).mean().backward(); copt.step()
        gm = GradSampleModule(C.Net(self.K).to(C.DEV)); gm._module.load_state_dict(cal.state_dict())
        cj = torch.randperm(d['Xp'].shape[0], generator=torch.Generator().manual_seed(s * 104729 + 999))[:self.calib_n].to(C.DEV)
        norms = []
        for s0 in range(0, len(cj), 128):
            jj = cj[s0:s0 + 128]; gm.zero_grad()
            C.per_sample_loss(gm(d['Xp'][jj].float()), *[y[jj] for y in d['Yp']]).mean().backward()
            norms.append(torch.cat([q.grad_sample.flatten(1) for q in gm.parameters()], 1).norm(dim=1))
        self.C_star = float(torch.quantile(torch.cat(norms), 0.5))
        torch.manual_seed(s)
        self.r, self.sigmas, self.steps = 0, [], {c: 0 for c in d['clients']}
        self.prev_loss = None
        return self.observe()

    # ── one federated round at noise multiplier sigma ─────────────────────────────
    def step(self, sigma):
        from opacus import PrivacyEngine
        d, s, r, B = self.d, self.seed, self.r, self.B
        gstate = {k: v.detach().clone() for k, v in self.glob.state_dict().items()}
        new = {}
        for ci, (c, idx) in enumerate(d['clients'].items()):
            torch.manual_seed(s * 1000003 + r * 1009 + ci)
            local = C.Net(self.K).to(C.DEV); local.load_state_dict(gstate)
            opt = torch.optim.SGD(local.parameters(), lr=self.lr, momentum=0.9)
            dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(len(idx))), batch_size=B,
                                             generator=torch.Generator().manual_seed(s * 7919 + r * 31 + ci))
            local, opt, dl = PrivacyEngine(accountant='rdp').make_private(
                module=local, optimizer=opt, data_loader=dl, noise_multiplier=float(sigma),
                max_grad_norm=self.C_star, poisson_sampling=True, grad_sample_mode='hooks')
            n = 0
            while n < self.it:                         # exactly local_iters steps, cycling epochs
                for b in dl:
                    if n == self.it:
                        break
                    n += 1
                    j = idx[b[0].to(C.DEV)]
                    if len(j) == 0:
                        continue
                    opt.zero_grad()
                    C.per_sample_loss(local(d['X'][j].float()), *[y[j] for y in d['Y']]).mean().backward()
                    if not opt.pre_step():
                        continue
                    opt.original_optimizer.step(); self.steps[c] += 1
                    GUARD.tick()
            new[c] = {k: v.detach() for k, v in local._module.state_dict().items()}
        tot = sum(d['sizes'].values())
        self.glob.load_state_dict({k: sum(new[c][k].float() * d['sizes'][c] for c in d['clients']) / tot
                                   for k in gstate})
        self.sigmas.append(float(sigma)); self.r += 1
        return self.observe()

    # ── server-side observation on PUBLIC proxy frames (zero privacy cost) ────────
    def observe(self):
        from opacus import GradSampleModule
        d = self.d
        m = self.glob.eval()
        with torch.no_grad():
            loss = float(np.mean([C.per_sample_loss(m(d['Xf'][i:i + 256].float()), *[y[i:i + 256] for y in d['Yf']]).mean().item()
                                  for i in range(0, d['Xf'].shape[0], 256)]))
        self.glob.train()
        gm = GradSampleModule(C.Net(self.K).to(C.DEV)); gm._module.load_state_dict(self.glob.state_dict())
        jj = torch.arange(min(256, d['Xf'].shape[0]), device=C.DEV)
        gm.zero_grad(); C.per_sample_loss(gm(d['Xf'][jj].float()), *[y[jj] for y in d['Yf']]).mean().backward()
        gnorm = float(torch.cat([q.grad_sample.flatten(1) for q in gm.parameters()], 1).norm(dim=1).median())
        dl_ = 0.0 if self.prev_loss is None else loss - self.prev_loss
        self.prev_loss = loss
        return {'round': self.r, 'proxy_loss': loss, 'proxy_gnorm': gnorm, 'proxy_dloss': dl_}

    def evaluate(self):
        m = self.glob.eval(); bx, sc = [], []
        with torch.no_grad():
            for s0 in range(0, self.d['Xe'].shape[0], 512):
                b, p = C.decode(m(self.d['Xe'][s0:s0 + 512].float()))
                bx += b.cpu().tolist(); sc += p.cpu().tolist()
        self.glob.train()
        return C.map50(self.d['Be'], bx, sc)

    def epsilon(self):
        return {c: budget.eps_client(self.sigmas, 64 / self.d['sizes'][c], exact=True) for c in self.d['clients']}
