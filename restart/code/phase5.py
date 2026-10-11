# =============================================================================
# Phase 5 (restart/phase5/PREREG.md, approved 2026-10-11): Byzantine robustness under DP-FL at P.
# FLEnv's client loop verbatim (seed formulas -> common random numbers), then an omniscient attack
# by c3 + c6 and one of FedAvg / FLTrust / median / Krum.
#   python phase5.py run [--worker i --workers n]   (resumable; reproduction check first)
#   python phase5.py analyse
# =============================================================================
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, budget, fl_env, phase4 as P4
from gpu_guard import GUARD

OUT = os.path.join(C.RESULTS, 'phase5')
P4OUT = os.path.join(C.RESULTS, 'phase4')
SEEDS = range(200, 220)
MAL = ('c3', 'c6')
SCHED = ('A1', 'A2', 'A3', 'A5')
ATTACKS = ('none', 'ALIE', 'IPM', 'Fang')
N, M = 8, len(MAL)
Z_ALIE = float(stats.norm.ppf((N - (N // 2 + 1 - M)) / N))     # 0.319 (PREREG §4)
EPS_IPM, B_FANG, F_KRUM = 0.5, 2.0, 2


# ── attacks (rows = updates of the 6 honest clients) ─────────────────────────
def attack(kind, H, g):
    mu = H.mean(0)
    if kind == 'ALIE':
        return mu - Z_ALIE * H.std(0)
    if kind == 'IPM':
        return -EPS_IPM * mu
    if kind == 'Fang':
        lo, hi = H.min(0).values, H.max(0).values
        u = torch.rand(lo.shape, generator=g, device=lo.device)
        dn = torch.where(lo > 0, lo / B_FANG + u * (lo - lo / B_FANG), B_FANG * lo + u * (lo - B_FANG * lo))   # <= w_min
        up = torch.where(hi > 0, hi + u * (B_FANG * hi - hi), hi / B_FANG + u * (hi - hi / B_FANG))          # >= w_max
        return torch.where(mu > 0, dn, up)
    raise ValueError(kind)


def krum(U, f=F_KRUM):
    d2 = torch.cdist(U, U) ** 2
    k = len(U) - f - 2
    score = torch.sort(d2, 1).values[:, 1:k + 1].sum(1)
    i = int(score.argmin())
    return U[i], i


class Env(fl_env.FLEnv):
    def __init__(self, data, seed, attack='none', agg='fedavg'):
        super().__init__(data, seed)
        self.attack, self.agg, self.log = attack, agg, []

    def step(self, sigma):
        from opacus import PrivacyEngine
        d, s, r, B = self.d, self.seed, self.r, self.B
        gstate = {k: v.detach().clone() for k, v in self.glob.state_dict().items()}
        new = {}
        for ci, (c, idx) in enumerate(d['clients'].items()):          # ── verbatim fl_env.FLEnv.step
            torch.manual_seed(s * 1000003 + r * 1009 + ci)
            local = C.Net(self.K).to(C.DEV); local.load_state_dict(gstate)
            opt = torch.optim.SGD(local.parameters(), lr=self.lr, momentum=0.9)
            dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(len(idx))), batch_size=B,
                                             generator=torch.Generator().manual_seed(s * 7919 + r * 31 + ci))
            local, opt, dl = PrivacyEngine(accountant='rdp').make_private(
                module=local, optimizer=opt, data_loader=dl, noise_multiplier=float(sigma),
                max_grad_norm=self.C_star, poisson_sampling=True, grad_sample_mode='hooks')
            n = 0
            while n < self.it:
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
        keys = list(gstate)
        flat = lambda sd: torch.cat([sd[k].float().flatten() for k in keys])
        g0 = flat(gstate)
        names = list(d['clients'])
        rec = {'round': r + 1, 'sigma': float(sigma)}
        if self.attack != 'none':                                    # omniscient, colluding (PREREG §4)
            H = torch.stack([flat(new[c]) - g0 for c in names if c not in MAL])
            gen = torch.Generator(device=C.DEV).manual_seed(s * 1000003 + r * 1009 + 77)
            mal = attack(self.attack, H, gen)
            for c in MAL:
                new[c] = self._unflat(g0 + mal, gstate)
        tot = sum(d['sizes'].values())
        U = torch.stack([flat(new[c]) - g0 for c in names])
        rec['norm'] = dict(zip(names, U.norm(dim=1).tolist()))
        if self.agg == 'fedavg':                                     # Phase 4 formula, bit-identical
            self.glob.load_state_dict({k: sum(new[c][k].float() * d['sizes'][c] for c in d['clients']) / tot for k in gstate})
        else:
            if self.agg == 'fltrust':
                gr = flat(self._root_update(gstate)) - g0
                cos = torch.nn.functional.cosine_similarity(U, gr.unsqueeze(0), dim=1)
                ts = torch.relu(cos)
                rec.update({'trust': dict(zip(names, ts.tolist())), 'cos': dict(zip(names, cos.tolist())), 'root_norm': float(gr.norm())})
                Uh = U * (gr.norm() / U.norm(dim=1).clamp_min(1e-12)).unsqueeze(1)
                upd = (ts.unsqueeze(1) * Uh).sum(0) / ts.sum() if ts.sum() > 0 else torch.zeros_like(g0)
            elif self.agg == 'median':
                srt = torch.sort(U, 0).values
                upd = 0.5 * (srt[N // 2 - 1] + srt[N // 2])
            elif self.agg == 'krum':
                upd, i = krum(U); rec['krum_pick'] = names[i]
            self.glob.load_state_dict(self._unflat(g0 + upd, gstate))
        self.log.append(rec)
        self.sigmas.append(float(sigma)); self.r += 1
        return self.observe()

    @staticmethod
    def _unflat(v, ref):
        out, o = {}, 0
        for k, t in ref.items():
            out[k] = v[o:o + t.numel()].view_as(t).to(t.dtype); o += t.numel()
        return out

    def _root_update(self, gstate):
        """FLTrust server update (Amendment 1): the clients' procedure on the root set, clipped at
        C*, noise multiplier 0."""
        from opacus import PrivacyEngine
        d, s, r = self.d, self.seed, self.r
        torch.manual_seed(s * 1000003 + r * 1009 + 99)
        m = C.Net(self.K).to(C.DEV); m.load_state_dict(gstate)
        opt = torch.optim.SGD(m.parameters(), lr=self.lr, momentum=0.9)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(d['Xr'].shape[0])), batch_size=self.B,
                                         generator=torch.Generator().manual_seed(s * 7919 + r * 31 + 99))
        m, opt, dl = PrivacyEngine(accountant='rdp').make_private(
            module=m, optimizer=opt, data_loader=dl, noise_multiplier=0.0,
            max_grad_norm=self.C_star, poisson_sampling=True, grad_sample_mode='hooks')
        n = 0
        while n < self.it:
            for b in dl:
                if n == self.it:
                    break
                n += 1
                j = b[0].to(C.DEV)
                if len(j) == 0:
                    continue
                opt.zero_grad()
                C.per_sample_loss(m(d['Xr'][j].float()), *[y[j] for y in d['Yr']]).mean().backward()
                if not opt.pre_step():
                    continue
                opt.original_optimizer.step()
        return {k: v.detach() for k, v in m._module.state_dict().items()}


# ── runs ──────────────────────────────────────────────────────────────────────
_D, _POL = {}, {}


def data():
    if not _D:
        d = fl_env.build_data('real', 'test')
        rt = fl_env._load('root')
        d['Xr'], d['Yr'] = rt['feats'].to(C.DEV), [rt[k].to(C.DEV) for k in ('obj', 'reg', 'cell', 'has')]
        _D.update(d)
    return _D


def run(block, sched, agg, att, seed):
    name = '%s_%s_%s_%s_seed%d' % (block, sched, agg, att, seed)
    f = os.path.join(OUT, name + '.json')
    if os.path.exists(f):
        return json.load(open(f))
    d, t0 = data(), time.time()
    env = Env(d, seed, att, agg)
    if sched == 'A5':
        if 'A5' not in _POL:
            import phase4_agents as AG
            _POL['A5'] = AG.load_policy(os.path.join(P4OUT, 'agent_A5_P_best.pt'), 'A5')
        pol, tgt, R = _POL['A5'], P4.target_eps('P', d['qs']), P4.OPS['P'][0]
        obs = [env.reset()]; pol.reset()
        for t in range(R):                                           # = phase4.run_policy
            spent = budget.eps_worst(env.sigmas, d['qs']) / tgt if env.sigmas else 0.0
            a = float(pol.act(obs, t, R, spent))
            obs.append(env.step(budget.sigma_for_round(env.sigmas, 0.25 * 16 ** a, R - t, tgt, d['qs'])))
    else:
        if sched == 'noDP':
            sig = [0.0] * 12
        else:
            k = json.load(open(os.path.join(P4OUT, 'tune_P.json')))['best_k'] if sched == 'A2' else None
            sig = P4.schedule_for(sched, 'P', d['qs'], k)[0]
        env.reset()
        for sg in sig:
            env.step(sg)
    eps = env.epsilon() if min(env.sigmas) > 0 else None
    rec = {'block': block, 'schedule': sched, 'agg': agg, 'attack': att, 'seed': seed, 'malicious': MAL,
           'sigmas': env.sigmas, 'test_map50': env.evaluate(), 'eps_max': max(eps.values()) if eps else float('inf'),
           'C_star': env.C_star, 'rounds': env.log, 'seconds': time.time() - t0, 'meta': C.meta()}
    C.save_json(rec, f)
    print('%s: mAP %.4f eps %.4f (%.0f s)' % (name, rec['test_map50'], rec['eps_max'], rec['seconds']), flush=True)
    return rec


def jobs():
    J = [('main', s, a, t) for t in ATTACKS for a in ('fedavg', 'fltrust') for s in SCHED]
    J += [('ref', 'A1', a, t) for t in ATTACKS for a in ('median', 'krum')]
    J += [('nodp', 'noDP', a, t) for t in ATTACKS for a in ('fedavg', 'fltrust')]
    return [(b, s, a, t, seed) for (b, s, a, t) in J for seed in SEEDS]


def repro_check():
    bad = []
    for seed in (200, 201, 202):
        v = run('main', 'A1', 'fedavg', 'none', seed)['test_map50']
        ref = json.load(open(os.path.join(P4OUT, 'eval_P_A1_seed%d.json' % seed)))['test_map50']
        print('repro seed %d: %.6f vs Phase 4 %.6f %s' % (seed, v, ref, 'OK' if v == ref else 'MISMATCH'), flush=True)
        if v != ref:
            bad.append(seed)
    return not bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'analyse'])
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'analyse':
        import analyse_phase5; analyse_phase5.main(); sys.exit(0)
    if a.worker == 0:
        if not repro_check():
            C.save_json({'repro': 'MISMATCH'}, os.path.join(OUT, 'STOPPED.json')); sys.exit(1)
        C.save_json({'repro': 'OK', 'meta': C.meta()}, os.path.join(OUT, 'repro_ok.json'))
    else:
        while not os.path.exists(os.path.join(OUT, 'repro_ok.json')):
            if os.path.exists(os.path.join(OUT, 'STOPPED.json')):
                sys.exit(1)
            time.sleep(10)
    for i, j in enumerate(jobs()):
        if i % a.workers == a.worker:
            run(*j)
    print('worker %d DONE' % a.worker, flush=True)
