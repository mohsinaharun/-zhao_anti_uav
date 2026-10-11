# =============================================================================
# Phase 5c (restart/phase5c/PREREG.md, approved 2026-10-12): Variant N follow-up checks.
# Part 1: Variant N at sigma 0, 0.5, 1, 4 under none / ALIE / IPM / Fang.
# Part 2: adaptive "trust-capture" attack (knows Variant N's rule and g0) at sigma 2 against
#         Variant N, FedAvg and plain FLTrust.  Phase 5 / 5b are imported read-only.
#   python phase5c.py run [--worker i --workers n]   (resumable; reproduction check first)
#   python phase5c.py analyse
# =============================================================================
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, phase5 as P5, phase5b as P5B
from phase5 import MAL, ATTACKS, SEEDS, N, attack, krum
from phase5b import noise_var, schedule
from gpu_guard import GUARD

OUT = os.path.join(C.RESULTS, 'phase5c')
LEVELS = ('0', '0.5', '1', '4')
C_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def trust_capture(H, gr, sigma, C_star, lr, B, K, Dn):
    """PREREG §4: the update both attackers send, plus the chosen c and S."""
    gh = gr / gr.norm()
    mu = H.mean(0)
    perp = mu - (mu @ gh) * gh
    h = -perp / perp.norm().clamp_min(1e-30)
    S = float(torch.relu(torch.nn.functional.cosine_similarity(H, gr.unsqueeze(0), dim=1)).sum())
    f = 2 * C_GRID * np.sqrt(1 - C_GRID ** 2) / (2 * C_GRID + S)
    c = float(C_GRID[int(np.argmax(f))])
    Nt = torch.sqrt(gr.norm() ** 2 + Dn * noise_var(sigma, C_star, lr, B, K))
    return Nt * (c * gh + np.sqrt(1 - c * c) * h), c, S


class Env(P5B.Env):
    def step(self, sigma):
        from opacus import PrivacyEngine
        d, s, r, B = self.d, self.seed, self.r, self.B
        gstate = {k: v.detach().clone() for k, v in self.glob.state_dict().items()}
        new = {}
        for ci, (c, idx) in enumerate(d['clients'].items()):          # ── verbatim phase5b.Env.step
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
        Dn = g0.numel()
        v = noise_var(float(sigma), self.C_star, self.lr, B, self.it)
        gr = flat(self._root_update(gstate)) - g0 if (self.agg.startswith('fltrust') or self.attack == 'adaptive') else None
        if self.attack != 'none':
            H = torch.stack([flat(new[c]) - g0 for c in names if c not in MAL])
            if self.attack == 'adaptive':
                mal, cc, S = trust_capture(H, gr, float(sigma), self.C_star, self.lr, B, self.it, Dn)
                rec.update({'adv_c': cc, 'adv_S': S})
            else:
                gen = torch.Generator(device=C.DEV).manual_seed(s * 1000003 + r * 1009 + 77)
                mal = attack(self.attack, H, gen)
            for c in MAL:
                new[c] = self._unflat(g0 + mal, gstate)
        tot = sum(d['sizes'].values())
        U = torch.stack([flat(new[c]) - g0 for c in names])
        rec['norm'] = dict(zip(names, U.norm(dim=1).tolist()))
        if self.agg == 'fedavg':
            self.glob.load_state_dict({k: sum(new[c][k].float() * d['sizes'][c] for c in d['clients']) / tot for k in gstate})
            rec['upd_norm'] = float((flat(self.glob.state_dict()) - g0).norm())
        else:
            if self.agg in ('fltrust', 'fltrust_N'):
                cos = torch.nn.functional.cosine_similarity(U, gr.unsqueeze(0), dim=1)
                ts = torch.relu(cos)
                target = gr.norm() if self.agg == 'fltrust' else torch.sqrt(gr.norm() ** 2 + Dn * v)
                rec.update({'trust': dict(zip(names, ts.tolist())), 'cos': dict(zip(names, cos.tolist())),
                            'root_norm': float(gr.norm()), 'noise_var': v, 'target_norm': float(target)})
                Uh = U * (target / U.norm(dim=1).clamp_min(1e-12)).unsqueeze(1)
                upd = (ts.unsqueeze(1) * Uh).sum(0) / ts.sum() if ts.sum() > 0 else torch.zeros_like(g0)
            else:
                raise ValueError(self.agg)
            rec['upd_norm'] = float(upd.norm())
            self.glob.load_state_dict(self._unflat(g0 + upd, gstate))
        self.log.append(rec)
        self.sigmas.append(float(sigma)); self.r += 1
        return self.observe()


def run(block, level, agg, att, seed):
    name = '%s_s%s_%s_%s_seed%d' % (block, level, agg, att, seed)
    f = os.path.join(OUT, name + '.json')
    if os.path.exists(f):
        return json.load(open(f))
    d, t0 = P5.data(), time.time()
    env = Env(d, seed, att, agg)
    env.reset()
    for sg in schedule(level, d['qs']) if level != '0' else [0.0] * 12:
        env.step(sg)
    eps = env.epsilon() if min(env.sigmas) > 0 else None
    rec = {'block': block, 'level': level, 'agg': agg, 'attack': att, 'seed': seed, 'malicious': MAL,
           'sigmas': env.sigmas, 'test_map50': env.evaluate(), 'eps_max': max(eps.values()) if eps else float('inf'),
           'C_star': env.C_star, 'rounds': env.log, 'seconds': time.time() - t0, 'meta': C.meta()}
    C.save_json(rec, f)
    print('%s: mAP %.4f eps %.4f (%.0f s)' % (name, rec['test_map50'], rec['eps_max'], rec['seconds']), flush=True)
    return rec


def jobs():
    J = [('levels', l, 'fltrust_N', t) for l in LEVELS for t in ATTACKS]
    J += [('adaptive', '2', a, 'adaptive') for a in ('fltrust_N', 'fedavg', 'fltrust')]
    return [(b, l, a, t, seed) for (b, l, a, t) in J for seed in SEEDS]


def repro_check():
    bad = []
    for seed in (200, 201, 202):
        v = run('check', '2', 'fltrust_N', 'none', seed)['test_map50']
        ref = json.load(open(os.path.join(P5B.OUT, 'variant_s2_fltrust_N_none_seed%d.json' % seed)))['test_map50']
        print('repro Variant N seed %d: %.6f vs Phase 5b %.6f %s' % (seed, v, ref, 'OK' if v == ref else 'MISMATCH'), flush=True)
        bad += [seed] if v != ref else []
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'analyse'])
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'analyse':
        import analyse_phase5c; analyse_phase5c.main(); sys.exit(0)
    if a.worker == 0:
        bad = repro_check()
        if bad:
            C.save_json({'repro': 'MISMATCH', 'bad': bad}, os.path.join(OUT, 'STOPPED.json')); sys.exit(1)
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
