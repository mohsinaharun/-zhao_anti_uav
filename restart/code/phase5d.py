# =============================================================================
# Phase 5d (restart/phase5d/PREREG.md, approved 2026-10-12 with +A and +B): per-client trust-weight
# cap on Variant N at sigma 2 (eps 1.86). Cap kappa = m / K, m in {1.5, 2, 1}; water-filling, excess
# dropped when too few clients have positive trust (PREREG §3). Attacks: none / ALIE / IPM / Fang,
# Phase 5c's trust-capture ('adaptive'), and (+A) a cap-aware trust-capture ('adaptive_cap').
# Phase 5 / 5b / 5c are imported read-only.
#   python phase5d.py run [--worker i --workers n]   (resumable; reproduction check first)
#   python phase5d.py analyse
# =============================================================================
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, phase5 as P5, phase5b as P5B, phase5c as P5C
from phase5 import MAL, ATTACKS, SEEDS, attack
from phase5b import noise_var, schedule
from phase5c import trust_capture, C_GRID
from gpu_guard import GUARD

OUT = os.path.join(C.RESULTS, 'phase5d')
K_CLIENTS = 8
MARGINS = ('1.5', '2', '1')
CONDS = ATTACKS + ('adaptive',)


def cap_weights(ts, kappa):
    """PREREG §3 steps 2-5. ts: 1-D numpy array of trust scores (>= 0, sum > 0). Returns (w, bound)."""
    ts = np.asarray(ts, float)
    w = ts / ts.sum()
    if w.max() <= kappa:
        return w, False
    fixed = np.zeros(len(w), bool)
    while True:
        over = (w > kappa) & ~fixed
        if not over.any():
            break
        fixed |= over
        w[fixed] = kappa
        free = ~fixed & (ts > 0)
        rem = 1.0 - kappa * fixed.sum()
        if not free.any() or rem <= 0:
            w[~fixed] = 0.0
            break
        w[free] = rem * ts[free] / ts[free].sum()
    return w, True


def trust_capture_cap(H, gr, sigma, C_star, lr, B, K, Dn, kappa, n_mal):
    """(+A): same update family as phase5c.trust_capture, c chosen knowing the cap:
    maximise (attackers' capped weight) * sqrt(1 - c^2) on the same c grid."""
    gh = gr / gr.norm()
    mu = H.mean(0)
    perp = mu - (mu @ gh) * gh
    h = -perp / perp.norm().clamp_min(1e-30)
    th = torch.relu(torch.nn.functional.cosine_similarity(H, gr.unsqueeze(0), dim=1)).cpu().numpy().astype(float)
    S = float(th.sum())
    best, bc = -1.0, None
    for c in C_GRID:
        w, _ = cap_weights(np.concatenate([th, np.full(n_mal, c)]), kappa)
        f = w[-n_mal:].sum() * np.sqrt(1 - c * c)
        if f > best + 1e-15:
            best, bc = f, float(c)
    Nt = torch.sqrt(gr.norm() ** 2 + Dn * noise_var(sigma, C_star, lr, B, K))
    return Nt * (bc * gh + np.sqrt(1 - bc * bc) * h), bc, S


class Env(P5C.Env):
    kappa = None                                                       # set per run

    def step(self, sigma):
        from opacus import PrivacyEngine
        d, s, r, B = self.d, self.seed, self.r, self.B
        gstate = {k: v.detach().clone() for k, v in self.glob.state_dict().items()}
        new = {}
        for ci, (c, idx) in enumerate(d['clients'].items()):          # ── verbatim phase5c.Env.step
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
        gr = flat(self._root_update(gstate)) - g0
        if self.attack != 'none':
            H = torch.stack([flat(new[c]) - g0 for c in names if c not in MAL])
            if self.attack == 'adaptive':
                mal, cc, S = trust_capture(H, gr, float(sigma), self.C_star, self.lr, B, self.it, Dn)
                rec.update({'adv_c': cc, 'adv_S': S})
            elif self.attack == 'adaptive_cap':
                mal, cc, S = trust_capture_cap(H, gr, float(sigma), self.C_star, self.lr, B, self.it, Dn, self.kappa, len(MAL))
                rec.update({'adv_c': cc, 'adv_S': S})
            else:
                gen = torch.Generator(device=C.DEV).manual_seed(s * 1000003 + r * 1009 + 77)
                mal = attack(self.attack, H, gen)
            for c in MAL:
                new[c] = self._unflat(g0 + mal, gstate)
        U = torch.stack([flat(new[c]) - g0 for c in names])
        rec['norm'] = dict(zip(names, U.norm(dim=1).tolist()))
        assert self.agg == 'fltrust_N'
        cos = torch.nn.functional.cosine_similarity(U, gr.unsqueeze(0), dim=1)
        ts = torch.relu(cos)
        target = torch.sqrt(gr.norm() ** 2 + Dn * v)
        rec.update({'trust': dict(zip(names, ts.tolist())), 'cos': dict(zip(names, cos.tolist())),
                    'root_norm': float(gr.norm()), 'noise_var': v, 'target_norm': float(target)})
        Uh = U * (target / U.norm(dim=1).clamp_min(1e-12)).unsqueeze(1)
        if ts.sum() > 0:
            w, bound = cap_weights(ts.cpu().numpy(), self.kappa)
            if not bound:                                              # Variant N's expression, verbatim
                upd = (ts.unsqueeze(1) * Uh).sum(0) / ts.sum()
            else:
                wt = torch.tensor(w, dtype=Uh.dtype, device=Uh.device)
                upd = (wt.unsqueeze(1) * Uh).sum(0)
            mi = [i for i, c in enumerate(names) if c in MAL]
            w0 = ts.cpu().numpy() / float(ts.sum())
            rec.update({'cap_bound': bool(bound), 'w': dict(zip(names, w.tolist())), 'w_sum': float(w.sum()),
                        'n_pos': int((ts > 0).sum()), 'mal_share_uncapped': float(w0[mi].sum()), 'mal_share': float(w[mi].sum())})
        else:
            upd = torch.zeros_like(g0)
            rec.update({'cap_bound': False, 'w_sum': 0.0, 'n_pos': 0, 'mal_share_uncapped': 0.0, 'mal_share': 0.0})
        rec['upd_norm'] = float(upd.norm())
        self.glob.load_state_dict(self._unflat(g0 + upd, gstate))
        self.log.append(rec)
        self.sigmas.append(float(sigma)); self.r += 1
        return self.observe()


def run(block, m, att, seed):
    name = '%s_m%s_%s_seed%d' % (block, m, att, seed)
    f = os.path.join(OUT, name + '.json')
    if os.path.exists(f):
        return json.load(open(f))
    d, t0 = P5.data(), time.time()
    env = Env(d, seed, att, 'fltrust_N')
    env.kappa = float(m) / K_CLIENTS
    env.reset()
    for sg in schedule('2', d['qs']):
        env.step(sg)
    eps = env.epsilon()
    rec = {'block': block, 'm': m, 'kappa': env.kappa, 'agg': 'fltrust_N_cap', 'attack': att, 'seed': seed,
           'malicious': MAL, 'sigmas': env.sigmas, 'test_map50': env.evaluate(), 'eps_max': max(eps.values()),
           'C_star': env.C_star, 'rounds': env.log, 'seconds': time.time() - t0, 'meta': C.meta()}
    C.save_json(rec, f)
    print('%s: mAP %.4f eps %.4f (%.0f s)' % (name, rec['test_map50'], rec['eps_max'], rec['seconds']), flush=True)
    return rec


def jobs():
    J = [(m, t) for m in MARGINS for t in CONDS] + [(m, 'adaptive_cap') for m in ('1.5', '2')]
    return [('grid', m, t, seed) for (m, t) in J for seed in SEEDS]


def repro_check():
    """Cap disabled (m = K, kappa = 1): must reproduce Phase 5b Variant N (none) and Phase 5c (trust-capture)."""
    bad = []
    for seed in (200, 201, 202):
        for att, ref_f in (('none', os.path.join(P5B.OUT, 'variant_s2_fltrust_N_none_seed%d.json' % seed)),
                           ('adaptive', os.path.join(P5C.OUT, 'adaptive_s2_fltrust_N_adaptive_seed%d.json' % seed))):
            v = run('check', str(K_CLIENTS), att, seed)['test_map50']
            ref = json.load(open(ref_f))['test_map50']
            print('repro cap-off %s seed %d: %.6f vs stored %.6f %s' % (att, seed, v, ref, 'OK' if v == ref else 'MISMATCH'), flush=True)
            bad += [(att, seed)] if v != ref else []
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'analyse'])
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'analyse':
        import analyse_phase5d; analyse_phase5d.main(); sys.exit(0)
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
