# =============================================================================
# Phase 5b (restart/phase5b/PREREG.md, approved 2026-10-11): FLTrust robustness checks.
# Part 1: Phase 5's A1 comparison (FedAvg / FLTrust / median / Krum x none / ALIE / IPM / Fang) at
#         uniform sigma 0.5, 1, 4.  Part 2: DP-aware FLTrust variants L (noise-corrected cosine) and
#         N (noise-aware norm target) at sigma 2 (eps 1.86).  Phase 5 is imported read-only.
#   python phase5b.py run [--worker i --workers n]   (resumable; reproduction checks first)
#   python phase5b.py analyse
# =============================================================================
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, phase4 as P4, phase5 as P5
from phase5 import MAL, ATTACKS, SEEDS, N, attack, krum
from gpu_guard import GUARD

OUT = os.path.join(C.RESULTS, 'phase5b')
P5OUT = P5.OUT
LEVELS = ('0.5', '1', '4')                       # Part 1 (sigma 2 and 0 are Phase 5's, read-only)
AGGS = ('fedavg', 'fltrust', 'median', 'krum')
VARIANTS = ('fltrust_L', 'fltrust_N')            # Part 2, at sigma 2


def noise_var(sigma, C_star, lr, B, K, mom=0.9):
    """Known per-coordinate noise variance of an honest update (PREREG §4)."""
    a = [sum(mom ** k for k in range(K - j + 1)) for j in range(1, K + 1)]
    return (lr * sigma * C_star / B) ** 2 * sum(x * x for x in a)


class Env(P5.Env):
    def step(self, sigma):
        from opacus import PrivacyEngine
        d, s, r, B = self.d, self.seed, self.r, self.B
        gstate = {k: v.detach().clone() for k, v in self.glob.state_dict().items()}
        new = {}
        for ci, (c, idx) in enumerate(d['clients'].items()):          # ── verbatim phase5.Env.step
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
        if self.attack != 'none':
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
            rec['upd_norm'] = float((flat(self.glob.state_dict()) - g0).norm())
        else:
            if self.agg.startswith('fltrust'):
                gr = flat(self._root_update(gstate)) - g0
                cos = torch.nn.functional.cosine_similarity(U, gr.unsqueeze(0), dim=1)
                v = noise_var(float(sigma), self.C_star, self.lr, B, self.it)
                Dn = g0.numel()
                target = gr.norm()
                if self.agg == 'fltrust':                            # Phase 5 rule, unchanged
                    ts = torch.relu(cos)
                elif self.agg == 'fltrust_L':                        # noise-corrected cosine (PREREG §4 L)
                    dot = (U * gr.unsqueeze(0)).sum(1)
                    sh = (U.norm(dim=1) ** 2 - Dn * v).clamp_min(0).sqrt()
                    c_ = torch.where(sh > 0, dot / (gr.norm() * sh.clamp_min(1e-30)), torch.sign(dot)).clamp(-1, 1)
                    ts = torch.relu(c_)
                    rec['cos_corrected'] = dict(zip(names, c_.tolist()))
                elif self.agg == 'fltrust_N':                        # noise-aware norm target (PREREG §4 N)
                    ts = torch.relu(cos)
                    target = torch.sqrt(gr.norm() ** 2 + Dn * v)
                else:
                    raise ValueError(self.agg)
                rec.update({'trust': dict(zip(names, ts.tolist())), 'cos': dict(zip(names, cos.tolist())),
                            'root_norm': float(gr.norm()), 'noise_var': v, 'target_norm': float(target)})
                Uh = U * (target / U.norm(dim=1).clamp_min(1e-12)).unsqueeze(1)
                upd = (ts.unsqueeze(1) * Uh).sum(0) / ts.sum() if ts.sum() > 0 else torch.zeros_like(g0)
            elif self.agg == 'median':
                srt = torch.sort(U, 0).values
                upd = 0.5 * (srt[N // 2 - 1] + srt[N // 2])
            elif self.agg == 'krum':
                upd, i = krum(U); rec['krum_pick'] = names[i]
            rec['upd_norm'] = float(upd.norm())
            self.glob.load_state_dict(self._unflat(g0 + upd, gstate))
        self.log.append(rec)
        self.sigmas.append(float(sigma)); self.r += 1
        return self.observe()


def schedule(level, qs):
    if level == '2':
        return P4.schedule_for('A1', 'P', qs, None)[0]               # exactly Phase 5's A1
    if level == '4':
        return P4.schedule_for('A1', 'S2', qs, None)[0]              # exactly Phase 4's S2 uniform
    return [float(level)] * 12


def run(block, level, agg, att, seed):
    name = '%s_s%s_%s_%s_seed%d' % (block, level, agg, att, seed)
    f = os.path.join(OUT, name + '.json')
    if os.path.exists(f):
        return json.load(open(f))
    d, t0 = P5.data(), time.time()
    env = Env(d, seed, att, agg)
    env.reset()
    for sg in schedule(level, d['qs']):
        env.step(sg)
    eps = env.epsilon()
    rec = {'block': block, 'level': level, 'agg': agg, 'attack': att, 'seed': seed, 'malicious': MAL,
           'sigmas': env.sigmas, 'test_map50': env.evaluate(), 'eps_max': max(eps.values()),
           'C_star': env.C_star, 'rounds': env.log, 'seconds': time.time() - t0, 'meta': C.meta()}
    C.save_json(rec, f)
    print('%s: mAP %.4f eps %.4f (%.0f s)' % (name, rec['test_map50'], rec['eps_max'], rec['seconds']), flush=True)
    return rec


def jobs():
    J = [('sweep', l, a, t) for l in LEVELS for t in ATTACKS for a in AGGS]
    J += [('variant', '2', v, t) for t in ATTACKS for v in VARIANTS]
    return [(b, l, a, t, seed) for (b, l, a, t) in J for seed in SEEDS]


def repro_check():
    """PREREG §3: FedAvg/none at sigma 4 == Phase 4 S2 uniform, all 20 seeds.
    PREREG §4: original FLTrust rule in this code == Phase 5 FLTrust/none, seeds 200-202."""
    bad = []
    for seed in SEEDS:
        v = run('sweep', '4', 'fedavg', 'none', seed)['test_map50']
        ref = json.load(open(os.path.join(P5.P4OUT, 'eval_S2_A1_seed%d.json' % seed)))['test_map50']
        print('repro S2 seed %d: %.6f vs Phase 4 %.6f %s' % (seed, v, ref, 'OK' if v == ref else 'MISMATCH'), flush=True)
        bad += [('S2', seed)] if v != ref else []
    for seed in (200, 201, 202):
        v = run('check', '2', 'fltrust', 'none', seed)['test_map50']
        ref = json.load(open(os.path.join(P5OUT, 'main_A1_fltrust_none_seed%d.json' % seed)))['test_map50']
        print('repro FLTrust seed %d: %.6f vs Phase 5 %.6f %s' % (seed, v, ref, 'OK' if v == ref else 'MISMATCH'), flush=True)
        bad += [('fltrust', seed)] if v != ref else []
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'analyse'])
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'analyse':
        import analyse_phase5b; analyse_phase5b.main(); sys.exit(0)
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
