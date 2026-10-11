# =============================================================================
# Phase 4f (restart/phase4f/PREREG.md): VisDrone generalization check. Single class "car",
# 2 x 2 tiles letterboxed to 640 x 512, clients = exact drone sizes (so sigma 2 <-> eps 1.8569).
# Reuses Phase 4c's FL environment, stages and analysis (imported read-only).
#   python phase4f.py prepare     (split, features, scoring check)
#   python phase4f.py auto        (gate -> pulse -> tune -> test -> analysis; resumable per run)
#   python phase4f.py analyse
# =============================================================================
import os, sys, json, glob, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, budget, phase4c as P4C

OUT = os.path.join(C.RESULTS, 'phase4f')
DIR = os.path.join(C.CACHE, 'ext', 'visdrone')
SPLIT_F = os.path.join(C.ROOT, 'restart', 'data', 'phase4f_visdrone_split.json')
SPLIT_SEED = 20261012
CAR, IGNORED = 4, 0
PROXY_MAX, PROXY_MIN = 738, 600
R = P4C.R
EPS_P = budget.eps_worst([2] * R, [64 / n for n in P4C.REAL_SIZES], exact=True)   # drone P budget


# ── tiles ─────────────────────────────────────────────────────────────────────
def _read_ann(split, name):
    f = os.path.join(DIR, 'VisDrone2019-DET-%s' % split, 'annotations', name + '.txt')
    if not os.path.getsize(f):
        return np.zeros((0, 8))
    return np.loadtxt(f, delimiter=',', ndmin=2, usecols=range(8))


def _tile_geom(W0, H0, ti):
    tw, th = W0 // 2, H0 // 2
    tx0, ty0 = (ti % 2) * tw, (ti // 2) * th
    return tx0, ty0, tw, th, min(C.W / tw, C.H / th)


def tile_ann(a, W0, H0, ti):
    """[(xywh in the 640x512 letterboxed tile, ignore flag)]: a car is a target (0) if >= 50% of its
    box lies in the tile (box clipped to it), else an ignore box (1); ignored regions are ignore boxes."""
    tx0, ty0, tw, th, s = _tile_geom(W0, H0, ti)
    out = []
    for x, y, w, h, sc, cat, _, _ in a:
        cat = int(cat)
        if cat not in (CAR, IGNORED) or w <= 0 or h <= 0:
            continue
        ix0, iy0, ix1, iy1 = max(x, tx0), max(y, ty0), min(x + w, tx0 + tw), min(y + h, ty0 + th)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        vis = (ix1 - ix0) * (iy1 - iy0) / (w * h)
        ign = int(cat == IGNORED or sc == 0 or vis < 0.5)
        out.append(([(ix0 - tx0) * s, (iy0 - ty0) * s, (ix1 - ix0) * s, (iy1 - iy0) * s], ign))
    return out


def _images(split):
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(DIR, 'VisDrone2019-DET-%s' % split, 'images', '*.jpg')))


def _size(split, name):
    from PIL import Image
    return Image.open(os.path.join(DIR, 'VisDrone2019-DET-%s' % split, 'images', name + '.jpg')).size


def anns(split):
    """All tiles of a split in file-name order: names ['img#t'], annotations."""
    names, out = [], []
    for n in _images(split):
        a, (W0, H0) = _read_ann(split, n), _size(split, n)
        for ti in range(4):
            names.append('%s#%d' % (n, ti)); out.append(tile_ann(a, W0, H0, ti))
    return names, out


# ── task ──────────────────────────────────────────────────────────────────────
class VisDrone:
    name, metric = 'visdrone', 'mAP@0.5 (car, tile-level)'
    DECODE_K = 100                                                # COCO maxDets; tiles hold up to ~100 cars
    trivial, M = None, None
    net, loss = P4C.Voc.net, P4C.Voc.loss

    @classmethod
    @torch.no_grad()
    def score(cls, model, X, Y):
        bx, sc = [], []
        for s0 in range(0, X.shape[0], 256):
            b, p = C.decode(model(X[s0:s0 + 256].float()), k=cls.DECODE_K)
            bx += b.cpu().tolist(); sc += p.cpu().tolist()
        return P4C.map50_multi(Y[0], bx, sc)

    @classmethod
    def split(cls):
        if os.path.exists(SPLIT_F):
            sp = json.load(open(SPLIT_F)); cls.trivial = sp['trivial_val']; return sp
        names, A = anns('train')
        seq = [n.split('_')[0] for n in names]
        by = {}
        for i, s in enumerate(seq):
            by.setdefault(s, []).append(i)
        rng = np.random.default_rng(SPLIT_SEED)
        order = [str(s) for s in rng.permutation(sorted(by))]
        proxy, pseq = [], []
        for s in order:
            if len(proxy) >= PROXY_MIN:
                break
            if len(proxy) + len(by[s]) <= PROXY_MAX:
                proxy += by[s]; pseq.append(s)
        rest = [i for s in order if s not in pseq for i in by[s]]
        clients, o = {}, 0
        for k, n in enumerate(P4C.REAL_SIZES):
            clients['c%d' % k] = rest[o:o + n]; o += n
        cb = np.array([b for k in clients.values() for i in k for b, g in A[i] if not g])
        tb = [float(np.median(cb[:, 0] + cb[:, 2] / 2)), float(np.median(cb[:, 1] + cb[:, 3] / 2)),
              float(np.median(cb[:, 2])), float(np.median(cb[:, 3]))]
        tb = [tb[0] - tb[2] / 2, tb[1] - tb[3] / 2, tb[2], tb[3]]
        _, vA = anns('val')
        triv = P4C.map50_multi(vA, [[tb]] * len(vA), [[1.0]] * len(vA))
        cseq = {k: sorted(set(seq[i] for i in v)) for k, v in clients.items()}
        sp = {'seed': SPLIT_SEED, 'tiles': names, 'proxy': proxy, 'proxy_sequences': pseq, 'clients': clients,
              'client_sequences': cseq, 'unused': len(rest) - o, 'trivial_box': tb, 'trivial_val': triv,
              'val': 'official val, tiled (%d tiles)' % len(vA), 'test': 'official test-dev, tiled',
              'note': 'Phase 4f PREREG §3: clients = drone REAL_SIZES, filled by whole sequences in random order'}
        C.save_json(sp, SPLIT_F)
        cls.trivial = triv
        return sp

    @classmethod
    def cache(cls, K):
        f = os.path.join(C.CACHE, 'p4f_visdrone_K%d.pt' % K)
        if os.path.exists(f):
            return torch.load(f, weights_only=False)
        import cv2
        body = C.frozen_prefix(K).to(C.DEV)

        def run(split):
            feats, A, buf = [], [], []

            def flush():
                x = (torch.stack(buf).float().div(255) - C.MEAN) / C.STD
                with torch.no_grad():
                    feats.append(body(x.to(C.DEV)).half().cpu())
                buf.clear()
            for n in _images(split):
                im = cv2.cvtColor(cv2.imread(os.path.join(DIR, 'VisDrone2019-DET-%s' % split, 'images', n + '.jpg')), cv2.COLOR_BGR2RGB)
                H0, W0 = im.shape[:2]
                a = _read_ann(split, n)
                for ti in range(4):
                    tx0, ty0, tw, th, s = _tile_geom(W0, H0, ti)
                    t = cv2.resize(im[ty0:ty0 + th, tx0:tx0 + tw], (min(C.W, int(round(tw * s))), min(C.H, int(round(th * s)))),
                                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                    canvas = np.zeros((C.H, C.W, 3), np.uint8); canvas[:t.shape[0], :t.shape[1]] = t
                    buf.append(torch.from_numpy(canvas).permute(2, 0, 1)); A.append(tile_ann(a, W0, H0, ti))
                    if len(buf) == 64:
                        flush()
            if buf:
                flush()
            return torch.cat(feats), A
        d = {}
        for sp in ('train', 'val', 'test-dev'):
            d[sp], d['ann_' + sp] = run(sp)
            print('cached %s: %d tiles' % (sp, len(d['ann_' + sp])), flush=True)
        torch.save(d, f)
        return d

    @classmethod
    def targets(cls, A, idx):
        return [t.to(C.DEV) for t in P4C.encode_multi([[b for b, g in A[i] if not g] for i in idx], cls.M)]

    @classmethod
    def build(cls, K, eval_pool):
        sp, c = cls.split(), cls.cache(K)
        cls.M = max(len([b for b, g in a if not g]) for a in c['ann_train'])
        cl = {k: torch.tensor(v) for k, v in sp['clients'].items()}
        allidx = torch.cat(list(cl.values()))
        offs = np.cumsum([0] + [len(v) for v in cl.values()])
        remap = {k: torch.arange(offs[i], offs[i + 1]) for i, k in enumerate(cl)}
        ep = 'test-dev' if eval_pool == 'test' else 'val'
        d = {'X': c['train'][allidx].to(C.DEV), 'Y': cls.targets(c['ann_train'], allidx.tolist()),
             'clients': {k: r.to(C.DEV) for k, r in remap.items()},
             'Xp': c['train'][torch.tensor(sp['proxy'])].to(C.DEV), 'Yp': cls.targets(c['ann_train'], sp['proxy']),
             'Xe': c[ep].to(C.DEV), 'Ye': [c['ann_' + ep]]}
        d['sizes'] = {k: len(r) for k, r in remap.items()}
        return d


class Ctx(P4C.Ctx):
    def __init__(self):
        self.task, self.T, self.out = 'visdrone', VisDrone, OUT
        self.cfg_f = os.path.join(OUT, 'config.json')
        self.cfg = json.load(open(self.cfg_f)) if os.path.exists(self.cfg_f) else {'K': 9, 'B': 64, 'log': []}
        self._data = {}


# ── scoring check (PREREG §5.3) ───────────────────────────────────────────────
def scoring_check():
    f = os.path.join(OUT, 'scoring_check.json')
    if os.path.exists(f):
        return json.load(open(f))
    VisDrone.split()
    c = VisDrone.cache(9)
    G = c['ann_val']
    gt_direct = P4C.map50_multi(G, [[b for b, g in a if not g] for a in G], [[1.0] * len([b for b, g in a if not g]) for a in G])
    VisDrone.M = max(len([b for b, g in a if not g]) for a in c['ann_train'] + G)
    obj, reg, cell, has = P4C.encode_multi([[b for b, g in a if not g] for a in G], VisDrone.M)
    n = len(G)
    pred = torch.zeros(n, 5, C.GH, C.GW)
    pred[:, 0] = torch.where(obj > 0, 20.0, -20.0)
    off = torch.logit(reg[..., :2].clamp(1e-4, 1 - 1e-4))
    for k in range(n):
        for m in range(int(has[k].sum())):
            gj, gi = divmod(int(cell[k, m]), C.GW)
            pred[k, 1:3, gj, gi] = off[k, m]; pred[k, 3:5, gj, gi] = reg[k, m, 2:]
    bx, sc = [], []
    for s0 in range(0, n, 256):
        b, p = C.decode(pred[s0:s0 + 256], k=VisDrone.DECODE_K)
        bx += b.tolist(); sc += p.tolist()
    ceiling = P4C.map50_multi(G, bx, sc)
    ncar = sum(len([b for b, g in a if not g]) for a in G)
    rec = {'gt_direct_map50': gt_direct, 'encode_decode_ceiling_map50': ceiling, 'val_tiles': n, 'val_cars': ncar,
           'cars_sharing_a_cell': int(ncar - sum(int(o.sum()) for o in obj)), 'trivial_val': VisDrone.trivial,
           'pass': gt_direct >= 0.99, 'meta': C.meta()}
    C.save_json(rec, f)
    return rec


# ── gate (PREREG §5) ──────────────────────────────────────────────────────────
def stage_gate(cx):
    while True:
        triv = VisDrone.trivial
        bar = max(0.104, triv + 0.10)
        s0 = [cx.run('sweep_%s_s0_seed%d' % (cx.tag(), s), 'val', s, [0] * R)['score'] for s in P4C.TUNE_SEEDS]
        ok = float(np.mean(s0[:3])) >= bar
        rec = {'config': cx.tag(), 'sigma0_scores': s0, 'sigma0_mean_100_102': float(np.mean(s0[:3])), 'trivial': triv,
               'bar': bar, 'cleared': ok}
        cx.cfg['log'].append({'time': time.strftime('%Y-%m-%d %H:%M'), 'quality': rec}); cx.save_cfg()
        print('[visdrone] quality %s: sigma-0 %.4f vs bar %.4f -> %s' % (cx.tag(), rec['sigma0_mean_100_102'], bar,
              'CLEARED' if ok else 'not cleared'), flush=True)
        if ok:
            break
        i = P4C.FALLBACK.index((cx.cfg['K'], cx.cfg['B']))
        if i + 1 == len(P4C.FALLBACK):
            cx.cfg['gates'] = 'FAILED'; cx.save_cfg()
            print('[visdrone] bar not cleared under any pre-registered configuration: STOP (capacity limit).', flush=True)
            return False
        cx.cfg['K'], cx.cfg['B'] = P4C.FALLBACK[i + 1]; cx.save_cfg()
        print('[visdrone] fallback -> %s' % cx.tag(), flush=True)
    qs = cx.qs()
    sig = 2.0 if cx.cfg['B'] == 64 else float(budget.scale_to_budget(np.ones(R), EPS_P, qs)[0][0])
    m = {}
    for sg in P4C.SWEEP:                                          # reported only (PREREG §5.5)
        m[sg] = float(np.mean([cx.run('sweep_%s_s%g_seed%d' % (cx.tag(), sg, s), 'val', s, [sg] * R)['score'] for s in P4C.TUNE_SEEDS]))
    g2 = (m[0] - m[8]) >= 0.30 * (m[0] - VisDrone.trivial)
    eps = budget.eps_worst([sig] * R, qs, exact=True)
    cx.cfg.update({'sigma': sig, 'eps': eps, 'gates': 'passed', 'sweep_mean': {str(k): v for k, v in m.items()}, 'G2': g2})
    cx.save_cfg()
    print('[visdrone] sweep %s | G2 %s | operating sigma %.4f eps %.4f (target %.4f)' % (
        ' '.join('%g:%.3f' % (k, v) for k, v in m.items()), g2, sig, eps, EPS_P), flush=True)
    return True


# ── analysis: Phase 4c rule + drone/CIFAR classification (PREREG §7) ──────────
def profile(drop, base):
    drop = np.asarray(drop, float)
    t = np.arange(1, len(drop) + 1)
    return {'slope_over_base': float(np.polyfit(t, drop, 1)[0] / base), 'pooled_over_base': float(drop.mean() / base),
            'last_third_share': float(drop[-len(drop) // 3:].sum() / drop.sum()) if drop.sum() > 0 else None, 'baseline': float(base)}


def analyse(cx):
    res = P4C.analyse(cx)
    pa = json.load(open(os.path.join(OUT, 'analysis_pulse.json')))
    base = np.mean([json.load(open(f))['score'] for f in glob.glob(os.path.join(OUT, 'pulse_%s_base_seed*.json' % cx.tag()))])
    d3 = json.load(open(os.path.join(C.RESULTS, 'phase3', 'analysis_V1.json')))['cells']['2']
    c4 = os.path.join(C.RESULTS, 'phase4c', 'cifar10')
    cb = np.mean([json.load(open(f))['score'] for f in glob.glob(os.path.join(c4, 'pulse_K9_B64_base_seed*.json'))])
    prof = {'drone (Phase 3 V1, sigma 2)': profile(d3['drop_mean'], float(d3['baseline_map'])),
            'cifar10 (Phase 4c)': profile(json.load(open(os.path.join(c4, 'analysis_pulse.json')))['drop_mean'], cb),
            'visdrone (Phase 4f)': profile(pa['drop_mean'], base)}
    A = res['arms']
    best = max((a for a in ('A2', 'A3') if a in A), key=lambda a: A[a]['mean'])
    g = (A[best]['mean'] - A['A1']['mean']) / A['A1']['mean']
    win = any(res['pairs'].get(k, {}).get('verdict') == 'WIN' for k in ('A2-A1', 'A3-A1'))
    if pa['verdict'] != 'VARIES':
        cls = 'NEITHER'
    elif win and g >= 0.10:
        cls = 'DRONE-LIKE'
    elif not win or g < 0.05:
        cls = 'CIFAR-LIKE'
    else:
        cls = 'INTERMEDIATE'
    res['phase4f'] = {'better_decay_arm': best, 'relative_gain': g, 'decay_win': win, 'pulse_verdict': pa['verdict'],
                      'class': cls, 'profiles': prof, 'reference_gain': {'drone': 0.24, 'cifar10': 0.017}}
    C.save_json(res, os.path.join(OUT, 'analysis.json'))
    print('\nPhase 4f: better decay arm %s, relative gain %+.1f%%, decay WIN %s, pulse %s -> %s' % (best, 100 * g, win, pa['verdict'], cls))
    for k, v in prof.items():
        print('  %-28s baseline %.4f  slope/base %+.4f  pooled/base %+.4f  last-third share %s' % (
            k, v['baseline'], v['slope_over_base'], v['pooled_over_base'],
            '%.2f' % v['last_third_share'] if v['last_third_share'] is not None else 'n/a'))
    plot(cx, res, pa, prof)
    return res


def plot(cx, res, pa, prof):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    d3 = json.load(open(os.path.join(C.RESULTS, 'phase3', 'analysis_V1.json')))['cells']['2']['drop_mean']
    c4 = json.load(open(os.path.join(C.RESULTS, 'phase4c', 'cifar10', 'analysis_pulse.json')))['drop_mean']
    for (lab, v), dm in zip(prof.items(), (d3, c4, pa['drop_mean'])):
        ax[0].plot(range(1, R + 1), np.array(dm) / v['baseline'], marker='o', label=lab)
    ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xlabel('pulsed round'); ax[0].set_ylabel('drop / baseline score')
    ax[0].set_title('Noise-pulse sensitivity a_t (relative)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    arms = list(res['arms'])
    ax[1].bar(arms, [res['arms'][a]['mean'] for a in arms], color=['#7f7f7f', '#1f77b4', '#2ca02c', '#d62728'][:len(arms)])
    for i, a in enumerate(arms):
        ax[1].text(i, res['arms'][a]['mean'], '%.4f' % res['arms'][a]['mean'], ha='center', va='bottom', fontsize=8)
    ax[1].set_title('VisDrone car, TEST, 20 seeds, eps %.2f' % res['target_eps']); ax[1].set_ylabel('mAP@0.5')
    fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, 'plots', 'phase4f_visdrone.png'), dpi=150)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['prepare', 'auto', 'analyse']); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cx = Ctx()
    if a.mode == 'prepare':
        sp = VisDrone.split()
        print({k: len(v) for k, v in sp['clients'].items()}, 'proxy', len(sp['proxy']), 'unused', sp['unused'],
              'trivial %.4f' % VisDrone.trivial, 'eps(sigma 2) %.4f' % budget.eps_worst([2] * R, cx.qs(), exact=True), 'target %.4f' % EPS_P)
        print(scoring_check())
        sys.exit(0)
    if a.mode == 'analyse':
        analyse(cx); sys.exit(0)
    sc = scoring_check()
    if not sc['pass']:
        print('[visdrone] scoring check FAILED: %s; stop.' % sc, flush=True); sys.exit(1)
    if cx.cfg.get('gates') == 'FAILED' or (cx.cfg.get('gates') != 'passed' and not stage_gate(cx)):
        sys.exit(0)
    P4C.stage_pulse(cx)
    k = P4C.stage_tune(cx)
    P4C.stage_eval(cx, k)
    analyse(cx)
    print('[visdrone] PHASE 4f DONE', flush=True)
