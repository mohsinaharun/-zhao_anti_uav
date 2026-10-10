# =============================================================================
# Phase 4c (restart/phase4c/PREREG.md, approved 2026-10-11): does "less noise late beats
# uniform" replicate on CIFAR-10 and Pascal VOC? Same FL / DP-SGD / C* / CRN machinery as
# fl_env.FLEnv (seed formulas copied verbatim); only data, head, loss and score differ.
#   python phase4c.py auto --task cifar10     (every stage in order; resumable per run)
#   python phase4c.py analyse --task cifar10
# =============================================================================
import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, budget, phase4 as P4
from gpu_guard import GUARD

EXT = os.path.join(C.CACHE, 'ext')                               # raw downloads (gitignored)
SPLIT_SEED = 20261011
REAL_SIZES = [1970, 1884, 1500, 1300, 6031, 6162, 2999, 3300]
R = 12
SWEEP = [0, 0.5, 1, 2, 3, 4, 6, 8]
TUNE_SEEDS, PULSE_SEEDS, TEST_SEEDS = range(100, 105), range(0, 20), range(200, 220)
FALLBACK = [(9, 64), (7, 64), (7, 32)]                           # PREREG §3 G2, in order


def out_dir(task):
    return os.path.join(C.RESULTS, 'phase4c', task)


# ── task: CIFAR-10 ───────────────────────────────────────────────────────────
class CHead(nn.Module):
    """Drone head's two conv + GroupNorm layers, then global average pool + linear."""
    def __init__(self, cin=48, c=64, ncls=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, c, 3, padding=1), nn.GroupNorm(8, c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.ReLU())
        self.fc = nn.Linear(c, ncls)

    def forward(self, x):
        return self.fc(self.net(x).mean((2, 3)))


class CNet(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.body = C.fused_body(K) if K < C.CUT_MAX else nn.Identity()
        self.head = CHead()

    def forward(self, f):
        return self.head(self.body(f))


class Cifar10:
    name, trivial, metric = 'cifar10', 0.10, 'top-1 accuracy'
    PX = 128

    @staticmethod
    def net(K):
        return CNet(K)

    @staticmethod
    def loss(pred, y):
        return F.cross_entropy(pred, y, reduction='none')

    @staticmethod
    @torch.no_grad()
    def score(model, X, Y):
        hit = 0
        for s in range(0, X.shape[0], 1024):
            hit += (model(X[s:s + 1024].float()).argmax(1) == Y[0][s:s + 1024]).sum().item()
        return hit / X.shape[0]

    # data -------------------------------------------------------------------
    @classmethod
    def raw(cls):
        import torchvision
        tr = torchvision.datasets.CIFAR10(os.path.join(EXT, 'cifar10'), train=True, download=True)
        te = torchvision.datasets.CIFAR10(os.path.join(EXT, 'cifar10'), train=False, download=True)
        return (tr.data, np.array(tr.targets)), (te.data, np.array(te.targets))

    @classmethod
    def split(cls):
        f = os.path.join(C.ROOT, 'restart', 'data', 'phase4c_cifar10_split.json')
        if os.path.exists(f):
            return json.load(open(f))
        (xtr, ytr), _ = cls.raw()
        rng = np.random.default_rng(SPLIT_SEED)
        perm = rng.permutation(len(ytr))
        proxy, val, pool = perm[:5000], perm[5000:10000], perm[10000:]
        avail = {c: [int(i) for i in pool if ytr[i] == c] for c in range(10)}
        clients, mix = {}, {}
        for k, n in enumerate(REAL_SIZES):
            p = rng.dirichlet([0.5] * 10)
            want = np.floor(p * n).astype(int)
            for j in np.argsort(-(p * n - want))[:n - want.sum()]:
                want[j] += 1
            while True:                                          # redistribute exhausted classes
                short = {c: want[c] - len(avail[c]) for c in range(10) if want[c] > len(avail[c])}
                if not short:
                    break
                for c, s in short.items():
                    want[c] -= s
                    others = [o for o in range(10) if len(avail[o]) > want[o]]
                    w = np.array([p[o] + 1e-9 for o in others]); w = w / w.sum()
                    add = np.floor(w * s).astype(int)
                    for j in np.argsort(-(w * s - add))[:s - add.sum()]:
                        add[j] += 1
                    for o, a in zip(others, add):
                        want[o] += a
            idx = []
            for c in range(10):
                idx += avail[c][:want[c]]; avail[c] = avail[c][want[c]:]
            clients['c%d' % k] = sorted(idx); mix['c%d' % k] = want.tolist()
        sp = {'seed': SPLIT_SEED, 'proxy': sorted(proxy.tolist()), 'val': sorted(val.tolist()),
              'clients': clients, 'client_class_counts': mix, 'test': 'official CIFAR-10 test set (10,000)',
              'note': 'Phase 4c PREREG §1: client sizes = drone REAL_SIZES; class mix Dirichlet(0.5)'}
        C.save_json(sp, f)
        return sp

    @classmethod
    def cache(cls, K):
        f = os.path.join(C.CACHE, 'p4c_cifar10_K%d.pt' % K)
        if os.path.exists(f):
            return torch.load(f, weights_only=False)
        (xtr, ytr), (xte, yte) = cls.raw()
        body = C.frozen_prefix(K).to(C.DEV)

        def feats(x):
            out = []
            with torch.no_grad():
                for s in range(0, len(x), 500):
                    t = torch.from_numpy(x[s:s + 500]).permute(0, 3, 1, 2).float().div(255).to(C.DEV)
                    t = F.interpolate(t, size=(cls.PX, cls.PX), mode='bilinear', align_corners=False)
                    t = (t - C.MEAN.to(C.DEV)) / C.STD.to(C.DEV)
                    out.append(body(t).half().cpu())
            return torch.cat(out)
        d = {'train': feats(xtr), 'ytrain': torch.from_numpy(ytr).long(),
             'test': feats(xte), 'ytest': torch.from_numpy(yte).long()}
        torch.save(d, f)
        return d

    @classmethod
    def build(cls, K, eval_pool):
        sp, c = cls.split(), cls.cache(K)
        X, Y = c['train'], c['ytrain']
        cl = {k: torch.tensor(v) for k, v in sp['clients'].items()}
        allidx = torch.cat(list(cl.values()))
        remap = {k: torch.arange(sum(len(v) for v in list(cl.values())[:i]),
                                 sum(len(v) for v in list(cl.values())[:i + 1])) for i, k in enumerate(cl)}
        px = torch.tensor(sp['proxy'])
        if eval_pool == 'test':
            Xe, Ye = c['test'], c['ytest']
        else:
            v = torch.tensor(sp['val']); Xe, Ye = X[v], Y[v]
        d = {'X': X[allidx].to(C.DEV), 'Y': [Y[allidx].to(C.DEV)],
             'clients': {k: r.to(C.DEV) for k, r in remap.items()},
             'Xp': X[px].to(C.DEV), 'Yp': [Y[px].to(C.DEV)], 'Xe': Xe.to(C.DEV), 'Ye': [Ye.to(C.DEV)]}
        d['sizes'] = {k: len(r) for k, r in remap.items()}
        return d


# ── task: Pascal VOC, class-agnostic detection ───────────────────────────────
def encode_multi(boxes_per_img, M):
    """Drone targets extended to several objects per image (PREREG §2): objectness 1 at every
    object's centre cell; box regression at each object (padded to M, mask `has`). With one box
    per image this is exactly common.encode."""
    n = len(boxes_per_img)
    obj = torch.zeros(n, C.GH, C.GW)
    reg, cell, has = torch.zeros(n, M, 4), torch.zeros(n, M, dtype=torch.long), torch.zeros(n, M, dtype=torch.bool)
    for k, bs in enumerate(boxes_per_img):
        for m, (x, y, w, h) in enumerate(bs[:M]):
            cx, cy = x + w / 2, y + h / 2
            gi, gj = min(max(int(cx // C.STRIDE), 0), C.GW - 1), min(max(int(cy // C.STRIDE), 0), C.GH - 1)
            obj[k, gj, gi] = 1
            reg[k, m] = torch.tensor([cx / C.STRIDE - gi, cy / C.STRIDE - gj,
                                      np.log(max(w, 1.0) / C.STRIDE), np.log(max(h, 1.0) / C.STRIDE)])
            cell[k, m] = gj * C.GW + gi
            has[k, m] = True
    return obj, reg, cell, has


def loss_multi(pred, obj, reg, cell, has, alpha=0.25, gamma=2.0):
    """common.per_sample_loss with the box term summed over each image's objects."""
    logit = pred[:, 0]
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, obj, reduction='none')
    pt = p * obj + (1 - p) * (1 - obj)
    w = alpha * obj + (1 - alpha) * (1 - obj)
    focal = (w * (1 - pt) ** gamma * ce).flatten(1).sum(1)
    r = pred[:, 1:].flatten(2)                                    # N x 4 x cells
    rp = r.gather(2, cell.unsqueeze(1).expand(-1, 4, -1))         # N x 4 x M
    rp = torch.cat([torch.sigmoid(rp[:, :2]), rp[:, 2:]], 1).transpose(1, 2)   # N x M x 4
    box = ((rp - reg).abs().sum(2) * has.float()).sum(1)
    return focal + box


def map50_multi(gts, det_boxes, det_scores):
    """common.map50 for any number of GT boxes per image; VOC 'difficult' objects -> iscrowd
    (neither hits nor misses), the COCO analogue of VOC's ignore rule."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import io, contextlib
    images = [{'id': i, 'width': C.W, 'height': C.H} for i in range(len(gts))]
    anns, aid = [], 1
    for i, g in enumerate(gts):
        for b, diff in g:
            anns.append({'id': aid, 'image_id': i, 'category_id': 1, 'bbox': list(b), 'area': b[2] * b[3], 'iscrowd': int(diff)})
            aid += 1
    dets = [{'image_id': i, 'category_id': 1, 'bbox': [float(v) for v in bb], 'score': float(s)}
            for i in range(len(gts)) for bb, s in zip(det_boxes[i], det_scores[i])]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {'images': images, 'annotations': anns, 'categories': [{'id': 1, 'name': 'object'}]}
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else COCO()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1])


class Voc:
    name, metric = 'voc', 'mAP@0.5 (class-agnostic)'
    DIR = os.path.join(EXT, 'voc')
    SCALE = 0.515
    DECODE_K = 20                                                 # detections per image (drone: 5, one target)
    trivial = None                                                # median-box predictor on VAL, set in split()

    @staticmethod
    def net(K):
        return C.Net(K)

    @staticmethod
    def loss(pred, *Y):
        return loss_multi(pred, *Y)

    @classmethod
    @torch.no_grad()
    def score(cls, model, X, Y):
        bx, sc = [], []
        for s0 in range(0, X.shape[0], 256):
            b, p = C.decode(model(X[s0:s0 + 256].float()), k=cls.DECODE_K)
            bx += b.cpu().tolist(); sc += p.cpu().tolist()
        return map50_multi(Y[0], bx, sc)

    @classmethod
    def _extract(cls):
        import tarfile
        root = os.path.join(cls.DIR, 'VOCdevkit')
        if all(os.path.isdir(os.path.join(root, y)) for y in ('VOC2007', 'VOC2012')) and \
                os.path.exists(os.path.join(root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')):
            return root
        for t in ('VOCtrainval_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar', 'VOCtrainval_11-May-2012.tar'):
            with tarfile.open(os.path.join(cls.DIR, t)) as tf:
                tf.extractall(cls.DIR)
        return root

    @classmethod
    def _ann(cls, year, iid):
        """Letterbox (top-left anchored) to 640 x 512; returns [(xywh, difficult)], scale, dominant class."""
        import xml.etree.ElementTree as ET
        r = ET.parse(os.path.join(cls.DIR, 'VOCdevkit', 'VOC%s' % year, 'Annotations', iid + '.xml')).getroot()
        w, h = float(r.find('size/width').text), float(r.find('size/height').text)
        s = min(C.W / w, C.H / h)
        objs, best = [], (-1, None)
        for o in r.findall('object'):
            bb = o.find('bndbox')
            x1, y1, x2, y2 = (float(bb.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
            diff = int(o.find('difficult').text) if o.find('difficult') is not None else 0
            box = [(x1 - 1) * s, (y1 - 1) * s, (x2 - x1 + 1) * s, (y2 - y1 + 1) * s]
            objs.append((box, diff))
            area = box[2] * box[3] * (2 if not diff else 1)           # prefer non-difficult for the dominant class
            if area > best[0]:
                best = (area, o.find('name').text)
        return objs, s, best[1]

    @classmethod
    def _ids(cls, year, split):
        cls._extract()
        return [l.strip() for l in open(os.path.join(cls.DIR, 'VOCdevkit', 'VOC%s' % year, 'ImageSets', 'Main', split + '.txt'))]

    @classmethod
    def split(cls):
        f = os.path.join(C.ROOT, 'restart', 'data', 'phase4c_voc_split.json')
        if os.path.exists(f):
            sp = json.load(open(f)); cls.trivial = sp['trivial_val']; return sp
        tv = [('2007', i) for i in cls._ids('2007', 'trainval')] + [('2012', i) for i in cls._ids('2012', 'trainval')]
        dom = [cls._ann(y, i)[2] for y, i in tv]
        classes = sorted(set(dom))
        rng = np.random.default_rng(SPLIT_SEED)
        perm = rng.permutation(len(tv))
        proxy, val, pool = perm[:1500], perm[1500:3500], perm[3500:]
        sizes = [int(round(n * cls.SCALE)) for n in REAL_SIZES]
        avail = {c: [int(i) for i in pool if dom[i] == c] for c in classes}
        clients, mix = {}, {}
        nc = len(classes)
        for k, n in enumerate(sizes):
            p = rng.dirichlet([0.5] * nc)
            want = np.floor(p * n).astype(int)
            for j in np.argsort(-(p * n - want))[:n - want.sum()]:
                want[j] += 1
            while True:
                short = {j: want[j] - len(avail[classes[j]]) for j in range(nc) if want[j] > len(avail[classes[j]])}
                if not short:
                    break
                for j, s in short.items():
                    want[j] -= s
                    others = [o for o in range(nc) if len(avail[classes[o]]) > want[o]]
                    w = np.array([p[o] + 1e-9 for o in others]); w = w / w.sum()
                    add = np.floor(w * s).astype(int)
                    for jj in np.argsort(-(w * s - add))[:s - add.sum()]:
                        add[jj] += 1
                    for o, a_ in zip(others, add):
                        want[o] += a_
            idx = []
            for j, c in enumerate(classes):
                idx += avail[c][:want[j]]; avail[c] = avail[c][want[j]:]
            clients['c%d' % k] = sorted(idx); mix['c%d' % k] = dict(zip(classes, want.tolist()))
        # trivial control (Phase 1 rule): one median box per image, from the clients' boxes, scored on VAL
        cb = np.array([b for k in clients.values() for i in k for b, d in cls._ann(*tv[i])[0] if not d])
        tb = [float(np.median(cb[:, 0] + cb[:, 2] / 2)), float(np.median(cb[:, 1] + cb[:, 3] / 2)),
              float(np.median(cb[:, 2])), float(np.median(cb[:, 3]))]
        tb = [tb[0] - tb[2] / 2, tb[1] - tb[3] / 2, tb[2], tb[3]]
        vg = [cls._ann(*tv[i])[0] for i in sorted(val.tolist())]
        triv = map50_multi(vg, [[tb]] * len(vg), [[1.0]] * len(vg))
        sp = {'seed': SPLIT_SEED, 'images': ['%s/%s' % t for t in tv], 'proxy': sorted(proxy.tolist()),
              'val': sorted(val.tolist()), 'clients': clients, 'client_dominant_class_counts': mix,
              'unused': len(pool) - sum(sizes), 'trivial_box': tb, 'trivial_val': triv,
              'test': 'VOC2007 test (4,952 images)',
              'note': 'Phase 4c PREREG §1: drone REAL_SIZES x 0.515; dominant-class mix Dirichlet(0.5)'}
        C.save_json(sp, f)
        cls.trivial = triv
        return sp

    @classmethod
    def cache(cls, K):
        f = os.path.join(C.CACHE, 'p4c_voc_K%d.pt' % K)
        if os.path.exists(f):
            return torch.load(f, weights_only=False)
        import cv2
        sp = cls.split()
        body = C.frozen_prefix(K).to(C.DEV)
        tv = [s.split('/') for s in sp['images']]
        te = [('2007', i) for i in cls._ids('2007', 'test')]

        def run(items):
            feats, anns = [], []
            for s0 in range(0, len(items), 64):
                ims = []
                for y, i in items[s0:s0 + 64]:
                    objs, s, _ = cls._ann(y, i); anns.append(objs)
                    im = cv2.cvtColor(cv2.imread(os.path.join(cls.DIR, 'VOCdevkit', 'VOC%s' % y, 'JPEGImages', i + '.jpg')), cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, (int(round(im.shape[1] * s)), int(round(im.shape[0] * s))), interpolation=cv2.INTER_LINEAR)
                    canvas = np.zeros((C.H, C.W, 3), np.uint8); canvas[:im.shape[0], :im.shape[1]] = im[:C.H, :C.W]
                    ims.append(torch.from_numpy(canvas).permute(2, 0, 1))
                x = (torch.stack(ims).float().div(255) - C.MEAN) / C.STD
                with torch.no_grad():
                    feats.append(body(x.to(C.DEV)).half().cpu())
            return torch.cat(feats), anns
        Xtr, Atr = run(tv)
        Xte, Ate = run(te)
        d = {'train': Xtr, 'ann_train': Atr, 'test': Xte, 'ann_test': Ate}
        torch.save(d, f)
        return d

    @classmethod
    def build(cls, K, eval_pool):
        sp, c = cls.split(), cls.cache(K)
        cl = {k: torch.tensor(v) for k, v in sp['clients'].items()}
        allidx = torch.cat(list(cl.values()))
        offs = np.cumsum([0] + [len(v) for v in cl.values()])
        remap = {k: torch.arange(offs[i], offs[i + 1]) for i, k in enumerate(cl)}
        tgt = lambda idx: [t.to(C.DEV) for t in encode_multi([[b for b, d in c['ann_train'][i] if not d] for i in idx], cls.M)]
        cls.M = max(len([b for b, d in a if not d]) for a in c['ann_train'])
        px = sp['proxy']
        if eval_pool == 'test':
            Xe, Ye = c['test'], c['ann_test']
        else:
            Xe, Ye = c['train'][torch.tensor(sp['val'])], [c['ann_train'][i] for i in sp['val']]
        d = {'X': c['train'][allidx].to(C.DEV), 'Y': tgt(allidx.tolist()),
             'clients': {k: r.to(C.DEV) for k, r in remap.items()},
             'Xp': c['train'][torch.tensor(px)].to(C.DEV), 'Yp': tgt(px), 'Xe': Xe.to(C.DEV), 'Ye': [Ye]}
        d['sizes'] = {k: len(r) for k, r in remap.items()}
        return d


TASKS = {'cifar10': Cifar10, 'voc': Voc}


# ── FL environment (fl_env.FLEnv with pluggable task; seed formulas verbatim) ──
class Env:
    def __init__(self, task, data, seed, B=64, local_iters=20, lr=0.03, K=9, calib_n=512):
        self.T, self.d, self.seed, self.B, self.it, self.lr, self.K, self.calib_n = task, data, seed, B, local_iters, lr, K, calib_n

    def reset(self):
        from opacus import GradSampleModule
        d, s, T = self.d, self.seed, self.T
        torch.manual_seed(s)
        self.glob = T.net(self.K).to(C.DEV)
        cal = T.net(self.K).to(C.DEV); cal.load_state_dict(self.glob.state_dict())
        copt = torch.optim.SGD(cal.parameters(), lr=self.lr, momentum=0.9)
        cg = torch.Generator().manual_seed(s * 15485863)
        for _ in range(2 * self.it):
            jj = torch.randint(0, d['Xp'].shape[0], (self.B,), generator=cg).to(C.DEV)
            copt.zero_grad(); T.loss(cal(d['Xp'][jj].float()), *[y[jj] for y in d['Yp']]).mean().backward(); copt.step()
        gm = GradSampleModule(T.net(self.K).to(C.DEV)); gm._module.load_state_dict(cal.state_dict())
        cj = torch.randperm(d['Xp'].shape[0], generator=torch.Generator().manual_seed(s * 104729 + 999))[:self.calib_n].to(C.DEV)
        norms = []
        for s0 in range(0, len(cj), 128):
            jj = cj[s0:s0 + 128]; gm.zero_grad()
            T.loss(gm(d['Xp'][jj].float()), *[y[jj] for y in d['Yp']]).mean().backward()
            norms.append(torch.cat([q.grad_sample.flatten(1) for q in gm.parameters()], 1).norm(dim=1))
        self.C_star = float(torch.quantile(torch.cat(norms), 0.5))
        torch.manual_seed(s)
        self.r, self.sigmas = 0, []

    def step(self, sigma):
        from opacus import PrivacyEngine
        d, s, r, B, T = self.d, self.seed, self.r, self.B, self.T
        gstate = {k: v.detach().clone() for k, v in self.glob.state_dict().items()}
        new = {}
        for ci, (c, idx) in enumerate(d['clients'].items()):
            torch.manual_seed(s * 1000003 + r * 1009 + ci)
            local = T.net(self.K).to(C.DEV); local.load_state_dict(gstate)
            opt = torch.optim.SGD(local.parameters(), lr=self.lr, momentum=0.9)
            dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(len(idx))), batch_size=B,
                                             generator=torch.Generator().manual_seed(s * 7919 + r * 31 + ci))
            local, opt, dl = PrivacyEngine(accountant='rdp').make_private(
                module=local, optimizer=opt, data_loader=dl, noise_multiplier=float(sigma),
                max_grad_norm=self.C_star, poisson_sampling=True,
                grad_sample_mode='hooks' if self.K == C.CUT_MAX else 'ew')
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
                    T.loss(local(d['X'][j].float()), *[y[j] for y in d['Y']]).mean().backward()
                    if not opt.pre_step():
                        continue
                    opt.original_optimizer.step()
                    GUARD.tick()
            new[c] = {k: v.detach() for k, v in local._module.state_dict().items()}
        tot = sum(d['sizes'].values())
        self.glob.load_state_dict({k: sum(new[c][k].float() * d['sizes'][c] for c in d['clients']) / tot for k in gstate})
        self.sigmas.append(float(sigma)); self.r += 1

    def evaluate(self):
        m = self.glob.eval()
        v = self.T.score(m, self.d['Xe'], self.d['Ye'])
        self.glob.train()
        return float(v)


# ── runs ─────────────────────────────────────────────────────────────────────
class Ctx:
    def __init__(self, task):
        self.task, self.T, self.out = task, TASKS[task], out_dir(task)
        self.cfg_f = os.path.join(self.out, 'config.json')
        self.cfg = json.load(open(self.cfg_f)) if os.path.exists(self.cfg_f) else {'K': 9, 'B': 64, 'log': []}
        self._data = {}

    def save_cfg(self):
        C.save_json(self.cfg, self.cfg_f)

    def data(self, pool):
        key = (pool, self.cfg['K'])
        if key not in self._data:
            self._data = {key: self.T.build(self.cfg['K'], pool)}
        return self._data[key]

    def qs(self):
        sz = self.T.split()['clients']
        return [self.cfg['B'] / len(v) for v in sz.values()]

    def run(self, name, pool, seed, sigmas):
        f = os.path.join(self.out, name + '.json')
        if os.path.exists(f):
            return json.load(open(f))
        t0 = time.time()
        env = Env(self.T, self.data(pool), seed, B=self.cfg['B'], K=self.cfg['K']); env.reset()
        for s in sigmas:
            env.step(s)
        qs = self.qs()
        eps = budget.eps_worst(sigmas, qs, exact=True) if min(sigmas) > 0 else float('inf')
        rec = {'task': self.task, 'name': name, 'pool': pool, 'seed': seed, 'sigmas': list(map(float, sigmas)),
               'score': env.evaluate(), 'metric': self.T.metric, 'eps_max': eps, 'C_star': env.C_star,
               'K': self.cfg['K'], 'B': self.cfg['B'], 'seconds': time.time() - t0, 'meta': C.meta()}
        C.save_json(rec, f)
        print('[%s] %s: %s %.4f  eps %.4f  (%.0f s)' % (self.task, name, self.T.metric, rec['score'], eps, rec['seconds']), flush=True)
        return rec

    def tag(self):
        return 'K%d_B%d' % (self.cfg['K'], self.cfg['B'])


def ci95(x):
    x = np.asarray(x, float); m = x.mean()
    h = 2.093 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) == 20 else \
        __import__('scipy.stats', fromlist=['t']).t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(m), [float(m - h), float(m + h)]


def stage_sweep(cx):
    """G1 + G2 + operating point, with the pre-registered fallback chain."""
    while True:
        res = {}
        for sg in SWEEP:
            res[sg] = [cx.run('sweep_%s_s%g_seed%d' % (cx.tag(), sg, s), 'val', s, [sg] * R)['score'] for s in TUNE_SEEDS]
        m = {sg: float(np.mean(v)) for sg, v in res.items()}
        s0, triv = m[0], cx.T.trivial
        g1 = float(np.mean(res[0][:3])) >= triv + 0.10
        g2 = (s0 - m[8]) >= 0.30 * (s0 - triv)
        mid = (s0 + triv) / 2
        op = min((sg for sg in SWEEP if sg > 0), key=lambda sg: abs(m[sg] - mid))
        rec = {'config': cx.tag(), 'mean': {str(k): v for k, v in m.items()}, 'scores': {str(k): v for k, v in res.items()},
               'G1': g1, 'G2': g2, 'midpoint': mid, 'operating_sigma': op,
               'eps_operating': budget.eps_worst([op] * R, cx.qs(), exact=True)}
        cx.cfg['log'].append({'time': time.strftime('%Y-%m-%d %H:%M'), 'sweep': rec})
        print('[%s] sweep %s: %s | G1 %s G2 %s -> sigma %g (eps %.3f)' % (cx.task, cx.tag(),
              ' '.join('%g:%.3f' % (k, v) for k, v in m.items()), g1, g2, op, rec['eps_operating']), flush=True)
        if g1 and g2:
            cx.cfg.update({'sigma': op, 'eps': rec['eps_operating'], 'gates': 'passed'}); cx.save_cfg()
            return True
        i = FALLBACK.index((cx.cfg['K'], cx.cfg['B']))
        if i + 1 == len(FALLBACK):
            cx.cfg['gates'] = 'FAILED'; cx.save_cfg()
            print('[%s] G1/G2 failed under every pre-registered configuration; dataset stops.' % cx.task, flush=True)
            return False
        cx.cfg['K'], cx.cfg['B'] = FALLBACK[i + 1]; cx.save_cfg()
        print('[%s] fallback -> %s' % (cx.task, cx.tag()), flush=True)


def stage_pulse(cx):
    sg = cx.cfg['sigma']
    base = {s: cx.run('pulse_%s_base_seed%d' % (cx.tag(), s), 'val', s, [sg] * R)['score'] for s in PULSE_SEEDS}
    drop = np.zeros((len(PULSE_SEEDS), R))
    for k in range(R):
        for i, s in enumerate(PULSE_SEEDS):
            sig = [sg] * R; sig[k] = 2 * sg
            drop[i, k] = base[s] - cx.run('pulse_%s_k%d_seed%d' % (cx.tag(), k + 1, s), 'val', s, sig)['score']
    ks = np.arange(1, R + 1)
    slopes = [np.polyfit(ks, row, 1)[0] for row in drop]
    third = R // 3
    shape = drop[:, :third].mean(1) - drop[:, -third:].mean(1)
    pooled = drop.mean(1)
    a = {'sigma': sg, 'drop_mean': drop.mean(0).tolist(), 'drop_ci': [ci95(drop[:, k])[1] for k in range(R)],
         'slope': ci95(slopes), 'first_minus_last_third': ci95(shape), 'pooled': ci95(pooled)}
    a['informative'] = a['pooled'][1][0] > 0
    a['trend'] = not (a['slope'][1][0] <= 0 <= a['slope'][1][1])
    a['shape_varies'] = not (a['first_minus_last_third'][1][0] <= 0 <= a['first_minus_last_third'][1][1])
    a['verdict'] = 'not identifiable' if not a['informative'] else ('VARIES' if a['trend'] or a['shape_varies'] else 'FLAT')
    C.save_json(a, os.path.join(cx.out, 'analysis_pulse.json'))
    print('[%s] pulse: drop/round %s | slope %.4f %s | first-last %.4f %s | pooled %.4f %s -> %s' % (
        cx.task, ' '.join('%.4f' % x for x in a['drop_mean']), a['slope'][0], np.round(a['slope'][1], 4),
        a['first_minus_last_third'][0], np.round(a['first_minus_last_third'][1], 4), a['pooled'][0],
        np.round(a['pooled'][1], 4), a['verdict']), flush=True)
    return a


def schedules(cx, k=None):
    sg, qs = cx.cfg['sigma'], cx.qs()
    tgt = budget.eps_worst([sg] * R, qs, exact=True)
    out = {'A1': [sg] * R}
    if k is not None:
        out['A2'] = budget.scale_to_budget(P4.shape_exp(R, k), tgt, qs)[0]
    pa = os.path.join(cx.out, 'analysis_pulse.json')
    if os.path.exists(pa) and json.load(open(pa))['informative']:
        a = np.array(json.load(open(pa))['drop_mean'])
        floor = 0.1 * np.median(a)
        a = np.maximum(a, floor if floor > 0 else 1e-4)
        out['A3'] = budget.scale_to_budget(a ** -0.25, tgt, qs)[0]
    h = []                                                        # A6: Kiani et al. spend-as-you-go
    for t in range(R):
        share = (0.5 / 0.9) ** 2 if t < R // 2 else 1.0
        h.append(budget.sigma_for_round(h, share, R - t, tgt, qs))
    out['A6'] = h
    return out, tgt


def stage_tune(cx):
    f = os.path.join(cx.out, 'tune.json')
    if os.path.exists(f):
        return json.load(open(f))['best_k']
    sg, qs = cx.cfg['sigma'], cx.qs()
    tgt = budget.eps_worst([sg] * R, qs, exact=True)
    res = {}
    for k in P4.KS:
        sig = budget.scale_to_budget(P4.shape_exp(R, k), tgt, qs)[0]
        if sig is None:
            res[str(k)] = None; continue
        res[str(k)] = float(np.mean([cx.run('tune_%s_k%g_seed%d' % (cx.tag(), k, s), 'val', s, sig)['score'] for s in TUNE_SEEDS]))
        print('[%s] tune k=%g: %.4f' % (cx.task, k, res[str(k)]), flush=True)
    best = float(max((k for k in res if res[k] is not None), key=lambda k: res[k]))
    C.save_json({'grid': res, 'best_k': best, 'meta': C.meta()}, f)
    return best


def stage_eval(cx, k):
    sch, tgt = schedules(cx, k)
    C.save_json({'schedules': sch, 'target_eps': tgt, 'best_k': k, 'config': cx.tag()}, os.path.join(cx.out, 'schedules.json'))
    for arm, sig in sch.items():
        for s in TEST_SEEDS:
            cx.run('eval_%s_seed%d' % (arm, s), 'test', s, sig)


# ── analysis (Phase 4 rule) ──────────────────────────────────────────────────
def compare(a, b):
    from scipy import stats
    d = np.array([a[s] - b[s] for s in sorted(set(a) & set(b))]); n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return {'n': n, 'diff': float(m), 'ci95': [float(m - h), float(m + h)],
            'p': float(stats.ttest_1samp(d, 0).pvalue) if se > 0 else 1.0,
            'wins': int((d > 0).sum()), 'losses': int((d < 0).sum())}


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj


def verdict(c, p_adj):
    if p_adj < 0.05:
        return 'WIN' if c['diff'] > 0 else 'LOSS'
    return 'TIE (EQUIVALENT)' if -0.02 <= c['ci95'][0] and c['ci95'][1] <= 0.02 else 'TIE'


def analyse(cx):
    sch = json.load(open(os.path.join(cx.out, 'schedules.json')))
    sc = {arm: {s: json.load(open(os.path.join(cx.out, 'eval_%s_seed%d.json' % (arm, s))))['score'] for s in TEST_SEEDS
                if os.path.exists(os.path.join(cx.out, 'eval_%s_seed%d.json' % (arm, s)))} for arm in sch['schedules']}
    res = {'task': cx.task, 'config': sch['config'], 'target_eps': sch['target_eps'], 'best_k': sch['best_k'],
           'arms': {}, 'pairs': {}}
    print('\n=== Phase 4c %s  (%s, %s, sigma %g, eps %.4f) ===' % (cx.task, cx.T.metric, sch['config'], cx.cfg['sigma'], sch['target_eps']))
    for arm, v in sc.items():
        x = np.array(list(v.values()))
        eps = budget.eps_worst(sch['schedules'][arm], cx.qs(), exact=True)
        res['arms'][arm] = {'n': len(x), 'mean': float(x.mean()), 'sd': float(x.std(ddof=1)), 'eps': eps, 'sigmas': sch['schedules'][arm]}
        print('%-3s n=%d  %.4f  sd %.4f  eps %.4f  sigma %s' % (arm, len(x), x.mean(), x.std(ddof=1), eps,
              ' '.join('%.2f' % s for s in sch['schedules'][arm])))
    fam = [p for p in (('A2', 'A1'), ('A3', 'A1'), ('A6', 'A1')) if p[0] in sc]
    for x, y in fam + [p for p in (('A3', 'A2'), ('A6', 'A2'), ('A6', 'A3')) if p[0] in sc and p[1] in sc]:
        res['pairs']['%s-%s' % (x, y)] = compare(sc[x], sc[y])
    adj = holm(np.array([res['pairs']['%s-%s' % p]['p'] for p in fam]))
    for p, pa in zip(fam, adj):
        k = '%s-%s' % p
        res['pairs'][k].update({'family': 'primary', 'p_holm': float(pa), 'verdict': verdict(res['pairs'][k], pa)})
    for k, c in res['pairs'].items():
        print('%-6s diff %+.4f [%+.4f, %+.4f]  W/L %2d/%-2d  p %.4f  %s' % (k, c['diff'], c['ci95'][0], c['ci95'][1],
              c['wins'], c['losses'], c['p'], 'p_holm %.4f %s' % (c['p_holm'], c['verdict']) if 'p_holm' in c else '(secondary)'))
    v = [res['pairs'][k]['verdict'] for k in ('A2-A1', 'A3-A1') if k in res['pairs']]
    res['replication'] = 'REVERSED' if 'LOSS' in v else ('REPLICATES' if 'WIN' in v else 'FAILS TO REPLICATE')
    print('"decay beats uniform": %s' % res['replication'])
    res['meta'] = C.meta()
    C.save_json(res, os.path.join(cx.out, 'analysis.json'))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['auto', 'prepare', 'analyse'])
    ap.add_argument('--task', choices=list(TASKS), required=True)
    a = ap.parse_args()
    cx = Ctx(a.task)
    os.makedirs(cx.out, exist_ok=True)
    if a.mode == 'prepare':
        sp = cx.T.split(); cx.T.cache(9)
        print({k: len(v) for k, v in sp['clients'].items()}, 'trivial', cx.T.trivial)
        sys.exit(0)
    if a.mode == 'analyse':
        analyse(cx); sys.exit(0)
    if cx.cfg.get('gates') != 'passed' and not stage_sweep(cx):
        sys.exit(0)
    stage_pulse(cx)
    k = stage_tune(cx)
    stage_eval(cx, k)
    analyse(cx)
    print('[%s] PHASE 4c DONE' % a.task, flush=True)
