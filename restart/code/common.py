# =============================================================================
# Shared pieces for the restart pipeline (Phase 1 onwards). Every choice here is
# fixed by restart/phase1/PREREG.md; change it only with a dated amendment there.
# =============================================================================
import os, sys, io, json, time, platform, subprocess, contextlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)                                   # gpu_guard lives at the repo root
DATA = os.path.join(ROOT, '3rd_Anti-UAV_train_val')
SPLIT = json.load(open(os.path.join(ROOT, 'restart', 'data', 'split.json')))
CACHE = os.path.join(ROOT, 'restart', 'cache')
RESULTS = os.path.join(ROOT, 'restart', 'results')
DEV = 'cuda:0'

W, H, STRIDE = 640, 512, 16
GW, GH = W // STRIDE, H // STRIDE                          # 40 x 32 grid
CUT_MAX = 9                                                # blocks 0..8 = stride-16 output
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ── data ─────────────────────────────────────────────────────────────────────
def frames(pool, step=5):
    """[(jpg path, xywh box or None)] for every `step`-th frame of a pool, in a
    deterministic order (session, folder, frame)."""
    out, mismatch = [], 0
    for sess in sorted(SPLIT['pools'][pool]):
        for f in sorted(SPLIT['pools'][pool][sess]['folders'], key=lambda d: d['folder']):
            d = os.path.join(DATA, f['official_split'], f['folder'])
            imgs = sorted(x for x in os.listdir(d) if x.endswith('.jpg'))
            lab = json.load(open(os.path.join(d, 'IR_label.json')))
            rects, ex = lab.get('gt_rect', []), lab.get('exist', [])
            n = min(len(imgs), len(rects), len(ex))
            mismatch += len(imgs) - n
            for i in range(0, n, step):
                gt = rects[i]
                ok = ex[i] and gt and len(gt) == 4 and gt[2] > 0 and gt[3] > 0
                out.append((os.path.join(d, imgs[i]), [float(v) for v in gt] if ok else None))
    return out, mismatch


def load_batch(paths):
    ims = []
    for p in paths:
        im = cv2.imread(p)
        if im.shape[:2] != (H, W):
            raise ValueError('unexpected size %s: %s' % (im.shape, p))
        ims.append(torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
    x = torch.stack(ims).float().div_(255)
    return (x - MEAN) / STD


_FOLDER_LIST = {}


def load_batch_temporal(paths, gap=5):
    """Causal 3-frame input: channels = grayscale frames (t-2*gap, t-gap, t), clamped
    at the folder start (Phase 1 Amendment 2)."""
    ims = []
    for p in paths:
        d, name = os.path.split(p)
        lst = _FOLDER_LIST.get(d)
        if lst is None:
            lst = _FOLDER_LIST[d] = sorted(x for x in os.listdir(d) if x.endswith('.jpg'))
        i = lst.index(name)
        ch = [cv2.imread(os.path.join(d, lst[max(0, i - k * gap)]), cv2.IMREAD_GRAYSCALE) for k in (2, 1, 0)]
        if any(c.shape != (H, W) for c in ch):
            raise ValueError('unexpected size in %s' % d)
        ims.append(torch.from_numpy(np.stack(ch)))
    x = torch.stack(ims).float().div_(255)
    return (x - MEAN) / STD


def encode(boxes):
    """Targets on the stride-16 grid. boxes: list of xywh or None."""
    n = len(boxes)
    obj = torch.zeros(n, GH, GW)
    reg = torch.zeros(n, 4)
    cell = torch.zeros(n, dtype=torch.long)
    has = torch.zeros(n, dtype=torch.bool)
    for k, b in enumerate(boxes):
        if b is None:
            continue
        x, y, w, h = b
        cx, cy = x + w / 2, y + h / 2
        gi, gj = min(max(int(cx // STRIDE), 0), GW - 1), min(max(int(cy // STRIDE), 0), GH - 1)
        obj[k, gj, gi] = 1
        reg[k] = torch.tensor([cx / STRIDE - gi, cy / STRIDE - gj,
                               np.log(max(w, 1.0) / STRIDE), np.log(max(h, 1.0) / STRIDE)])
        cell[k] = gj * GW + gi
        has[k] = True
    return obj, reg, cell, has


# ── model ────────────────────────────────────────────────────────────────────
def mobilenet():
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    return mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features


def frozen_prefix(K):
    m = mobilenet()[:K].eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class Head(nn.Module):
    def __init__(self, cin=48, c=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, c, 3, padding=1), nn.GroupNorm(8, c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.ReLU(),
            nn.Conv2d(c, 5, 1))
        nn.init.constant_(self.net[-1].bias, 0.0)
        with torch.no_grad():
            self.net[-1].bias[0] = -4.6                    # prior p ~ 0.01 for objectness

    def forward(self, x):
        return self.net(x)


def fused_body(K):
    """MobileNet blocks K..8 with every BatchNorm folded into its preceding conv
    (eval-mode statistics). Identical function and pretrained weights, but no
    BatchNorm left, so per-sample DP-SGD can train it (Phase 3 variant V3)."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    from torchvision.ops.misc import Conv2dNormActivation
    body = mobilenet()[K:CUT_MAX].eval()
    for m in body.modules():
        if isinstance(m, Conv2dNormActivation) and isinstance(m[1], nn.BatchNorm2d):
            m[0] = fuse_conv_bn_eval(m[0], m[1])
            m[0].bias.requires_grad_(True)   # fuse_conv_bn_eval leaves a new bias frozen
            m[1] = nn.Identity()
    import types
    from torchvision.models.mobilenetv3 import InvertedResidual

    def _fwd(self, x):                       # torchvision does `result += input` (in place)
        out = self.block(x)
        return out + x if self.use_res_connect else out
    for m in body.modules():                 # Opacus hooks forbid in-place ops
        if hasattr(m, 'inplace'):
            m.inplace = False
        if isinstance(m, InvertedResidual):
            m.forward = types.MethodType(_fwd, m)
    return body


def body_channels_in(K):
    return mobilenet()[:K](torch.zeros(1, 3, H, W)).shape[1]


class Net(nn.Module):
    """Trainable part: MobileNet blocks K..8 (empty for K = 9; BatchNorm folded) + head."""
    def __init__(self, K):
        super().__init__()
        self.body = fused_body(K) if K < CUT_MAX else nn.Identity()
        self.head = Head()

    def forward(self, f):
        return self.head(self.body(f))


def n_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ── loss (per-sample, so it is valid under DP-SGD's mean reduction) ──────────
def per_sample_loss(pred, obj, reg, cell, has, alpha=0.25, gamma=2.0):
    logit = pred[:, 0]
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, obj, reduction='none')
    pt = p * obj + (1 - p) * (1 - obj)
    w = alpha * obj + (1 - alpha) * (1 - obj)
    focal = (w * (1 - pt) ** gamma * ce).flatten(1).sum(1)
    r = pred[:, 1:].flatten(2)                              # N x 4 x cells
    rp = r.gather(2, cell.view(-1, 1, 1).expand(-1, 4, 1)).squeeze(2)
    rp = torch.cat([torch.sigmoid(rp[:, :2]), rp[:, 2:]], 1)
    box = (rp - reg).abs().sum(1) * has.float()
    return focal + box


# ── decoding + mAP@0.5 ───────────────────────────────────────────────────────
@torch.no_grad()
def decode(pred, k=5):
    heat = torch.sigmoid(pred[:, 0])
    peak = heat * (F.max_pool2d(heat.unsqueeze(1), 3, 1, 1).squeeze(1) == heat)
    sc, idx = peak.flatten(1).topk(k, 1)
    gj, gi = idx // GW, idx % GW
    r = pred[:, 1:].flatten(2)
    rr = r.gather(2, idx.unsqueeze(1).expand(-1, 4, -1))    # N x 4 x k
    cx = (gi + torch.sigmoid(rr[:, 0])) * STRIDE
    cy = (gj + torch.sigmoid(rr[:, 1])) * STRIDE
    w, h = torch.exp(rr[:, 2]) * STRIDE, torch.exp(rr[:, 3]) * STRIDE
    return torch.stack([cx - w / 2, cy - h / 2, w, h], 2), sc   # N x k x 4 (xywh), N x k


def map50(gt_boxes, det_boxes, det_scores):
    """gt_boxes: list (per frame) of xywh or None; det_*: per frame arrays."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images = [{'id': i, 'width': W, 'height': H} for i in range(len(gt_boxes))]
    anns = [{'id': i + 1, 'image_id': i, 'category_id': 1, 'bbox': list(b),
             'area': b[2] * b[3], 'iscrowd': 0} for i, b in enumerate(gt_boxes) if b is not None]
    dets = [{'image_id': i, 'category_id': 1, 'bbox': [float(v) for v in bb], 'score': float(s)}
            for i in range(len(gt_boxes)) for bb, s in zip(det_boxes[i], det_scores[i])]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {'images': images, 'annotations': anns, 'categories': [{'id': 1, 'name': 'uav'}]}
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else COCO()
        ev = COCOeval(gt, dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1])                               # AP @ IoU 0.50, all areas


# ── run bookkeeping ──────────────────────────────────────────────────────────
def meta():
    commit = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', 'restart/code'],
                                capture_output=True, text=True).stdout.strip())
    return {'code_commit': commit, 'code_dirty': dirty, 'machine': platform.node(),
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'time': time.strftime('%Y-%m-%d %H:%M:%S')}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)
