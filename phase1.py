#!/usr/bin/env python
"""
Phase 1 experiments (COD10K + simulated LWIR from sgn.py).

Configs
  rgb_only        RGB backbone only
  concat          two backbones, 1x1-conv concat fusion (paper baseline)
  concat_drop     concat + modality dropout (thermal image blacked out, p=0.25)
  concat_drop_d0  concat + "crossover dropout" (delta forced to 0, p=0.25)
  concat_d0_p10   concat + crossover dropout (p=0.10)
  concat_d0_p50   concat + crossover dropout (p=0.50)
  gate_oracle     concat + theoretical oracle gate (perfect alpha per delta)
  gate_raw        concat + thermal gate fed by RAW-LWIR statistics, seg loss only
  gate_sup        gate_raw + gate supervised by a delta-derived target
  gate_sup_drop   gate_sup + modality dropout
Extra (--rule): threshold-rule baseline and an ORACLE switch (upper bound for any gate),
built from the trained rgb_only and concat checkpoints.

Usage
  python -u phase1.py --smoke
  python -u phase1.py --configs rgb_only,concat,concat_drop,concat_drop_d0,concat_d0_p10,concat_d0_p50 --seeds 0,1,2
  python -u phase1.py --configs gate_raw,gate_sup,gate_sup_drop,gate_oracle --seeds 0,1,2
  python -u phase1.py --rule --seeds 0,1,2
  python -u phase1.py --report
"""
import os, sys, glob, json, time, random, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image

# Import core simulation from sgn.py
from sgn import simulate_lwir

ROOT = '/kaggle/input/datasets/ivanomelchenkoim11/cod10k-dataset/COD10K-v3'
BASE = '/kaggle/working'
CACHE = f'{BASE}/phase1_cache'
SIZE = 352
DELTAS_TEST = [8, 4, 2, 1, 0]
DELTAS_TRAIN = [0, 0.5, 1, 2, 4, 8]
EVAL_SEED = 12345
LAMBDA_SUP, K_SIG, D0_SIG = 1.0, 4.0, 0.5     # gate-target sigmoid(K*(delta-D0)); fixed a priori
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = DEV == 'cuda'
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

LAST_DELTA = {}
CONFIGS = {
    'rgb_only':       dict(fusion='none'),
    'concat':         dict(fusion='concat'),
    'concat_drop':    dict(fusion='concat', drop=0.25, mode='zero'),
    'concat_drop_d0': dict(fusion='concat', drop=0.25, mode='d0'),
    'concat_d0_p10':  dict(fusion='concat', drop=0.10, mode='d0'),
    'concat_d0_p50':  dict(fusion='concat', drop=0.50, mode='d0'),
    'gate_oracle':    dict(fusion='oracle'),
    'gate_raw':       dict(fusion='gate'),
    'gate_sup':       dict(fusion='gate', sup=True),
    'gate_sup_drop':  dict(fusion='gate', sup=True, drop=0.25, mode='zero'),
}

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ----------------------------------------------------------------------------- data
def load_pairs(files):
    imgs = np.empty((len(files), SIZE, SIZE, 3), np.uint8)
    msks = np.empty((len(files), SIZE, SIZE), np.uint8)
    for i, p in enumerate(files):
        im = Image.open(p).convert('RGB').resize((SIZE, SIZE), Image.BILINEAR)
        mp = p.replace('/Image/', '/GT_Object/').replace('.jpg', '.png')
        mk = Image.open(mp).convert('L').resize((SIZE, SIZE), Image.NEAREST)
        imgs[i] = np.asarray(im); msks[i] = (np.asarray(mk) > 127)
        if i % 500 == 0:
            print(f'  caching {i}/{len(files)}', flush=True)
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(msks).unsqueeze(1).contiguous()

def get_data(root=ROOT):
    os.makedirs(CACHE, exist_ok=True)
    path = f'{CACHE}/cod10k_cam.pt'
    if os.path.exists(path):
        d = torch.load(path)
        print('loaded cache', {k: tuple(v[0].shape) for k, v in d.items()}, flush=True)
        return d
    files = sorted(glob.glob(f'{root}/**/Image/COD10K-CAM-*.jpg', recursive=True))
    tr = [f for f in files if '/train/' in f.lower()]
    te = [f for f in files if '/test/' in f.lower()]
    print(f'camouflaged images found: train={len(tr)} test={len(te)} (expected 3040 / 2026)', flush=True)
    if len(tr) != 3040 or len(te) != 2026:
        raise RuntimeError(f'Unexpected counts. Sample path: {files[:1]}. Fix the glob/split logic.')
    d = {'train': load_pairs(tr), 'test': load_pairs(te)}
    torch.save(d, path)
    return d

# ----------------------------------------------------------------------------- simulation helpers
def make_lwir(rgb, mask, delta):
    LAST_DELTA['d'] = delta
    lw = simulate_lwir(rgb, mask, delta=delta)
    if lw.dim() == 3:
        lw = lw.unsqueeze(1)
    lw = lw.float()
    if lw.max() > 1.5:
        lw = lw / 255.
    return lw.clamp(0, 1)

def shadow(rgb, p=0.3):
    B = rgb.size(0)
    eta = torch.randn(B, 1, 8, 8, device=rgb.device)
    m = 0.25 + 0.75 * torch.sigmoid(2 * F.interpolate(eta, size=rgb.shape[-2:], mode='bilinear', align_corners=False))
    sel = (torch.rand(B, 1, 1, 1, device=rgb.device) < p)
    return torch.where(sel, rgb * m, rgb)

def norm_rgb(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)

def norm_lw(lw):
    return norm_rgb(lw.repeat(1, 3, 1, 1))

# ----------------------------------------------------------------------------- raw-LWIR descriptors
def _rp(x, p):
    return F.pad(x, (p, p, p, p), mode='replicate')

def descriptors(lw):
    """lw: B,1,H,W raw thermal in [0,1]. Returns B,4 = log[local std, grad mean, high-pass energy, grad p99]."""
    x = lw.float()
    k = torch.ones(1, 1, 7, 7, device=x.device) / 49.
    m = F.conv2d(_rp(x, 3), k); v = (F.conv2d(_rp(x * x, 3), k) - m * m).clamp_min(1e-10)
    lstd = v.sqrt().mean((1, 2, 3))
    sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).view(1, 1, 3, 3)
    sy = sx.transpose(2, 3)
    xp = _rp(x, 1)
    g = torch.sqrt(F.conv2d(xp, sx) ** 2 + F.conv2d(xp, sy) ** 2 + 1e-10)
    gmean = g.mean((1, 2, 3))
    gp99 = torch.quantile(g.flatten(1), 0.99, dim=1)
    hp = (x - F.avg_pool2d(_rp(x, 2), 5, 1)).abs().mean((1, 2, 3))
    return torch.log(torch.stack([lstd, gmean, hp, gp99], 1) + 1e-5)

# ----------------------------------------------------------------------------- model
def feats(bb, x):
    x = bb.maxpool(bb.relu(bb.bn1(bb.conv1(x))))
    x = bb.layer1(x); f3 = bb.layer2(x); f4 = bb.layer3(f3); f5 = bb.layer4(f4)
    return [f3, f4, f5]

def backbone():
    w = None if os.environ.get('PHASE1_NOPRETRAIN') else tvm.ResNet18_Weights.DEFAULT
    bb = tvm.resnet18(weights=w)
    bb.fc = nn.Identity()
    return bb

class Net(nn.Module):
    def __init__(self, fusion):
        super().__init__()
        self.fusion = fusion
        chans = [128, 256, 512]
        self.rgb = backbone()
        if fusion != 'none':
            self.ir = backbone()
            self.mix = nn.ModuleList([nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True)) for c in chans])
        if fusion == 'gate':
            self.register_buffer('mu', torch.zeros(4)); self.register_buffer('sd', torch.ones(4))
            self.gate = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 3))
            nn.init.zeros_(self.gate[2].weight); nn.init.zeros_(self.gate[2].bias)   # alpha starts at 0.5
        self.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in chans])
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, rgb_n, lw_n=None, lw_raw=None):
        fr = feats(self.rgb, rgb_n)
        alpha = None
        if self.fusion != 'none':
            fi = feats(self.ir, lw_n)
            if self.fusion == 'oracle':
                alpha = torch.sigmoid(K_SIG * (LAST_DELTA['d'].float() - D0_SIG)).view(-1, 1).expand(-1, 3)
            if self.fusion == 'gate':
                with torch.autocast(DEV, enabled=False):
                    z = (descriptors(lw_raw) - self.mu) / self.sd
                    alpha = torch.sigmoid(self.gate(z))          # B,3 (one thermal weight per stage)
            fr = [self.mix[i](torch.cat([fr[i], fi[i] * (1. if alpha is None else alpha[:, i].view(-1, 1, 1, 1).to(fi[i].dtype))], 1))
                  for i in range(3)]
        s = fr[0].shape[-2:]
        y = sum(F.interpolate(l(f), size=s, mode='bilinear', align_corners=False) for l, f in zip(self.lat, fr))
        y = self.out(self.head(y))
        return F.interpolate(y, size=rgb_n.shape[-2:], mode='bilinear', align_corners=False), alpha

def seg_loss(logit, m):
    bce = F.binary_cross_entropy_with_logits(logit, m)
    p = torch.sigmoid(logit)
    inter = (p * m).sum((1, 2, 3)); union = (p + m - p * m).sum((1, 2, 3))
    return bce + (1 - (inter + 1) / (union + 1)).mean()

def iou_per_image(logit, m):
    pred = (torch.sigmoid(logit.float()) > 0.5).float()
    inter = (pred * m).sum((1, 2, 3)); union = (pred + m - pred * m).sum((1, 2, 3))
    return (inter + 1e-6) / (union + 1e-6)

@torch.no_grad()
def calibrate_gate(net, tr_rgb, tr_msk, n=256):
    n = min(n, len(tr_rgb))
    idx = torch.randperm(len(tr_rgb), device=tr_rgb.device)[:n]
    zs = []
    DT = torch.tensor(DELTAS_TRAIN, device=tr_rgb.device)
    for i in range(0, n, 32):
        j = idx[i:i + 32]
        d = DT[torch.randint(0, len(DT), (len(j),), device=tr_rgb.device)]
        zs.append(descriptors(make_lwir(tr_rgb[j].float() / 255, tr_msk[j].float(), d)))
    z = torch.cat(zs)
    net.mu.copy_(z.mean(0)); net.sd.copy_(z.std(0) + 1e-3)

# ----------------------------------------------------------------------------- train / eval
def train_one(name, seed, data, args, out):
    cfg = CONFIGS[name]
    seed_all(seed)
    net = Net(cfg['fusion']).to(DEV)
    tr_rgb, tr_msk = data['train'][0].to(DEV), data['train'][1].to(DEV)
    if cfg['fusion'] == 'gate':
        calibrate_gate(net, tr_rgb, tr_msk)
    B, N = 16, len(tr_rgb)
    ipe = N // B
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=args.epochs * ipe)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    DT = torch.tensor(DELTAS_TRAIN, device=DEV)
    t0 = time.time(); step = 0
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(N, device=DEV)
        run = 0.
        for it in range(ipe):
            idx = perm[it * B:(it + 1) * B]
            rgb = tr_rgb[idx].float() / 255; msk = tr_msk[idx].float()
            flip = (torch.rand(B, device=DEV) < 0.5).view(-1, 1, 1, 1)
            rgb = torch.where(flip, rgb.flip(-1), rgb); msk = torch.where(flip, msk.flip(-1), msk)
            delta = DT[torch.randint(0, len(DT), (B,), device=DEV)]
            drop = torch.rand(B, device=DEV) < cfg.get('drop', 0.0)
            delta_eff = torch.where(drop, torch.zeros_like(delta), delta)
            with torch.no_grad():
                lw = make_lwir(rgb, msk, delta_eff if cfg.get('mode') == 'd0' else delta) if cfg['fusion'] != 'none' else None
                if lw is not None and cfg.get('mode') == 'zero':
                    lw = lw * (~drop).float().view(-1, 1, 1, 1)
                rgb = shadow(rgb)
            with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
                logit, alpha = net(norm_rgb(rgb), None if lw is None else norm_lw(lw), lw)
            loss = seg_loss(logit.float(), msk)
            if cfg.get('sup') and alpha is not None:
                tgt = torch.sigmoid(K_SIG * (delta_eff - D0_SIG)).view(-1, 1).expand_as(alpha)
                loss = loss + LAMBDA_SUP * F.mse_loss(alpha, tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); step += 1
            if step % 50 == 0 or step == 1:
                el = time.time() - t0
                tot = args.epochs * ipe
                print(f'[{name} s{seed}] ep {ep + 1}/{args.epochs} it {it + 1}/{ipe} loss {run / (it + 1):.4f} '
                      f'{1000 * el / step:.0f} ms/it elapsed {el / 60:.1f}m ETA {(tot - step) * el / step / 60:.1f}m', flush=True)
    os.makedirs(f'{out}/ckpt', exist_ok=True)
    torch.save(net.state_dict(), f'{out}/ckpt/{name}_s{seed}.pt')
    res = evaluate(net, cfg, data['test'])
    res.update(cfg=name, seed=seed, train_min=round((time.time() - t0) / 60, 2))
    print(f'[{name} s{seed}] DONE mIoU', {k: round(v, 1) for k, v in res['miou'].items()},
          'alpha(P3)', None if res['alpha'] is None else {k: round(v[0], 2) for k, v in res['alpha'].items()}, flush=True)
    return res

@torch.no_grad()
def evaluate(net, cfg, te, bs=32):
    net.eval()
    te_rgb, te_msk = te
    miou, alph = {}, {}
    for d in DELTAS_TEST:
        seed_all(EVAL_SEED)
        ious, als = [], []
        for i in range(0, len(te_rgb), bs):
            rgb = te_rgb[i:i + bs].to(DEV).float() / 255; msk = te_msk[i:i + bs].to(DEV).float()
            lw = None if cfg['fusion'] == 'none' else make_lwir(rgb, msk, torch.full((len(rgb),), float(d), device=DEV))
            with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
                logit, alpha = net(norm_rgb(rgb), None if lw is None else norm_lw(lw), lw)
            ious.append(iou_per_image(logit, msk).cpu())
            if alpha is not None:
                als.append(alpha.float().cpu())
        miou[str(d)] = 100 * torch.cat(ious).mean().item()
        if als:
            alph[str(d)] = torch.cat(als).mean(0).tolist()
    return dict(miou=miou, alpha=alph or None)

def load_net(name, seed, out):
    net = Net(CONFIGS[name]['fusion']).to(DEV)
    net.load_state_dict(torch.load(f'{out}/ckpt/{name}_s{seed}.pt', map_location=DEV)); net.eval()
    return net

@torch.no_grad()
def run_rule(seed, data, out):
    a, b = load_net('rgb_only', seed, out), load_net('concat', seed, out)
    tr_rgb, tr_msk = data['train'][0].to(DEV), data['train'][1].to(DEV)
    seed_all(seed)
    sub = torch.randperm(len(tr_rgb), device=DEV)[:min(400, len(tr_rgb))]
    stats, labels = [], []
    for d in DELTAS_TRAIN:
        for i in range(0, len(sub), 50):
            j = sub[i:i + 50]
            lw = make_lwir(tr_rgb[j].float() / 255, tr_msk[j].float(), torch.full((len(j),), float(d), device=DEV))
            stats.append(descriptors(lw)[:, 3].cpu()); labels.append(torch.full((len(j),), float(d >= 1)))
    s = torch.cat(stats).numpy(); y = torch.cat(labels).numpy() > 0.5

    def bacc(t):
        p = s >= t
        return 0.5 * ((p & y).sum() / max(y.sum(), 1) + ((~p) & (~y)).sum() / max((~y).sum(), 1))
    tau = max(np.quantile(s, np.linspace(0.02, 0.98, 97)), key=bacc)
    print(f'[rule s{seed}] tau={tau:.3f} balanced-acc on train subset={bacc(tau):.3f}', flush=True)

    te_rgb, te_msk = data['test']
    R = {'rule': {}, 'oracle': {}}; frac = {}
    for d in DELTAS_TEST:
        seed_all(EVAL_SEED)
        ir, io, fr = [], [], []
        for i in range(0, len(te_rgb), 32):
            rgb = te_rgb[i:i + 32].to(DEV).float() / 255; msk = te_msk[i:i + 32].to(DEV).float()
            lw = make_lwir(rgb, msk, torch.full((len(rgb),), float(d), device=DEV))
            with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
                la, _ = a(norm_rgb(rgb))
                lb, _ = b(norm_rgb(rgb), norm_lw(lw), lw)
            use_thermal = descriptors(lw)[:, 3] >= tau
            lr = torch.where(use_thermal.view(-1, 1, 1, 1), lb.float(), la.float())
            lo = lb.float() if d >= 1 else la.float()
            ir.append(iou_per_image(lr, msk).cpu()); io.append(iou_per_image(lo, msk).cpu())
            fr.append((~use_thermal).float().cpu())
        R['rule'][str(d)] = 100 * torch.cat(ir).mean().item()
        R['oracle'][str(d)] = 100 * torch.cat(io).mean().item()
        frac[str(d)] = torch.cat(fr).mean().item()
    print(f'[rule s{seed}] rule  ', {k: round(v, 1) for k, v in R['rule'].items()}, 'routed-to-RGB', {k: round(v, 2) for k, v in frac.items()})
    print(f'[rule s{seed}] oracle', {k: round(v, 1) for k, v in R['oracle'].items()}, flush=True)
    return [dict(cfg='rule', seed=seed, miou=R['rule'], alpha=None, tau=float(tau), routed_rgb=frac),
            dict(cfg='oracle', seed=seed, miou=R['oracle'], alpha=None)]

# ----------------------------------------------------------------------------- report
def report(out):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in open(f'{out}/results.jsonl')]
    order = list(CONFIGS) + ['rule', 'oracle']
    cfgs = [c for c in order if any(r['cfg'] == c for r in rows)]
    print('mIoU % (mean +- std over seeds)   delta:', DELTAS_TEST)
    M = {}
    for c in cfgs:
        A = np.array([[r['miou'][str(d)] for d in DELTAS_TEST] for r in rows if r['cfg'] == c])
        M[c] = A
        print(f'{c:15s} n={len(A)} ' + '  '.join(f'{m:5.1f}+-{s:3.1f}' for m, s in zip(A.mean(0), A.std(0))))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c in cfgs:
        ax.errorbar(DELTAS_TEST, M[c].mean(0), yerr=M[c].std(0), marker='o', capsize=2, label=c, ls='--' if c == 'oracle' else '-')
    ax.invert_xaxis(); ax.set_xlabel('thermal contrast delta (K)'); ax.set_ylabel('mIoU (%)'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f'{out}/miou_vs_delta.png', dpi=200)
    gates = [c for c in cfgs if c.startswith('gate')]
    if gates:
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
        for st, ax in enumerate(axs):
            for c in gates:
                A = np.array([[r['alpha'][str(d)][st] for d in DELTAS_TEST] for r in rows if r['cfg'] == c])
                ax.errorbar(DELTAS_TEST, A.mean(0), yerr=A.std(0), marker='o', capsize=2, label=c)
            ax.set_title(f'P{st + 3}'); ax.invert_xaxis(); ax.set_xlabel('delta (K)'); ax.grid(alpha=.3)
        axs[0].set_ylabel('mean thermal gate'); axs[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f'{out}/alpha_vs_delta.png', dpi=200)
    print('saved figures to', out)

# ----------------------------------------------------------------------------- main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', default=','.join(CONFIGS))
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--rule', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--root', default=ROOT)
    args = ap.parse_args()
    out = f'{BASE}/phase1' + ('_smoke' if args.smoke else '')
    os.makedirs(out, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    if args.report:
        report(out); sys.exit()
    data = get_data(root=args.root)
    if args.smoke:
        args.epochs = 1
        data = {'train': (data['train'][0][:160], data['train'][1][:160]), 'test': (data['test'][0][:64], data['test'][1][:64])}
    seeds = [int(s) for s in args.seeds.split(',')]
    res_path = f'{out}/results.jsonl'

    def done():
        return {(r['cfg'], r['seed']) for r in map(json.loads, open(res_path))} if os.path.exists(res_path) else set()

    if args.rule:
        for s in seeds:
            if ('rule', s) in done():
                continue
            with open(res_path, 'a') as f:
                for r in run_rule(s, data, out):
                    f.write(json.dumps(r) + '\n')
        sys.exit()
    for s in seeds:
        for c in args.configs.split(','):
            if (c, s) in done():
                print(f'skip {c} s{s} (already done)'); continue
            r = train_one(c, s, data, args, out)
            with open(res_path, 'a') as f:
                f.write(json.dumps(r) + '\n')
            torch.cuda.empty_cache()