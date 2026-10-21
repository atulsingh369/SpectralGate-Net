#!/usr/bin/env python
"""SpectralGate-Net: minimal study on simulated thermal crossover built from COD10K.

  python sgn.py train --mode full   --root COD10K-v3
  python sgn.py train --mode fixed  --root COD10K-v3
  python sgn.py train --mode concat --root COD10K-v3
  python sgn.py train --mode rgb    --root COD10K-v3
  python sgn.py report                      # writes LaTeX tables + PDF plots

  python sgn.py train --mode full --smoke   # 1-minute CPU/GPU sanity check, synthetic data
"""
import argparse, glob, json, math, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
import torchvision
from torchvision.ops import deform_conv2d

DELTAS = [8, 4, 2, 1, 0]            # test-time target-minus-background temperature (K)
TRAIN_D = [0, 0.5, 1, 2, 4, 8]      # training-time contrast values (sampled per image)

# ------------------------------------------------------------------ thermal simulation
H_PL, C_PL, K_PL = 6.626e-34, 2.998e8, 1.381e-23


def planck_band(T):
    """Band-integrated (8-14 um) blackbody radiance, arbitrary units. T in kelvin."""
    lam = torch.linspace(8e-6, 14e-6, 13, dtype=torch.float64, device=T.device).view(1, -1, 1, 1)
    T = T.double()
    w = 2 * H_PL * C_PL ** 2 / lam ** 5 / torch.expm1(H_PL * C_PL / (lam * K_PL * T))
    return w.sum(1, keepdim=True)


def smooth_noise(B, H, W, res, dev, gen=None):
    n = torch.randn(B, 1, res, res, device=dev, generator=gen)
    return Fn.interpolate(n, size=(H, W), mode="bicubic", align_corners=False)


def gauss_blur(x, sigma=1.0, k=5):
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    x = Fn.conv2d(Fn.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return Fn.conv2d(Fn.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


@torch.no_grad()
def simulate_lwir(rgb, mask, delta, gen=None, T0=295.0, netd=0.05):
    """rgb (B,3,H,W) in [0,1]; mask (B,1,H,W) in {0,1}; delta (B,) kelvin -> LWIR (B,1,H,W) in [0,1].
    Tbg = T0 + 2K * spatially-correlated field + 2K * (luminance - 0.5)   [scene-dependent term]
    Ttarget = Tbg + delta inside mask; Gaussian PSF (sigma 1 px); NETD 50 mK; fixed 280-320 K -> 8-bit range."""
    B, _, H, W = rgb.shape
    lum = rgb.mean(1, keepdim=True)
    T = T0 + 2.0 * smooth_noise(B, H, W, 12, rgb.device, gen) + 2.0 * (lum - 0.5)
    T = T + delta.view(B, 1, 1, 1).to(rgb.device) * mask
    T = gauss_blur(T)
    T = T + netd * torch.randn(T.shape, device=T.device, generator=gen)
    lo = planck_band(torch.full((1, 1, 1, 1), 280.0, device=T.device))
    hi = planck_band(torch.full((1, 1, 1, 1), 320.0, device=T.device))
    g = ((planck_band(T) - lo) / (hi - lo)).clamp(0, 1).float()
    return torch.round(g * 255) / 255


def shadow_rgb(rgb, gen=None):
    """Dappled-shadow darkening of the visible image."""
    B, _, H, W = rgb.shape
    s = torch.sigmoid(2.0 * smooth_noise(B, H, W, 8, rgb.device, gen))
    return rgb * (0.25 + 0.75 * s)


# ------------------------------------------------------------------ Haar DWT / IDWT
def dwt(x):
    a, b, c, d = x[..., 0::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2


def idwt(ll, lh, hl, hh):
    a = (ll + lh + hl + hh) / 2; b = (ll - lh + hl - hh) / 2
    c = (ll + lh - hl - hh) / 2; d = (ll - lh - hl + hh) / 2
    B, C, H, W = ll.shape
    return Fn.pixel_shuffle(torch.stack([a, b, c, d], 2).reshape(B, C * 4, H, W), 2)


# ------------------------------------------------------------------ gate
class SoftEntropy(nn.Module):
    def __init__(self, M=16, tau=0.2):
        super().__init__()
        self.e = nn.Parameter(torch.linspace(-1, 1, M)); self.tau = tau; self.M = M

    def forward(self, t):                      # t in [-1,1], (B,C,H,W)
        v = t.float().reshape(t.shape[0], -1, 1)
        p = torch.softmax(-(v - self.e.view(1, 1, -1)) ** 2 / (2 * self.tau ** 2), dim=-1)
        ph = p.mean(1)
        return -(ph * torch.log(ph + 1e-7)).sum(-1) / math.log(self.M)   # normalised to [0,1]


class Gate(nn.Module):
    def __init__(self, extra=False):
        super().__init__()
        self.ent = SoftEntropy(); self.extra = extra
        self.mlp = nn.Sequential(nn.Linear(3 if extra else 2, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.mlp[2].weight); nn.init.zeros_(self.mlp[2].bias)   # start at alpha = 0.5

    def forward(self, f, raw_std=None):
        t = torch.tanh(f.float())
        var = t.var(dim=(2, 3)).mean(1) * 10.0
        z = [var, self.ent(t)]
        if self.extra:
            z.append(raw_std.float())
        return torch.sigmoid(self.mlp(torch.stack(z, 1))).squeeze(1)      # (B,)


# ------------------------------------------------------------------ fusion blocks
class WaveFuse(nn.Module):
    def __init__(self, C, mode, heads=4, extra=False):
        super().__init__()
        self.mode = mode
        self.off = nn.Conv2d(2 * C, 27, 3, padding=1)
        nn.init.zeros_(self.off.weight); nn.init.zeros_(self.off.bias)
        self.w_ll = nn.Parameter(torch.empty(C, C, 3, 3)); nn.init.kaiming_uniform_(self.w_ll, a=math.sqrt(5))
        self.w_hf = nn.Parameter(torch.empty(C, C, 3, 3)); nn.init.kaiming_uniform_(self.w_hf, a=math.sqrt(5))
        self.g_rgb, self.g_ir = Gate(extra), Gate(extra)
        mk = lambda: nn.ModuleList([nn.MultiheadAttention(C, heads, batch_first=True) for _ in range(3)])
        self.r2i, self.i2r = mk(), mk()                       # separate projections per direction
        self.ln = nn.ModuleList([nn.LayerNorm(C) for _ in range(3)])

    def forward(self, fr, fi, raw_r=None, raw_i=None):
        llr, lhr, hlr, hhr = dwt(fr)
        lli, lhi, hli, hhi = dwt(fi)
        with torch.autocast(device_type=fr.device.type, enabled=False):      # deform conv in fp32
            o = self.off(torch.cat([llr, lli], 1).float())
            off, m = o[:, :18], 2 * torch.sigmoid(o[:, 18:])
            lli = deform_conv2d(lli.float(), off, self.w_ll, padding=1, mask=m)
            lhi, hli, hhi = [deform_conv2d(x.float(), off, self.w_hf, padding=1, mask=m) for x in (lhi, hli, hhi)]
        B = fr.shape[0]
        if self.mode == "fixed":
            a_r = a_i = torch.full((B,), 0.5, device=fr.device)
        else:
            a_r, a_i = self.g_rgb(llr, raw_r), self.g_ir(lli, raw_i)
        self.gates = (a_r, a_i)
        ar4, ai4 = a_r.view(-1, 1, 1, 1), a_i.view(-1, 1, 1, 1)
        ll = ar4 * llr + ai4 * lli
        outs = []
        for k, (X, Y) in enumerate([(lhr, lhi), (hlr, hli), (hhr, hhi)]):
            Bc, C, h, w = X.shape
            xt, yt = X.flatten(2).transpose(1, 2), Y.to(X.dtype).flatten(2).transpose(1, 2)
            A1 = self.r2i[k](xt, yt, yt, need_weights=False)[0]                # Q from RGB, K/V from LWIR
            A2 = self.i2r[k](yt, xt, xt, need_weights=False)[0]
            o_ = a_i.view(-1, 1, 1) * A1 + a_r.view(-1, 1, 1) * A2 + self.ln[k](xt * yt)
            outs.append(o_.transpose(1, 2).reshape(Bc, C, h, w))
        return idwt(ll, *outs)


class GatedSum(nn.Module):
    def __init__(self, C, mode, extra=False):
        super().__init__()
        self.mode = mode; self.g_rgb, self.g_ir = Gate(extra), Gate(extra)

    def forward(self, fr, fi, raw_r=None, raw_i=None):
        B = fr.shape[0]
        if self.mode == "fixed":
            a_r = a_i = torch.full((B,), 0.5, device=fr.device)
        else:
            a_r, a_i = self.g_rgb(fr, raw_r), self.g_ir(fi, raw_i)
        self.gates = (a_r, a_i)
        return a_r.view(-1, 1, 1, 1) * fr + a_i.view(-1, 1, 1, 1) * fi


class ConcatFuse(nn.Module):
    def __init__(self, C):
        super().__init__(); self.c = nn.Sequential(nn.Conv2d(2 * C, C, 1), nn.BatchNorm2d(C), nn.ReLU())

    def forward(self, fr, fi, *_):
        return self.c(torch.cat([fr, fi], 1))


# ------------------------------------------------------------------ network
def run_backbone(m, x):
    x = m.maxpool(m.relu(m.bn1(m.conv1(x)))); x = m.layer1(x)
    f3 = m.layer2(x); f4 = m.layer3(f3); f5 = m.layer4(f4)
    return f3, f4, f5


class Net(nn.Module):
    def __init__(self, mode, pretrained=True, extra=False):
        super().__init__()
        self.mode, self.extra = mode, extra
        w = "IMAGENET1K_V1" if pretrained else None
        self.b_rgb = torchvision.models.resnet18(weights=w)
        if mode != "rgb":
            self.b_ir = torchvision.models.resnet18(weights=w)
        ch = (128, 256, 512)
        if mode in ("full", "fixed"):
            self.fuse = nn.ModuleList([WaveFuse(128, mode, extra=extra), WaveFuse(256, mode, extra=extra),
                                       GatedSum(512, mode, extra=extra)])
        elif mode == "concat":
            self.fuse = nn.ModuleList([ConcatFuse(c) for c in ch])
        self.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in ch])
        self.out = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, rgb, ir):
        size = rgb.shape[-2:]
        fr = run_backbone(self.b_rgb, (rgb - self.mean) / self.std)
        if self.mode == "rgb":
            fused, gates = fr, []
        else:
            fi = run_backbone(self.b_ir, ((ir - 0.45) / 0.225).expand(-1, 3, -1, -1))
            raw_r = rgb.flatten(1).std(1) * 10 if self.extra else None
            raw_i = ir.flatten(1).std(1) * 10 if self.extra else None
            fused = [f(a, b, raw_r, raw_i) for f, a, b in zip(self.fuse, fr, fi)]
            gates = [f.gates for f in self.fuse] if self.mode in ("full", "fixed") else []
        p = [l(f) for l, f in zip(self.lat, fused)]
        up = lambda t: Fn.interpolate(t, size=p[0].shape[-2:], mode="bilinear", align_corners=False)
        x = p[0] + up(p[1]) + up(p[2])
        return Fn.interpolate(self.out(x), size=size, mode="bilinear", align_corners=False), gates


# ------------------------------------------------------------------ data
def load_split(root, split, size):
    """Finds COD10K camouflaged (COD10K-CAM-*) images + GT_Object masks anywhere under root.
    Non-camouflaged / background images are skipped (they have no camouflage target)."""
    cache = f"cache_{split}_{size}.npz"
    if os.path.exists(cache):
        d = np.load(cache); return d["x"], d["m"]
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    in_split = lambda p: split.lower() in [c.lower() for c in p.split(os.sep)]
    ims = sorted(p for p in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
                 if in_split(p) and os.path.basename(p).startswith("COD10K-CAM-"))
    gts = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
           if in_split(p) and "object" in p.lower() and "instance" not in p.lower()}
    ims = [p for p in ims if os.path.splitext(os.path.basename(p))[0] in gts]
    assert ims, f"no COD10K-CAM images+GT_Object masks found for split {split} under {root}"
    print(f"{split}: {len(ims)} camouflaged images", flush=True)

    def ld(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        x = np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))
        m = np.asarray(Image.open(gts[stem]).convert("L").resize((size, size), Image.NEAREST)) > 127
        return x, m

    with ThreadPoolExecutor(8) as ex:
        r = list(ex.map(ld, ims))
    x = np.stack([a for a, _ in r]); m = np.stack([b for _, b in r]).astype(np.uint8)
    np.savez(cache, x=x, m=m)
    return x, m


def synthetic(n, size):
    x = np.random.randint(0, 255, (n, size, size, 3), dtype=np.uint8)
    m = np.zeros((n, size, size), np.uint8)
    for i in range(n):
        a, b = np.random.randint(8, size // 2, 2); m[i, a:a + size // 4, b:b + size // 4] = 1
    return x, m


def get_batch(x, m, idx, dev):
    rgb = torch.from_numpy(x[idx]).to(dev).permute(0, 3, 1, 2).float() / 255
    mk = torch.from_numpy(m[idx]).to(dev).unsqueeze(1).float()
    return rgb, mk


# ------------------------------------------------------------------ train / eval
def loss_fn(logits, mk, gates, mode, lgate=0.05):
    bce = Fn.binary_cross_entropy_with_logits(logits, mk)
    p = torch.sigmoid(logits)
    iou = 1 - ((p * mk).sum((1, 2, 3)) + 1) / ((p + mk - p * mk).sum((1, 2, 3)) + 1)
    loss = bce + iou.mean()
    if mode == "full" and gates:
        lg = sum(-0.5 * ((ar - 0.5) ** 2 + (ai - 0.5) ** 2).mean() for ar, ai in gates) / len(gates)
        loss = loss + lgate * lg
    return loss


@torch.no_grad()
def evaluate(model, x, m, dev, bs=32):
    model.eval(); res = {}
    for s in (0, 1):
        for d in DELTAS:
            gen = torch.Generator(device=dev).manual_seed(1234)
            iou = f1 = mae = n = 0.0; ar = [0.0] * 3; ai = [0.0] * 3
            for i in range(0, len(x), bs):
                idx = np.arange(i, min(i + bs, len(x)))
                rgb, mk = get_batch(x, m, idx, dev)
                ir = simulate_lwir(rgb, mk, torch.full((len(idx),), float(d)), gen)
                if s:
                    rgb = shadow_rgb(rgb, gen)
                logits, gates = model(rgb, ir)
                p = torch.sigmoid(logits); pb = (p > 0.5).float()
                inter = (pb * mk).sum((1, 2, 3)); uni = (pb + mk - pb * mk).sum((1, 2, 3))
                iou += (inter / (uni + 1e-6)).sum().item()
                f1 += (2 * inter / (pb.sum((1, 2, 3)) + mk.sum((1, 2, 3)) + 1e-6)).sum().item()
                mae += (p - mk).abs().mean((1, 2, 3)).sum().item(); n += len(idx)
                for j, (a_r, a_i) in enumerate(gates):
                    ar[j] += a_r.sum().item(); ai[j] += a_i.sum().item()
            res[f"d{d}_s{s}"] = dict(iou=iou / n, f1=f1 / n, mae=mae / n,
                                     alpha_rgb=[v / n for v in ar], alpha_ir=[v / n for v in ai])
            print(f"  delta={d} shadow={s}  IoU={iou/n:.4f}  F1={f1/n:.4f}  MAE={mae/n:.4f}  alpha_ir={[round(v/n,3) for v in ai]}", flush=True)
    return res


@torch.no_grad()
def latency(model, size, dev):
    model.eval(); x = torch.rand(1, 3, size, size, device=dev); y = torch.rand(1, 1, size, size, device=dev)
    for _ in range(10): model(x, y)
    if dev.type == "cuda": torch.cuda.synchronize()
    t = time.time()
    for _ in range(50): model(x, y)
    if dev.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t) / 50 * 1000


def train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.smoke:
        a.size, a.epochs, a.bs = 96, 1, 8
        xtr, mtr = synthetic(32, a.size); xte, mte = synthetic(16, a.size)
    else:
        xtr, mtr = load_split(a.root, "Train", a.size); xte, mte = load_split(a.root, "Test", a.size)
    print(f"train {len(xtr)}  test {len(xte)}  device {dev}", flush=True)
    model = Net(a.mode, pretrained=not a.smoke, extra=a.raw_desc).to(dev)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(xtr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    use_amp = dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    t0 = time.time(); it = 0
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(len(xtr))
        for i in range(0, len(xtr) - a.bs + 1, a.bs):
            idx = np.sort(perm[i:i + a.bs])
            rgb, mk = get_batch(xtr, mtr, idx, dev)
            if random.random() < 0.5:
                rgb, mk = rgb.flip(-1), mk.flip(-1)
            delta = torch.tensor(np.random.choice(TRAIN_D, a.bs), dtype=torch.float32)
            ir = simulate_lwir(rgb, mk, delta)
            sh = (torch.rand(a.bs, device=dev) < 0.3).view(-1, 1, 1, 1)
            rgb = torch.where(sh, shadow_rgb(rgb), rgb)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits, gates = model(rgb, ir)
            loss = loss_fn(logits.float(), mk, gates, a.mode, a.lgate)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); it += 1
            if it % 50 == 0:
                print(f"ep {ep} it {it}/{steps} loss {loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)
    print("evaluating ...", flush=True)
    res = evaluate(model, xte, mte, dev)
    out = dict(mode=a.mode, seed=a.seed, epochs=a.epochs, size=a.size, params_M=nparam,
               latency_ms_bs1=latency(model, a.size, dev), device=str(torch.cuda.get_device_name(0) if use_amp else "cpu"),
               raw_desc=a.raw_desc, lgate=a.lgate, res=res)
    name = f"results_{a.mode}{'_raw' if a.raw_desc else ''}{'_nolg' if a.lgate == 0 else ''}_seed{a.seed}.json"
    json.dump(out, open("smoke_" + name if a.smoke else name, "w"), indent=1)
    print("saved", name)


# ------------------------------------------------------------------ report
def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = {}
    for f in glob.glob("results_*_seed*.json"):
        d = json.load(open(f)); R.setdefault(d["mode"] + ("_raw" if d.get("raw_desc") else "") + ("_nolg" if d.get("lgate", 0.05) == 0 else ""), []).append(d)
    names = {"rgb": "RGB only", "concat": "Early concatenation", "fixed": r"Fixed gates ($\alpha=0.5$)",
             "full": "SpectralGate-Net", "full_nolg": r"SpectralGate-Net, no $L_{gate}$",
             "full_raw": "SpectralGate-Net + raw descriptor", "full_raw_nolg": r"SpectralGate-Net + raw desc., no $L_{gate}$"}
    order = [k for k in names if k in R]

    def st(k, key, fld="iou"):
        v = np.array([r["res"][key][fld] for r in R[k]]) * 100
        return v.mean(), v.std()

    cell = lambda k, key: (f"{st(k,key)[0]:.1f}" if len(R[k]) == 1 else f"{st(k,key)[0]:.1f}$\\pm${st(k,key)[1]:.1f}")
    for s, tag in ((0, "clean"), (1, "shadow")):
        lines = [r"\begin{tabular}{l" + "c" * len(DELTAS) + "}", r"\toprule",
                 "Method & " + " & ".join(rf"$\delta={d}$" for d in DELTAS) + r" \\", r"\midrule"]
        for k in order:
            lines.append(names[k] + " & " + " & ".join(cell(k, f"d{d}_s{s}") for d in DELTAS) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(f"table_sim_{tag}.tex", "w").write("\n".join(lines))
    eff = [r"\begin{tabular}{lcc}", r"\toprule", r"Method & Params (M) & Latency (ms, bs=1) \\", r"\midrule"]
    for k in order:
        eff.append(f"{names[k]} & {R[k][0]['params_M']:.1f} & {R[k][0]['latency_ms_bs1']:.1f} \\\\")
    eff += [r"\bottomrule", r"\end{tabular}"]
    open("table_eff.tex", "w").write("\n".join(eff))
    print("device:", R[order[0]][0]["device"])

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for s in (0, 1):
        for k in order:
            ax[s].plot(DELTAS, [st(k, f"d{d}_s{s}")[0] for d in DELTAS], marker="o", label=k)
        ax[s].set_xlabel(r"thermal contrast $\delta$ (K)"); ax[s].set_ylabel("mIoU (%)")
        ax[s].set_title("clean RGB" if s == 0 else "RGB with shadow"); ax[s].invert_xaxis()
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig("fig_iou_vs_delta.pdf")

    gk = [k for k in ("full", "full_nolg", "full_raw", "full_raw_nolg") if k in R]
    if gk:
        fig, ax = plt.subplots(figsize=(4, 3))
        for k in gk:
            for s in (0, 1):
                for j, stg in enumerate(("P3", "P4", "P5")):
                    y = [np.mean([r["res"][f"d{d}_s{s}"]["alpha_ir"][j] for r in R[k]]) for d in DELTAS]
                    ax.plot(DELTAS, y, marker="o", ls="-" if s == 0 else "--", label=f"{k} {stg} {'shadow' if s else 'clean'}")
        ax.set_xlabel(r"$\delta$ (K)"); ax.set_ylabel(r"mean $\alpha_{ir}$"); ax.invert_xaxis(); ax.legend(fontsize=5)
        plt.tight_layout(); plt.savefig("fig_alpha_vs_delta.pdf")
    print("wrote table_sim_clean.tex table_sim_shadow.tex table_eff.tex fig_iou_vs_delta.pdf fig_alpha_vs_delta.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "report"])
    ap.add_argument("--mode", default="full", choices=["rgb", "concat", "fixed", "full"])
    ap.add_argument("--root", default="COD10K-v3")
    ap.add_argument("--size", type=int, default=352)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw_desc", action="store_true", help="add raw-input contrast as 3rd gate feature")
    ap.add_argument("--lgate", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else report()
