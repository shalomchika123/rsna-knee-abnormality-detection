# =============================================================================
# C - TRAIN THE NEW MODEL (GPU T4 x2, Internet ON - this is not the submission notebook)
#
# How to run:
#   1. Create -> New Notebook, name it "knee-new-model". Accelerator GPU T4 x2, Internet ON.
#   2. Add Input: the competition data, knee-cache-0 ... knee-cache-4 (step A outputs) and
#      knee-labels-final (step B output).
#   3. Paste this into one cell. Save & Run All (Commit).
#   4. CHECKPOINT: after fold 0 (~45-70 min) the log prints "FOLD 0 vs RADIOLOGISTS". Send it
#      to Claude. The 58 radiologist studies are never trained on, so it is a clean score.
#      Target: 0.90+ after one fold (the 5-fold average adds ~0.01-0.02).
#
# Model: 3-slice windows from every series -> EfficientNetV2-S (a different family from the
# public CoAtNet and ConvNeXt models) -> tokens tagged with series slot and slice position ->
# 2-layer transformer over the whole study -> one attention pool per finding.
# Architecture adapted from goodpjw2008/rsna-knee-2-5d-convnext-reader (Apache-2.0).
# =============================================================================
BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"   # replaced by the faster one below if it is too slow
FAST_BACKBONES = ["convnext_nano.in12k_ft_in1k", "convnext_nano.d1h_in1k", "resnet50d.ra2_in1k"]
RES = 256              # network input; the cache is 256 px over a 153.6 mm field of view
K_TRAIN, K_EVAL = 8, 16  # 3-slice windows per series
EPOCHS = 12            # upper limit; the real number is planned from the measured speed (see PLAN)
MIN_EPOCHS = 5         # a fold gets at least this many epochs, otherwise fewer folds are trained
BATCH = 4              # studies per step (each up to 6 series x 8 windows)
LR = 2e-4
WD = 0.05
FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]
SEED = 2026
import os
NUM_WORKERS = max(2, min(4, os.cpu_count() or 4))
READ_THREADS = 16      # parallel disk reads inside each loader worker (hides disk latency)

import os, sys, glob, json, math, time, hashlib, subprocess, mmap
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm

T0 = time.time()
if os.environ.get("FOLDS"):                       # set by the driver: train only these folds
    FOLDS_TO_TRAIN = [int(v) for v in os.environ["FOLDS"].split(",")]
TAG = "" if not os.environ.get("FOLDS") else "_f" + "_".join(map(str, FOLDS_TO_TRAIN))
def log(m): print(f"[{(time.time() - T0) / 60:6.1f} min] {m}", flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)
OUT = "/kaggle/working"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
N_SLOTS = 6
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# ---- finding files fast: the competition's DICOM folders hold ~1 million files, never walk them --
SKIP_DIRS = {"train_series", "test_series", "train_images", "test_images"}
def find(name):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if name in files:
            hits.append(os.path.join(root, name))
    return sorted(hits)

# ---- the cache shards ------------------------------------------------------------------------
shards = sorted(os.path.dirname(p) for p in find("cache_meta.json"))
assert shards, "attach knee-cache-0 ... knee-cache-4"
SER = {}      # series -> (memmap, offset, n)
STUDY = {}    # study -> list of 6 slots, each a list of series
for d in shards:
    meta = json.load(open(f"{d}/cache_meta.json"))
    S = meta["size"]
    mm = f"{d}/data.bin"      # read with pread: nothing is mapped into memory
    idx = pd.read_csv(f"{d}/index.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    for st, sr, sl, n, off in idx.itertuples(index=False):
        SER[sr] = (mm, int(off), int(n))
        STUDY.setdefault(st, [[] for _ in range(N_SLOTS)])[int(sl)].append(sr)
    log(f"shard {meta['shard']}: {len(idx)} series")
for slots in STUDY.values():          # missing fat-sat / non-fat-sat series: use its counterpart
    for p in range(3):
        if not slots[p] and slots[p + 3]:
            slots[p] = list(slots[p + 3])
        elif not slots[p + 3] and slots[p]:
            slots[p + 3] = list(slots[p])
log(f"{len(shards)} shards, {len(STUDY)} studies, {len(SER)} series, {S} px")
assert len(shards) == 5, f"expected 5 cache shards, found {len(shards)}"

# ---- labels and folds ------------------------------------------------------------------------
lab_path = find("final_labels.csv")
assert lab_path, "attach knee-labels-final"
lab = pd.read_csv(lab_path[0], dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
comp = os.path.dirname([p for p in find("train.csv") if os.path.exists(os.path.join(os.path.dirname(p), "train_series.csv"))][0])
rep = pd.read_csv(f"{comp}/train.csv", dtype={"StudyInstanceUID": str}).drop_duplicates("StudyInstanceUID") \
        .set_index("StudyInstanceUID")["Report"].fillna("")
studies = [s for s in lab.index if s in STUDY]
gold_ids = [s for s in studies if lab.at[s, "is_gold"] == 1]
train_ids = [s for s in studies if lab.at[s, "is_gold"] == 0]       # the 58 are NEVER trained on
fold_of = {s: int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % 5 for s in train_ids}
Y = lab[LABELS].astype(np.float32)
G = lab.loc[gold_ids, [f"gold_{t}" for t in LABELS]].values.astype(int)
log(f"train pool {len(train_ids)}, radiologist studies {len(gold_ids)} (held out of every fold)")

# ---- data ------------------------------------------------------------------------------------
def window_centres(n, k, train, lo=0.04, hi=0.96):
    a, b = max(1.0, lo * (n - 1)), min(n - 2.0, hi * (n - 1))
    if b < a:
        a = b = (n - 1) / 2
    if train:
        e = np.linspace(a, b, k + 1)
        c = e[:-1] + np.random.rand(k) * (e[1:] - e[:-1])
    else:
        c = np.linspace(a, b, k)
    return np.clip(np.round(c).astype(int), 1, max(1, n - 2))

_FDS = {}
_POOL = {}
def _pool():
    if os.getpid() not in _POOL:
        _POOL[os.getpid()] = ThreadPoolExecutor(READ_THREADS)
    return _POOL[os.getpid()]
_DIRECT = set()          # handles opened in direct mode: reads skip the system's file cache
ALIGN = 4096
def _fd(path):
    """One read-only file handle per data file per process (each loader worker opens its own).
    Tries direct mode first (the bytes never enter the file cache, so RAM cannot fill up);
    falls back to a normal handle if the disk does not support it."""
    key = (os.getpid(), path)
    if key not in _FDS:
        fd = None
        if hasattr(os, "O_DIRECT"):
            try:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                _direct_read(fd, ALIGN, 0)                    # test that direct reads work here
                _DIRECT.add(fd)
            except OSError:
                if fd is not None:
                    os.close(fd)
                fd = None
        _FDS[key] = fd if fd is not None else os.open(path, os.O_RDONLY)
    return _FDS[key]

def _direct_read(fd, n, off):
    a0 = off - off % ALIGN
    a1 = -(-(off + n) // ALIGN) * ALIGN
    buf = mmap.mmap(-1, a1 - a0)                              # page-aligned memory, as direct reads require
    got = os.preadv(fd, [buf], a0)
    out = bytes(buf[off - a0: min(off - a0 + n, got)])
    buf.close()
    return out

def _pread_drop(fd, n, off):
    """Read n bytes at off without leaving them in the file cache."""
    if fd in _DIRECT:
        return _direct_read(fd, n, off)
    buf = os.pread(fd, n, off)
    try:
        os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
    except Exception:
        pass
    return buf

DATA_FILES = sorted({v[0] for v in SER.values()})
def _ram_frac():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return 1 - vm.available / vm.total
    except Exception:
        return 0.0

def ram_guard():
    """Every 25 steps: if RAM is getting full, drop the cached input data; report if it stays full."""
    before = _ram_frac()
    if before < 0.55:
        return "ok"
    for path in DATA_FILES:
        try:
            fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        except Exception:
            pass
    import gc; gc.collect()
    after = _ram_frac()
    log(f"    RAM guard: {before:.0%} -> {after:.0%} after dropping cached input data")
    return "critical" if after > 0.88 else "ok"

def _mem():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return f"RAM used {vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} GB"
    except Exception:
        return ""

class StudySet(torch.utils.data.Dataset):
    def __init__(self, sids, k, train):
        self.sids, self.k, self.train = list(sids), k, train
    def __len__(self):
        return len(self.sids)
    def __getitem__(self, i):
        st = self.sids[i]
        slots = STUDY.get(st, [[] for _ in range(N_SLOTS)])
        x = np.zeros((N_SLOTS, self.k, 3, S, S), np.uint8)
        mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, self.k), np.float32)
        plane = S * S
        reads = []                                         # (slot, window, fd, offset)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            sr = cands[np.random.randint(len(cands))] if self.train else cands[0]
            path, off, n = SER[sr]
            if n < 3:
                continue
            fd = _fd(path)
            c = window_centres(n, self.k, self.train)      # centres lie in 1..n-2
            for w, j in enumerate(c):                      # 3 neighbouring slices = one contiguous read
                reads.append((s, w, fd, off + (int(j) - 1) * plane))
            mask[s], pos[s] = True, c / max(n - 1, 1)
        # all reads of this study at once: the disk answers many requests in parallel
        bufs = _pool().map(lambda r: _pread_drop(r[2], 3 * plane, r[3]), reads)
        for (s, w, _, _), buf in zip(reads, bufs):
            x[s, w] = np.frombuffer(buf, np.uint8).reshape(3, S, S)
        y = Y.loc[st].values.astype(np.float32) if st in Y.index else np.zeros(len(LABELS), np.float32)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos), torch.from_numpy(y)

def loader(sids, k, train, bs):
    return torch.utils.data.DataLoader(StudySet(sids, k, train), batch_size=bs, shuffle=train,
                                       num_workers=NUM_WORKERS, pin_memory=True, drop_last=train,
                                       persistent_workers=False, prefetch_factor=4)

# ---- model -------------------------------------------------------------------------------------
def make_views(x, res, train, group, fov=0.92):
    n = x.shape[0]; ns = n // group; dev = x.device
    x = x.float().div_(255.0)
    if train:
        ang = (torch.rand(ns, device=dev) - 0.5) * 2 * math.radians(12)
        sc = fov * (1 + (torch.rand(ns, device=dev) - 0.5) * 0.24)
        tx = (torch.rand(ns, device=dev) - 0.5) * 0.12
        ty = (torch.rand(ns, device=dev) - 0.5) * 0.12
        gamma = torch.exp((torch.rand(ns, device=dev) - 0.5) * 0.5)
        gain = 1 + (torch.rand(ns, device=dev) - 0.5) * 0.2
    else:
        ang = torch.zeros(ns, device=dev); sc = torch.full((ns,), fov, device=dev)
        tx = ty = torch.zeros(ns, device=dev); gamma = gain = None
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    theta = theta.repeat_interleave(group, 0)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if train:
        g = gamma.repeat_interleave(group).view(n, 1, 1, 1)
        x = x.clamp_min(1e-4).pow(g) * gain.repeat_interleave(group).view(n, 1, 1, 1)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std

class KneeNet(nn.Module):
    def __init__(self, backbone, pretrained=True, dim=384, depth=2, drop=0.2, drop_path=0.1):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(N_SLOTS, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(LABELS)))
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))
    def head(self, feat, mask, pos):
        b, s, k, _ = feat.shape
        slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)
        h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        h = h.view(b, s * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)
        h = self.tf(h, src_key_padding_mask=pad)
        a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
        pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b
    def forward(self, x, mask, pos, res=256, train=False):
        b, s, k = x.shape[:3]
        keep = mask.reshape(-1)
        xv = x.reshape(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
        f = self.enc(make_views(xv, res, train, k).contiguous(memory_format=torch.channels_last))
        feat = f.new_zeros(b * s, k, f.shape[-1])
        feat[keep] = f.view(-1, k, f.shape[-1])
        return self.head(feat.view(b, s, k, -1), mask, pos)

# ---- helpers -----------------------------------------------------------------------------------
def aucs(y, p):
    return np.array([roc_auc_score(y[:, j], p[:, j]) if 0 < y[:, j].sum() < len(y) else np.nan
                     for j in range(y.shape[1])])

@torch.no_grad()
def predict(model, sids, bs=4):
    model.eval(); P = []
    for x, m, pos, _ in loader(sids, K_EVAL, False, bs):
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x.cuda(non_blocking=True), m.cuda(), pos.cuda(), res=RES, train=False)
        P.append(torch.sigmoid(z.float()).cpu().numpy())
    return np.concatenate(P)

# ---- training ----------------------------------------------------------------------------------
n_gpu = torch.cuda.device_count()
bs = BATCH                      # DataParallel splits it across the two GPUs
SESSION_BUDGET_H = float(os.environ.get("BUDGET_H", 11.0))   # Kaggle stops sessions at 12 h; everything must be saved before
WARM = 300                      # warm-up steps
PLAN = None                     # decided after measuring the speed of the first 40 steps of fold 0

def make_plan(step_s, n_steps):
    """Pick the most folds that still get >= MIN_EPOCHS each inside the session budget."""
    epoch_s = step_s * n_steps
    avail = SESSION_BUDGET_H * 3600 - (time.time() - T0)
    for nf in range(len(FOLDS_TO_TRAIN), 0, -1):
        e = int(avail / nf / epoch_s - 0.7)        # ~0.7 epoch per fold for its predictions (safe side)
        if e >= MIN_EPOCHS or nf == 1:
            return {"folds": FOLDS_TO_TRAIN[:nf], "epochs": int(max(1, min(EPOCHS, e))),
                    "epoch_min": epoch_s / 60, "avail_h": avail / 3600}
def bench(backbone, n_gpu, steps=6):
    """Seconds per training step for this backbone, on a fake full batch (fp16, both GPUs)."""
    core = KneeNet(backbone).cuda().to(memory_format=torch.channels_last)
    model = nn.DataParallel(core) if n_gpu > 1 else core
    opt = torch.optim.AdamW(core.parameters(), lr=1e-5)
    x = torch.randint(0, 255, (BATCH, N_SLOTS, K_TRAIN, 3, S, S), dtype=torch.uint8).cuda()
    m = torch.ones(BATCH, N_SLOTS, dtype=torch.bool).cuda(); pos = torch.rand(BATCH, N_SLOTS, K_TRAIN).cuda()
    times = []
    for i in range(steps):
        torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x, m, pos, res=RES, train=True).float().mean()
        opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
        times.append(time.time() - t)
    del model, core, opt, x; torch.cuda.empty_cache()
    return float(np.median(times[2:]))

n_steps = len([s for s in train_ids if fold_of[s] != 0]) // BATCH
avail = SESSION_BUDGET_H * 3600 - (time.time() - T0)
choice = (os.environ["BACKBONE"], 99) if os.environ.get("BACKBONE") else None
for bb in ([] if choice else [BACKBONE] + FAST_BACKBONES):
    try:
        t = bench(bb, n_gpu)
    except Exception as e:
        log(f"backbone {bb}: unavailable ({type(e).__name__}: {str(e)[:80]})"); continue
    fits = int(avail / (t * n_steps * (MIN_EPOCHS + 0.7)))
    log(f"backbone {bb}: {t:.2f} s/step (pure GPU), {t * n_steps / 60:.0f} min/epoch, "
        f"room for {fits} fold(s) x {MIN_EPOCHS} epochs")
    if choice is None or fits > choice[1]:
        choice = (bb, fits)
    if fits >= 3:
        break                                  # first in the list that fits 3+ folds wins
BACKBONE = choice[0]
log(f"USING BACKBONE {BACKBONE}")

oof = pd.DataFrame(np.nan, index=train_ids, columns=LABELS)
gold_preds = []
report = []
HALT = False
for fold in FOLDS_TO_TRAIN:
    if HALT:
        log(f"fold {fold} not started: RAM could not be freed (the saved folds are usable)")
        continue
    if PLAN is not None and fold not in PLAN["folds"]:
        log(f"fold {fold} skipped: no session time for it (the saved folds are usable)")
        continue
    tr = [s for s in train_ids if fold_of[s] != fold]
    va = [s for s in train_ids if fold_of[s] == fold]
    log(f"===== fold {fold}: train {len(tr)}, holdout {len(va)}, radiologist studies {len(gold_ids)} =====")
    core = KneeNet(BACKBONE).cuda().to(memory_format=torch.channels_last)
    model = nn.DataParallel(core) if n_gpu > 1 else core
    opt = torch.optim.AdamW(core.parameters(), lr=LR, weight_decay=WD)
    dl = loader(tr, K_TRAIN, True, bs)
    TOTAL = [(PLAN["epochs"] if PLAN else EPOCHS) * len(dl)]     # can be reset once speed is known
    def lr_at(step):
        if step < WARM:
            return (step + 1) / WARM
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - WARM) / max(1, TOTAL[0] - WARM))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda")
    ep = 0; STOP = False
    while ep < (PLAN["epochs"] if PLAN else EPOCHS) and not STOP:
        model.train(); t0 = time.time(); tot = nb = 0; t_wait = 0.0; t_last = time.time()
        for x, m, pos, y in dl:
            t_wait += time.time() - t_last
            x, m, pos, y = x.cuda(non_blocking=True), m.cuda(), pos.cuda(), y.cuda()
            w = torch.isfinite(y).float(); y = torch.nan_to_num(y)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(x, m, pos, res=RES, train=True)
            loss = (F.binary_cross_entropy_with_logits(z.float(), y, reduction="none") * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(core.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); nb += 1
            if PLAN is None and nb == 40:
                PLAN = make_plan((time.time() - t0) / nb, len(dl))
                TOTAL[0] = PLAN["epochs"] * len(dl)
                log(f"PLAN: {PLAN['epoch_min']:.0f} min per epoch, {PLAN['avail_h']:.1f} h left -> "
                    f"folds {PLAN['folds']} x {PLAN['epochs']} epochs each")
            if nb % 25 == 0 and ram_guard() == "critical":
                log("    RAM nearly full even after dropping caches: saving and ending this fold early")
                STOP = HALT = True
                break
            if nb % 50 == 0:
                el = time.time() - t0
                log(f"    fold {fold} epoch {ep + 1} step {nb}/{len(dl)}  loss {tot / nb:.4f}  "
                    f"{el / nb:.2f} s/step ({t_wait / el:.0%} waiting for data), "
                    f"epoch ~{el / nb * len(dl) / 60:.1f} min  {_mem()}")
            t_last = time.time()
        E = PLAN["epochs"] if PLAN else EPOCHS
        msg = f"  fold {fold} epoch {ep + 1}/{E} loss {tot / max(nb, 1):.4f} ({(time.time() - t0) / 60:.1f} min)"
        if ep in (E // 2 - 1, E - 1):
            pg = predict(model, gold_ids)
            msg += f"  vs-radiologists {np.nanmean(aucs(G, pg)):.4f}"
        log(msg)
        ep += 1
        # safety copy after every epoch: if the session were ever cut short, the latest weights exist
        torch.save({"model": core.state_dict(), "args": {"backbone": BACKBONE, "res": RES, "k": K_EVAL,
                    "size": S, "mm_per_px": 0.6}, "fold": fold, "epoch": ep}, f"{OUT}/newmodel_f{fold}_partial.pt")
    pv = predict(model, va)
    oof.loc[va] = pv
    pg = predict(model, gold_ids)
    gold_preds.append(pg)
    a_rep = aucs((Y.loc[va].fillna(0).values > 0.5).astype(int), pv)
    a_g = aucs(G, pg)
    line = (f"FOLD {fold} vs RADIOLOGISTS {np.nanmean(a_g):.4f} | vs labels {np.nanmean(a_rep):.4f}\n  " +
            ", ".join(f"{t} {v:.3f}" for t, v in zip(LABELS, a_g)))
    log(line); report.append(line)
    torch.save({"model": core.state_dict(), "args": {"backbone": BACKBONE, "res": RES, "k": K_EVAL,
                "size": S, "mm_per_px": 0.6}, "fold": fold}, f"{OUT}/newmodel_f{fold}.pt")
    if os.path.exists(f"{OUT}/newmodel_f{fold}_partial.pt"):
        os.remove(f"{OUT}/newmodel_f{fold}_partial.pt")
    oof.reset_index().rename(columns={"index": "StudyInstanceUID"}).to_csv(f"{OUT}/newmodel_oof{TAG}.csv", index=False)
    ens = np.mean(gold_preds, axis=0)
    a_e = aucs(G, ens)
    line = f"ENSEMBLE of {len(gold_preds)} fold(s) vs RADIOLOGISTS {np.nanmean(a_e):.4f}"
    log(line); report.append(line)
    pd.DataFrame(ens, index=gold_ids, columns=LABELS).to_csv(f"{OUT}/newmodel_gold_preds{TAG}.csv")
    del model, core, opt; torch.cuda.empty_cache()

ens = np.mean(gold_preds, axis=0)
tab = pd.DataFrame({"vs_radiologists": aucs(G, ens), "positives": G.sum(0)}, index=LABELS)
txt = "\n".join(report) + "\n\n" + tab.sort_values("vs_radiologists").to_string() + \
      f"\n\nFINAL: {len(gold_preds)}-fold ensemble vs radiologists {tab.vs_radiologists.mean():.4f}" \
      "\n(bar to help the 0.958 blend: ~0.93; previous DINOv2 models 0.844)"
print(txt); open(f"{OUT}/newmodel_report{TAG}.txt", "w").write(txt)
log("STEP C DONE")
