# =============================================================================
# D - PART C FOR THE 0.958 NOTEBOOK: add the new model to the final submission
#
# Paste as ONE NEW CELL at the very END of your fork of the 0.958 notebook (after Part B,
# which writes /kaggle/working/submission.csv). Add Input: your "knee-new-model" notebook
# (its output holds newmodel_f0.pt ... newmodel_f4.pt). Internet stays OFF.
#
# It decodes the hidden test studies exactly like the training cache (step A), runs the
# fold models, and rank-blends them into submission.csv with one weight per finding.
# If anything fails, or time is short, submission.csv is left exactly as Part B wrote it.
# =============================================================================
# Weight of the new model per finding. The 0.958 notebook's own extra model (about as strong as ours)
# was tuned by its author to only 0.003-0.035 per finding; our 0.05-0.25 scored 0.956 (vs 0.958).
# So: small and flat. Change NEW_W only, everything else follows.
NEW_W = 0.05
NEW_W_PER_TARGET = {t: NEW_W for t in ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
                                      "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
                                      "Contusion", "Fracture"]}
NEW_GLOB = "newmodel_f*.pt"
SKIP_IF_ELAPSED_H = 7.0

import time as _n_time
_n_elapsed = (_n_time.time() - globals().get("NB_T0", _n_time.time())) / 3600
if _n_elapsed > SKIP_IF_ELAPSED_H:
    print(f"Part C skipped: {_n_elapsed:.1f} h used; submission.csv kept as is", flush=True)
else:
    import os as _n_os, glob as _n_glob, shutil as _n_shutil, subprocess as _n_sp, sys as _n_sys
    import pandas as _n_pd
    _n_sub, _n_bak = "/kaggle/working/submission.csv", "/kaggle/working/_before_new_model.csv"
    _n_shutil.copy(_n_sub, _n_bak)
    _n_code = r'''
import os, sys, glob, math, numpy as np, pandas as pd, pydicom, cv2, torch, torch.nn as nn, torch.nn.functional as F, timm
import fnmatch
def find(pattern):        # walk the inputs but never the folders holding ~1 million scan files
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series", "train_images", "test_images")]
        hits += [os.path.join(root, f) for f in files if fnmatch.fnmatch(f, pattern)]
    return sorted(hits)
COMP = os.path.dirname([p for p in find("test_series.csv") if os.path.isdir(os.path.join(os.path.dirname(p), "test_series"))][0])
CKPTS, _seen = [], set()
for p in find(sys.argv[2]):                     # one checkpoint per fold, even if attached twice
    if "partial" not in p and os.path.basename(p) not in _seen:
        _seen.add(os.path.basename(p)); CKPTS.append(p)
print("new model checkpoints:", [os.path.basename(p) for p in CKPTS], flush=True)
OUT_CSV = sys.argv[1]
assert CKPTS, "no newmodel checkpoints attached"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
N_SLOTS, MEAN, STD = 6, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ck0 = torch.load(CKPTS[0], map_location="cpu", weights_only=False)
A = ck0["args"]; SIZE, MM_PER_PX, RES, K = A["size"], A["mm_per_px"], A["res"], A["k"]
MAX_SLICES = 48
PLANES = ["Sagittal", "Coronal", "Axial"]
CANON = {0: (+1, (1, +1), (2, -1)), 1: (+1, (0, +1), (2, -1)), 2: (-1, (0, +1), (1, +1))}

def _read(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    slope, inter = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
    if slope is not None and inter is not None:
        arr = arr * float(slope) + float(inter)
    iop = [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
    ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
    ps = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
    inst = float(ds.get("InstanceNumber") or 0)
    frames = list(arr) if arr.ndim == 3 else [arr]
    return [(f, iop, ipp, ps, inst + i * 1e-3) for i, f in enumerate(frames)]

def _canon_slice(img, iop, ps, plane):
    ps = ps or [MM_PER_PX * img.shape[0] / SIZE] * 2
    if iop is None:
        return img, ps
    r, c = np.array(iop[:3]), np.array(iop[3:])
    _, (cax, csgn), (rax, rsgn) = CANON[plane]
    if abs(r[cax]) < abs(r[rax]):
        img, r, c, ps = img.T, c, r, ps[::-1]
    if r[cax] * csgn < 0:
        img = img[:, ::-1]
    if c[rax] * rsgn < 0:
        img = img[::-1]
    return img, ps

def _resample(img, ps):
    out = np.zeros((SIZE, SIZE), np.float32)
    h, w = img.shape
    ch = min(h, int(round(SIZE * MM_PER_PX / ps[0]))); cw = min(w, int(round(SIZE * MM_PER_PX / ps[1])))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    nh = max(1, min(SIZE, int(round(ch * ps[0] / MM_PER_PX)))); nw = max(1, min(SIZE, int(round(cw * ps[1] / MM_PER_PX))))
    res = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR)
    oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = res
    return out

def load_series(sdir, plane_name=None):
    slices = []
    for fn in os.listdir(sdir):
        try:
            slices.extend(_read(os.path.join(sdir, fn)))
        except Exception:
            pass
    if not slices:
        return None
    iop = next((s[1] for s in slices if s[1] is not None), None)
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:]); plane = int(np.abs(normal).argmax())
    else:
        normal, plane = None, PLANES.index(plane_name) if plane_name in PLANES else 0
    if normal is not None and all(s[2] is not None for s in slices):
        sgn = CANON[plane][0] * np.sign(normal[plane])
        slices.sort(key=lambda s: float(np.dot(s[2], normal)) * sgn)
    else:
        slices.sort(key=lambda s: s[4])
    if len(slices) > MAX_SLICES:
        keep = np.linspace(0, len(slices) - 1, MAX_SLICES).round().astype(int)
        slices = [slices[i] for i in keep]
    vol = np.stack([_resample(*_canon_slice(s[0], s[1], s[3], plane)) for s in slices])
    lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    return (vol * 255).round().astype(np.uint8)

def _is_fs(v):
    return str(v).strip().lower() in ("1", "1.0", "true", "yes", "y", "t")

se = pd.read_csv(f"{COMP}/test_series.csv", dtype=str)
fs_col = "Fat_Suppression" if "Fat_Suppression" in se.columns else "Fluid_Sensitive"
studies = pd.read_csv(f"{COMP}/test.csv", dtype=str).StudyInstanceUID.tolist()
se["nfiles"] = [len(os.listdir(f"{COMP}/test_series/{a}/{b}")) if os.path.isdir(f"{COMP}/test_series/{a}/{b}") else 0
                for a, b in zip(se.StudyInstanceUID, se.SeriesInstanceUID)]
se["slot"] = [PLANES.index(p) + 3 * (1 - int(_is_fs(f))) if p in PLANES else -1 for p, f in zip(se.Anatomical_Plane, se[fs_col])]
se = se[(se.slot >= 0) & (se.nfiles >= 3)].sort_values("nfiles", ascending=False)
TABLE = {}
for st, sr, sl, pl in zip(se.StudyInstanceUID, se.SeriesInstanceUID, se.slot, se.Anatomical_Plane):
    slots = TABLE.setdefault(st, [None] * N_SLOTS)
    if slots[int(sl)] is None:
        slots[int(sl)] = (sr, pl)
for slots in TABLE.values():
    for p in range(3):
        if slots[p] is None and slots[p + 3] is not None: slots[p] = slots[p + 3]
        elif slots[p + 3] is None and slots[p] is not None: slots[p + 3] = slots[p]

def centres(n, k, lo=0.04, hi=0.96):
    a, b = max(1.0, lo * (n - 1)), min(n - 2.0, hi * (n - 1))
    if b < a: a = b = (n - 1) / 2
    return np.clip(np.round(np.linspace(a, b, k)).astype(int), 1, max(1, n - 2))

class TestSet(torch.utils.data.Dataset):
    def __len__(self): return len(studies)
    def __getitem__(self, i):
        st = studies[i]
        x = np.zeros((N_SLOTS, K, 3, SIZE, SIZE), np.uint8)
        mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, K), np.float32)
        cache = {}
        for s, item in enumerate(TABLE.get(st, [None] * N_SLOTS)):
            if item is None: continue
            sr, pl = item
            try:
                vol = cache[sr] if sr in cache else load_series(f"{COMP}/test_series/{st}/{sr}", pl)
            except Exception:
                vol = None
            cache[sr] = vol
            if vol is None or len(vol) < 3: continue
            n = len(vol); c = centres(n, K)
            x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
            mask[s], pos[s] = True, c / max(n - 1, 1)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos)

def make_views(x, res, group, fov=0.92):
    n = x.shape[0]; ns = n // group; dev = x.device
    x = x.float().div_(255.0)
    sc = torch.full((ns,), fov, device=dev); z = torch.zeros(ns, device=dev)
    theta = torch.stack([torch.stack([sc, z, z], 1), torch.stack([z, sc, z], 1)], 1).repeat_interleave(group, 0)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (x - torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)) / torch.tensor(STD, device=dev).view(1, 3, 1, 1)

class KneeNet(nn.Module):
    def __init__(self, backbone, dim=384, depth=2, drop=0.2):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(N_SLOTS, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(LABELS)))
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))
    def forward(self, x, mask, pos, res):
        b, s, k = x.shape[:3]
        keep = mask.reshape(-1)
        xv = x.reshape(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
        f = self.enc(make_views(xv, res, k).contiguous(memory_format=torch.channels_last))
        feat = f.new_zeros(b * s, k, f.shape[-1]); feat[keep] = f.view(-1, k, f.shape[-1])
        feat = feat.view(b, s, k, -1)
        slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)
        h = (self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))).view(b, s * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)
        h = self.tf(h, src_key_padding_mask=pad)
        a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
        pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

models = []
for p in CKPTS:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = KneeNet(ck["args"]["backbone"])
    m.load_state_dict(ck["model"]); models.append(m.cuda().to(memory_format=torch.channels_last).eval())
print(f"new model: {len(models)} folds, {len(studies)} test studies", flush=True)
dl = torch.utils.data.DataLoader(TestSet(), batch_size=2, num_workers=4, pin_memory=True)
P = []
with torch.no_grad():
    for x, m, pos in dl:
        x, m, pos = x.cuda(non_blocking=True), m.cuda(), pos.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            P.append(torch.stack([mm(x, m, pos, RES).float().sigmoid() for mm in models]).mean(0).cpu().numpy())
pd.DataFrame(np.concatenate(P), index=pd.Index(studies, name="StudyInstanceUID"), columns=LABELS).to_csv(OUT_CSV)
print("wrote", OUT_CSV, flush=True)
'''
    try:
        open("/kaggle/working/_newmodel_infer.py", "w").write(_n_code)
        _n_env = dict(_n_os.environ)
        if _n_os.path.isdir("/kaggle/working/_own_env"):     # Part B's offline DICOM decoders
            _n_env["PYTHONPATH"] = "/kaggle/working/_own_env" + _n_os.pathsep + _n_env.get("PYTHONPATH", "")
        try:
            import gc as _n_gc, torch as _n_torch
            _n_gc.collect(); _n_torch.cuda.empty_cache()
            _n_free = [_n_torch.cuda.mem_get_info(i)[0] for i in range(_n_torch.cuda.device_count())]
            _n_env["CUDA_VISIBLE_DEVICES"] = str(max(range(len(_n_free)), key=lambda i: _n_free[i]))
        except Exception:
            pass
        _n_t0 = _n_time.time()
        _n_sp.run([_n_sys.executable, "/kaggle/working/_newmodel_infer.py", "/kaggle/working/_newmodel.csv", NEW_GLOB],
                  check=True, env=_n_env, timeout=int(max(0.5, 8.6 - _n_elapsed - 0.4) * 3600))
        _n_stack = _n_pd.read_csv(_n_bak, dtype={"StudyInstanceUID": str})
        _n_new = _n_pd.read_csv("/kaggle/working/_newmodel.csv", dtype={"StudyInstanceUID": str})
        _n_new = _n_new.set_index("StudyInstanceUID").loc[_n_stack.StudyInstanceUID].reset_index()
        _n_labels = [c for c in _n_stack.columns if c != "StudyInstanceUID"]
        assert _n_new[_n_labels].notna().all().all(), "new model predictions contain NaN"
        _n_rs = _n_stack[_n_labels].rank(method="average", pct=True)
        _n_rn = _n_new[_n_labels].rank(method="average", pct=True)
        _n_final = _n_stack.copy()
        for _c in _n_labels:
            _w = float(NEW_W_PER_TARGET.get(_c, 0.0))
            _n_final[_c] = ((1 - _w) * _n_rs[_c] + _w * _n_rn[_c]).to_numpy()
        _n_final[_n_labels] = _n_final[_n_labels].rank(method="average", pct=True)
        _n_final.to_csv(_n_sub, index=False)
        print(f"new model blended in {_n_time.time() - _n_t0:.0f}s with weights {NEW_W_PER_TARGET}; "
              f"mean rank corr with the stack {_n_rs.corrwith(_n_rn).mean():.3f}", flush=True)
    except Exception as _n_e:
        _n_shutil.copy(_n_bak, _n_sub)
        print("new model FAILED, submission.csv kept as Part B wrote it:", repr(_n_e), flush=True)
