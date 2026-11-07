# =============================================================================
# A - IMAGE CACHE SHARD (CPU notebook, free: no GPU quota)
#
# Builds canonical volumes for every training series a study uses, at a fixed physical
# scale (0.6 mm/pixel, 256 px = 153.6 mm field of view, the same field of view as the
# public 2.5D reader), slices sorted by position along the slice normal, standard in-plane
# orientation, per-series intensity window. Volumes are appended to one binary file;
# index.csv says where each series lives.
#
# The training set is split into 5 shards so each output stays far under Kaggle's 20 GB.
# Make FIVE notebooks: knee-cache-0 ... knee-cache-4, each with this code and SHARD set
# to 0,1,2,3,4. Each:  Create -> New Notebook, Add Input -> competition data,
# Accelerator None, Internet off, paste this, set SHARD, Save & Run All (Commit).
# Expect roughly 30-60 min each. They can run at the same time.
#
# Preprocessing adapted from goodpjw2008/rsna-knee-2-5d-convnext-reader (Apache-2.0).
# =============================================================================
SHARD = 0            # <-- 0, 1, 2, 3 or 4
N_SHARDS = 5

import os, glob, json, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pydicom
import cv2

T0 = time.time()
COMP = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train_series.csv", recursive=True))
OUT = "/kaggle/working"
MM_PER_PX = 0.6
SIZE = 256
MAX_SLICES = 48
PLANES = ["Sagittal", "Coronal", "Axial"]
CANON = {0: (+1, (1, +1), (2, -1)), 1: (+1, (0, +1), (2, -1)), 2: (-1, (0, +1), (1, +1))}
N_SLOTS = 6


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
    ch = min(h, int(round(SIZE * MM_PER_PX / ps[0])))
    cw = min(w, int(round(SIZE * MM_PER_PX / ps[1])))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    nh = max(1, min(SIZE, int(round(ch * ps[0] / MM_PER_PX))))
    nw = max(1, min(SIZE, int(round(cw * ps[1] / MM_PER_PX))))
    interp = cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR
    res = cv2.resize(crop, (nw, nh), interpolation=interp)
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
        normal = np.cross(iop[:3], iop[3:])
        plane = int(np.abs(normal).argmax())
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


def job(args):
    st, sr, plane = args
    try:
        v = load_series(f"{COMP}/train_series/{st}/{sr}", plane)
        return st, sr, v, ""
    except Exception as e:
        return st, sr, None, f"{type(e).__name__}: {e}"[:150]


# ---- which series: for every study, the largest series in each of the 6 slots -------------
se = pd.read_csv(f"{COMP}/train_series.csv", dtype=str)
fs_col = "Fat_Suppression" if "Fat_Suppression" in se.columns else "Fluid_Sensitive"
studies = sorted(se.StudyInstanceUID.unique())
mine = set(studies[SHARD::N_SHARDS])
se = se[se.StudyInstanceUID.isin(mine)].copy()
se["nfiles"] = [len(os.listdir(f"{COMP}/train_series/{a}/{b}")) if os.path.isdir(f"{COMP}/train_series/{a}/{b}") else 0
                for a, b in zip(se.StudyInstanceUID, se.SeriesInstanceUID)]
def _is_fs(v):
    return str(v).strip().lower() in ("1", "1.0", "true", "yes", "y", "t")
se["slot"] = [PLANES.index(p) + 3 * (1 - int(_is_fs(f))) if p in PLANES else -1
              for p, f in zip(se.Anatomical_Plane, se[fs_col])]
se = se[(se.slot >= 0) & (se.nfiles >= 3)]
pick = se.sort_values("nfiles", ascending=False).groupby(["StudyInstanceUID", "slot"]).head(1)
jobs = list(zip(pick.StudyInstanceUID, pick.SeriesInstanceUID, pick.Anatomical_Plane))
slot_of = dict(zip(pick.SeriesInstanceUID, pick.slot))
print(f"shard {SHARD}: {len(mine)} studies, {len(jobs)} series to cache ({fs_col} used for fat-sat)", flush=True)

# ---- decode in parallel, append to one binary file ----------------------------------------
rows, errors, offset = [], 0, 0
with open(f"{OUT}/data.bin", "wb") as fh, Pool(os.cpu_count()) as pool:
    for i, (st, sr, v, err) in enumerate(pool.imap(job, jobs, chunksize=2)):
        if v is None:
            errors += 1
            continue
        fh.write(v.tobytes())
        rows.append((st, sr, int(slot_of[sr]), int(len(v)), offset))
        offset += v.nbytes
        if i % 500 == 0:
            el = time.time() - T0
            print(f"  {i}/{len(jobs)} series, {offset / 1e9:.1f} GB, {el / 60:.0f} min, "
                  f"~{el / max(i, 1) * (len(jobs) - i) / 60:.0f} min left", flush=True)
idx = pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID", "slot", "n", "offset"])
idx.to_csv(f"{OUT}/index.csv", index=False)
json.dump({"shard": SHARD, "n_shards": N_SHARDS, "size": SIZE, "mm_per_px": MM_PER_PX,
           "max_slices": MAX_SLICES, "series": len(idx), "errors": errors, "bytes": offset},
          open(f"{OUT}/cache_meta.json", "w"), indent=1)
print(f"SHARD {SHARD} DONE: {len(idx)} series, {errors} unreadable, {offset / 1e9:.1f} GB, "
      f"{(time.time() - T0) / 60:.0f} min; slices per series median {idx.n.median():.0f}", flush=True)
