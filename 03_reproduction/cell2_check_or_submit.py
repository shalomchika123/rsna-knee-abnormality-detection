# =============================================================================
# CELL 2 - run OUR classifier through the 1st-place inference engine (cell 1 wrote infer.py)
#
# MODE = "check":  predicts the 58 radiologist studies of the TRAINING set and prints the AUC.
#                  It must come out close to the training notebook's last line (gold 58 0.9356).
#                  If it does, the engine reads our model correctly. (GPU T4 x2, ~10 min)
# MODE = "submit": predicts the hidden test set and writes /kaggle/working/submission.csv.
#                  (GPU T4 x2, Internet OFF - required for submissions)
#
# Inputs to attach (Add Input):
#   - the competition (RSNA Knee Abnormality Detection)
#   - knee-models-common            (tascj's detector + side classifiers)
#   - your "RSNA 2026 Knee Box-Pooling Classifier" copy (its output holds fold0_ema.pth)
#   - your "knee-classifier-full" notebook             (its output holds fold5_ema.pth)
# =============================================================================
MODE = "check"
MODELS = "both"     # "fold0" = our first model only | "full" = the all-data model only | "both" = their average

import os, sys, glob, subprocess, shutil
from pathlib import Path
import numpy as np, pandas as pd

W = Path("/kaggle/working")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]

# ---- 1. find our trained weights and the competition data -----------------------------------
def find(name):           # walk the inputs but never the folders holding ~1 million scan files
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series", "train_images", "test_images")]
        hits += [os.path.join(root, name)] if name in files or name in dirs else []
    return sorted(hits)

RUNS, DIRS = [], []
if MODELS in ("fold0", "both"):
    hits = find("fold0_ema.pth")
    assert hits, "STOPPED: fold0_ema.pth not found - attach your Box-Pooling Classifier notebook's output"
    RUNS.append("ours_256"); DIRS.append(str(Path(hits[0]).parent))
if MODELS in ("full", "both"):
    hits = find("fold5_ema.pth")
    assert hits, "STOPPED: fold5_ema.pth not found - attach your knee-classifier-full notebook's output"
    RUNS.append("ours_full"); DIRS.append(str(Path(hits[0]).parent))
COMP = next(p for p in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
                        "/kaggle/input/rsna-knee-abnormality-detection"] if os.path.exists(p))
assert find("detector.pth"), "STOPPED: attach the dataset knee-models-common"
print("our models:", list(zip(RUNS, DIRS)), flush=True)

# ---- 2. the arguments both GPUs share -----------------------------------------------------------
out = W / "ours"
out.mkdir(exist_ok=True)
common = ["--run", *RUNS, "--weights", *DIRS, "--out", str(out), "--competition", COMP]   # 2 runs = equal average
if MODE == "check":
    train = pd.read_csv(f"{COMP}/train.csv")
    gold = train.dropna(subset=LABELS).drop_duplicates("StudyInstanceUID")
    gold[["StudyInstanceUID"]].to_csv(W / "gold58.csv", index=False)
    common += ["--split", "train", "--studies", str(W / "gold58.csv")]
    print(f"checking on {len(gold)} radiologist studies", flush=True)

# ---- 3. one process per GPU, each takes every other study; then merge ------------------------
procs = [subprocess.Popen([sys.executable, str(W / "infer.py"), *common, "--shard", str(k)],
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))) for k in (0, 1)]
codes = [p.wait() for p in procs]
print("shards finished with exit codes", codes, flush=True)
subprocess.run([sys.executable, str(W / "infer.py"), *common, "--merge"], check=True)
pred = pd.read_csv(out / "submission.csv")

# ---- 4. check: AUC against the radiologists | submit: copy to submission.csv -------------------
if MODE == "check":
    from sklearn.metrics import roc_auc_score
    g = gold.set_index("StudyInstanceUID").loc[pred.StudyInstanceUID, LABELS].astype(int)
    p = pred.set_index("StudyInstanceUID")[LABELS]
    a = [roc_auc_score(g[t], p[t]) for t in LABELS if g[t].nunique() == 2]
    print(f"\nOUR MODEL THROUGH THE ENGINE vs RADIOLOGISTS: {np.mean(a):.4f}  (fold0 alone said 0.9356-0.9360, full alone 0.9385)")
    print(", ".join(f"{t} {v:.3f}" for t, v in zip(LABELS, a)))
else:
    shutil.copy(out / "submission.csv", W / "submission.csv")
    print("submission.csv written:", pred.shape, "| any missing:", int(pred[LABELS].isna().any(axis=1).sum()))
