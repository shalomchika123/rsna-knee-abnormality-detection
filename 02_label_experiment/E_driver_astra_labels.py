# =============================================================================
# E - TRAIN ON THE ASTRA LABELS (second cell of the notebook "knee-astra-model")
#
# Cell 1 of this notebook = the line  %%writefile /kaggle/working/train.py
#                           followed by the whole of C_train.py
# Cell 2 = this file.
#
# Each fold runs in its own fresh Python process, so the memory that slowly built up in the
# last run (about 12 GB per fold) is handed back after every fold. Folds 0 and 1 are already
# trained in "knee-new-model"; attach that notebook's output too and the final line reports
# the 5-fold ensemble against the radiologists.
# =============================================================================
import os, sys, time, glob, subprocess
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

FOLDS = [0, 1, 2]        # session 1. For session 2 change this line to: FOLDS = [3, 4]
BACKBONE = "resnet50d.ra2_in1k"      # the winner of last run's speed test (no need to test again)
SESSION_H = 11.2
T0 = time.time()

def find(name):           # walk the inputs but never the folders holding ~1 million scan files
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series", "train_images", "test_images")]
        hits += [os.path.join(root, name)] if name in files else []
    return sorted(hits)

# ---- check every input BEFORE spending GPU time ------------------------------------------------
missing = []
if len(find("cache_meta.json")) != 5: missing.append(f"knee-cache-0 ... knee-cache-4 (found {len(find('cache_meta.json'))} of 5)")
if not find("final_labels.csv"):      missing.append("knee-label-check (the better_labels output)")
if len(find("final_labels.csv")) > 1:  raise SystemExit("STOPPED: two final_labels.csv files are attached. Remove knee-labels-final, keep only knee-label-check.")
if not find("train_series.csv"):      missing.append("the competition data")
if not os.path.exists("/kaggle/working/train.py"): missing.append("cell 1 (%%writefile /kaggle/working/train.py + C_train.py)")
if missing:
    raise SystemExit("STOPPED BEFORE TRAINING. Add Input -> attach: " + "; ".join(missing))
print("all inputs found, starting", flush=True)

for f in FOLDS:
    left = SESSION_H - (time.time() - T0) / 3600
    if left < 1.9:                   # one fold needs about 1 h 45 min
        print(f"fold {f} not started: only {left:.1f} h left in the session", flush=True)
        continue
    env = dict(os.environ, FOLDS=str(f), BACKBONE=BACKBONE, BUDGET_H=f"{left:.2f}")
    print(f"\n######## starting fold {f} in a fresh process ({left:.1f} h left) ########", flush=True)
    r = subprocess.run([sys.executable, "-u", "/kaggle/working/train.py"], env=env)
    print(f"######## fold {f} finished with exit code {r.returncode} ########", flush=True)
    if r.returncode != 0 and not os.path.exists(f"/kaggle/working/newmodel_f{f}.pt"):
        print("fold crashed without saving: stopping here (the next folds would fail the same way). "
              "Send the error above to Claude.", flush=True)
        break

# ---- all folds together vs the radiologists ---------------------------------------------------
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
lab = pd.read_csv(find("final_labels.csv")[0], index_col=0)
parts, weights = [], []
for p in sorted(glob.glob("/kaggle/working/newmodel_gold_preds_f*.csv")):
    parts.append(pd.read_csv(p, index_col=0)); weights.append(1)
for p in find("newmodel_gold_preds_f0.csv") + find("newmodel_gold_preds_f1.csv") + find("newmodel_gold_preds_f2.csv"):
    parts.append(pd.read_csv(p, index_col=0)); weights.append(1)    # session 2: folds from session 1 (attach it)
if parts:
    ids = parts[0].index
    ens = sum(w * d.loc[ids, LABELS].values for d, w in zip(parts, weights)) / sum(weights)
    G = lab.loc[ids, [f"gold_{t}" for t in LABELS]].values.astype(int)
    a = [roc_auc_score(G[:, j], ens[:, j]) for j in range(len(LABELS))]
    print(f"\nENSEMBLE of {sum(weights)} folds vs RADIOLOGISTS {np.mean(a):.4f}")
    print(", ".join(f"{t} {v:.3f}" for t, v in zip(LABELS, a)))
print("E DONE. Checkpoints:", sorted(os.path.basename(p) for p in glob.glob("/kaggle/working/newmodel_f*.pt")))
