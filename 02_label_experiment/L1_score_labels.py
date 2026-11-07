# =============================================================================
# L1 - WHICH TRAINING LABELS ARE BEST? (CPU only, about 1 minute)
#
# WHY: every top-5 team said the same thing: this competition was won by better LABELS, not by
# bigger models. This cell scores every label table you attach against the 58 radiologist studies,
# the same way the 1st-place team judged theirs (AUC, log loss and "confident errors"), and writes
# the best one in the exact format our training script reads (final_labels.csv), so we can retrain
# our own model on better labels and see how much it moves.
#
# Kaggle notebook "knee-label-check": Accelerator NONE, Internet OFF is fine.
# Add Input:
#   - knee-labels-final (OUR labels; also holds the 58 radiologist answers - required)
#   - any public label tables you want to compare (Kaggle datasets with a CSV that has a study ID
#     column and the 12 findings as columns, e.g. the "Labels" dataset linked in the 1st-place writeup)
# Paste into one cell -> Run All. Send Claude everything from "LABEL TABLES FOUND" down.
# =============================================================================
import os, re, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SKIP_DIRS = {"train_series", "test_series", "train_images", "test_images"}

def key(s):                       # "Medial_Meniscus", "medial meniscus", "MEDIAL-MENISCUS" -> "medialmeniscus"
    return re.sub(r"[^a-z]", "", str(s).lower())
LKEY = {key(t): t for t in LABELS} | {"bakers": "Baker's", "bakerscyst": "Baker's"}

# ---- 1. our labels + the 58 radiologist answers ------------------------------------------------
csvs = []
for root, dirs, files in os.walk("/kaggle/input"):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    csvs += [os.path.join(root, f) for f in files if f.endswith(".csv")]
ours = [p for p in csvs if os.path.basename(p) == "final_labels.csv"]
assert ours, "STOPPED: attach knee-labels-final (it holds the 58 radiologist answers)"
OUR = pd.read_csv(ours[0], dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
gold_ids = OUR.index[OUR.is_gold == 1].tolist()
G = OUR.loc[gold_ids, [f"gold_{t}" for t in LABELS]].astype(int)
G.columns = LABELS

# ---- 2. find every other table that looks like labels ----------------------------------------
def load_table(path):
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return None
    id_col = next((c for c in df.columns if key(c) in ("studyinstanceuid", "studyuid", "studyid", "study")), None)
    found = {c: LKEY[key(c)] for c in df.columns if key(c) in LKEY}
    if id_col is None or len(set(found.values())) < 12:
        return None
    df = pd.read_csv(path, dtype={id_col: str})
    df = df.rename(columns={id_col: "StudyInstanceUID", **found}).drop_duplicates("StudyInstanceUID")
    df = df.set_index("StudyInstanceUID")[LABELS].apply(pd.to_numeric, errors="coerce")
    if df.max().max() > 1.0001 or df.min().min() < -0.0001:      # logits or 0-100 scores, not probabilities
        return None
    return df

tables = {"OURS (knee-labels-final)": OUR[LABELS].astype(float)}
for p in csvs:
    if p in ours:
        continue
    t = load_table(p)
    if t is not None:
        tables[os.path.relpath(p, "/kaggle/input")] = t
print("LABEL TABLES FOUND:", len(tables))

# ---- 3. score each one on the 58, the way the 1st-place team did -------------------------------
def score(T):
    ids = [s for s in gold_ids if s in T.index]
    P, Y = T.loc[ids, LABELS].clip(1e-4, 1 - 1e-4), G.loc[ids]
    aucs = []
    for t in LABELS:
        m = P[t].notna()
        if Y.loc[m, t].nunique() == 2:
            aucs.append(roc_auc_score(Y.loc[m, t], P.loc[m, t]))
    p, y = P.values, Y.values.astype(float)
    ok = ~np.isnan(p)
    ll = -np.mean(y[ok] * np.log(p[ok]) + (1 - y[ok]) * np.log(1 - p[ok]))
    confident_wrong = int(((p >= 0.8) & (y == 0)).sum() + ((p <= 0.2) & (y == 1)).sum())
    return np.mean(aucs), ll, confident_wrong, len(ids), aucs

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def calibrate(T, k): return 1 / (1 + np.exp(-k * logit(T)))

def best_k(T, ids):              # one "temperature" k for all findings, fitted by log loss
    ks = np.linspace(0.3, 2.0, 69)
    P, Y = T.loc[ids, LABELS].values, G.loc[ids].values.astype(float)
    ok = ~np.isnan(P)
    losses = []
    for k in ks:
        q = np.clip(calibrate(P[ok], k), 1e-4, 1 - 1e-4)
        losses.append(-np.mean(Y[ok] * np.log(q) + (1 - Y[ok]) * np.log(1 - q)))
    return ks[int(np.argmin(losses))]

rows = []
for name, T in tables.items():
    auc, ll, cw, n, aucs = score(T)
    ids = [s for s in gold_ids if s in T.index]
    # out-of-fold calibration: fit k on 4/5 of the 58, apply to the other 1/5, so the score is honest
    rng = np.random.RandomState(0); order = rng.permutation(ids); oof = []
    for f in range(5):
        hold = list(order[f::5]); fit = [s for s in ids if s not in hold]
        oof.append(calibrate(T.loc[hold, LABELS], best_k(T, fit)))
    _, ll_cal, cw_cal, _, _ = score(pd.concat(oof))
    rows.append(dict(table=name, studies=len(T), gold_covered=n, AUC=round(auc, 4), log_loss=round(ll, 3),
                     confident_errors=cw, log_loss_calibrated=round(ll_cal, 3),
                     confident_errors_calibrated=cw_cal, k=round(best_k(T, ids), 2),
                     **{t: round(a, 3) for t, a in zip(LABELS, aucs)}))
R = pd.DataFrame(rows).sort_values(["AUC", "log_loss_calibrated"], ascending=[False, True])
pd.set_option("display.width", 250, "display.max_columns", 30)
print("\nSCORES ON THE 58 RADIOLOGIST STUDIES (higher AUC, lower log loss, fewer confident errors = better)")
print(R[["table", "studies", "gold_covered", "AUC", "log_loss", "confident_errors",
         "log_loss_calibrated", "confident_errors_calibrated", "k"]].to_string(index=False))
print("\nPER FINDING AUC:")
print(R[["table"] + LABELS].to_string(index=False))
R.to_csv("/kaggle/working/label_scores.csv", index=False)

# ---- 4. write the best table in our training format -------------------------------------------
best = R.iloc[0]
T = tables[best.table]
if best.table.startswith("OURS"):
    print("\nOur own labels are still the best attached table - nothing new written.")
else:
    cal = calibrate(T[LABELS], best.k)
    missing = [s for s in OUR.index if s not in cal.index]          # studies the new table lacks keep OUR label
    out = pd.concat([cal, OUR.loc[missing, LABELS]]).loc[OUR.index]
    out = out.fillna(OUR[LABELS].astype(float))
    for c in ["is_gold"] + [f"gold_{t}" for t in LABELS]:
        out[c] = OUR[c]
    os.makedirs("/kaggle/working/better_labels", exist_ok=True)
    out.reset_index().to_csv("/kaggle/working/better_labels/final_labels.csv", index=False)
    print(f"\nWROTE better_labels/final_labels.csv from '{best.table}' (calibrated with k={best.k}; "
          f"{len(missing)} studies it lacked keep our label).")
print("L1 DONE - send Claude everything from 'LABEL TABLES FOUND' down.")
