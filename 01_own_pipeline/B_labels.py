# =============================================================================
# B - BEST LABELS PER FINDING (CPU notebook, free)
#
# How to run:
#   1. Make a copy of "RSNA Knee baseline v1" (it has the rule-based reader `extract`).
#      Name it "knee-labels-final". Accelerator None, Internet off.
#   2. Add Input: knee-llm-labels (Qwen labels), stept-teacher-py (teacher_oof.csv), and
#      EVERY public report-label dataset you can find (Datasets tab; authors credited in the
#      0.958 notebook: stevenleehans, Pilkwang Kim, lixin73, laymond).
#   3. Delete the last cell (main()), paste this, Save & Run All (Commit). A few minutes.
#
# Every CSV that has study IDs and the twelve findings is treated as one label source.
# Each source is scored against the 58 radiologist-annotated studies, per finding; for each
# finding the labels are the average of its k best sources, with k chosen on the same 58.
# Output: final_labels.csv (soft labels 0..1 for every training study) + label_report.txt
# =============================================================================
from sklearn.metrics import roc_auc_score

OUT = Path("/kaggle/working")
train_df = pd.read_csv(ROOT / "train.csv").drop_duplicates("StudyInstanceUID").reset_index(drop=True)
ids = train_df["StudyInstanceUID"].astype(str).values
idset = set(ids)
gold = train_df.set_index("StudyInstanceUID")[TARGETS]
gold = (gold[gold.notna().all(axis=1)] > 0.5).astype(int)
gidx = gold.index
print(f"{len(ids)} training studies, {len(gidx)} radiologist-annotated")

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

KEYS = {"ACL": ["acl", "anteriorcruciate"], "MCL": ["mcl", "medialcollateral"],
        "Medial Meniscus": ["medialmeniscus", "medmeniscus"],
        "Lateral Meniscus": ["lateralmeniscus", "latmeniscus"],
        "Medial OA": ["medialoa", "medoa", "medialosteoarthritis"],
        "Lateral OA": ["lateraloa", "latoa", "lateralosteoarthritis"],
        "PF OA": ["pfoa", "patellofemoral"],
        "Effusion": ["effusion"], "Synovitis": ["synovitis"], "Baker's": ["baker"],
        "Contusion": ["contusion", "bonebruise"], "Fracture": ["fracture"]}
SKIP_COL = ("conf", "code", "reason", "evidence", "note", "text", "gold", "weight", "fold", "mask", "llm_", "w_")
WORDS = {"yes": 1, "true": 1, "positive": 1, "present": 1, "no": 0, "false": 0, "negative": 0, "absent": 0}

def to_num(s):
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(s.astype(str).str.strip().str.lower().map(WORDS))

def match(cols):
    m = {}
    for t, keys in KEYS.items():
        for k in keys:
            hits = [c for c in cols if k in _norm(c) and not any(x in str(c).lower() for x in SKIP_COL)]
            if hits:
                m[t] = min(hits, key=lambda c: len(str(c)))
                break
    return m

def minmax(v):
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v) if np.isfinite(v).any() else 0, np.nanmax(v) if np.isfinite(v).any() else 1
    return (v - lo) / (hi - lo) if hi > lo else v * 0

# ---- collect sources ---------------------------------------------------------------------
sources = {}
lex = pd.DataFrame([extract(r) for r in train_df["Report"].fillna("")])
lex.index = ids
sources["rule_reader"] = lex[TARGETS].astype(float)

SKIP_FILES = {"train.csv", "test.csv", "train_series.csv", "test_series.csv", "sample_submission.csv",
              "labels.csv", "oof.csv", "oof_annotated.csv", "index.csv"}
for root, dirs, files in os.walk("/kaggle/input"):
    dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
    for f in files:
        if not f.endswith(".csv") or f in SKIP_FILES:
            continue
        p = Path(root) / f
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False)
        except Exception:
            continue
        idc = max(df.columns, key=lambda c: df[c].astype(str).isin(idset).sum())
        if df[idc].astype(str).isin(idset).sum() < 1000:
            continue
        m = match([c for c in df.columns if c != idc])
        if len(m) < 8:
            print(f"  skip {p.name}: only {len(m)} findings recognised ({list(df.columns)[:15]})")
            continue
        df = df.drop_duplicates(idc).set_index(df[idc].astype(str).values)
        name = f"{p.parent.name[:28]}/{p.stem[:28]}"
        sources[name] = pd.DataFrame({t: to_num(df[m[t]]) if t in m else np.nan for t in TARGETS})
        print(f"  source {name}: {df[idc].astype(str).isin(idset).sum()} studies, {len(m)} findings")

print(f"\n{len(sources)} label sources: {list(sources)}")
for k in list(sources):
    sources[k] = sources[k].reindex(ids).astype(float)
    for t in TARGETS:
        sources[k][t] = minmax(sources[k][t].values)

# ---- score every source per finding on the 58 radiologist studies ------------------------
def auc_on_gold(v, t):
    s = pd.Series(v, index=ids).reindex(gidx)
    ok = s.notna().values
    y = gold[t].values[ok]
    return roc_auc_score(y, s.values[ok]) if ok.sum() >= 40 and 0 < y.sum() < len(y) else np.nan

# Leak guard: a table that reproduces the radiologist labels on the 58 studies (copied gold, or
# predictions of a model trained on them) would look perfect here and teach nothing. Drop it.
for k in list(sources):
    if k == "rule_reader":
        continue
    v = sources[k].reindex(gidx)[TARGETS]
    ok = v.notna().all(axis=1)
    if ok.sum() < 40:
        continue
    same = ((v[ok].values > 0.5).astype(int) == gold[ok].values).all(axis=1).mean()
    aucs_k = [auc_on_gold(sources[k][t].values, t) for t in TARGETS]
    if same > 0.8 or np.nanmean(aucs_k) > 0.985:
        print(f"  LEAK GUARD: dropping {k} (matches the radiologist labels on {same:.0%} of the 58 "
              f"studies, mean AUC {np.nanmean(aucs_k):.3f}): it has seen the answers")
        del sources[k]

table = pd.DataFrame({k: [auc_on_gold(sources[k][t].values, t) for t in TARGETS] for k in sources},
                     index=TARGETS)
pd.set_option("display.float_format", lambda v: f"{v:.3f}"); pd.set_option("display.width", 250)
print("\n==== each label source vs radiologists (AUC per finding) ====")
print(table.to_string())
print("mean: " + "  ".join(f"{k} {table[k].mean():.3f}" for k in table.columns))

# ---- per finding: average of the k best sources, k picked on the 58 ----------------------
final = pd.DataFrame(index=ids)
choice, best_auc = {}, {}
for t in TARGETS:
    order = table.loc[t].dropna().sort_values(ascending=False).index.tolist()
    best = (-1, None, None)
    for k in range(1, min(5, len(order)) + 1):
        v = np.nanmean(np.stack([sources[s][t].values for s in order[:k]]), axis=0)
        a = auc_on_gold(v, t)
        if a > best[0] + 0.003:            # a bigger mix must earn its place
            best = (a, k, v)
    if best[1] is None:
        best = (np.nan, 1, sources["rule_reader"][t].values)
        order = ["rule_reader"]
    final[t] = best[2]
    choice[t] = order[:best[1]]
    best_auc[t] = best[0]
final[TARGETS] = final[TARGETS].fillna(sources["rule_reader"][TARGETS])

lines = [f"{t:17s} {best_auc[t]:.3f}  <- mean of {choice[t]}" for t in TARGETS]
summary = "\n".join(lines) + f"\n\nFINAL LABELS vs radiologists, mean AUC: {np.nanmean(list(best_auc.values())):.3f}" \
          "\n(our previous best: 0.864)"
print("\n==== chosen labels per finding ====\n" + summary)

final.index.name = "StudyInstanceUID"
final["is_gold"] = final.index.isin(gidx).astype(int)
for t in TARGETS:
    final[f"gold_{t}"] = gold[t].reindex(final.index)
final.reset_index().to_csv(OUT / "final_labels.csv", index=False)
(OUT / "label_report.txt").write_text(table.to_string() + "\n\n" + summary)
print("\nsaved final_labels.csv")
