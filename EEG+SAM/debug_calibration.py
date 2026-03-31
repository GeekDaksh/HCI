"""
debug_calibration.py
====================
Run this standalone to diagnose why Stage C shows no subjects.
Place in your HCI/ folder and run: python debug_calibration.py
"""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb

data        = np.load("processed/dataset.npz", allow_pickle=True)
X           = data["X"].astype(np.float32)
y_class     = data["y_class"].astype(np.int32)
subject_ids = data["subject_ids"].astype(np.int32)
subjects    = data["subjects"]
games       = data["games"]

print("=== RAW GAMES ARRAY ===")
print(f"dtype  : {games.dtype}")
print(f"length : {len(games)}")
print(f"sample : {games[:12]}")
print(f"unique : {np.unique(games)[:20]}")

# Apply binary filter in ONE pass — same array, same rows
keep        = y_class != 1
X_b         = X[keep]
y_b         = y_class[keep].copy()
y_b[y_b==2] = 1
sids_b      = subject_ids[keep]
subs_b      = subjects[keep]
games_b     = games[keep]            # ← aligned, same filter

# Remove subjects missing a class
valid = np.zeros(len(y_b), dtype=bool)
for sid in np.unique(sids_b):
    m  = sids_b == sid
    nl = (y_b[m]==0).sum()
    nh = (y_b[m]==1).sum()
    if nl >= 50 and nh >= 50:
        valid |= m

X_b     = X_b[valid]
y_b     = y_b[valid]
sids_b  = sids_b[valid]
subs_b  = subs_b[valid]
games_b = games_b[valid]

# Remap IDs
remap  = {old: new for new, old in enumerate(np.unique(sids_b))}
sids_b = np.array([remap[s] for s in sids_b])

print(f"\n=== AFTER BINARY FILTER ===")
print(f"Windows  : {len(X_b)}")
print(f"Subjects : {len(np.unique(sids_b))}")
print(f"games_b sample : {games_b[:12]}")
print(f"games_b unique : {np.unique(games_b)[:20]}")

print(f"\n=== PER-SUBJECT GAME BREAKDOWN ===")
for sid in np.unique(sids_b):
    m            = sids_b == sid
    sub_name     = str(subs_b[m][0])
    unique_games = np.unique(games_b[m])
    print(f"  {sub_name}: {len(unique_games)} games — {unique_games}")

print(f"\n=== RUNNING CALIBRATION ===")
rows = []
for sid in np.unique(sids_b):
    m            = sids_b == sid
    sub_name     = str(subs_b[m][0])
    X_sub        = X_b[m]
    y_sub        = y_b[m]
    games_sub    = games_b[m]
    unique_games = np.unique(games_sub)

    if len(unique_games) < 3:
        print(f"  SKIP {sub_name}: only {len(unique_games)} unique games")
        continue

    logo        = LeaveOneGroupOut()
    fold_kappas = []

    for tr, te in logo.split(X_sub, y_sub, groups=games_sub):
        y_tr = y_sub[tr]
        y_te = y_sub[te]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        sc      = StandardScaler()
        X_tr_sc = sc.fit_transform(X_sub[tr])
        X_te_sc = sc.transform(X_sub[te])

        sw    = compute_sample_weight("balanced", y_tr)
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        model.fit(X_tr_sc, y_tr, sample_weight=sw)
        fold_kappas.append(cohen_kappa_score(y_te, model.predict(X_te_sc)))

    if fold_kappas:
        mk = np.mean(fold_kappas)
        rows.append(mk)
        print(f"  {sub_name}: Kappa={mk:.3f}  [{len(fold_kappas)} folds]")
    else:
        print(f"  {sub_name}: no valid folds (all games single-class)")

if rows:
    print(f"\nMean within-subject Kappa: {np.mean(rows):.4f}")
else:
    print("\nNo subjects completed calibration — check game breakdown above")
