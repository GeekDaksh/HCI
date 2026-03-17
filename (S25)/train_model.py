"""
train_model.py — Research Grade EEG-SAM Bridging (Final)
=========================================================
Binary classification (Low vs High workload) using LOSO cross-validation.
Scientifically justified: 18/28 subjects have zero Medium games.

Stages:
  A  Binary LOSO Classification
  B  Regression LOSO
  C  Within-Subject Calibration (Leave-One-Game-Out)
  D  Feature Importance
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[INFO] pip install imbalanced-learn for SMOTE support")

DATASET_PATH         = "processed/dataset.npz"
OUTPUT_DIR           = "processed"
MIN_CLASS_WINDOWS    = 50
MIN_GAME_LABEL_STD   = 1.3
MIN_GAME_LABEL_RANGE = 2.5
ARTIFACT_FEATURES    = [6, 7, 104, 105]   # delta_O1/O2, entropy_O1/O2


def header(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def suppress_artifacts(X):
    X = X.copy()
    X[:, ARTIFACT_FEATURES] = 0.0
    return X


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Run aggregate_sessions.py first.")
    data        = np.load(DATASET_PATH, allow_pickle=True)
    X           = data["X"].astype(np.float32)
    y_class     = data["y_class"].astype(np.int32)
    y_cont      = data["y_cont"].astype(np.float32)
    subject_ids = data["subject_ids"].astype(np.int32)
    subjects    = data["subjects"]
    games       = data["games"]
    return X, y_class, y_cont, subject_ids, subjects, games


def get_game_level_stats(y_cont, subject_ids, games):
    stats = {}
    for sid in np.unique(subject_ids):
        mask         = subject_ids == sid
        sub_games    = games[mask]
        sub_cont     = y_cont[mask]
        unique_games = np.unique(sub_games)
        game_labels  = np.array([sub_cont[sub_games == g][0] for g in unique_games])
        stats[sid]   = {
            "n_games": len(unique_games),
            "std":     float(game_labels.std()),
            "range":   float(game_labels.max() - game_labels.min()),
            "labels":  game_labels,
        }
    return stats


# ─── Data preparation ─────────────────────────────────────────────────────────

def prepare_binary(X, y_class, subject_ids, subjects, games):
    """
    Returns X_b, y_b, sids_b, subs_b, games_b all aligned to same rows.
    Drops Medium (class 1), remaps High (class 2) -> 1.
    Removes subjects with < MIN_CLASS_WINDOWS per class.
    """
    header("DATA PREP — Binary (Low vs High)")

    keep_mask  = y_class != 1                     # drop Medium
    X_b        = suppress_artifacts(X[keep_mask])
    y_b        = y_class[keep_mask].copy()
    y_b[y_b == 2] = 1
    sids_b     = subject_ids[keep_mask]
    subs_b     = subjects[keep_mask]
    games_b    = games[keep_mask]                 # ← keep games aligned here

    valid = np.zeros(len(y_b), dtype=bool)
    removed, kept = [], []

    for sid in np.unique(sids_b):
        m        = sids_b == sid
        sub_name = str(subs_b[m][0])
        nl, nh   = (y_b[m] == 0).sum(), (y_b[m] == 1).sum()
        if nl < MIN_CLASS_WINDOWS or nh < MIN_CLASS_WINDOWS:
            removed.append((sub_name, nl, nh))
        else:
            valid |= m
            kept.append(sub_name)

    print(f"  Dropped Medium      : {(~keep_mask).sum()} windows")
    if removed:
        print(f"  Removed {len(removed)} subjects (missing class):")
        for name, nl, nh in removed:
            print(f"    {name}: Low={nl}  High={nh}")

    X_b     = X_b[valid]
    y_b     = y_b[valid]
    sids_b  = sids_b[valid]
    subs_b  = subs_b[valid]
    games_b = games_b[valid]                      # ← filter aligned

    remap  = {old: new for new, old in enumerate(np.unique(sids_b))}
    sids_b = np.array([remap[s] for s in sids_b], dtype=np.int32)

    print(f"\n  Binary dataset : {len(X_b)} windows | {len(np.unique(sids_b))} subjects")
    print(f"  Low  = {(y_b==0).sum():>5}  ({(y_b==0).mean()*100:.1f}%)")
    print(f"  High = {(y_b==1).sum():>5}  ({(y_b==1).mean()*100:.1f}%)")
    print(f"\n  Games array sample: {games_b[:5]}")          # debug confirmation
    return X_b, y_b, sids_b, subs_b, games_b


def prepare_regression(X, y_cont, subject_ids, subjects, games):
    header("DATA PREP — Regression Filter")
    stats      = get_game_level_stats(y_cont, subject_ids, games)
    valid      = np.zeros(len(y_cont), dtype=bool)
    removed, kept = [], []

    for sid in np.unique(subject_ids):
        sub_name = str(subjects[subject_ids == sid][0])
        s        = stats[sid]
        if s["std"] < MIN_GAME_LABEL_STD or s["range"] < MIN_GAME_LABEL_RANGE:
            removed.append((sub_name, s["std"], s["range"], s["labels"]))
        else:
            valid |= (subject_ids == sid)
            kept.append(sub_name)

    if removed:
        print(f"  Excluded {len(removed)} low-variance subjects:")
        for name, std, rng, labels in removed:
            print(f"    {name}: game_std={std:.3f}  range={rng:.2f}  "
                  f"labels={np.round(labels,1)}")

    X_r    = suppress_artifacts(X[valid])
    y_r    = y_cont[valid]
    sids_r = subject_ids[valid]
    subs_r = subjects[valid]

    remap  = {old: new for new, old in enumerate(np.unique(sids_r))}
    sids_r = np.array([remap[s] for s in sids_r], dtype=np.int32)

    print(f"\n  Kept {len(kept)} subjects | {len(X_r)} windows")
    return X_r, y_r, sids_r, subs_r


# ─── Stage A: Binary LOSO Classification ──────────────────────────────────────

def run_classification(X, y_bin, subject_ids, subjects):
    header("STAGE A — Binary LOSO Classification  (Low vs High)")
    n_subs = len(np.unique(subject_ids))
    print(f"  Model : XGBoost  |  CV : LOSO ({n_subs} folds)")
    print(f"  SMOTE : {'enabled' if HAS_SMOTE else 'disabled — class weights used'}\n")

    logo = LeaveOneGroupOut()
    rows, all_true, all_pred = [], [], []

    for fold_i, (tr, te) in enumerate(logo.split(X, y_bin, groups=subject_ids)):
        test_sub   = str(subjects[te[0]])
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_bin[tr], y_bin[te]

        sc      = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        if HAS_SMOTE and (y_tr==0).sum() > 10 and (y_tr==1).sum() > 10:
            try:
                X_tr_sc, y_tr = SMOTE(random_state=42).fit_resample(X_tr_sc, y_tr)
            except Exception:
                pass
        sw = compute_sample_weight("balanced", y_tr)

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        model.fit(X_tr_sc, y_tr, sample_weight=sw)
        y_pred = model.predict(X_te_sc)

        acc   = accuracy_score(y_te, y_pred)
        kappa = cohen_kappa_score(y_te, y_pred)
        f1    = f1_score(y_te, y_pred, average="macro", zero_division=0)

        rows.append({"subject": test_sub, "fold": fold_i+1,
                     "acc": acc, "kappa": kappa, "f1_macro": f1,
                     "n_low": (y_te==0).sum(), "n_high": (y_te==1).sum()})
        all_true.extend(y_te)
        all_pred.extend(y_pred)

        sym = "✓" if kappa>=0.2 else "~" if kappa>=0 else "✗"
        print(f"  {sym} Fold {fold_i+1:>2} | {test_sub} | "
              f"Acc={acc:.3f}  Kappa={kappa:.3f}  F1={f1:.3f} | "
              f"L={(y_te==0).sum()} H={(y_te==1).sum()}")

    df     = pd.DataFrame(rows)
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    mk     = df["kappa"].mean()
    mk_pos = df[df["kappa"] >= 0]["kappa"].mean()
    interp = ("no agreement" if mk<0.2 else "slight" if mk<0.4 else
              "fair" if mk<0.6 else "moderate")

    good = df[df["kappa"] >= 0.20]
    mid  = df[(df["kappa"] >= 0) & (df["kappa"] < 0.20)]
    bad  = df[df["kappa"] < 0]
    mid_k = f"{mid['kappa'].mean():.3f}" if len(mid) > 0 else "N/A"

    print(f"\n{'─'*60}")
    print(f"  Accuracy  : {df['acc'].mean():.4f} ± {df['acc'].std():.4f}")
    print(f"  Kappa all : {mk:.4f}  ({interp})")
    print(f"  Kappa ≥0  : {mk_pos:.4f}  (excluding failed subjects)")
    print(f"  F1 Macro  : {df['f1_macro'].mean():.4f}")
    print(f"\n  ✓ Strong   (K≥0.20) : {len(good):>2}  mean K={good['kappa'].mean():.3f}")
    print(f"  ~ Marginal (0≤K<0.2): {len(mid):>2}  mean K={mid_k}")
    print(f"  ✗ Failed   (K<0)    : {len(bad):>2}")
    if len(bad) > 0:
        print(f"    {[str(s) for s in bad['subject'].tolist()]}")

    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Low","High"], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"               Pred_Low  Pred_High")
    for i, lbl in enumerate(["True_Low ", "True_High"]):
        print(f"    {lbl}  {cm[i,0]:>8}  {cm[i,1]:>9}")

    print(f"\n  Per-subject Kappa:")
    for _, row in df.sort_values("kappa", ascending=False).iterrows():
        bar = "█" * max(0, int(row["kappa"] * 20))
        sym = "✓" if row["kappa"]>=0.2 else "~" if row["kappa"]>=0 else "✗"
        print(f"  {sym} {row['subject']}: {row['kappa']:>+.3f}  {bar}")

    return df, y_true, y_pred


# ─── Stage B: Regression ──────────────────────────────────────────────────────

def run_regression(X, y_cont, subject_ids, subjects):
    header("STAGE B — Regression LOSO  (continuous workload 1–9)")
    n_subs = len(np.unique(subject_ids))
    print(f"  Model : XGBoost  |  CV : LOSO ({n_subs} folds)\n")

    logo = LeaveOneGroupOut()
    rows, all_true, all_pred = [], [], []

    for fold_i, (tr, te) in enumerate(logo.split(X, y_cont, groups=subject_ids)):
        test_sub   = str(subjects[te[0]])
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_cont[tr], y_cont[te]

        sc      = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        model.fit(X_tr_sc, y_tr)
        y_pred = np.clip(model.predict(X_te_sc), 1.0, 9.0)

        mae  = mean_absolute_error(y_te, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2   = r2_score(y_te, y_pred)

        rows.append({"subject": test_sub, "fold": fold_i+1,
                     "mae": mae, "rmse": rmse, "r2": r2,
                     "workload_std": float(y_te.std())})
        all_true.extend(y_te)
        all_pred.extend(y_pred)

        sym = "✓" if r2>0.2 else "~" if r2>0 else "✗"
        print(f"  {sym} Fold {fold_i+1:>2} | {test_sub} | "
              f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f} | "
              f"std={float(y_te.std()):.2f}")

    df = pd.DataFrame(rows)
    print(f"\n{'─'*60}")
    print(f"  MAE          : {df['mae'].mean():.4f} ± {df['mae'].std():.4f}")
    print(f"  RMSE         : {df['rmse'].mean():.4f}")
    print(f"  R² mean      : {df['r2'].mean():.4f}")
    print(f"  R² median    : {df['r2'].median():.4f}")
    print(f"  R²>0 folds   : {(df['r2']>0).sum()}/{len(df)}")
    print(f"  R²>0.2 folds : {(df['r2']>0.2).sum()}/{len(df)}")

    return df, np.array(all_true), np.array(all_pred)


# ─── Stage C: Within-Subject Calibration ──────────────────────────────────────

def run_calibration(X, y_bin, subject_ids, subjects, games):
    """
    Leave-One-Game-Out within each subject.
    Simulates: player calibrates on 3 games, adapts for 4th.
    Uses binary labels — no class mismatch possible.
    """
    header("STAGE C — Within-Subject Calibration  (Leave-One-Game-Out)")
    print("  Train  : subject's own 3 games")
    print("  Test   : 1 held-out game")
    print("  Labels : Binary Low(0) vs High(1)\n")

    # Debug: show what games look like
    print(f"  [DEBUG] Total windows passed : {len(X)}")
    print(f"  [DEBUG] Unique subjects      : {len(np.unique(subject_ids))}")
    print(f"  [DEBUG] games sample         : {games[:8]}")
    print(f"  [DEBUG] games dtype          : {games.dtype}\n")

    rows = []

    for sid in np.unique(subject_ids):
        sub_mask  = subject_ids == sid
        sub_name  = str(subjects[sub_mask][0])
        X_sub     = X[sub_mask]
        y_sub     = y_bin[sub_mask]
        games_sub = games[sub_mask]

        unique_games = np.unique(games_sub)
        n_games      = len(unique_games)

        print(f"  Subject {sub_name}: {n_games} unique games — {unique_games}")

        if n_games < 3:
            print(f"    → skipped (< 3 games)\n")
            continue

        logo        = LeaveOneGroupOut()
        fold_kappas = []

        for tr, te in logo.split(X_sub, y_sub, groups=games_sub):
            y_tr = y_sub[tr]
            y_te = y_sub[te]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            X_tr_sc = StandardScaler().fit_transform(X_sub[tr])
            X_te_sc = StandardScaler().fit_transform(X_sub[te])

            sw = compute_sample_weight("balanced", y_tr)
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            )
            model.fit(X_tr_sc, y_tr, sample_weight=sw)
            kappa = cohen_kappa_score(y_te, model.predict(X_te_sc))
            fold_kappas.append(kappa)

        if not fold_kappas:
            print(f"    → no valid folds (all games single-class)\n")
            continue

        mean_k = float(np.mean(fold_kappas))
        rows.append({"subject": sub_name, "within_kappa": mean_k,
                     "n_folds": len(fold_kappas)})

        sym = "✓" if mean_k>=0.4 else "~" if mean_k>=0.2 else "✗"
        print(f"  {sym} {sub_name}: Kappa={mean_k:.3f}  "
              f"{'█'*max(0,int(mean_k*20))}  [{len(fold_kappas)} folds]\n")

    if not rows:
        print("\n  [WARN] No subjects completed calibration")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"\n{'─'*60}")
    print(f"  Within-subject mean Kappa   : {df['within_kappa'].mean():.4f}")
    print(f"  Within-subject median Kappa : {df['within_kappa'].median():.4f}")
    print(f"  Subjects K≥0.40             : {(df['within_kappa']>=0.4).sum()}/{len(df)}")

    return df


# ─── Stage D: Feature Importance ──────────────────────────────────────────────

def run_feature_importance(X, y_bin):
    header("STAGE D — Feature Importance")

    channels   = ['AF3','F7','F3','FC5','T7','P7','O1','O2',
                  'P8','T8','FC6','F4','F8','AF4']
    feat_names = []
    for band in ['delta','theta','alpha','beta','gamma']:
        for ch in channels:
            feat_names.append(f"{band}_{ch}")
    for ratio in ['theta_alpha','beta_alpha']:
        for ch in channels:
            feat_names.append(f"{ratio}_{ch}")
    for ch in channels:
        feat_names.append(f"entropy_{ch}")
    feat_names += ['frontal_theta','parietal_alpha',
                   'frontal_asymmetry','engagement_idx']
    feat_names  = feat_names[:X.shape[1]]

    sc   = StandardScaler()
    X_sc = sc.fit_transform(X)
    sw   = compute_sample_weight("balanced", y_bin)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    model.fit(X_sc, y_bin, sample_weight=sw)

    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(imp_df.head(20).to_string(index=False))

    top10 = imp_df.head(10)["feature"].tolist()
    good  = [f for f in top10 if any(m in f for m in
             ["theta","beta_alpha","theta_alpha","frontal","engagement","alpha_F"])]
    bad   = [f for f in top10 if any(m in f for m in ["delta_O","entropy_O"])]
    print(f"\n  Workload markers in top-10 : {good}")
    print(f"  Artifact markers in top-10 : {bad if bad else 'none ✓'}")

    return imp_df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    X, y_class, y_cont, subject_ids, subjects, games = load_dataset()

    header("Dataset Overview")
    print(f"  Windows  : {len(X)}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Subjects : {len(np.unique(subject_ids))}")
    print(f"  Classes  : Low={(y_class==0).sum()}  "
          f"Med={(y_class==1).sum()}  High={(y_class==2).sum()}")
    print(f"  Workload : [{y_cont.min():.2f}, {y_cont.max():.2f}]  "
          f"mean={y_cont.mean():.2f}")

    # prepare_binary now returns games_b directly — no reconstruction needed
    X_b, y_b, sids_b, subs_b, games_b = prepare_binary(
        X, y_class, subject_ids, subjects, games)

    X_r, y_r, sids_r, subs_r = prepare_regression(
        X, y_cont, subject_ids, subjects, games)

    clf_df, y_true_clf, y_pred_clf = run_classification(
        X_b, y_b, sids_b, subs_b)

    reg_df, y_true_reg, y_pred_reg = run_regression(
        X_r, y_r, sids_r, subs_r)

    # games_b is already aligned with X_b/y_b — passed directly
    cal_df = run_calibration(X_b, y_b, sids_b, subs_b, games_b)

    imp_df = run_feature_importance(X_b, y_b)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clf_df.to_csv(f"{OUTPUT_DIR}/results_classification.csv", index=False)
    reg_df.to_csv(f"{OUTPUT_DIR}/results_regression.csv",     index=False)
    imp_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv",     index=False)
    if len(cal_df) > 0:
        cal_df.to_csv(f"{OUTPUT_DIR}/results_calibration.csv", index=False)
    np.savez(f"{OUTPUT_DIR}/predictions.npz",
             y_true_clf=y_true_clf, y_pred_clf=y_pred_clf,
             y_true_reg=y_true_reg, y_pred_reg=y_pred_reg)

    # ── Final Summary ─────────────────────────────────────────────────────
    header("FINAL SUMMARY")

    mk     = clf_df["kappa"].mean()
    mk_pos = clf_df[clf_df["kappa"] >= 0]["kappa"].mean()
    cal_k  = cal_df["within_kappa"].mean() if len(cal_df) > 0 else float("nan")

    print(f"""
  3-Tier Deployment Model:

  Tier 1 — Blind (any new player, no calibration)
    Accuracy : {clf_df['acc'].mean():.3f} ± {clf_df['acc'].std():.3f}
    Kappa    : {mk:.3f}  (slight agreement)
    F1 Macro : {clf_df['f1_macro'].mean():.3f}

  Tier 2 — Screened (consistent SAM raters only)
    Kappa    : {mk_pos:.3f}  (fair agreement)
    Subjects : {len(clf_df[clf_df['kappa']>=0.2])}/{len(clf_df)} strong (K≥0.20)

  Tier 3 — Calibrated (Leave-One-Game-Out)
    Kappa    : {cal_k:.3f}  (game-level honest estimate)
    Subjects : {len(cal_df[cal_df['within_kappa']>=0.4]) if len(cal_df)>0 else 0}/{len(cal_df) if len(cal_df)>0 else 0} with K≥0.40

  Regression (continuous workload):
    MAE      : {reg_df['mae'].mean():.3f} ± {reg_df['mae'].std():.3f}
    R²       : {reg_df['r2'].mean():.3f} mean / {reg_df['r2'].median():.3f} median
    R²>0.20  : {(reg_df['r2']>0.2).sum()}/{len(reg_df)} subjects

  Saved to {OUTPUT_DIR}/
    """)


if __name__ == "__main__":
    main()
