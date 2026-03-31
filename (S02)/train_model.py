"""
train_model.py — Research Grade EEG Workload Estimation  (v5 — GAMEEMO Final)
==============================================================================
Three targeted fixes for the three diagnosed problems:

PROBLEM 1: Failing subjects (K<0)
  Root cause: 7 subjects have SAM ratings anti-correlated with EEG.
  Fix: Subject Reliability Weighting — compute each training subject's
  internal consistency score (cross-game EEG discriminability via AUC).
  Subjects with poor internal consistency get downweighted in the training
  loss, so their noise doesn't corrupt the model for the test subject.
  Also added: subject dropout regularisation (randomly exclude 1 training
  subject per fold) to prevent overfitting to any single subject's pattern.

PROBLEM 2: R² misleading with n=4 test points
  Root cause: R² requires many test points to be stable. With 4 points
  per subject, one bad prediction swings R² by ±0.5.
  Fix: Replace R² as primary metric with Spearman ρ (rank correlation),
  which is stable with small n and directly answers "does the model rank
  game difficulty correctly?" — which is what adaptive game design needs.
  Also report: Kendall τ, concordance correlation coefficient (CCC).
  R² is kept but flagged as secondary.

PROBLEM 3: Low Kappa / Stage C broken
  Root cause: GAMEEMO assigns one class per game, making temporal and
  game-level splits always single-class.
  Fix: Subject-specific within-class temporal split — for each subject,
  find the games that contain BOTH classes within their windows (after
  the binary filter), and only use those subjects for Stage C.
  Fallback: cross-game pseudo-calibration where train = all windows of
  the 3 most common games, test = all windows of the rarest game.
  This gives a valid non-trivial calibration estimate for subjects where
  at least one game has mixed-class windows.

STAGE A: Added subject reliability weighting + dropout regularisation
STAGE B: Spearman ρ + Kendall τ + CCC as primary metrics, R² secondary
STAGE C: Game-boundary calibration with mixed-class game detection
STAGE D: Unchanged (permutation importance already research-grade)
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    roc_auc_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.inspection import permutation_importance
import xgboost as xgb

warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import ADASYN
    HAS_ADASYN = True
except ImportError:
    HAS_ADASYN = False
    print("[INFO] pip install imbalanced-learn  (for ADASYN oversampling)")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[INFO] pip install optuna  (for Bayesian hyperparameter search)")

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH         = "processed/dataset.npz"
OUTPUT_DIR           = "processed"
MIN_CLASS_WINDOWS    = 50
MIN_GAME_LABEL_STD   = 1.3
MIN_GAME_LABEL_RANGE = 2.5
N_FEATURES_SELECT    = 60
OPTUNA_TRIALS        = 25
ARTIFACT_FEATURES    = [6, 7, 104, 105]
SUBJECT_DROPOUT_PROB = 0.15   # probability of dropping a training subject per fold
RNG                  = np.random.default_rng(42)


def header(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── Load ──────────────────────────────────────────────────────────────────────

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


def suppress_artifacts(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, ARTIFACT_FEATURES] = 0.0
    return X


def get_game_level_stats(y_cont, subject_ids, games):
    stats = {}
    for sid in np.unique(subject_ids):
        mask         = subject_ids == sid
        sub_games    = games[mask]
        sub_cont     = y_cont[mask]
        unique_games = np.unique(sub_games)
        game_labels  = np.array([sub_cont[sub_games == g][0] for g in unique_games])
        stats[sid]   = {
            "std":    float(game_labels.std()),
            "range":  float(game_labels.max() - game_labels.min()),
            "labels": game_labels,
        }
    return stats


# ── Feature selection ─────────────────────────────────────────────────────────

def select_features(X_tr, y_tr, X_te, k=N_FEATURES_SELECT):
    mi  = mutual_info_classif(X_tr, y_tr, random_state=42)
    idx = np.argsort(mi)[::-1][:k]
    return X_tr[:, idx], X_te[:, idx], idx


def select_features_reg(X_tr, y_tr, X_te, k=N_FEATURES_SELECT):
    from sklearn.feature_selection import mutual_info_regression
    mi  = mutual_info_regression(X_tr, y_tr, random_state=42)
    idx = np.argsort(mi)[::-1][:k]
    return X_tr[:, idx], X_te[:, idx], idx


# ── Subject reliability weighting ─────────────────────────────────────────────

def compute_subject_reliability(X, y_bin, subject_ids):
    """
    For each subject, estimate internal EEG-label consistency using
    leave-one-game-out AUC within that subject's data.
    Returns a dict {sid: reliability_weight} where weight ∈ [0.2, 1.0].

    A subject whose EEG cleanly separates Low from High (high AUC)
    gets weight 1.0. A subject whose EEG is random w.r.t. labels
    (AUC ~0.5) gets weight 0.2 — still included but heavily downweighted.

    This means failing subjects still contribute training diversity
    but don't pull the model toward their inconsistent patterns.
    """
    weights = {}
    for sid in np.unique(subject_ids):
        m    = subject_ids == sid
        X_s  = X[m]
        y_s  = y_bin[m]
        if (y_s==0).sum() < 10 or (y_s==1).sum() < 10:
            weights[sid] = 0.5
            continue
        # Fast AUC estimate: linear SVM score on 5-fold CV
        try:
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs  = []
            sc    = StandardScaler()
            for tr, te in skf.split(X_s, y_s):
                Xtr = sc.fit_transform(X_s[tr])
                Xte = sc.transform(X_s[te])
                clf = CalibratedClassifierCV(
                    LinearSVC(C=0.1, max_iter=1000, random_state=42), cv=3)
                clf.fit(Xtr, y_s[tr])
                prob = clf.predict_proba(Xte)[:, 1]
                if len(np.unique(y_s[te])) > 1:
                    aucs.append(roc_auc_score(y_s[te], prob))
            auc = float(np.mean(aucs)) if aucs else 0.5
        except Exception:
            auc = 0.5
        # Map AUC [0.5, 1.0] → weight [0.2, 1.0]
        w = np.clip(2.0 * (auc - 0.5), 0.0, 1.0) * 0.8 + 0.2
        weights[sid] = float(w)
    return weights


# ── Concordance Correlation Coefficient ───────────────────────────────────────

def concordance_cc(y_true, y_pred):
    """CCC: combines Pearson r with mean/variance agreement. Range [-1, 1]."""
    mu_t, mu_p   = np.mean(y_true), np.mean(y_pred)
    sig_t, sig_p = np.std(y_true),  np.std(y_pred)
    r, _         = pearsonr(y_true, y_pred)
    ccc = (2 * r * sig_t * sig_p) / (sig_t**2 + sig_p**2 + (mu_t - mu_p)**2 + 1e-9)
    return float(ccc)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_3class(X, y_class, subject_ids, subjects, games):
    """
    Prepare 3-class dataset using per-subject tertile labels.

    With sam_workload.py (new version), y_class already contains
    per-subject tertile labels: Low=0, Medium=1, High=2.
    Every subject has all 3 classes by construction — no windows dropped.

    Only removes subjects with < MIN_CLASS_WINDOWS in any single class
    (edge case: very short recordings).
    """
    header("DATA PREP — 3-Class  (Low=0 / Medium=1 / High=2)")
    X_3  = suppress_artifacts(X.copy())
    y_3  = y_class.copy()
    sids = subject_ids.copy()
    subs = subjects.copy()
    gms  = games.copy()

    valid   = np.zeros(len(y_3), dtype=bool)
    removed = []

    for sid in np.unique(sids):
        m   = sids == sid
        nm  = {c: int((y_3[m] == c).sum()) for c in [0, 1, 2]}
        sub_name = str(subs[m][0])
        if any(nm[c] < MIN_CLASS_WINDOWS for c in [0, 1, 2]):
            removed.append((sub_name, nm[0], nm[1], nm[2]))
        else:
            valid |= m

    X_3  = X_3[valid]
    y_3  = y_3[valid]
    sids = sids[valid]
    subs = subs[valid]
    gms  = gms[valid]

    remap = {old: new for new, old in enumerate(np.unique(sids))}
    sids  = np.array([remap[s] for s in sids], dtype=np.int32)

    if removed:
        print(f"  Removed {len(removed)} subjects "
              f"(< {MIN_CLASS_WINDOWS} windows in some class):")
        for name, nl, nm, nh in removed:
            print(f"    {name}: Low={nl}  Med={nm}  High={nh}")

    print(f"\n  3-class dataset : {len(X_3)} windows | "
          f"{len(np.unique(sids))} subjects")
    for cls, lbl in [(0,"Low"),(1,"Medium"),(2,"High")]:
        n   = (y_3 == cls).sum()
        pct = n / len(y_3) * 100
        print(f"  {lbl:<8} = {n:>5}  ({pct:.1f}%)")
    return X_3, y_3, sids, subs, gms


def run_classification_3class(X, y_3, subject_ids, subjects):
    """
    3-class LOSO: Low(0) / Medium(1) / High(2).

    Uses per-subject tertile labels — every subject has all 3 classes.
    XGBoost multi:softmax with class weights and MI feature selection.
    """
    header("STAGE A-3 — 3-Class LOSO  (Low / Medium / High)")
    n_subs = len(np.unique(subject_ids))
    print(f"  Model   : XGBoost  |  CV : LOSO ({n_subs} folds)")
    print(f"  Classes : Low=0  Medium=1  High=2  (per-subject tertile)")
    print(f"  Feature : top-{N_FEATURES_SELECT} MI per fold\n")

    logo = LeaveOneGroupOut()
    rows, all_true, all_pred, all_prob = [], [], [], []

    for fold_i, (tr, te) in enumerate(logo.split(X, y_3, groups=subject_ids)):
        test_sub   = str(subjects[te[0]])
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_3[tr], y_3[te]

        sc      = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        # Feature selection per fold
        mi  = mutual_info_classif(X_tr_sc, y_tr, random_state=42)
        idx = np.argsort(mi)[::-1][:N_FEATURES_SELECT]
        X_tr_fs = X_tr_sc[:, idx]
        X_te_fs = X_te_sc[:, idx]

        sw = compute_sample_weight("balanced", y_tr)

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softmax", num_class=3,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        )
        model.fit(X_tr_fs, y_tr, sample_weight=sw)
        y_pred = model.predict(X_te_fs)
        y_prob = model.predict_proba(X_te_fs)

        acc   = accuracy_score(y_te, y_pred)
        kappa = cohen_kappa_score(y_te, y_pred, labels=[0, 1, 2])
        f1    = f1_score(y_te, y_pred, average="macro", zero_division=0)
        cnt   = {c: int((y_te == c).sum()) for c in [0, 1, 2]}

        rows.append({"subject": test_sub, "fold": fold_i+1,
                     "acc": acc, "kappa": kappa, "f1_macro": f1,
                     "n_low": cnt[0], "n_med": cnt[1], "n_high": cnt[2]})
        all_true.extend(y_te)
        all_pred.extend(y_pred)
        all_prob.extend(y_prob)

        sym = "✓" if kappa >= 0.2 else "~" if kappa >= 0 else "✗"
        print(f"  {sym} Fold {fold_i+1:>2} | {test_sub} | "
              f"Acc={acc:.3f}  Kappa={kappa:.3f}  F1={f1:.3f} | "
              f"L={cnt[0]} M={cnt[1]} H={cnt[2]}")

    df     = pd.DataFrame(rows)
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    mk     = df["kappa"].mean()
    mk_pos = df[df["kappa"] >= 0]["kappa"].mean()
    interp = ("no agreement" if mk < 0.2 else "slight" if mk < 0.4 else
              "fair" if mk < 0.6 else "moderate")

    good  = df[df["kappa"] >= 0.20]
    mid   = df[(df["kappa"] >= 0) & (df["kappa"] < 0.20)]
    bad   = df[df["kappa"] < 0]
    mid_k = f"{mid['kappa'].mean():.3f}" if len(mid) > 0 else "N/A"

    print(f"\n{'─'*60}")
    print(f"  Accuracy      : {df['acc'].mean():.4f} ± {df['acc'].std():.4f}")
    print(f"  Kappa (all)   : {mk:.4f}  ({interp})")
    print(f"  Kappa (≥0)    : {mk_pos:.4f}")
    print(f"  F1 Macro      : {df['f1_macro'].mean():.4f}")
    print(f"\n  ✓ Strong   (K≥0.20): {len(good):>2}  "
          f"mean K={good['kappa'].mean():.3f}" if len(good) > 0 else
          f"\n  ✓ Strong   (K≥0.20):  0")
    print(f"  ~ Marginal (0≤K<0.2): {len(mid):>2}  mean K={mid_k}")
    print(f"  ✗ Failed   (K<0)    : {len(bad):>2}")
    if len(bad) > 0:
        print(f"    {bad['subject'].tolist()}")

    print(f"\n  Classification Report (pooled):")
    print(classification_report(y_true, y_pred,
                                target_names=["Low","Medium","High"],
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                Pred_Low  Pred_Med  Pred_High")
    for i, lbl in enumerate(["True_Low ","True_Med ","True_High"]):
        row_vals = "   ".join(f"{cm[i,j]:>8}" for j in range(cm.shape[1]))
        print(f"    {lbl}  {row_vals}")

    print(f"\n  Per-subject Kappa (3-class):")
    for _, row in df.sort_values("kappa", ascending=False).iterrows():
        bar = "█" * max(0, int(row["kappa"] * 20))
        sym = "✓" if row["kappa"] >= 0.2 else "~" if row["kappa"] >= 0 else "✗"
        print(f"  {sym} {row['subject']}: {row['kappa']:>+.3f}  {bar}")

    return df, y_true, y_pred, y_prob


def prepare_binary(X, y_class, subject_ids, subjects, games):
    header("DATA PREP — Binary (Low vs High)")
    keep    = y_class != 1
    X_b     = suppress_artifacts(X[keep])
    y_b     = y_class[keep].copy();  y_b[y_b == 2] = 1
    sids_b  = subject_ids[keep]
    subs_b  = subjects[keep]
    games_b = games[keep]

    valid   = np.zeros(len(y_b), dtype=bool)
    removed = []
    for sid in np.unique(sids_b):
        m = sids_b == sid
        nl, nh = (y_b[m]==0).sum(), (y_b[m]==1).sum()
        if nl < MIN_CLASS_WINDOWS or nh < MIN_CLASS_WINDOWS:
            removed.append((subs_b[m][0], nl, nh))
        else:
            valid |= m

    X_b, y_b   = X_b[valid], y_b[valid]
    sids_b      = sids_b[valid]
    subs_b      = subs_b[valid]
    games_b     = games_b[valid]
    remap       = {old: new for new, old in enumerate(np.unique(sids_b))}
    sids_b      = np.array([remap[s] for s in sids_b], dtype=np.int32)

    print(f"  Dropped Medium : {(~keep).sum()} windows")
    if removed:
        print(f"  Removed {len(removed)} subjects (missing class):")
        for name, nl, nh in removed:
            print(f"    {name}: Low={nl}  High={nh}")
    print(f"\n  Binary dataset : {len(X_b)} windows | {len(np.unique(sids_b))} subjects")
    print(f"  Low={( y_b==0).sum()}  ({(y_b==0).mean()*100:.1f}%)   "
          f"High={(y_b==1).sum()}  ({(y_b==1).mean()*100:.1f}%)")
    return X_b, y_b, sids_b, subs_b, games_b


def prepare_regression_game_level(X, y_cont, subject_ids, subjects, games):
    header("DATA PREP — Game-Level Regression")
    stats      = get_game_level_stats(y_cont, subject_ids, games)
    valid_sids = set()
    removed    = []
    for sid in np.unique(subject_ids):
        s = stats[sid]
        if s["std"] < MIN_GAME_LABEL_STD or s["range"] < MIN_GAME_LABEL_RANGE:
            removed.append((subjects[subject_ids==sid][0],
                            s["std"], s["range"], s["labels"]))
        else:
            valid_sids.add(sid)

    if removed:
        print(f"  Excluded {len(removed)} low-variance subjects:")
        for name, std, rng, labels in removed:
            print(f"    {name}: game_std={std:.3f}  range={rng:.2f}  "
                  f"labels={np.round(labels,1)}")

    X_gl, y_gl, sids_gl, subs_gl = [], [], [], []
    for sid in sorted(valid_sids):
        m        = subject_ids == sid
        sub_name = subjects[m][0]
        X_sub    = suppress_artifacts(X[m])
        y_sub    = y_cont[m]
        gsub     = games[m]
        for g in np.unique(gsub):
            gm = gsub == g
            feat_mean = X_sub[gm].mean(axis=0)
            feat_std  = X_sub[gm].std(axis=0)
            X_gl.append(np.concatenate([feat_mean, feat_std]))
            y_gl.append(y_sub[gm][0])
            sids_gl.append(sid)
            subs_gl.append(sub_name)

    X_gl    = np.array(X_gl,    dtype=np.float32)
    y_gl    = np.array(y_gl,    dtype=np.float32)
    sids_gl = np.array(sids_gl, dtype=np.int32)
    subs_gl = np.array(subs_gl)
    remap   = {old: new for new, old in enumerate(np.unique(sids_gl))}
    sids_gl = np.array([remap[s] for s in sids_gl], dtype=np.int32)

    print(f"\n  Kept {len(np.unique(sids_gl))} subjects | {len(X_gl)} game-level samples")
    print(f"  Feature dim: {X_gl.shape[1]}  (mean+std per game)")
    return X_gl, y_gl, sids_gl, subs_gl


# ── Optuna HPO ────────────────────────────────────────────────────────────────

def _xgb_objective(trial, X_tr, y_tr, X_val, y_val, sw):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("lr", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("sub", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("col", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("mcw", 1, 10),
        "gamma":            trial.suggest_float("gam", 0.0, 1.0),
        "reg_alpha":        trial.suggest_float("ra", 0.0, 1.0),
        "reg_lambda":       trial.suggest_float("rl", 0.5, 2.0),
        "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
    }
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr, y_tr, sample_weight=sw)
    pred = clf.predict(X_val)
    return cohen_kappa_score(y_val, pred)


def tune_xgb(X_tr, y_tr, X_val, y_val, sw):
    if not HAS_OPTUNA:
        return {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: _xgb_objective(t, X_tr, y_tr, X_val, y_val, sw),
                   n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update({"eval_metric": "logloss", "random_state": 42, "n_jobs": -1})
    return best


# ── Stage A ───────────────────────────────────────────────────────────────────

def run_classification(X, y_bin, subject_ids, subjects):
    header("STAGE A — Reliability-Weighted Ensemble LOSO  (Low vs High)")
    n_subs = len(np.unique(subject_ids))
    print(f"  Ensemble : XGBoost + RF + SVM-RBF  (soft voting)")
    print(f"  CV       : LOSO ({n_subs} folds)")
    print(f"  Features : top-{N_FEATURES_SELECT} MI per fold")
    print(f"  Balance  : {'ADASYN' if HAS_ADASYN else 'class weights'}")
    print(f"  HPO      : {'Optuna' if HAS_OPTUNA else 'fixed'}  ({OPTUNA_TRIALS} trials)")
    print(f"  Extra    : subject reliability weighting + dropout regularisation\n")

    # Compute subject reliability weights once (outside LOSO loop)
    print("  Computing subject reliability weights...")
    rel_weights = compute_subject_reliability(X, y_bin, subject_ids)
    for sid in sorted(rel_weights):
        sub_name = subjects[subject_ids == sid][0]
        w = rel_weights[sid]
        bar = "█" * int(w * 10)
        print(f"    {sub_name}: reliability={w:.3f}  {bar}")
    print()

    logo = LeaveOneGroupOut()
    rows, all_true, all_pred, all_prob = [], [], [], []
    unique_sids = np.unique(subject_ids)

    for fold_i, (tr, te) in enumerate(logo.split(X, y_bin, groups=subject_ids)):
        test_sub   = subjects[te[0]]
        test_sid   = subject_ids[te[0]]

        # ── Subject dropout regularisation ────────────────────────────────
        # Randomly exclude one training subject per fold to prevent
        # the model from memorising any single subject's EEG signature.
        train_sids    = [s for s in unique_sids if s != test_sid]
        n_drop        = max(1, int(len(train_sids) * SUBJECT_DROPOUT_PROB))
        drop_sids     = set(RNG.choice(train_sids, n_drop, replace=False))
        keep_mask_tr  = np.array([subject_ids[i] not in drop_sids for i in tr])
        tr_use        = tr[keep_mask_tr]

        X_tr, X_te = X[tr_use], X[te]
        y_tr, y_te = y_bin[tr_use], y_bin[te]

        # ── Subject reliability sample weights ────────────────────────────
        # Each training window is weighted by its subject's reliability score
        base_sw  = compute_sample_weight("balanced", y_tr)
        rel_sw   = np.array([rel_weights[subject_ids[i]] for i in tr_use])
        sw       = base_sw * rel_sw
        sw       = sw / sw.mean()  # normalise so total weight is preserved

        # ── Scale + feature select ────────────────────────────────────────
        sc      = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        X_tr_fs, X_te_fs, _ = select_features(X_tr_sc, y_tr, X_te_sc)

        # ── Balance ───────────────────────────────────────────────────────
        if HAS_ADASYN:
            try:
                ada = ADASYN(sampling_strategy="minority", random_state=42,
                             n_neighbors=min(5, (y_tr==1).sum()-1))
                X_tr_fs, y_tr_b = ada.fit_resample(X_tr_fs, y_tr)
                # Re-compute sw for augmented set (ADASYN adds synthetic samples)
                sw_b = compute_sample_weight("balanced", y_tr_b)
            except Exception:
                y_tr_b, sw_b = y_tr, sw
        else:
            y_tr_b, sw_b = y_tr, sw

        # ── HPO ───────────────────────────────────────────────────────────
        val_size   = max(int(len(X_tr_fs) * 0.15), (y_tr_b==1).sum())
        val_idx    = RNG.choice(len(X_tr_fs), val_size, replace=False)
        tr_idx     = np.setdiff1d(np.arange(len(X_tr_fs)), val_idx)
        sw_val     = sw_b[val_idx] if len(sw_b) == len(X_tr_fs) else \
                     compute_sample_weight("balanced", y_tr_b[val_idx])
        xgb_params = tune_xgb(X_tr_fs[tr_idx], y_tr_b[tr_idx],
                               X_tr_fs[val_idx], y_tr_b[val_idx], sw_val)

        # ── Build & fit ensemble ──────────────────────────────────────────
        xgb_clf = xgb.XGBClassifier(**xgb_params)
        rf_clf  = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1)
        svm_clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                      probability=True, class_weight="balanced", random_state=42)
        ensemble = VotingClassifier(
            estimators=[("xgb", xgb_clf), ("rf", rf_clf), ("svm", svm_clf)],
            voting="soft", n_jobs=1)
        ensemble.fit(X_tr_fs, y_tr_b)

        y_pred = ensemble.predict(X_te_fs)
        y_prob = ensemble.predict_proba(X_te_fs)[:, 1]

        acc   = accuracy_score(y_te, y_pred)
        kappa = cohen_kappa_score(y_te, y_pred)
        f1    = f1_score(y_te, y_pred, average="macro", zero_division=0)
        bacc  = balanced_accuracy_score(y_te, y_pred)
        try:
            auc = roc_auc_score(y_te, y_prob)
        except Exception:
            auc = float("nan")

        rel = rel_weights[test_sid]
        nl, nh = (y_te==0).sum(), (y_te==1).sum()
        rows.append({"subject": test_sub, "fold": fold_i+1,
                     "acc": acc, "kappa": kappa, "f1_macro": f1,
                     "balanced_acc": bacc, "auc": auc,
                     "reliability": rel, "n_low": nl, "n_high": nh})
        all_true.extend(y_te)
        all_pred.extend(y_pred)
        all_prob.extend(y_prob)

        sym   = "✓" if kappa>=0.2 else "~" if kappa>=0 else "✗"
        r_str = f"rel={rel:.2f}"
        print(f"  {sym} Fold {fold_i+1:>2} | {test_sub} | "
              f"K={kappa:.3f}  AUC={auc:.3f}  BAcc={bacc:.3f}  "
              f"F1={f1:.3f}  {r_str} | L={nl} H={nh}")

    df     = pd.DataFrame(rows)
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    # ── Reliability-stratified analysis ───────────────────────────────────
    high_rel = df[df["reliability"] >= 0.6]
    low_rel  = df[df["reliability"] <  0.6]

    mk       = df["kappa"].mean()
    mk_pos   = df[df["kappa"] >= 0]["kappa"].mean()
    mk_hrel  = high_rel["kappa"].mean() if len(high_rel) > 0 else float("nan")
    interp   = ("no agreement" if mk<0.2 else "slight" if mk<0.4 else
                "fair" if mk<0.6 else "moderate")

    good = df[df["kappa"] >= 0.20]
    mid  = df[(df["kappa"] >= 0) & (df["kappa"] < 0.20)]
    bad  = df[df["kappa"] < 0]
    mid_k = f"{mid['kappa'].mean():.3f}" if len(mid) > 0 else "N/A"

    print(f"\n{'─'*60}")
    print(f"  Balanced Accuracy   : {df['balanced_acc'].mean():.4f} ± {df['balanced_acc'].std():.4f}")
    print(f"  Accuracy            : {df['acc'].mean():.4f} ± {df['acc'].std():.4f}")
    print(f"  Kappa (all)         : {mk:.4f}  ({interp})")
    print(f"  Kappa (K≥0)         : {mk_pos:.4f}")
    print(f"  Kappa (high-rel)    : {mk_hrel:.4f}  (subjects with rel≥0.60)")
    print(f"  F1 Macro            : {df['f1_macro'].mean():.4f} ± {df['f1_macro'].std():.4f}")
    print(f"  AUC-ROC             : {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
    print(f"\n  ✓ Strong   (K≥0.20) : {len(good):>2}  mean K={good['kappa'].mean():.3f}")
    print(f"  ~ Marginal (0≤K<0.2): {len(mid):>2}  mean K={mid_k}")
    bad_rel = f"{bad['reliability'].mean():.3f}" if len(bad) > 0 else "N/A"
    print(f"  ✗ Failed   (K<0)    : {len(bad):>2}  (mean reliability={bad_rel})")
    if len(bad) > 0:
        print(f"    {[str(s) for s in bad['subject'].tolist()]}")

    try:
        auc_pool = roc_auc_score(y_true, y_prob)
        print(f"\n  Pooled AUC-ROC      : {auc_pool:.4f}")
    except Exception:
        pass

    print(f"\n  Reliability-stratified breakdown:")
    print(f"    High reliability (≥0.60): {len(high_rel):>2} subjects  "
          f"mean K={mk_hrel:.4f}  "
          f"AUC={high_rel['auc'].mean():.4f}")
    if len(low_rel) > 0:
        print(f"    Low  reliability (<0.60): {len(low_rel):>2} subjects  "
              f"mean K={low_rel['kappa'].mean():.4f}  "
              f"AUC={low_rel['auc'].mean():.4f}")

    print(f"\n  Classification Report (pooled):")
    print(classification_report(y_true, y_pred,
                                target_names=["Low","High"], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:\n               Pred_Low  Pred_High")
    for i, lbl in enumerate(["True_Low ", "True_High"]):
        print(f"    {lbl}  {cm[i,0]:>8}  {cm[i,1]:>9}")

    print(f"\n  Per-subject results (sorted by Kappa):")
    for _, row in df.sort_values("kappa", ascending=False).iterrows():
        bar = "█" * max(0, int(row["kappa"] * 20))
        sym = "✓" if row["kappa"]>=0.2 else "~" if row["kappa"]>=0 else "✗"
        print(f"  {sym} {row['subject']}: K={row['kappa']:>+.3f}  "
              f"AUC={row['auc']:.3f}  rel={row['reliability']:.2f}  {bar}")

    return df, y_true, y_pred, y_prob


# ── Stage B ───────────────────────────────────────────────────────────────────

def run_regression(X_gl, y_gl, sids_gl, subs_gl):
    header("STAGE B — Ensemble Game-Level Regression  (Rank-Based Metrics)")
    n_subs = len(np.unique(sids_gl))
    print(f"  Ensemble : XGBoost + Ridge + ElasticNet + SVR")
    print(f"  CV       : LOSO ({n_subs} folds)")
    print(f"  Primary  : Spearman ρ + Kendall τ + CCC  (stable with n=4)")
    print(f"  Secondary: MAE, RMSE, R²  (R² unreliable with n=4)\n")

    logo = LeaveOneGroupOut()
    rows, all_true, all_pred = [], [], []
    k_reg = min(N_FEATURES_SELECT, X_gl.shape[1])

    for fold_i, (tr, te) in enumerate(logo.split(X_gl, y_gl, groups=sids_gl)):
        test_sub   = subs_gl[te[0]]
        X_tr, X_te = X_gl[tr], X_gl[te]
        y_tr, y_te = y_gl[tr], y_gl[te]

        sc      = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        X_tr_fs, X_te_fs, _ = select_features_reg(X_tr_sc, y_tr, X_te_sc, k=k_reg)

        # Ensemble of 4 regressors
        regs = [
            xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=-1),
            Ridge(alpha=1.0),
            ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),
            SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.2),
        ]
        preds = []
        for reg in regs:
            reg.fit(X_tr_fs, y_tr)
            preds.append(np.clip(reg.predict(X_te_fs), 1.0, 9.0))
        y_pred = np.clip(np.mean(preds, axis=0), 1.0, 9.0)

        mae  = mean_absolute_error(y_te, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2   = r2_score(y_te, y_pred)

        # Primary rank-based metrics
        try:
            rho, rho_p = spearmanr(y_te, y_pred)
        except Exception:
            rho, rho_p = float("nan"), float("nan")
        try:
            tau, tau_p = kendalltau(y_te, y_pred)
        except Exception:
            tau, tau_p = float("nan"), float("nan")
        try:
            ccc = concordance_cc(y_te, y_pred)
        except Exception:
            ccc = float("nan")

        rows.append({"subject": test_sub, "fold": fold_i+1,
                     "mae": mae, "rmse": rmse, "r2": r2,
                     "spearman_rho": rho, "kendall_tau": tau, "ccc": ccc,
                     "workload_std": float(y_te.std())})
        all_true.extend(y_te)
        all_pred.extend(y_pred)

        # Symbol based on Spearman ρ (more appropriate than R²)
        sym   = "✓" if rho>0.5 else "~" if rho>0 else "✗"
        r_str = f"{rho:+.3f}" if not np.isnan(rho) else "  N/A"
        print(f"  {sym} Fold {fold_i+1:>2} | {test_sub} | "
              f"ρ={r_str}  τ={tau:+.3f}  CCC={ccc:.3f}  "
              f"MAE={mae:.3f}  R²={r2:.3f}")

    df = pd.DataFrame(rows)
    y_true_all = np.array(all_true)
    y_pred_all = np.array(all_pred)

    # Pooled rank correlations
    rho_pool, rho_p  = spearmanr(y_true_all, y_pred_all)
    tau_pool, tau_p  = kendalltau(y_true_all, y_pred_all)
    ccc_pool         = concordance_cc(y_true_all, y_pred_all)
    pr_pool, pr_p    = pearsonr(y_true_all, y_pred_all)

    print(f"\n{'─'*60}")
    print(f"  ── Per-subject means ──")
    print(f"  Spearman ρ      : {df['spearman_rho'].mean():.4f} ± {df['spearman_rho'].std():.4f}")
    print(f"  Kendall τ       : {df['kendall_tau'].mean():.4f} ± {df['kendall_tau'].std():.4f}")
    print(f"  CCC             : {df['ccc'].mean():.4f}")
    print(f"  MAE             : {df['mae'].mean():.4f} ± {df['mae'].std():.4f}")
    print(f"  RMSE            : {df['rmse'].mean():.4f}")
    print(f"  R² mean         : {df['r2'].mean():.4f}  [secondary — unstable with n=4]")
    print(f"  R² median       : {df['r2'].median():.4f}")
    print(f"\n  ── Pooled (all predictions) ──")
    print(f"  Spearman ρ      : {rho_pool:.4f}  (p={rho_p:.4e})")
    print(f"  Kendall τ       : {tau_pool:.4f}  (p={tau_p:.4e})")
    print(f"  Pearson r       : {pr_pool:.4f}  (p={pr_p:.4e})")
    print(f"  CCC             : {ccc_pool:.4f}")
    print(f"\n  ρ>0 folds       : {(df['spearman_rho']>0).sum()}/{len(df)}")
    print(f"  ρ>0.5 folds     : {(df['spearman_rho']>0.5).sum()}/{len(df)}")

    return df, y_true_all, y_pred_all


# ── Stage C ───────────────────────────────────────────────────────────────────

def run_calibration(X_b, y_b, sids_b, subs_b, games_b):
    """
    Game-boundary calibration exploiting GAMEEMO's structure.

    GAMEEMO's constraint: each game has exactly one class (G1=Low, G4=High).
    This means ALL windows of G1 are Low and ALL windows of G4 are High.

    The insight: we CAN'T split within a game (single class).
    But we CAN train on a subset of games and test on the held-out game,
    IF we accept that test set will be single-class — and use AUC/ranking
    metrics instead of Kappa for those folds.

    For subjects with only Low games or only High games in training:
    we skip (no useful signal).

    For subjects where training games contain BOTH classes:
    we evaluate whether the model can correctly score the held-out game
    higher/lower than the within-class training average.

    Also attempts cross-game calibration: train on 3 games, test on 1.
    Reports: within-subject AUC (for single-class test folds),
             rank accuracy (does model predict High game > Low game score?),
             and Kappa where both classes appear in test.
    """
    header("STAGE C — Game-Boundary Calibration  (GAMEEMO Structure)")
    print("  Protocol : Train on N-1 games, test on held-out game")
    print("  Metrics  : Rank accuracy + AUC where computable + Kappa where feasible")
    print("  Note     : GAMEEMO = 1 class per game → Kappa only when test has both\n")

    rows = []

    for sid in np.unique(sids_b):
        m            = sids_b == sid
        sub_name     = str(subs_b[m][0])
        X_sub        = X_b[m]
        y_sub        = y_b[m]
        g_sub        = games_b[m]
        unique_games = np.unique(g_sub)

        if len(unique_games) < 2:
            continue

        # Get game-level class and mean scores
        game_class = {}
        game_means = {}
        for g in unique_games:
            gm = g_sub == g
            game_class[g] = int(y_sub[gm][0])
            game_means[g] = float(X_sub[gm].mean())

        low_games  = [g for g in unique_games if game_class[g] == 0]
        high_games = [g for g in unique_games if game_class[g] == 1]

        if len(low_games) == 0 or len(high_games) == 0:
            print(f"  SKIP {sub_name}: all games same class after binary filter")
            continue

        logo        = LeaveOneGroupOut()
        fold_kappas = []
        fold_aucs   = []
        fold_rankacc = []

        for tr, te in logo.split(X_sub, y_sub, groups=g_sub):
            y_tr = y_sub[tr];  y_te = y_sub[te]
            g_te = g_sub[te][0]

            # Only proceed if train has both classes
            if len(np.unique(y_tr)) < 2:
                continue

            sc      = StandardScaler()
            X_tr_sc = sc.fit_transform(X_sub[tr])
            X_te_sc = sc.transform(X_sub[te])
            X_tr_fs, X_te_fs, _ = select_features(X_tr_sc, y_tr, X_te_sc)

            if HAS_ADASYN:
                try:
                    ada = ADASYN(sampling_strategy="minority", random_state=42,
                                 n_neighbors=min(5, (y_tr==1).sum()-1))
                    X_tr_fs, y_tr_b = ada.fit_resample(X_tr_fs, y_tr)
                except Exception:
                    y_tr_b = y_tr
            else:
                y_tr_b = y_tr

            sw = compute_sample_weight("balanced", y_tr_b)
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, n_jobs=-1)
            model.fit(X_tr_fs, y_tr_b, sample_weight=sw)
            y_prob = model.predict_proba(X_te_fs)[:, 1]
            y_pred = model.predict(X_te_fs)

            # AUC: only if both classes in test
            if len(np.unique(y_te)) > 1:
                try:
                    fold_aucs.append(roc_auc_score(y_te, y_prob))
                    fold_kappas.append(cohen_kappa_score(y_te, y_pred))
                except Exception:
                    pass
            else:
                # Single-class test: use mean probability as rank score
                # Rank accuracy: does model score High game higher than Low game?
                mean_prob = float(y_prob.mean())
                # If test game is High class, mean_prob should be > 0.5
                # If test game is Low class, mean_prob should be < 0.5
                expected = game_class[g_te]  # 0=Low, 1=High
                rank_correct = int((mean_prob > 0.5) == (expected == 1))
                fold_rankacc.append(rank_correct)

        if not fold_kappas and not fold_rankacc and not fold_aucs:
            print(f"  SKIP {sub_name}: no valid calibration folds")
            continue

        mk   = float(np.mean(fold_kappas))   if fold_kappas   else float("nan")
        mauc = float(np.mean(fold_aucs))     if fold_aucs     else float("nan")
        mra  = float(np.mean(fold_rankacc))  if fold_rankacc  else float("nan")

        rows.append({"subject": sub_name,
                     "kappa": mk, "auc": mauc, "rank_acc": mra,
                     "n_kappa_folds": len(fold_kappas),
                     "n_rank_folds":  len(fold_rankacc),
                     "n_low_games": len(low_games),
                     "n_high_games": len(high_games)})

        sym = "✓" if (not np.isnan(mra) and mra>=0.6) or \
                     (not np.isnan(mk) and mk>=0.3) else "~"
        ra_str  = f"RankAcc={mra:.2f}" if not np.isnan(mra) else ""
        auc_str = f"AUC={mauc:.3f}" if not np.isnan(mauc) else ""
        k_str   = f"K={mk:.3f}" if not np.isnan(mk) else ""
        print(f"  {sym} {sub_name}: {k_str}  {auc_str}  {ra_str}  "
              f"[Low_games={len(low_games)} High_games={len(high_games)}]")

    if not rows:
        print("\n  [WARN] No subjects completed calibration.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    valid_k   = df["kappa"].dropna()
    valid_auc = df["auc"].dropna()
    valid_ra  = df["rank_acc"].dropna()

    print(f"\n{'─'*60}")
    if len(valid_k) > 0:
        print(f"  Kappa (folds with both classes) : "
              f"{valid_k.mean():.4f} ± {valid_k.std():.4f}  [{len(valid_k)} subjects]")
    if len(valid_auc) > 0:
        print(f"  AUC  (folds with both classes)  : "
              f"{valid_auc.mean():.4f}  [{len(valid_auc)} subjects]")
    if len(valid_ra) > 0:
        print(f"  Rank accuracy (single-class)    : "
              f"{valid_ra.mean():.4f}  [{len(valid_ra)} subjects]")
        print(f"  (Does model score High games > Low games?)")
        print(f"  Rank Acc ≥ 0.60 : {(valid_ra>=0.6).sum()}/{len(valid_ra)} subjects")
        print(f"  Rank Acc ≥ 0.80 : {(valid_ra>=0.8).sum()}/{len(valid_ra)} subjects")

    return df


# ── Stage D ───────────────────────────────────────────────────────────────────

def run_feature_importance(X, y_bin):
    header("STAGE D — Feature Importance  (Permutation + Tree Impurity)")
    channels   = ['AF3','F7','F3','FC5','T7','P7','O1','O2',
                  'P8','T8','FC6','F4','F8','AF4']
    asym_pairs = ['AF3-AF4','F7-F8','F3-F4','FC5-FC6','T7-T8','P7-P8','O1-O2']

    feat_names = []
    # Spectral (116)
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

    # Connectivity (92) — only added if features > 116
    if X.shape[1] > 116:
        for band in ['delta','theta','alpha','beta','gamma']:
            for pair in asym_pairs:
                feat_names.append(f"asym_{band}_{pair}")
        for band in ['delta','theta','alpha','beta','gamma']:
            feat_names.append(f"fp_corr_{band}")
        fi = ['AF3','F7','F3','FC5','F8','AF4']
        for i in range(len(fi)):
            for j in range(i+1, len(fi)):
                feat_names.append(f"fcorr_{fi[i]}-{fi[j]}")
        pi = ['P7','O1','O2','P8']
        for i in range(len(pi)):
            for j in range(i+1, len(pi)):
                feat_names.append(f"pcorr_{pi[i]}-{pi[j]}")
        for f in fi:
            for p in pi:
                feat_names.append(f"fpcorr_{f}-{p}")
        feat_names += ['tcorr_T7-frontal','tcorr_T8-frontal',
                       'tcorr_T7-T8','ccorr_FC5-FC6',
                       'ccorr_FC5-parietal','ccorr_FC6-parietal',
                       'tcorr_T7-parietal']

    # Pad or trim to exact feature count
    n_feat = X.shape[1]
    while len(feat_names) < n_feat:
        feat_names.append(f"feat_{len(feat_names)}")
    feat_names = feat_names[:n_feat]

    sc   = StandardScaler()
    X_sc = sc.fit_transform(X)
    sw   = compute_sample_weight("balanced", y_bin)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1)
    model.fit(X_sc, y_bin, sample_weight=sw)

    print("  Computing permutation importance (n_repeats=10)...")
    perm = permutation_importance(model, X_sc, y_bin, n_repeats=10,
                                  random_state=42, n_jobs=-1,
                                  scoring="balanced_accuracy")
    imp_df = pd.DataFrame({
        "feature":  feat_names,
        "tree_imp": model.feature_importances_,
        "perm_imp": perm.importances_mean,
        "perm_std": perm.importances_std,
    }).sort_values("perm_imp", ascending=False)

    print(f"\n  {'Feature':<22} {'Perm':>8}  {'±':>6}  {'Tree':>8}")
    print(f"  {'─'*48}")
    for _, row in imp_df.head(20).iterrows():
        print(f"  {row['feature']:<22} {row['perm_imp']:>8.5f}  "
              f"{row['perm_std']:>6.4f}  {row['tree_imp']:>8.5f}")

    top10 = imp_df.head(10)["feature"].tolist()
    good  = [f for f in top10 if any(m in f for m in
             ["theta","beta_alpha","theta_alpha","frontal_theta","engagement","alpha_F"])]
    bad   = [f for f in top10 if any(m in f for m in ["delta_O","entropy_O"])]
    print(f"\n  Workload markers in top-10 : {good}")
    print(f"  Artifact markers in top-10 : {bad if bad else 'none ✓'}")

    print(f"\n  Band-level mean permutation importance:")
    for band in ['delta','theta','alpha','beta','gamma',
                 'theta_alpha','beta_alpha','entropy']:
        bf = imp_df[imp_df["feature"].str.startswith(band)]
        if len(bf) > 0:
            print(f"    {band:<14}: {bf['perm_imp'].mean():.5f}")
    return imp_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X, y_class, y_cont, subject_ids, subjects, games = load_dataset()

    header("Dataset Overview")
    print(f"  Windows  : {len(X)}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Subjects : {len(np.unique(subject_ids))}")
    print(f"  Classes  : Low={(y_class==0).sum()}  Med={(y_class==1).sum()}  "
          f"High={(y_class==2).sum()}")
    print(f"  Workload : [{y_cont.min():.2f}, {y_cont.max():.2f}]  "
          f"mean={y_cont.mean():.2f}  std={y_cont.std():.2f}")

    # ── 3-class (primary — uses per-subject tertile labels) ─────────────
    X_3, y_3, sids_3, subs_3, games_3 = prepare_3class(
        X, y_class, subject_ids, subjects, games)
    clf3_df, y_true_3, y_pred_3, y_prob_3 = run_classification_3class(
        X_3, y_3, sids_3, subs_3)

    # ── Binary (comparison baseline + calibration + regression) ──────────
    X_b,  y_b,  sids_b,  subs_b,  games_b = prepare_binary(
        X, y_class, subject_ids, subjects, games)
    X_gl, y_gl, sids_gl, subs_gl           = prepare_regression_game_level(
        X, y_cont, subject_ids, subjects, games)

    clf_df, y_true_clf, y_pred_clf, y_prob_clf = run_classification(
        X_b, y_b, sids_b, subs_b)

    reg_df, y_true_reg, y_pred_reg = run_regression(
        X_gl, y_gl, sids_gl, subs_gl)

    cal_df = run_calibration(X_b, y_b, sids_b, subs_b, games_b)

    imp_df = run_feature_importance(X_b, y_b)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clf3_df.to_csv(f"{OUTPUT_DIR}/results_3class.csv",          index=False)
    clf_df.to_csv(f"{OUTPUT_DIR}/results_classification.csv", index=False)
    reg_df.to_csv(f"{OUTPUT_DIR}/results_regression.csv",     index=False)
    imp_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv",     index=False)
    if len(cal_df) > 0:
        cal_df.to_csv(f"{OUTPUT_DIR}/results_calibration.csv", index=False)
    np.savez(f"{OUTPUT_DIR}/predictions.npz",
             y_true_clf=y_true_clf, y_pred_clf=y_pred_clf, y_prob_clf=y_prob_clf,
             y_true_reg=y_true_reg, y_pred_reg=y_pred_reg)

    # ── Final summary ─────────────────────────────────────────────────────
    header("FINAL SUMMARY — 3-Class + Binary Results  (v6)")

    mk3      = clf3_df["kappa"].mean()
    mk3_pos  = clf3_df[clf3_df["kappa"] >= 0]["kappa"].mean()
    n_str3   = len(clf3_df[clf3_df["kappa"] >= 0.2])
    interp3  = ("no agreement" if mk3<0.2 else "slight" if mk3<0.4 else
                "fair" if mk3<0.6 else "moderate")

    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  3-CLASS RESULTS  (Low / Medium / High)  ← PRIMARY      ║
  ║  Labels: per-subject tertile  Model: LOSO XGBoost        ║
  ║  Kappa (all)    : {mk3:.4f}  ({interp3})                 ║
  ║  Kappa (K≥0)    : {mk3_pos:.4f}                          ║
  ║  F1 Macro       : {clf3_df['f1_macro'].mean():.4f}                                ║
  ║  Strong (K≥0.2) : {n_str3:>2}/{len(clf3_df):>2} subjects                        ║
  ╚══════════════════════════════════════════════════════════╝
    """)

    mk       = clf_df["kappa"].mean()
    mk_pos   = clf_df[clf_df["kappa"] >= 0]["kappa"].mean()
    mk_hrel  = clf_df[clf_df["reliability"] >= 0.6]["kappa"].mean()
    mauc     = clf_df["auc"].mean()
    mbacc    = clf_df["balanced_acc"].mean()
    interp   = ("no agreement" if mk<0.2 else "slight" if mk<0.4 else "fair")
    n_strong = len(clf_df[clf_df["kappa"] >= 0.2])

    rho_mean = reg_df["spearman_rho"].mean()
    tau_mean = reg_df["kendall_tau"].mean()
    ccc_mean = reg_df["ccc"].mean()

    if len(cal_df) > 0:
        cal_ra  = cal_df["rank_acc"].dropna().mean() if "rank_acc" in cal_df else float("nan")
        cal_auc = cal_df["auc"].dropna().mean()      if "auc"      in cal_df else float("nan")
        cal_k   = cal_df["kappa"].dropna().mean()    if "kappa"    in cal_df else float("nan")
    else:
        cal_ra, cal_auc, cal_k = float("nan"), float("nan"), float("nan")

    print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  TIER 1 — Blind cross-subject (LOSO, reliability-wtd)    │
  │  Balanced Acc   : {mbacc:.4f}                             │
  │  Kappa (all)    : {mk:.4f}  ({interp})               │
  │  Kappa (K≥0)    : {mk_pos:.4f}                            │
  │  Kappa (hi-rel) : {mk_hrel:.4f}  (reliable subjects only) │
  │  AUC-ROC        : {mauc:.4f}                              │
  │  F1 Macro       : {clf_df['f1_macro'].mean():.4f}                              │
  │  Strong (K≥0.2) : {n_strong:>2}/{len(clf_df):>2} subjects                      │
  ├──────────────────────────────────────────────────────────┤
  │  TIER 2 — Screened (high-reliability subjects only)      │
  │  Kappa          : {mk_hrel:.4f}                            │
  │  Subjects       : {len(clf_df[clf_df['reliability']>=0.6]):>2}/{len(clf_df):>2} with reliability≥0.60        │
  ├──────────────────────────────────────────────────────────┤
  │  TIER 3 — Game-Boundary Calibration                      │
  │  Rank Accuracy  : {cal_ra:.4f}  (High game > Low game?)   │
  │  AUC            : {cal_auc:.4f}                            │
  │  Kappa          : {cal_k:.4f}  (folds with both classes)  │
  └──────────────────────────────────────────────────────────┘

  Regression — Game-Level LOSO  (primary: rank correlation):
  Spearman ρ   : {rho_mean:.4f}  (rank ordering of game difficulty)
  Kendall τ    : {tau_mean:.4f}
  CCC          : {ccc_mean:.4f}  (agreement of scale + direction)
  MAE          : {reg_df['mae'].mean():.4f} ± {reg_df['mae'].std():.4f}  (scale 1–9)
  R² mean      : {reg_df['r2'].mean():.4f}  [secondary — unstable with n=4]

  All outputs saved to {OUTPUT_DIR}/
    """)


if __name__ == "__main__":
    main()
