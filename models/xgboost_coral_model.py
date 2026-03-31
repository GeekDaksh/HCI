import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from dataset_loader import load_dataset, loso_splits

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS     = ["Low", "Medium", "High"]
COLORS     = ["#4C72B0", "#DD8452", "#55A868"]
SEQ_COLORS = ["#534AB7", "#1D9E75", "#D85A30"]

# TCA hyperparameters
TCA_DIM    = 20      # subspace dimensionality (smaller = faster)
TCA_KERNEL = "rbf"   # kernel type
TCA_GAMMA  = 1.0     # RBF kernel bandwidth (1/sigma^2)
TCA_MU     = 1.0     # regularisation trade-off

# TCA subsampling — kernel matrix is O(n^2), must limit training samples
TCA_SUBSAMPLE = 1000   # kernel matrix is (n_sub x n_sub) — keep small for speed

# CORAL hyperparameters
# reg=0.5 and alpha=0.5 produced 73.63% mean accuracy (vs 77.65% plain XGBoost)
CORAL_REG   = 0.5    # covariance regularisation — prevents over-alignment
CORAL_ALPHA = 0.5    # blend factor: 0.5 * aligned + 0.5 * original

# XGBoost hyperparameters
XGB_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 6,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "mlogloss",
    random_state     = 42,
    n_jobs           = -1,
)

CHANNELS = ["AF3","AF4","F3","F4","F7","F8","FC5","FC6",
            "O1","O2","P7","P8","T7","T8"]
BANDS    = ["delta","theta","alpha","beta","gamma"]
FEAT_NAMES = (
    [f"{ch}_{b}" for ch in CHANNELS for b in BANDS] +
    ["eng_TLI_ratio", "eng_beta_alpha", "eng_theta_beta",
     "eng_FAA", "eng_engagement_idx", "eng_parietal_alpha"] +
    ["FTI_z"]
)


# ─────────────────────────────────────────────
#  CORAL IMPLEMENTATION
#  Sun & Saenko 2016 — ECCV
#  "Return of Frustratingly Easy Domain Adaptation"
#  https://arxiv.org/abs/1511.05547
# ─────────────────────────────────────────────

def coral_transform(X_src, X_tgt, reg=0.5, alpha=0.5):
    """
    CORAL — Correlation Alignment (Sun & Saenko 2016) with partial blending.

    Aligns source covariance to match target, then blends the aligned
    features with the original features using alpha as the blend factor.
    This prevents over-alignment — a common failure mode when upstream
    preprocessing (EA) has already removed most distribution shift.

    Parameters
    ----------
    X_src  : (n_src, d) — source (training) features
    X_tgt  : (n_tgt, d) — target (test) features — labels NOT needed
    reg    : float — covariance regularisation (higher = more stable)
    alpha  : float in [0,1] — blend factor
             0.0 = original features only (no CORAL)
             1.0 = fully aligned features (full CORAL)
             0.5 = equal blend (recommended with EA upstream)

    Returns
    -------
    X_blended : (n_src, d) — blended features, same shape as input

    How it works
    ------------
    1. Compute regularised covariances C_s and C_t
    2. Compute alignment matrix A = C_s^{-1/2} @ C_t^{1/2}
    3. X_aligned = X_src @ A  (source now has target covariance)
    4. X_blended = alpha * X_aligned + (1-alpha) * X_src
       Partial blending retains original discriminative structure
       while gently pushing toward target covariance.
    """
    from scipy.linalg import sqrtm

    d = X_src.shape[1]

    # Regularised covariances — higher reg prevents singular matrices
    C_s = np.cov(X_src.T) + reg * np.eye(d)
    C_t = np.cov(X_tgt.T) + reg * np.eye(d)

    # C_s^{-1/2} — whitens source covariance to identity
    try:
        C_s_sqrt     = sqrtm(C_s).real
        C_s_inv_sqrt = np.linalg.inv(C_s_sqrt)
        # Sanity check — inf/nan means matrix was near-singular
        if not np.isfinite(C_s_inv_sqrt).all():
            raise ValueError("Non-finite values in C_s_inv_sqrt")
    except Exception:
        # Fallback: diagonal whitening (always stable)
        diag = np.sqrt(np.diag(C_s))
        C_s_inv_sqrt = np.diag(1.0 / (diag + 1e-6))

    # C_t^{1/2} — recolours to match target covariance
    try:
        C_t_sqrt = sqrtm(C_t).real
        if not np.isfinite(C_t_sqrt).all():
            raise ValueError("Non-finite values in C_t_sqrt")
    except Exception:
        C_t_sqrt = np.diag(np.sqrt(np.diag(C_t)))

    # Alignment matrix: A maps source covariance to target covariance
    A = C_s_inv_sqrt @ C_t_sqrt   # (d, d)

    # Apply alignment
    X_aligned = (X_src @ A).astype(np.float32)

    # Clip extreme values caused by imperfect alignment
    q99 = np.percentile(np.abs(X_aligned), 99)
    q99_orig = np.percentile(np.abs(X_src), 99)
    clip_val = max(q99_orig * 3, q99)
    X_aligned = np.clip(X_aligned, -clip_val, clip_val)

    # Partial blend — alpha=0.5 keeps half original signal intact
    X_blended = alpha * X_aligned + (1.0 - alpha) * X_src

    return X_blended.astype(np.float32)


# ─────────────────────────────────────────────
#  LOSO TRAINING LOOP
# ─────────────────────────────────────────────



def run_loso(X, y_class, subject_ids, subjects):
    n_subjects = len(np.unique(subject_ids))
    fold_accs, fold_f1s, fold_kappas = [], [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    print(f"XGBoost + CORAL — LOSO cross-validation ({n_subjects} folds)")
    print(f"  CORAL reg={CORAL_REG}  alpha={CORAL_ALPHA}  (partial blending)")
    print(f"{'='*65}")

    for fold, test_subj, X_train, y_train, X_test, y_test in loso_splits(
        X, y_class, subject_ids
    ):
        subj_str = np.unique(subjects[subject_ids == test_subj])[0]

        # ── Standardize first (TCA works better on normalised features) ──
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # ── CORAL domain adaptation ──
        # Aligns source (train) covariance to match target (test) covariance.
        # Uses only UNLABELLED test features — no test labels needed.
        # O(d^2) complexity — runs in milliseconds per fold.
        try:
            Z_train = coral_transform(X_train, X_test, reg=CORAL_REG, alpha=CORAL_ALPHA)
        except Exception as e:
            print(f"  [WARN] CORAL failed fold {fold+1}: {e} — using raw features")
            Z_train = X_train
        Z_test = X_test   # test features unchanged

        # ── Train XGBoost on CORAL-aligned features ──
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(Z_train, y_train)

        y_pred = model.predict(Z_test)
        y_prob = model.predict_proba(Z_test)

        acc   = accuracy_score(y_test, y_pred)
        f1    = f1_score(y_test, y_pred, average="macro")
        kappa = cohen_kappa_score(y_test, y_pred)

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_kappas.append(kappa)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        print(f"  Fold {fold+1:>2} | {subj_str} | n={len(y_test):>4} | "
              f"acc={acc:.4f} | f1={f1:.4f} | kappa={kappa:.4f}")

    return (
        np.array(fold_accs), np.array(fold_f1s), np.array(fold_kappas),
        np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)
    )


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────

def plot_all(fold_accs, fold_f1s, fold_kappas,
             y_true, y_pred, y_prob, subjects, subject_ids):

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle("XGBoost + CORAL — EEG Cognitive Workload (Domain Adaptation)",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    unique_subjs = [np.unique(subjects[subject_ids == s])[0]
                    for s in np.unique(subject_ids)]

    # ── 1. Per-fold accuracy ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(fold_accs))
    bars = ax1.bar(x, fold_accs * 100, color=SEQ_COLORS[0],
                   alpha=0.8, edgecolor="white", linewidth=0.5)
    ax1.axhline(fold_accs.mean() * 100, color="red", linestyle="--",
                linewidth=1.5, label=f"Mean {fold_accs.mean()*100:.2f}%")
    ax1.set_xticks(x)
    ax1.set_xticklabels(unique_subjs, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Per-subject accuracy — XGBoost + CORAL (LOSO)")
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars, fold_accs):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{acc*100:.1f}", ha="center", va="bottom",
                 fontsize=6, rotation=90)

    # ── 2. Summary box ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    metrics = [
        ("Mean Accuracy",  f"{fold_accs.mean()*100:.2f}%"),
        ("Std",            f"±{fold_accs.std()*100:.2f}%"),
        ("Best Fold",      f"{fold_accs.max()*100:.2f}%"),
        ("Worst Fold",     f"{fold_accs.min()*100:.2f}%"),
        ("Mean F1 (macro)",f"{fold_f1s.mean():.4f}"),
        ("Mean Kappa",     f"{fold_kappas.mean():.4f}"),
        ("CORAL reg",      f"{CORAL_REG}"),
        ("CORAL alpha",    f"{CORAL_ALPHA}"),
    ]
    y_pos = 0.95
    ax2.text(0.5, 1.0, "Summary — XGB + CORAL",
             ha="center", va="top", fontsize=11, fontweight="bold",
             transform=ax2.transAxes)
    for label, val in metrics:
        ax2.text(0.05, y_pos, label, ha="left", fontsize=9,
                 color="gray", transform=ax2.transAxes)
        ax2.text(0.95, y_pos, val, ha="right", fontsize=9,
                 fontweight="bold", transform=ax2.transAxes)
        y_pos -= 0.11
    rect = plt.Rectangle((0, 0), 1, 1, fill=False,
                          edgecolor="#534AB7", linewidth=2,
                          transform=ax2.transAxes)
    ax2.add_patch(rect)

    # ── 3. Confusion matrix ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    im = ax3.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax3.set_xticks([0,1,2]); ax3.set_yticks([0,1,2])
    ax3.set_xticklabels(LABELS); ax3.set_yticklabels(LABELS)
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("True")
    ax3.set_title("Confusion matrix (%)")
    plt.colorbar(im, ax=ax3, fraction=0.046)
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, f"{cm_pct[i,j]:.1f}%\n({cm[i,j]})",
                     ha="center", va="center", fontsize=8,
                     color="white" if cm_pct[i,j] > 55 else "black")

    # ── 4. Per-class metrics ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred,
                                   target_names=LABELS, output_dict=True)
    metric_names = ["precision", "recall", "f1-score"]
    x_cls = np.arange(len(LABELS))
    width = 0.25
    for i, m in enumerate(metric_names):
        vals = [report[l][m] for l in LABELS]
        ax4.bar(x_cls + i * width, vals, width,
                label=m.replace("-score", ""), color=COLORS[i],
                alpha=0.85, edgecolor="white")
    ax4.set_xticks(x_cls + width)
    ax4.set_xticklabels(LABELS)
    ax4.set_ylabel("Score")
    ax4.set_title("Per-class precision, recall, F1")
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    # ── 5. ROC curves ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    y_prob_arr = np.array(y_prob)
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_arr[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_prob_arr[:, i])
            ax5.plot(fpr, tpr, color=color, lw=1.5,
                     label=f"{label} (AUC={auc:.3f})")
        except Exception:
            pass
    ax5.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax5.set_xlabel("False positive rate")
    ax5.set_ylabel("True positive rate")
    ax5.set_title("ROC curves (one-vs-rest)")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # ── 6. Accuracy comparison: XGBoost vs XGBoost+TCA ──────────
    ax6 = fig.add_subplot(gs[2, :])
    xgb_results_path = os.path.join(RESULTS_DIR, "xgboost_results.npz")
    if os.path.exists(xgb_results_path):
        xgb_data = np.load(xgb_results_path, allow_pickle=True)
        xgb_accs = xgb_data["fold_accs"]
        x = np.arange(len(fold_accs))
        w = 0.35
        ax6.bar(x - w/2, xgb_accs * 100, w, label="XGBoost",
                color="#888780", alpha=0.8, edgecolor="white")
        ax6.bar(x + w/2, fold_accs * 100, w, label="XGBoost + CORAL",
                color=SEQ_COLORS[0], alpha=0.8, edgecolor="white")
        ax6.axhline(xgb_accs.mean() * 100, color="#888780",
                    linestyle="--", linewidth=1, alpha=0.7)
        ax6.axhline(fold_accs.mean() * 100, color=SEQ_COLORS[0],
                    linestyle="--", linewidth=1.5)
        ax6.set_xticks(x)
        ax6.set_xticklabels(unique_subjs, rotation=45, ha="right", fontsize=8)
        ax6.set_ylabel("Accuracy (%)")
        ax6.set_title(f"Per-fold comparison: XGBoost ({xgb_accs.mean()*100:.2f}%) "
                      f"vs XGBoost+CORAL ({fold_accs.mean()*100:.2f}%)")
        ax6.legend(fontsize=9)
        ax6.grid(axis="y", alpha=0.3)
        ax6.set_ylim(0, 100)
    else:
        ax6.text(0.5, 0.5, "Run xgboost_model.py first for comparison",
                 ha="center", va="center", transform=ax6.transAxes,
                 fontsize=11, color="gray")
        ax6.axis("off")

    # ── 7. Accuracy distribution ─────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    vp = ax7.violinplot([fold_accs * 100], positions=[1],
                        showmeans=True, showmedians=True)
    for body in vp["bodies"]:
        body.set_facecolor(SEQ_COLORS[0])
        body.set_alpha(0.6)
    ax7.scatter([1] * len(fold_accs), fold_accs * 100,
                color="white", edgecolor=SEQ_COLORS[0], s=30, zorder=3)
    ax7.set_xticks([1])
    ax7.set_xticklabels(["Accuracy"])
    ax7.set_ylabel("Accuracy (%)")
    ax7.set_title("Fold accuracy distribution")
    ax7.grid(axis="y", alpha=0.3)

    # ── 8. F1 per fold ───────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.plot(range(1, len(fold_f1s)+1), fold_f1s,
             "o-", color="#1D9E75", linewidth=1.5, markersize=4)
    ax8.axhline(fold_f1s.mean(), color="red", linestyle="--",
                linewidth=1.5, label=f"Mean {fold_f1s.mean():.4f}")
    ax8.set_xlabel("Fold")
    ax8.set_ylabel("Macro F1")
    ax8.set_title("Macro F1 per fold")
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3)

    # ── 9. Kappa per fold ─────────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.plot(range(1, len(fold_kappas)+1), fold_kappas,
             "s-", color="#D85A30", linewidth=1.5, markersize=4)
    ax9.axhline(fold_kappas.mean(), color="red", linestyle="--",
                linewidth=1.5, label=f"Mean {fold_kappas.mean():.4f}")
    ax9.set_xlabel("Fold")
    ax9.set_ylabel("Cohen Kappa")
    ax9.set_title("Cohen's Kappa per fold")
    ax9.legend(fontsize=8)
    ax9.grid(alpha=0.3)

    out_path = os.path.join(RESULTS_DIR, "xgboost_coral_charts.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Charts saved → {out_path}")


# ─────────────────────────────────────────────
#  SAVE + SUMMARY
# ─────────────────────────────────────────────

def save_results(fold_accs, fold_f1s, fold_kappas, y_true, y_pred):
    out = os.path.join(RESULTS_DIR, "xgboost_coral_results.npz")
    np.savez(out,
             fold_accs   = fold_accs,
             fold_f1s    = fold_f1s,
             fold_kappas = fold_kappas,
             y_true      = y_true,
             y_pred      = y_pred,
             model_name  = np.array("XGBoost+CORAL"))
    print(f"  Results saved → {out}")


def print_summary(fold_accs, fold_f1s, fold_kappas, y_true, y_pred):
    print(f"\n{'='*65}")
    print(f"  XGBoost + CORAL — Final Results")
    print(f"{'='*65}")
    print(f"  Mean Accuracy  : {fold_accs.mean()*100:.2f}%  ±{fold_accs.std()*100:.2f}%")
    print(f"  Best Fold      : {fold_accs.max()*100:.2f}%")
    print(f"  Worst Fold     : {fold_accs.min()*100:.2f}%")
    print(f"  Mean F1 (macro): {fold_f1s.mean():.4f}")
    print(f"  Mean Kappa     : {fold_kappas.mean():.4f}")

    # Load XGBoost baseline for comparison
    xgb_path = os.path.join(RESULTS_DIR, "xgboost_results.npz")
    if os.path.exists(xgb_path):
        xgb = np.load(xgb_path, allow_pickle=True)
        delta = fold_accs.mean() - xgb["fold_accs"].mean()
        print(f"\n  vs XGBoost baseline:")
        print(f"  Accuracy gain  : {delta*100:+.2f}%")
        print(f"  XGBoost mean   : {xgb['fold_accs'].mean()*100:.2f}%")
        print(f"  CORAL mean     : {fold_accs.mean()*100:.2f}%")

    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred,
                                target_names=LABELS, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix:")
    print(f"  {'':>10}  " + "  ".join(f"{l:>10}" for l in LABELS))
    for i, row in enumerate(cm):
        print(f"  {LABELS[i]:>10}  " + "  ".join(f"{v:>10}" for v in row))


from sklearn.metrics import classification_report


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("Loading dataset...\n")
    X, y_class, y_cont, subject_ids, subjects, games = load_dataset()
    print(f"  X shape : {X.shape}")
    print(f"  Classes : {dict(zip(*np.unique(y_class, return_counts=True)))}\n")

    fold_accs, fold_f1s, fold_kappas, \
    y_true, y_pred, y_prob = run_loso(
        X, y_class, subject_ids, subjects
    )

    print_summary(fold_accs, fold_f1s, fold_kappas, y_true, y_pred)
    save_results(fold_accs, fold_f1s, fold_kappas, y_true, y_pred)
    plot_all(fold_accs, fold_f1s, fold_kappas,
             y_true, y_pred, y_prob,
             subjects, subject_ids)

    print(f"\n  Done. Next: run models/bilstm_model.py")


if __name__ == "__main__":
    main()
