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
#  TCA IMPLEMENTATION
#  Pan et al. 2011 — IEEE Trans. Knowledge and Data Engineering
#  "Domain Adaptation via Transfer Component Analysis"
#  DOI: 10.1109/TKDE.2010.199
# ─────────────────────────────────────────────

def rbf_kernel(X1, X2, gamma=1.0):
    """
    Radial Basis Function (RBF) kernel matrix.
    K(x, y) = exp(-gamma * ||x - y||^2)

    X1 : (n1, d)
    X2 : (n2, d)
    Returns K : (n1, n2)
    """
    # Efficient computation using ||x-y||^2 = ||x||^2 - 2x·y + ||y||^2
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)   # (n1, 1)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)   # (n2, 1)
    dist_sq = sq1 + sq2.T - 2.0 * X1 @ X2.T        # (n1, n2)
    dist_sq = np.clip(dist_sq, 0, None)             # numerical safety
    return np.exp(-gamma * dist_sq)


def tca_transform(X_src, X_tgt, n_components=30, gamma=1.0, mu=1.0):
    """
    Transfer Component Analysis — Pan et al. 2011.

    Finds a projection W such that the projected source and target
    distributions are as similar as possible (minimises MMD in the
    kernel space), while preserving sufficient variance.

    Parameters
    ----------
    X_src       : (n_src, d) — source (training) features
    X_tgt       : (n_tgt, d) — target (test) features, labels NOT needed
    n_components: dimensionality of shared subspace
    gamma       : RBF bandwidth
    mu          : regularisation weight

    Returns
    -------
    Z_src : (n_src, n_components) — projected source features
    Z_tgt : (n_tgt, n_components) — projected target features

    How it works
    ------------
    1. Stack X = [X_src; X_tgt] and compute the kernel matrix K (n×n)
    2. Build MMD matrix L: L_ij = 1/n_s^2 if both i,j are source;
                                   1/n_t^2 if both i,j are target;
                                  -1/(n_s*n_t) otherwise
    3. Solve the generalised eigenvalue problem:
       (K L K + mu I) W = K H K W lambda
       where H = I - (1/n) 11^T is the centering matrix
    4. Top n_components eigenvectors are the transfer components
    5. Project: Z = K W  (both source and target)
    """
    n_src = X_src.shape[0]
    n_tgt = X_tgt.shape[0]
    n     = n_src + n_tgt

    X_all = np.vstack([X_src, X_tgt])   # (n, d)

    # ── Step 1: Kernel matrix ──
    K = rbf_kernel(X_all, X_all, gamma=gamma)   # (n, n)
    K = K + 1e-5 * np.eye(n)                    # regularise for stability

    # ── Step 2: MMD matrix L ──
    L = np.zeros((n, n))
    L[:n_src, :n_src] =  1.0 / (n_src ** 2)
    L[n_src:, n_src:] =  1.0 / (n_tgt ** 2)
    L[:n_src, n_src:] = -1.0 / (n_src * n_tgt)
    L[n_src:, :n_src] = -1.0 / (n_src * n_tgt)

    # ── Step 3: Centering matrix H ──
    H = np.eye(n) - (1.0 / n) * np.ones((n, n))

    # ── Step 4: Generalised eigenvalue problem ──
    # (K L K + mu I) W = K H K W lambda
    # Rearranged to standard form: A W = lambda B W
    A = K @ L @ K + mu * np.eye(n)
    B = K @ H @ K

    try:
        # scipy generalised eigensolver
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(B, A,
                                         subset_by_index=[n - n_components, n - 1])
        W = eigenvectors[:, ::-1]   # descending order
    except Exception:
        # Fallback: standard eigensolver on A^{-1} B
        A_inv = np.linalg.pinv(A)
        M     = A_inv @ B
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        idx   = np.argsort(eigenvalues)[::-1]
        W     = eigenvectors[:, idx[:n_components]]

    W = W[:, :n_components]   # (n, n_components)

    # ── Step 5: Project ──
    Z_all = K @ W              # (n, n_components)
    Z_src = Z_all[:n_src]
    Z_tgt = Z_all[n_src:]

    return Z_src, Z_tgt


# ─────────────────────────────────────────────
#  LOSO TRAINING LOOP
# ─────────────────────────────────────────────

def run_loso(X, y_class, subject_ids, subjects):
    n_subjects = len(np.unique(subject_ids))
    fold_accs, fold_f1s, fold_kappas = [], [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    tca_dims_used = []

    print(f"XGBoost + TCA — LOSO cross-validation ({n_subjects} folds)")
    print(f"  TCA dim={TCA_DIM}  kernel={TCA_KERNEL}  gamma={TCA_GAMMA}  mu={TCA_MU}")
    print(f"{'='*65}")

    for fold, test_subj, X_train, y_train, X_test, y_test in loso_splits(
        X, y_class, subject_ids
    ):
        subj_str = np.unique(subjects[subject_ids == test_subj])[0]

        # ── Standardize first (TCA works better on normalised features) ──
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # ── TCA domain adaptation ──
        # CRITICAL: train and test MUST be projected together in the same
        # kernel call — TCA builds one kernel matrix over [train; test] and
        # the eigenvectors span that joint space. Projecting separately gives
        # incompatible spaces and random results.
        #
        # To keep it tractable we subsample BOTH train and test proportionally,
        # project together, then train XGBoost on projected train subsample
        # and predict on projected test subsample.
        #
        # Test subsample is stratified so class distribution is preserved.
        n_tr  = len(X_train)
        n_te  = len(X_test)
        rng   = np.random.default_rng(fold + 42)

        # Subsample train
        tr_size = min(n_tr, TCA_SUBSAMPLE)
        tr_idx  = rng.choice(n_tr, tr_size, replace=False)
        X_tr_s  = X_train[tr_idx]
        y_tr_s  = y_train[tr_idx]

        # Keep all test (it's already ~1580 which is manageable)
        X_te_s  = X_test
        y_te_s  = y_test

        try:
            Z_tr, Z_te = tca_transform(
                X_tr_s, X_te_s,
                n_components = TCA_DIM,
                gamma        = TCA_GAMMA,
                mu           = TCA_MU,
            )
        except Exception as e:
            print(f"  [WARN] TCA failed fold {fold+1}: {e} — using raw features")
            Z_tr, Z_te = X_tr_s, X_te_s

        tca_dims_used.append(Z_tr.shape[1])

        # ── Train XGBoost on TCA-projected features ──
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(Z_tr, y_tr_s)

        # Override y_test for this fold with the (possibly identical) subsample
        y_test = y_te_s

        y_pred = model.predict(Z_te)
        y_prob = model.predict_proba(Z_te)

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
    fig.suptitle("XGBoost + TCA — EEG Cognitive Workload (Domain Adaptation)",
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
    ax1.set_title("Per-subject accuracy — XGBoost + TCA (LOSO)")
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
        ("TCA dim",        f"{TCA_DIM}"),
    ]
    y_pos = 0.95
    ax2.text(0.5, 1.0, "Summary — XGB + TCA",
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
        ax6.bar(x + w/2, fold_accs * 100, w, label="XGBoost + TCA",
                color=SEQ_COLORS[0], alpha=0.8, edgecolor="white")
        ax6.axhline(xgb_accs.mean() * 100, color="#888780",
                    linestyle="--", linewidth=1, alpha=0.7)
        ax6.axhline(fold_accs.mean() * 100, color=SEQ_COLORS[0],
                    linestyle="--", linewidth=1.5)
        ax6.set_xticks(x)
        ax6.set_xticklabels(unique_subjs, rotation=45, ha="right", fontsize=8)
        ax6.set_ylabel("Accuracy (%)")
        ax6.set_title(f"Per-fold comparison: XGBoost ({xgb_accs.mean()*100:.2f}%) "
                      f"vs XGBoost+TCA ({fold_accs.mean()*100:.2f}%)")
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

    out_path = os.path.join(RESULTS_DIR, "xgboost_tca_charts.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Charts saved → {out_path}")


# ─────────────────────────────────────────────
#  SAVE + SUMMARY
# ─────────────────────────────────────────────

def save_results(fold_accs, fold_f1s, fold_kappas, y_true, y_pred):
    out = os.path.join(RESULTS_DIR, "xgboost_tca_results.npz")
    np.savez(out,
             fold_accs   = fold_accs,
             fold_f1s    = fold_f1s,
             fold_kappas = fold_kappas,
             y_true      = y_true,
             y_pred      = y_pred,
             model_name  = np.array("XGBoost+TCA"))
    print(f"  Results saved → {out}")


def print_summary(fold_accs, fold_f1s, fold_kappas, y_true, y_pred):
    print(f"\n{'='*65}")
    print(f"  XGBoost + TCA — Final Results")
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
        print(f"  TCA mean       : {fold_accs.mean()*100:.2f}%")

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
