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
    confusion_matrix, f1_score,
    roc_auc_score, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
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

LABELS      = ["Low", "Medium", "High"]
COLORS      = ["#4C72B0", "#DD8452", "#55A868"]
SEQ_COLORS  = ["#534AB7", "#1D9E75", "#D85A30"]

# XGBoost hyperparameters
PARAMS = dict(
    n_estimators      = 300,
    max_depth         = 6,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    eval_metric       = "mlogloss",
    random_state      = 42,
    n_jobs            = -1,
)

# Feature names for importance plot
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
#  LOSO TRAINING
# ─────────────────────────────────────────────

def run_loso(X, y_class, subject_ids, subjects):
    n_subjects = len(np.unique(subject_ids))
    fold_accs, fold_f1s, fold_kappas = [], [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    feat_importances = np.zeros(X.shape[1])

    print(f"XGBoost — LOSO cross-validation ({n_subjects} folds)")
    print(f"{'='*65}")

    for fold, test_subj, X_train, y_train, X_test, y_test in loso_splits(
        X, y_class, subject_ids
    ):
        subj_str = np.unique(subjects[subject_ids == test_subj])[0]

        # Standardize — fit on train only, apply to test
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = XGBClassifier(**PARAMS)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc   = accuracy_score(y_test, y_pred)
        f1    = f1_score(y_test, y_pred, average="macro")
        kappa = cohen_kappa_score(y_test, y_pred)

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_kappas.append(kappa)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        feat_importances += model.feature_importances_

        print(f"  Fold {fold+1:>2} | {subj_str} | n={len(y_test):>4} | "
              f"acc={acc:.4f} | f1={f1:.4f} | kappa={kappa:.4f}")

    feat_importances /= n_subjects

    return (
        np.array(fold_accs), np.array(fold_f1s), np.array(fold_kappas),
        np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob),
        feat_importances
    )


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────

def plot_all(fold_accs, fold_f1s, fold_kappas,
             y_true, y_pred, y_prob, feat_importances, subjects, subject_ids):

    fig = plt.figure(figsize=(20, 22))
    fig.suptitle("XGBoost — EEG Cognitive Workload Classification",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.45, wspace=0.35)

    # ── 1. Per-fold accuracy bar chart ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    unique_subjs = [np.unique(subjects[subject_ids == s])[0]
                    for s in np.unique(subject_ids)]
    x = np.arange(len(fold_accs))
    bars = ax1.bar(x, fold_accs * 100, color=SEQ_COLORS[0],
                   alpha=0.8, edgecolor="white", linewidth=0.5)
    ax1.axhline(fold_accs.mean() * 100, color="red",
                linestyle="--", linewidth=1.5,
                label=f"Mean {fold_accs.mean()*100:.2f}%")
    ax1.set_xticks(x)
    ax1.set_xticklabels(unique_subjs, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Per-subject (fold) accuracy — LOSO")
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars, fold_accs):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{acc*100:.1f}", ha="center", va="bottom",
                 fontsize=6, rotation=90)

    # ── 2. Metrics summary box ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    metrics = [
        ("Mean Accuracy",  f"{fold_accs.mean()*100:.2f}%"),
        ("Std Accuracy",   f"±{fold_accs.std()*100:.2f}%"),
        ("Best Fold",      f"{fold_accs.max()*100:.2f}%"),
        ("Worst Fold",     f"{fold_accs.min()*100:.2f}%"),
        ("Mean F1 (macro)",f"{fold_f1s.mean():.4f}"),
        ("Mean Kappa",     f"{fold_kappas.mean():.4f}"),
    ]
    y_pos = 0.95
    ax2.text(0.5, 1.0, "Summary Metrics", ha="center", va="top",
             fontsize=11, fontweight="bold",
             transform=ax2.transAxes)
    for label, val in metrics:
        ax2.text(0.05, y_pos, label, ha="left", fontsize=9,
                 color="gray", transform=ax2.transAxes)
        ax2.text(0.95, y_pos, val, ha="right", fontsize=9,
                 fontweight="bold", transform=ax2.transAxes)
        y_pos -= 0.13
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

    # ── 4. Per-class F1, Precision, Recall ──────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    report = classification_report(y_true, y_pred,
                                   target_names=LABELS, output_dict=True)
    metrics_per_class = ["precision", "recall", "f1-score"]
    x_cls = np.arange(len(LABELS))
    width = 0.25
    for i, m in enumerate(metrics_per_class):
        vals = [report[l][m] for l in LABELS]
        ax4.bar(x_cls + i * width, vals, width,
                label=m.replace("-score",""), color=COLORS[i],
                alpha=0.85, edgecolor="white")
    ax4.set_xticks(x_cls + width)
    ax4.set_xticklabels(LABELS)
    ax4.set_ylabel("Score")
    ax4.set_title("Per-class precision, recall, F1")
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    # ── 5. ROC-AUC curves ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    y_true_bin = label_binarize(y_true, classes=[0,1,2])
    y_prob_arr = np.array(y_prob)
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_arr[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_prob_arr[:, i])
            ax5.plot(fpr, tpr, color=color, lw=1.5,
                     label=f"{label} (AUC={auc:.3f})")
        except Exception:
            pass
    ax5.plot([0,1],[0,1],"k--", lw=1, alpha=0.4)
    ax5.set_xlabel("False positive rate")
    ax5.set_ylabel("True positive rate")
    ax5.set_title("ROC curves (one-vs-rest)")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # ── 6. Feature importance — top 20 ──────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    top_n = 20
    top_idx = np.argsort(feat_importances)[-top_n:][::-1]
    top_names = [FEAT_NAMES[i] for i in top_idx]
    top_vals  = feat_importances[top_idx]
    bar_colors = []
    for name in top_names:
        if "theta" in name or "FTI" in name: bar_colors.append("#534AB7")
        elif "alpha" in name:                bar_colors.append("#1D9E75")
        elif "beta"  in name:                bar_colors.append("#D85A30")
        else:                                bar_colors.append("#888888")
    bars6 = ax6.bar(range(top_n), top_vals, color=bar_colors,
                    alpha=0.85, edgecolor="white")
    ax6.set_xticks(range(top_n))
    ax6.set_xticklabels(top_names, rotation=45, ha="right", fontsize=8)
    ax6.set_ylabel("Mean importance score")
    ax6.set_title("Top 20 feature importances (averaged across LOSO folds)\n"
                  "Purple=theta  Green=alpha  Orange=beta  Gray=other")
    ax6.grid(axis="y", alpha=0.3)

    # ── 7. Accuracy distribution (violin) ───────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    vp = ax7.violinplot([fold_accs * 100], positions=[1],
                        showmeans=True, showmedians=True)
    for body in vp["bodies"]:
        body.set_facecolor("#534AB7")
        body.set_alpha(0.6)
    ax7.scatter([1] * len(fold_accs), fold_accs * 100,
                color="white", edgecolor="#534AB7", s=30, zorder=3)
    ax7.set_xticks([1])
    ax7.set_xticklabels(["Accuracy"])
    ax7.set_ylabel("Accuracy (%)")
    ax7.set_title("Fold accuracy distribution")
    ax7.grid(axis="y", alpha=0.3)

    # ── 8. F1 score per fold ─────────────────────────────────────
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

    # ── 9. Kappa per fold ────────────────────────────────────────
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

    out_path = os.path.join(RESULTS_DIR, "xgboost_charts.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Charts saved → {out_path}")


# ─────────────────────────────────────────────
#  SAVE RESULTS
# ─────────────────────────────────────────────

def save_results(fold_accs, fold_f1s, fold_kappas, y_true, y_pred):
    out = os.path.join(RESULTS_DIR, "xgboost_results.npz")
    np.savez(out,
             fold_accs   = fold_accs,
             fold_f1s    = fold_f1s,
             fold_kappas = fold_kappas,
             y_true      = y_true,
             y_pred      = y_pred,
             model_name  = np.array("XGBoost"))
    print(f"  Results saved → {out}")


# ─────────────────────────────────────────────
#  PRINT SUMMARY
# ─────────────────────────────────────────────

def print_summary(fold_accs, fold_f1s, fold_kappas, y_true, y_pred):
    print(f"\n{'='*65}")
    print(f"  XGBoost — Final Results")
    print(f"{'='*65}")
    print(f"  Mean Accuracy : {fold_accs.mean()*100:.2f}%  ±{fold_accs.std()*100:.2f}%")
    print(f"  Best Fold     : {fold_accs.max()*100:.2f}%")
    print(f"  Worst Fold    : {fold_accs.min()*100:.2f}%")
    print(f"  Mean F1 (macro): {fold_f1s.mean():.4f}")
    print(f"  Mean Kappa    : {fold_kappas.mean():.4f}")

    print(f"\n  Per-class report (aggregated across all folds):")
    print(classification_report(y_true, y_pred,
                                target_names=LABELS, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix (rows=true, cols=predicted):")
    print(f"  {'':>10}  " + "  ".join(f"{l:>10}" for l in LABELS))
    for i, row in enumerate(cm):
        print(f"  {LABELS[i]:>10}  " + "  ".join(f"{v:>10}" for v in row))


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("Loading dataset...\n")
    X, y_class, y_cont, subject_ids, subjects, games = load_dataset()
    print(f"  X shape: {X.shape}  |  Classes: {dict(zip(*np.unique(y_class, return_counts=True)))}\n")

    fold_accs, fold_f1s, fold_kappas, \
    y_true, y_pred, y_prob, feat_importances = run_loso(
        X, y_class, subject_ids, subjects
    )

    print_summary(fold_accs, fold_f1s, fold_kappas, y_true, y_pred)
    save_results(fold_accs, fold_f1s, fold_kappas, y_true, y_pred)
    plot_all(fold_accs, fold_f1s, fold_kappas,
             y_true, y_pred, y_prob,
             feat_importances, subjects, subject_ids)

    print(f"\n  Done. Next: run models/bilstm_model.py")


if __name__ == "__main__":
    main()
