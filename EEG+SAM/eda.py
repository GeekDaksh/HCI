"""
eda.py — Exploratory Data Analysis
===================================
Run this BEFORE train_model.py.
Diagnoses: class imbalance, label clustering, feature quality,
           per-subject variance, and SAM distribution problems.

Outputs saved to processed/eda/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import warnings
warnings.filterwarnings("ignore")

DATASET_PATH = "processed/dataset.npz"
EDA_DIR      = "processed/eda"
os.makedirs(EDA_DIR, exist_ok=True)

CHANNELS = ['AF3','F7','F3','FC5','T7','P7','O1','O2',
            'P8','T8','FC6','F4','F8','AF4']
BANDS    = ['delta','theta','alpha','beta','gamma']
LABELS   = {0: "Low", 1: "Medium", 2: "High"}
COLORS   = {0: "#3498db", 1: "#f39c12", 2: "#e74c3c"}


def load():
    data = np.load(DATASET_PATH, allow_pickle=True)
    return (data["X"], data["y_class"].astype(int),
            data["y_cont"].astype(float),
            data["subject_ids"].astype(int),
            data["subjects"], data["games"])


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ─── 1. Label Distribution ────────────────────────────────────────────────────

def plot_label_distribution(y_class, y_cont, subjects):
    section("1. LABEL DISTRIBUTION")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Label Distribution Analysis", fontsize=14, fontweight='bold')

    counts = [(y_class == c).sum() for c in [0, 1, 2]]
    ax = axes[0]
    bars = ax.bar([LABELS[c] for c in [0,1,2]], counts,
                  color=[COLORS[c] for c in [0,1,2]], edgecolor='black', linewidth=0.5)
    ax.set_title("Class Distribution (windows)")
    ax.set_ylabel("Count")
    for bar, count in zip(bars, counts):
        pct = count / len(y_class) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(counts) * 1.2)

    ax = axes[1]
    ax.hist(y_cont, bins=30, color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.axvline(y_cont.mean(), color='red', linestyle='--',
               label=f'Mean={y_cont.mean():.2f}')
    ax.set_title("Continuous Workload Distribution")
    ax.set_xlabel("Workload Score (1–9)")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[2]
    unique_subs = sorted(set(subjects))
    matrix = np.zeros((len(unique_subs), 3))
    for i, s in enumerate(unique_subs):
        mask = subjects == s
        total = mask.sum()
        for c in [0,1,2]:
            matrix[i, c] = (y_class[mask] == c).sum() / total * 100
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["Low","Med","High"])
    ax.set_yticks(range(len(unique_subs)))
    ax.set_yticklabels([s.replace('(','').replace(')','') for s in unique_subs], fontsize=8)
    ax.set_title("Per-Subject Class % (heatmap)")
    plt.colorbar(im, ax=ax, label='% of subject windows')
    for i in range(len(unique_subs)):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center',
                    fontsize=7, color='black' if matrix[i,j] < 60 else 'white')

    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/1_label_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {EDA_DIR}/1_label_distribution.png")

    ir = max(counts) / (min(counts) + 1)
    print(f"\n  Class counts   : Low={counts[0]}  Med={counts[1]}  High={counts[2]}")
    print(f"  Imbalance ratio: {ir:.1f}x  "
          f"({'⚠️  SEVERE' if ir > 5 else '✓ acceptable'})")
    print(f"  Workload mean  : {y_cont.mean():.2f}  std={y_cont.std():.2f}")
    if y_cont.std() < 1.0:
        print(f"  ⚠️  LOW VARIANCE — labels are clustered, poor separability")
    else:
        print(f"  ✓  Workload variance looks reasonable")


# ─── 2. Per-Subject Workload Variance ─────────────────────────────────────────

def plot_subject_variance(y_cont, y_class, subjects):
    section("2. PER-SUBJECT WORKLOAD VARIANCE")

    unique_subs = sorted(set(subjects))
    fig, axes   = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Per-Subject Workload Analysis", fontsize=14, fontweight='bold')

    data_by_sub = [y_cont[subjects == s] for s in unique_subs]
    bp = axes[0].boxplot(data_by_sub, patch_artist=True,
                         medianprops=dict(color='black', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
    axes[0].set_xticklabels([s.replace('(','').replace(')','') for s in unique_subs],
                             rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel("Workload Score (1–9)")
    axes[0].set_title("Workload Distribution per Subject")
    axes[0].axhline(y=3.67, color='blue',  linestyle=':', alpha=0.5, label='Low/Med')
    axes[0].axhline(y=6.33, color='red',   linestyle=':', alpha=0.5, label='Med/High')
    axes[0].legend(fontsize=8)

    means  = [y_cont[subjects == s].mean() for s in unique_subs]
    stds   = [y_cont[subjects == s].std()  for s in unique_subs]
    colors = ['#e74c3c' if st < 0.5 else '#f39c12' if st < 1.0 else '#2ecc71'
              for st in stds]
    axes[1].bar([s.replace('(','').replace(')','') for s in unique_subs],
                means, yerr=stds, color=colors, edgecolor='black',
                linewidth=0.5, capsize=4)
    axes[1].set_xticklabels([s.replace('(','').replace(')','') for s in unique_subs],
                             rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel("Mean Workload ± std")
    axes[1].set_title("Per-Subject Mean  (red=low variance ⚠️)")
    axes[1].set_ylim(1, 9)

    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/2_subject_variance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {EDA_DIR}/2_subject_variance.png")

    print(f"\n  Per-subject workload (mean ± std):")
    for s, m, st in zip(unique_subs, means, stds):
        flag = " ⚠️ " if st < 0.5 else "    "
        print(f"  {flag}{s}: mean={m:.2f}  std={st:.2f}")


# ─── 3. Feature Separability ──────────────────────────────────────────────────

def plot_feature_separability(X, y_class):
    section("3. FEATURE SEPARABILITY")

    feat_names = []
    for band in BANDS:
        for ch in CHANNELS:
            feat_names.append(f"{band}_{ch}")
    for ratio in ['theta_alpha','beta_alpha']:
        for ch in CHANNELS:
            feat_names.append(f"{ratio}_{ch}")
    for ch in CHANNELS:
        feat_names.append(f"entropy_{ch}")
    feat_names += ['frontal_theta','parietal_alpha','frontal_asymmetry','engagement_idx']
    feat_names = feat_names[:X.shape[1]]

    kw_stats = []
    for i in range(X.shape[1]):
        groups = [X[y_class == c, i] for c in [0,1,2]]
        if all(len(g) > 0 for g in groups):
            stat, p = kruskal(*groups)
            kw_stats.append((feat_names[i], stat, p))

    kw_df = pd.DataFrame(kw_stats, columns=["feature","kw_stat","p_value"])
    kw_df = kw_df.sort_values("kw_stat", ascending=False)
    kw_df["significant"] = kw_df["p_value"] < 0.05

    sig_count = kw_df["significant"].sum()
    total     = len(kw_df)
    print(f"\n  Significant features (p<0.05): {sig_count}/{total} ({sig_count/total*100:.1f}%)")
    if sig_count / total < 0.3:
        print(f"  ⚠️  <30% significant — EEG features weakly aligned with labels")
    else:
        print(f"  ✓  Good feature separability")

    top12     = kw_df.head(12)["feature"].tolist()
    top12_idx = [feat_names.index(f) for f in top12 if f in feat_names]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f"Top 12 Most Class-Separable Features\n"
                 f"({sig_count}/{total} features significant, Kruskal-Wallis)",
                 fontsize=13, fontweight='bold')

    for ax, feat_i, fname in zip(axes.flat, top12_idx, top12):
        data_by_class = [X[y_class == c, feat_i] for c in [0,1,2]]
        vp = ax.violinplot(data_by_class, positions=[0,1,2], showmedians=True)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(COLORS[i])
            body.set_alpha(0.7)
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(["Low","Med","High"], fontsize=8)
        ax.set_title(fname, fontsize=8, fontweight='bold')
        row = kw_df[kw_df["feature"] == fname]
        if not row.empty:
            p = row["p_value"].values[0]
            ax.set_xlabel(f"p={p:.4f}" + (" ✓" if p < 0.05 else " ✗"), fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/3_feature_separability.png", dpi=150, bbox_inches='tight')
    plt.close()
    kw_df.to_csv(f"{EDA_DIR}/feature_separability.csv", index=False)
    print(f"  Saved → {EDA_DIR}/3_feature_separability.png")
    print(f"  Table  → {EDA_DIR}/feature_separability.csv")

    return kw_df


# ─── 4. Band Power by Class ───────────────────────────────────────────────────

def plot_band_power_by_class(X, y_class):
    section("4. BAND POWER BY CLASS")

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle("Mean Frontal Band Power per Workload Class",
                 fontsize=13, fontweight='bold')

    frontal_ch = [0, 1, 2, 11, 12, 13]   # AF3,F7,F3,F4,F8,AF4

    for ax, (bi, band) in zip(axes, enumerate(BANDS)):
        band_feats    = X[:, bi*14 : bi*14+14]
        frontal_power = band_feats[:, frontal_ch].mean(axis=1)
        means = [frontal_power[y_class == c].mean() for c in [0,1,2]]
        stds  = [frontal_power[y_class == c].std()  for c in [0,1,2]]
        ax.bar([LABELS[c] for c in [0,1,2]], means, yerr=stds,
               color=[COLORS[c] for c in [0,1,2]],
               edgecolor='black', linewidth=0.5, capsize=4)
        ax.set_title(f"{band.capitalize()}", fontweight='bold')
        ax.set_ylabel("Mean Power")

    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/4_band_power_by_class.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {EDA_DIR}/4_band_power_by_class.png")

    theta_low  = X[y_class==0, 14:28].mean()
    theta_high = X[y_class==2, 14:28].mean()
    alpha_low  = X[y_class==0, 28:42].mean()
    alpha_high = X[y_class==2, 28:42].mean()

    print(f"\n  Theta Low→High: {theta_low:.4f} → {theta_high:.4f}  "
          f"{'✓ increases (expected)' if theta_high > theta_low else '⚠️  unexpected'}")
    print(f"  Alpha Low→High: {alpha_low:.4f} → {alpha_high:.4f}  "
          f"{'✓ decreases (expected)' if alpha_low > alpha_high else '⚠️  unexpected'}")


# ─── 5. SAM Distribution ──────────────────────────────────────────────────────

def plot_sam_distribution():
    section("5. SAM SCORE DISTRIBUTION")

    sam_path = "sam_all_subjects.csv"
    if not os.path.exists(sam_path):
        print(f"  [SKIP] sam_all_subjects.csv not found")
        return

    df = pd.read_csv(sam_path).dropna(subset=["valence","arousal","workload_continuous"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("SAM Score Analysis — Label Quality Diagnostic",
                 fontsize=13, fontweight='bold')

    axes[0,0].hist(df["valence"], bins=9, range=(0.5,9.5),
                   color='#9b59b6', edgecolor='black', linewidth=0.5)
    axes[0,0].set_title("Valence (1=horrible → 9=happy)")
    axes[0,0].set_xlabel("Score")
    axes[0,0].set_ylabel("Sessions")

    axes[0,1].hist(df["arousal"], bins=9, range=(0.5,9.5),
                   color='#e67e22', edgecolor='black', linewidth=0.5)
    axes[0,1].set_title("Arousal (1=calm → 9=excited)")
    axes[0,1].set_xlabel("Score")

    axes[0,2].hist(df["workload_continuous"], bins=20,
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    axes[0,2].axvline(df["workload_continuous"].mean(), color='red', linestyle='--',
                      label=f"mean={df['workload_continuous'].mean():.2f}")
    axes[0,2].set_title("Derived Workload Score")
    axes[0,2].set_xlabel("Score (1–9)")
    axes[0,2].legend()

    ax = axes[1,0]
    class_col = df["workload_class"].fillna(1).astype(int)
    for c in [0,1,2]:
        mask = class_col == c
        ax.scatter(df.loc[mask,"valence"], df.loc[mask,"arousal"],
                   color=COLORS[c], label=LABELS[c], alpha=0.8, s=60,
                   edgecolors='black', linewidth=0.5)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("Valence–Arousal Space (Russell's Circumplex)")
    ax.legend()
    ax.set_xlim(0.5, 9.5); ax.set_ylim(0.5, 9.5)
    ax.axvline(5, color='gray', linestyle=':', alpha=0.4)
    ax.axhline(5, color='gray', linestyle=':', alpha=0.4)
    ax.text(2, 8.5, "High workload", ha='center', fontsize=7, color='red')
    ax.text(7.5, 8.5, "Flow state",  ha='center', fontsize=7, color='green')
    ax.text(2, 1.5, "Bored",         ha='center', fontsize=7, color='blue')
    ax.text(7.5, 1.5, "Relaxed",     ha='center', fontsize=7, color='gray')

    game_means = df.groupby("game")["workload_continuous"].mean()
    game_stds  = df.groupby("game")["workload_continuous"].std()
    axes[1,1].bar(game_means.index, game_means.values, yerr=game_stds.values,
                  color='#3498db', edgecolor='black', linewidth=0.5, capsize=4)
    axes[1,1].set_title("Mean Workload per Game")
    axes[1,1].set_ylabel("Workload Score")
    axes[1,1].set_ylim(1, 9)

    sub_means  = df.groupby("subject")["workload_continuous"].mean()
    sub_stds   = df.groupby("subject")["workload_continuous"].std().fillna(0)
    sub_labels = [s.replace('(','').replace(')','') for s in sub_means.index]
    axes[1,2].bar(sub_labels, sub_means.values, yerr=sub_stds.values,
                  color='#e74c3c', edgecolor='black', linewidth=0.5, capsize=4)
    axes[1,2].set_xticklabels(sub_labels, rotation=45, ha='right', fontsize=7)
    axes[1,2].set_title("Mean Workload per Subject")
    axes[1,2].set_ylabel("Workload Score")
    axes[1,2].set_ylim(1, 9)

    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/5_sam_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {EDA_DIR}/5_sam_distribution.png")

    print(f"\n  Valence  : mean={df['valence'].mean():.2f}  std={df['valence'].std():.2f}")
    print(f"  Arousal  : mean={df['arousal'].mean():.2f}  std={df['arousal'].std():.2f}")
    print(f"  Workload : mean={df['workload_continuous'].mean():.2f}  "
          f"std={df['workload_continuous'].std():.2f}")

    unique_pairs = df[["valence","arousal"]].drop_duplicates()
    print(f"  Unique valence-arousal pairs: {len(unique_pairs)}/{len(df)} sessions")
    if len(unique_pairs) / len(df) < 0.5:
        print(f"  ⚠️  Many repeated rating pairs — low label diversity")


# ─── 6. Diagnosis ─────────────────────────────────────────────────────────────

def print_diagnosis(y_class, y_cont, subjects, kw_df):
    section("DIAGNOSIS SUMMARY — Read before running train_model.py")

    issues, tips = [], []

    counts = [(y_class == c).sum() for c in [0,1,2]]
    ir = max(counts) / (min(counts) + 1)
    if ir > 5:
        issues.append(f"Severe class imbalance ({ir:.1f}x)")
        tips.append("Use SMOTE oversampling or heavier class weights")

    if y_cont.std() < 1.0:
        issues.append(f"Low label variance (std={y_cont.std():.2f})")
        tips.append("Consider binary Low vs High classification, drop Medium")

    sig_pct = kw_df["significant"].mean() * 100
    if sig_pct < 30:
        issues.append(f"Only {sig_pct:.0f}% features significantly separate classes")
        tips.append("Review SAM extraction — labels may not reflect EEG state")

    unique_subs = sorted(set(subjects))
    for s in unique_subs:
        if y_cont[subjects == s].std() < 0.3:
            issues.append(f"{s} has near-zero workload variance")
            tips.append(f"Consider excluding {s} or flagging their sessions")

    if not issues:
        print("\n  ✅ Dataset looks healthy")
        print("  Expected LOSO: Accuracy 50–65%  Kappa 0.25–0.45")
    else:
        print(f"\n  Found {len(issues)} issue(s):\n")
        for i, (iss, tip) in enumerate(zip(issues, tips), 1):
            print(f"  {i}. ⚠️  {iss}")
            print(f"       Fix: {tip}\n")
        print(f"  ─── Expected impact on LOSO metrics ───────────────────")
        print(f"  Each unresolved issue above reduces Kappa by ~0.05–0.15")
        print(f"  Fix label issues before tuning the model")

    print(f"\n  All EDA plots → {EDA_DIR}/")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    section("Loading Dataset")
    X, y_class, y_cont, subject_ids, subjects, games = load()
    print(f"  Windows  : {len(X)}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Subjects : {len(np.unique(subject_ids))}")

    plot_label_distribution(y_class, y_cont, subjects)
    plot_subject_variance(y_cont, y_class, subjects)
    kw_df = plot_feature_separability(X, y_class)
    plot_band_power_by_class(X, y_class)
    plot_sam_distribution()
    print_diagnosis(y_class, y_cont, subjects, kw_df)


if __name__ == "__main__":
    main()