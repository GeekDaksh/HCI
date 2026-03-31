import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report

RESULTS_DIR = "results"
LABELS      = ["Low", "Medium", "High"]

MODELS = [
    {"name": "XGBoost",          "file": "xgboost_results.npz",              "color": "#888780", "fill": "#F2F2F2"},
    {"name": "XGBoost+CORAL",    "file": "xgboost_coral_results.npz",        "color": "#D85A30", "fill": "#FCE9D9"},
    {"name": "BiLSTM+Attention", "file": "bilstm_results.npz",               "color": "#7F77DD", "fill": "#EDE7F6"},
    {"name": "TCN",              "file": "tcn_results.npz",                  "color": "#1D9E75", "fill": "#E2EFDA"},
    {"name": "Transformer",      "file": "transformer_results.npz",          "color": "#2E75B6", "fill": "#D6E4F0"},
    {"name": "BiLSTM+CB-Attn",   "file": "bilstm_cb_attention_results.npz",  "color": "#9C27B0", "fill": "#F3E5F5"},
]


def load_all():
    loaded = []
    for m in MODELS:
        path = os.path.join(RESULTS_DIR, m["file"])
        if not os.path.exists(path):
            print(f"  [SKIP] {m['name']} — {path} not found")
            continue
        data = np.load(path, allow_pickle=True)
        loaded.append({
            "name":   m["name"],
            "color":  m["color"],
            "fill":   m["fill"],
            "accs":   data["fold_accs"],
            "f1s":    data["fold_f1s"],
            "kappas": data["fold_kappas"],
            "y_true": data["y_true"],
            "y_pred": data["y_pred"],
        })
    return loaded


def print_summary(results):
    baseline_acc = None
    for r in results:
        if r["name"] == "XGBoost":
            baseline_acc = r["accs"].mean()
            break

    print(f"\n{'='*95}")
    print(f"  {'Model':<22} {'Acc':>7}  {'+-Std':>6}  {'Best':>7}  {'Worst':>7}  {'F1':>7}  {'Kappa':>7}  {'vs XGB':>8}")
    print(f"{'='*95}")

    for r in results:
        acc   = r["accs"].mean()
        std   = r["accs"].std()
        best  = r["accs"].max()
        worst = r["accs"].min()
        f1    = r["f1s"].mean()
        kappa = r["kappas"].mean()
        delta = f"{(acc - baseline_acc)*100:+.2f}%" if baseline_acc is not None else "—"
        print(f"  {r['name']:<22} {acc*100:>6.2f}%  {std*100:>5.2f}%  "
              f"{best*100:>6.2f}%  {worst*100:>6.2f}%  "
              f"{f1:>7.4f}  {kappa:>7.4f}  {delta:>8}")
    print(f"{'='*95}")


def plot_comparison(results):
    n_models = len(results)
    names    = [r["name"]  for r in results]
    colors   = [r["color"] for r in results]

    fig = plt.figure(figsize=(24, 28))
    fig.suptitle("EEG Cognitive Workload — Model Comparison (LOSO)",
                 fontsize=17, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.32)

    # 1. Mean accuracy bar
    ax1 = fig.add_subplot(gs[0, :2])
    accs = [r["accs"].mean() * 100 for r in results]
    stds = [r["accs"].std()  * 100 for r in results]
    x    = np.arange(n_models)
    bars = ax1.bar(x, accs, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax1.errorbar(x, accs, yerr=stds, fmt="none",
                 color="black", capsize=5, linewidth=1.2, zorder=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax1.set_ylabel("Mean Accuracy (%)", fontsize=11)
    ax1.set_title("Mean LOSO Accuracy with +-1 Std Dev", fontsize=12)
    ax1.set_ylim(60, 100)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    for bar, acc, std in zip(bars, accs, stds):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + std + 0.4,
                 f"{acc:.2f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    # 2. Summary box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    best_model   = max(results, key=lambda r: r["accs"].mean())
    tight_model  = min(results, key=lambda r: r["accs"].std())
    lines = [
        ("Best accuracy",   best_model["name"]),
        ("",                f"{best_model['accs'].mean()*100:.2f}%"),
        ("Most consistent", tight_model["name"]),
        ("",                f"+-{tight_model['accs'].std()*100:.2f}%"),
        ("Models run",      str(n_models)),
        ("Total windows",   "44,240"),
        ("LOSO folds",      "28"),
        ("Features",        "77 per window"),
    ]
    y = 0.95
    ax2.text(0.5, 1.02, "Summary", ha="center", va="top",
             fontsize=12, fontweight="bold", transform=ax2.transAxes)
    for label, val in lines:
        if label:
            ax2.text(0.05, y, label, ha="left", fontsize=9,
                     color="gray", transform=ax2.transAxes)
        ax2.text(0.97, y, val, ha="right", fontsize=9,
                 fontweight="bold", transform=ax2.transAxes)
        y -= 0.10
    rect = plt.Rectangle((0,0), 1, 1, fill=False,
                          edgecolor="#2E75B6", linewidth=1.5,
                          transform=ax2.transAxes)
    ax2.add_patch(rect)

    # 3. Per-class F1
    ax3 = fig.add_subplot(gs[1, :])
    class_colors = ["#4C72B0", "#DD8452", "#55A868"]
    x_cls = np.arange(n_models)
    w     = 0.25
    for ci, (cls, clr) in enumerate(zip(LABELS, class_colors)):
        vals = []
        for r in results:
            rpt = classification_report(
                r["y_true"], r["y_pred"],
                target_names=LABELS, output_dict=True
            )
            vals.append(rpt[cls]["f1-score"] * 100)
        bars_cls = ax3.bar(x_cls + (ci - 1) * w, vals, w,
                           label=cls, color=clr, alpha=0.85,
                           edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars_cls, vals):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f"{v:.1f}", ha="center", va="bottom",
                     fontsize=7, rotation=90)
    ax3.set_xticks(x_cls)
    ax3.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax3.set_ylabel("F1 Score (%)", fontsize=11)
    ax3.set_title("Per-class F1 Score by Model", fontsize=12)
    ax3.set_ylim(50, 105)
    ax3.legend(fontsize=9, loc="lower right")
    ax3.grid(axis="y", alpha=0.3)

    # 4-6. Confusion matrices for top 3
    sorted_r = sorted(results, key=lambda r: r["accs"].mean(), reverse=True)[:3]
    for i, r in enumerate(sorted_r):
        ax = fig.add_subplot(gs[2, i])
        cm     = confusion_matrix(r["y_true"], r["y_pred"])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
        ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("True", fontsize=9)
        ax.set_title(f"{r['name']}\n{r['accs'].mean()*100:.2f}%", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for ii in range(3):
            for jj in range(3):
                ax.text(jj, ii, f"{cm_pct[ii,jj]:.1f}%\n({cm[ii,jj]})",
                        ha="center", va="center", fontsize=7,
                        color="white" if cm_pct[ii,jj] > 55 else "black")

    # 7. Violin plot
    ax7 = fig.add_subplot(gs[3, :2])
    data_vp = [r["accs"] * 100 for r in results]
    vp = ax7.violinplot(data_vp, positions=range(n_models),
                        showmeans=True, showmedians=True)
    for body_vp, color in zip(vp["bodies"], colors):
        body_vp.set_facecolor(color)
        body_vp.set_alpha(0.6)
    for i, r in enumerate(results):
        ax7.scatter([i]*len(r["accs"]), r["accs"]*100,
                    s=20, color=colors[i], alpha=0.7, zorder=3)
    ax7.set_xticks(range(n_models))
    ax7.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax7.set_ylabel("Fold Accuracy (%)", fontsize=11)
    ax7.set_title("Fold Accuracy Distribution (Violin Plot)", fontsize=12)
    ax7.grid(axis="y", alpha=0.3)

    # 8. Radar
    ax8 = fig.add_subplot(gs[3, 2], polar=True)
    radar_metrics = ["Accuracy", "F1 Macro", "Kappa", "Consistency", "Med F1"]
    N = len(radar_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(radar_metrics, fontsize=8)
    ax8.set_ylim(0, 1)
    ax8.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax8.set_yticklabels(["20","40","60","80","100"], fontsize=6)
    ax8.grid(alpha=0.3)
    for r in results:
        rpt = classification_report(r["y_true"], r["y_pred"],
                                    target_names=LABELS, output_dict=True)
        consistency = float(np.clip(1 - r["accs"].std() / 0.15, 0, 1))
        values = [r["accs"].mean(), r["f1s"].mean(), r["kappas"].mean(),
                  consistency, rpt["Medium"]["f1-score"]]
        values += values[:1]
        ax8.plot(angles, values, "o-", linewidth=1.5,
                 color=r["color"], label=r["name"], alpha=0.8)
        ax8.fill(angles, values, alpha=0.07, color=r["color"])
    ax8.set_title("Multi-metric Radar", fontsize=10, pad=12)
    ax8.legend(loc="upper right", bbox_to_anchor=(1.4, 1.2), fontsize=7)

    out_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved -> {out_path}")


def print_detailed(results):
    print(f"\n{'='*70}")
    print("  PER-MODEL DETAILED REPORT")
    print(f"{'='*70}")
    baseline_acc = next((r["accs"].mean() for r in results if r["name"] == "XGBoost"), None)
    for r in results:
        rpt = classification_report(r["y_true"], r["y_pred"],
                                    target_names=LABELS, output_dict=True)
        cm  = confusion_matrix(r["y_true"], r["y_pred"])
        print(f"\n  -- {r['name']} --")
        print(f"  Accuracy : {r['accs'].mean()*100:.2f}%  +-{r['accs'].std()*100:.2f}%")
        print(f"  F1 Macro : {r['f1s'].mean():.4f}   Kappa: {r['kappas'].mean():.4f}")
        print(f"  Best fold: {r['accs'].max()*100:.2f}%   Worst: {r['accs'].min()*100:.2f}%")
        if baseline_acc is not None:
            print(f"  vs XGBoost: {(r['accs'].mean()-baseline_acc)*100:+.2f}%")
        for cls in LABELS:
            print(f"    {cls:<8}: F1={rpt[cls]['f1-score']*100:.2f}%  "
                  f"P={rpt[cls]['precision']*100:.1f}%  R={rpt[cls]['recall']*100:.1f}%")
        total = cm.sum()
        print(f"  Low->High: {cm[0,2]} ({cm[0,2]/total*100:.2f}%)  "
              f"High->Low: {cm[2,0]} ({cm[2,0]/total*100:.2f}%)")


def print_statistics(results):
    print(f"\n{'='*70}")
    print("  KEY STATISTICAL FINDINGS")
    print(f"{'='*70}")
    xgb  = next((r for r in results if r["name"] == "XGBoost"), None)
    tcn  = next((r for r in results if r["name"] == "TCN"), None)
    tfmr = next((r for r in results if r["name"] == "Transformer"), None)
    bi   = next((r for r in results if r["name"] == "BiLSTM+Attention"), None)
    cb   = next((r for r in results if "CB" in r["name"]), None)
    best = max(results, key=lambda r: r["accs"].mean())

    print(f"\n  Best overall model: {best['name']}  {best['accs'].mean()*100:.2f}%")

    if xgb and tcn:
        print(f"  TCN vs XGBoost: +{(tcn['accs'].mean()-xgb['accs'].mean())*100:.2f}pp")
    if xgb and tfmr:
        print(f"  Transformer vs XGBoost: +{(tfmr['accs'].mean()-xgb['accs'].mean())*100:.2f}pp")
    if xgb and cb:
        print(f"  BiLSTM+CB-Attn vs XGBoost: +{(cb['accs'].mean()-xgb['accs'].mean())*100:.2f}pp")
    if tcn and tfmr:
        print(f"  TCN vs Transformer: {(tcn['accs'].mean()-tfmr['accs'].mean())*100:+.2f}pp  "
              f"Var: TCN +-{tcn['accs'].std()*100:.2f}%  "
              f"Tfmr +-{tfmr['accs'].std()*100:.2f}%")
    if bi and xgb:
        rpt_xgb = classification_report(xgb["y_true"], xgb["y_pred"],
                                        target_names=LABELS, output_dict=True)
        rpt_bi  = classification_report(bi["y_true"],  bi["y_pred"],
                                        target_names=LABELS, output_dict=True)
        print(f"  Medium F1 gain (BiLSTM vs XGB): "
              f"{rpt_xgb['Medium']['f1-score']*100:.1f}% -> "
              f"{rpt_bi['Medium']['f1-score']*100:.1f}%  "
              f"(temporal context +{(rpt_bi['Medium']['f1-score']-rpt_xgb['Medium']['f1-score'])*100:.1f}pp)")


def main():
    print("Loading model results...\n")
    results = load_all()
    if not results:
        print("[ERROR] No result files found in results/")
        return
    print(f"  Loaded {len(results)} models: {', '.join(r['name'] for r in results)}")
    print_summary(results)
    print_detailed(results)
    print_statistics(results)
    plot_comparison(results)


if __name__ == "__main__":
    main()
