import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, cohen_kappa_score,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from dataset_loader import load_dataset, loso_splits

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
LABELS     = ["Low", "Medium", "High"]
COLORS     = ["#4C72B0", "#DD8452", "#55A868"]
SEQ_COLORS = ["#085041", "#1D9E75", "#9FE1CB"]

SEQ_LEN    = 15
STRIDE     = 1

N_CHANNELS  = 14
N_BANDS     = 5
N_PSD       = N_CHANNELS * N_BANDS   # 70
N_ENG       = 6
N_FTI       = 1
INPUT_DIM   = N_PSD + N_ENG + N_FTI  # 77

BAND_NAMES    = ["delta", "theta", "alpha", "beta", "gamma"]
CHANNEL_NAMES = ["AF3","F7","F3","FC5","T7","P7","O1","O2",
                  "P8","T8","FC6","F4","F8","AF4"]

N_CLASSES   = 3
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
EPOCHS      = 30
BATCH_SIZE  = 256
LR          = 1e-3
PATIENCE    = 7


def build_sequences(X, y, seq_len=SEQ_LEN, stride=STRIDE):
    n = len(X)
    X_seq, y_seq = [], []
    for i in range(0, n - seq_len + 1, stride):
        X_seq.append(X[i: i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


def build_sequences_by_subject(X, y, subject_ids, seq_len=SEQ_LEN):
    X_all, y_all = [], []
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        X_s, y_s = build_sequences(X[mask], y[mask], seq_len)
        X_all.append(X_s)
        y_all.append(y_s)
    return np.vstack(X_all), np.concatenate(y_all)


class BandAttention(nn.Module):
    """
    Frequency Band Attention.
    Learns which of the 5 EEG bands is most informative per prediction.
    Theta upweighted (cognitive load marker), alpha downweighted (relaxation).
    Uses Sigmoid gates so multiple bands can be simultaneously attended.
    """
    def __init__(self, n_bands=N_BANDS):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_bands, n_bands * 2),
            nn.ReLU(),
            nn.Linear(n_bands * 2, n_bands),
            nn.Sigmoid(),
        )

    def forward(self, x_bands):
        band_summary = x_bands.mean(dim=-1).mean(dim=1)
        band_weights = self.fc(band_summary)
        gated = x_bands * band_weights.unsqueeze(1).unsqueeze(-1)
        return gated, band_weights


class ChannelAttention(nn.Module):
    """
    EEG Channel Attention.
    Learns which of the 14 electrodes is most informative.
    Expected: frontal (AF3,F7,F3,F4,F8,AF4) upweighted for theta.
              parietal (P7,P8) upweighted for alpha suppression.
              occipital (O1,O2) downweighted (visual artifact).
    """
    def __init__(self, n_channels=N_CHANNELS):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels * 2),
            nn.ReLU(),
            nn.Linear(n_channels * 2, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x_bands):
        chan_summary = x_bands.mean(dim=-2).mean(dim=1)
        chan_weights = self.fc(chan_summary)
        gated = x_bands * chan_weights.unsqueeze(1).unsqueeze(2)
        return gated, chan_weights


class ChannelBandAttention(nn.Module):
    """
    Dual 2D attention over (frequency band x electrode) space.
    Applied sequentially: band first, then channel within attended bands.
    Produces interpretable weights publishable as neuroscience results.
    """
    def __init__(self, n_bands=N_BANDS, n_channels=N_CHANNELS):
        super().__init__()
        self.band_attn    = BandAttention(n_bands)
        self.channel_attn = ChannelAttention(n_channels)

    def forward(self, psd_feats):
        B, T, _ = psd_feats.shape
        x = psd_feats.view(B, T, N_BANDS, N_CHANNELS)
        x, band_w = self.band_attn(x)
        x, chan_w = self.channel_attn(x)
        x_flat = x.view(B, T, N_BANDS * N_CHANNELS)
        return x_flat, band_w, chan_w


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H):
        scores  = self.v(torch.tanh(self.W(H)))
        weights = torch.softmax(scores, dim=1)
        context = (weights * H).sum(dim=1)
        return context, weights.squeeze(-1)


class BiLSTMChannelBandAttention(nn.Module):
    """
    BiLSTM + Channel-Band Attention (Model 6 — EEG-specific novelty).

    Three-level attention:
      1. Band attention    — which frequency band matters
      2. Channel attention — which electrode matters
      3. Temporal Bahdanau — which time window matters

    Architecture:
      PSD features → ChannelBandAttention → residual add → LayerNorm
      Merge with engineered features → BiLSTM x2 → BahdanauAttention
      → classifier → logits
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, n_classes):
        super().__init__()
        self.cb_attention = ChannelBandAttention()
        self.psd_norm     = nn.LayerNorm(N_PSD)
        self.merge_norm   = nn.LayerNorm(input_dim)
        self.input_norm   = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.dropout   = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim * 2)
        self.out_norm  = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, return_attention=False):
        psd_part = x[:, :, :N_PSD]
        eng_part = x[:, :, N_PSD:]
        attended_psd, band_w, chan_w = self.cb_attention(psd_part)
        attended_psd = self.psd_norm(attended_psd + psd_part)
        merged = torch.cat([attended_psd, eng_part], dim=-1)
        merged = self.merge_norm(merged)
        merged = self.input_norm(merged)
        H, _   = self.lstm(merged)
        H      = self.dropout(H)
        context, temporal_w = self.attention(H)
        context = self.out_norm(context)
        logits  = self.classifier(context)
        if return_attention:
            return logits, band_w, chan_w, temporal_w
        return logits


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, return_attention=False):
    model.eval()
    preds, probs, labels = [], [], []
    band_weights_all, chan_weights_all = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        if return_attention:
            logits, bw, cw, _ = model(X_batch, return_attention=True)
            band_weights_all.append(bw.cpu().numpy())
            chan_weights_all.append(cw.cpu().numpy())
        else:
            logits = model(X_batch)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        preds.extend(logits.argmax(1).cpu().numpy())
        probs.extend(p)
        labels.extend(y_batch.numpy())
    result = (np.array(labels), np.array(preds), np.array(probs))
    if return_attention:
        return result + (np.vstack(band_weights_all), np.vstack(chan_weights_all))
    return result


def run_loso(X, y_class, subject_ids, subjects):
    n_subjects = len(np.unique(subject_ids))
    fold_accs, fold_f1s, fold_kappas = [], [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    train_losses_all = []
    all_band_weights, all_chan_weights = [], []

    print(f"BiLSTM + Channel-Band Attention  LOSO ({n_subjects} folds)")
    print(f"  seq_len={SEQ_LEN}  hidden={HIDDEN_DIM}x2  layers={NUM_LAYERS}")
    print(f"  CB attention: {N_BANDS} bands x {N_CHANNELS} channels")
    print("=" * 65)

    for fold, test_subj, X_train, y_train, X_test, y_test in loso_splits(
        X, y_class, subject_ids
    ):
        subj_str = np.unique(subjects[subject_ids == test_subj])[0]
        scaler   = StandardScaler()
        X_train  = scaler.fit_transform(X_train)
        X_test   = scaler.transform(X_test)

        train_mask    = ~(subject_ids == test_subj)
        train_sub_ids = subject_ids[train_mask]

        X_tr_seq, y_tr_seq = build_sequences_by_subject(X_train, y_train, train_sub_ids, SEQ_LEN)
        X_te_seq, y_te_seq = build_sequences(X_test, y_test, SEQ_LEN)

        if len(X_tr_seq) == 0 or len(X_te_seq) == 0:
            print(f"  [SKIP] Fold {fold+1}")
            continue

        train_ds = TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq))
        test_ds  = TensorDataset(torch.from_numpy(X_te_seq), torch.from_numpy(y_te_seq))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

        model = BiLSTMChannelBandAttention(
            input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, dropout=DROPOUT, n_classes=N_CLASSES,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

        best_val_acc = 0.0
        best_state   = None
        patience_ctr = 0
        fold_tr_losses = []

        for epoch in range(EPOCHS):
            tr_loss, _ = train_epoch(model, train_loader, optimizer, criterion)
            val_labels, val_preds, _ = evaluate(model, test_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            scheduler.step(val_acc)
            fold_tr_losses.append(tr_loss)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        y_true_f, y_pred_f, y_prob_f, bw, cw = evaluate(model, test_loader, return_attention=True)

        acc   = accuracy_score(y_true_f, y_pred_f)
        f1    = f1_score(y_true_f, y_pred_f, average="macro")
        kappa = cohen_kappa_score(y_true_f, y_pred_f)

        fold_accs.append(acc); fold_f1s.append(f1); fold_kappas.append(kappa)
        all_y_true.extend(y_true_f); all_y_pred.extend(y_pred_f); all_y_prob.extend(y_prob_f)
        train_losses_all.append(fold_tr_losses)
        all_band_weights.append(bw.mean(axis=0))
        all_chan_weights.append(cw.mean(axis=0))

        print(f"  Fold {fold+1:>2} | {subj_str} | seq={len(X_te_seq):>4} | "
              f"acc={acc:.4f} | f1={f1:.4f} | kappa={kappa:.4f} | epochs={len(fold_tr_losses)}")

    mean_band_weights = np.mean(all_band_weights, axis=0)
    mean_chan_weights  = np.mean(all_chan_weights, axis=0)

    return (np.array(fold_accs), np.array(fold_f1s), np.array(fold_kappas),
            np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob),
            train_losses_all, mean_band_weights, mean_chan_weights)


def plot_all(fold_accs, fold_f1s, fold_kappas, y_true, y_pred, y_prob,
             subjects, subject_ids, train_losses_all,
             mean_band_weights, mean_chan_weights):

    fig = plt.figure(figsize=(20, 26))
    fig.suptitle("BiLSTM + Channel-Band Attention  EEG Cognitive Workload",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35)
    unique_subjs = [np.unique(subjects[subject_ids == s])[0] for s in np.unique(subject_ids)]

    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(fold_accs))
    bars = ax1.bar(x, fold_accs*100, color=SEQ_COLORS[1], alpha=0.8, edgecolor="white")
    ax1.axhline(fold_accs.mean()*100, color="red", linestyle="--", linewidth=1.5,
                label=f"Mean {fold_accs.mean()*100:.2f}%")
    ax1.set_xticks(x); ax1.set_xticklabels(unique_subjs[:len(fold_accs)], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Accuracy (%)"); ax1.set_title("Per-subject accuracy  BiLSTM+CB Attention (LOSO)")
    ax1.set_ylim(0, 100); ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars, fold_accs):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{acc*100:.1f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax2 = fig.add_subplot(gs[0, 2]); ax2.axis("off")
    metrics_list = [("Mean Accuracy", f"{fold_accs.mean()*100:.2f}%"),
                    ("Std", f"+-{fold_accs.std()*100:.2f}%"),
                    ("Best Fold", f"{fold_accs.max()*100:.2f}%"),
                    ("Worst Fold", f"{fold_accs.min()*100:.2f}%"),
                    ("Mean F1 (macro)", f"{fold_f1s.mean():.4f}"),
                    ("Mean Kappa", f"{fold_kappas.mean():.4f}"),
                    ("Seq length", f"{SEQ_LEN} windows ({SEQ_LEN*2}s)")]
    y_pos = 0.95
    ax2.text(0.5, 1.0, "BiLSTM + CB Attention", ha="center", va="top",
             fontsize=10, fontweight="bold", transform=ax2.transAxes)
    for label, val in metrics_list:
        ax2.text(0.05, y_pos, label, ha="left", fontsize=9, color="gray", transform=ax2.transAxes)
        ax2.text(0.95, y_pos, val, ha="right", fontsize=9, fontweight="bold", transform=ax2.transAxes)
        y_pos -= 0.11
    rect = plt.Rectangle((0,0), 1, 1, fill=False, edgecolor=SEQ_COLORS[1], linewidth=2, transform=ax2.transAxes)
    ax2.add_patch(rect)

    ax3 = fig.add_subplot(gs[1, 0])
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    im = ax3.imshow(cm_pct, cmap="Greens", vmin=0, vmax=100)
    ax3.set_xticks([0,1,2]); ax3.set_yticks([0,1,2])
    ax3.set_xticklabels(LABELS); ax3.set_yticklabels(LABELS)
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("True"); ax3.set_title("Confusion matrix (%)")
    plt.colorbar(im, ax=ax3, fraction=0.046)
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, f"{cm_pct[i,j]:.1f}%\n({cm[i,j]})", ha="center", va="center",
                     fontsize=8, color="white" if cm_pct[i,j] > 55 else "black")

    ax4 = fig.add_subplot(gs[1, 1])
    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)
    x_cls = np.arange(len(LABELS)); width = 0.25
    for i, m in enumerate(["precision", "recall", "f1-score"]):
        ax4.bar(x_cls + i*width, [report[l][m] for l in LABELS], width,
                label=m.replace("-score",""), color=COLORS[i], alpha=0.85, edgecolor="white")
    ax4.set_xticks(x_cls + width); ax4.set_xticklabels(LABELS)
    ax4.set_ylabel("Score"); ax4.set_title("Per-class precision, recall, F1")
    ax4.set_ylim(0, 1.05); ax4.legend(fontsize=8); ax4.grid(axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    y_true_bin = label_binarize(y_true, classes=[0,1,2]); y_prob_arr = np.array(y_prob)
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_prob_arr[:,i])
            auc = roc_auc_score(y_true_bin[:,i], y_prob_arr[:,i])
            ax5.plot(fpr, tpr, color=color, lw=1.5, label=f"{label} (AUC={auc:.3f})")
        except Exception: pass
    ax5.plot([0,1],[0,1],"k--",lw=1,alpha=0.4); ax5.set_xlabel("False positive rate")
    ax5.set_ylabel("True positive rate"); ax5.set_title("ROC curves (one-vs-rest)")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 0])
    band_colors = ["#4C72B0","#534AB7","#55A868","#DD8452","#D85A30"]
    bars_b = ax6.bar(BAND_NAMES, mean_band_weights, color=band_colors, alpha=0.85, edgecolor="white")
    ax6.set_ylabel("Mean attention weight"); ax6.set_title("Band attention weights\n(learned from EEG data)")
    ax6.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars_b, mean_band_weights):
        ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax7 = fig.add_subplot(gs[2, 1:])
    frontal_ch = {"AF3","F7","F3","F4","F8","AF4"}; temporal_ch = {"T7","T8"}
    parietal_ch = {"P7","O1","O2","P8"}
    ch_colors = ["#534AB7" if ch in frontal_ch else "#DD8452" if ch in temporal_ch
                 else "#55A868" if ch in parietal_ch else "#4C72B0" for ch in CHANNEL_NAMES]
    bars_c = ax7.bar(range(N_CHANNELS), mean_chan_weights, color=ch_colors, alpha=0.85, edgecolor="white")
    ax7.set_xticks(range(N_CHANNELS)); ax7.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=8)
    ax7.set_ylabel("Mean attention weight")
    ax7.set_title("Channel attention weights (purple=frontal, green=parietal, orange=temporal, blue=central)")
    ax7.grid(axis="y", alpha=0.3)

    ax8 = fig.add_subplot(gs[3, :])
    comparison_models = [("xgboost_results.npz","XGBoost","#888780"),
                          ("bilstm_results.npz","BiLSTM+Attention","#534AB7"),
                          ("tcn_results.npz","TCN","#D85A30"),
                          ("transformer_results.npz","Transformer","#7F77DD")]
    n_folds = len(fold_accs); x_ = np.arange(n_folds)
    n_bars = len(comparison_models) + 1; w = 0.8 / n_bars; plotted_count = 0
    for fname, mname, mcolor in comparison_models:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True); accs = data["fold_accs"][:n_folds]
            offset = (plotted_count - n_bars/2 + 0.5) * w
            ax8.bar(x_[:len(accs)] + offset, accs*100, w,
                    label=f"{mname} ({accs.mean()*100:.2f}%)", color=mcolor, alpha=0.8, edgecolor="white")
            plotted_count += 1
    offset = (plotted_count - n_bars/2 + 0.5) * w
    ax8.bar(x_ + offset, fold_accs*100, w, label=f"BiLSTM+CB ({fold_accs.mean()*100:.2f}%)",
            color=SEQ_COLORS[1], alpha=0.8, edgecolor="white")
    ax8.set_xticks(x_); ax8.set_xticklabels(unique_subjs[:n_folds], rotation=45, ha="right", fontsize=7)
    ax8.set_ylabel("Accuracy (%)"); ax8.set_title("Complete model comparison  all 6 models")
    ax8.legend(fontsize=8); ax8.grid(axis="y", alpha=0.3); ax8.set_ylim(0, 100)

    ax9 = fig.add_subplot(gs[4, 0])
    for i, losses in enumerate(train_losses_all[:]):
        ax9.plot(losses, alpha=0.7, linewidth=1, label=f"Fold {i+1}")
    ax9.set_xlabel("Epoch"); ax9.set_ylabel("Cross-entropy loss")
    ax9.set_title("Training loss"); ax9.legend(fontsize=7); ax9.grid(alpha=0.3)

    ax10 = fig.add_subplot(gs[4, 1])
    ax10.plot(range(1, len(fold_f1s)+1), fold_f1s, "o-", color="#1D9E75", linewidth=1.5, markersize=4)
    ax10.axhline(fold_f1s.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean {fold_f1s.mean():.4f}")
    ax10.set_xlabel("Fold"); ax10.set_ylabel("Macro F1"); ax10.set_title("Macro F1 per fold")
    ax10.legend(fontsize=8); ax10.grid(alpha=0.3)

    ax11 = fig.add_subplot(gs[4, 2])
    ax11.plot(range(1, len(fold_kappas)+1), fold_kappas, "s-", color=SEQ_COLORS[1], linewidth=1.5, markersize=4)
    ax11.axhline(fold_kappas.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean {fold_kappas.mean():.4f}")
    ax11.set_xlabel("Fold"); ax11.set_ylabel("Cohen Kappa"); ax11.set_title("Cohens Kappa per fold")
    ax11.legend(fontsize=8); ax11.grid(alpha=0.3)

    out_path = os.path.join(RESULTS_DIR, "bilstm_cb_attention_charts.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Charts saved  {out_path}")


def save_results(fold_accs, fold_f1s, fold_kappas, y_true, y_pred, mean_band_weights, mean_chan_weights):
    out = os.path.join(RESULTS_DIR, "bilstm_cb_attention_results.npz")
    np.savez(out, fold_accs=fold_accs, fold_f1s=fold_f1s, fold_kappas=fold_kappas,
             y_true=y_true, y_pred=y_pred, mean_band_weights=mean_band_weights,
             mean_chan_weights=mean_chan_weights, band_names=np.array(BAND_NAMES),
             channel_names=np.array(CHANNEL_NAMES), model_name=np.array("BiLSTM+CB-Attention"))
    print(f"  Results saved  {out}")


def print_summary(fold_accs, fold_f1s, fold_kappas, y_true, y_pred, mean_band_weights, mean_chan_weights):
    print("=" * 65)
    print("  BiLSTM + Channel-Band Attention  Final Results")
    print("=" * 65)
    print(f"  Mean Accuracy  : {fold_accs.mean()*100:.2f}%  +-{fold_accs.std()*100:.2f}%")
    print(f"  Best Fold      : {fold_accs.max()*100:.2f}%")
    print(f"  Worst Fold     : {fold_accs.min()*100:.2f}%")
    print(f"  Mean F1 (macro): {fold_f1s.mean():.4f}")
    print(f"  Mean Kappa     : {fold_kappas.mean():.4f}")
    for fname, mname in [("xgboost_results.npz","XGBoost"),("bilstm_results.npz","BiLSTM+Attention"),
                          ("tcn_results.npz","TCN"),("transformer_results.npz","Transformer")]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            prev  = np.load(path, allow_pickle=True)
            delta = fold_accs.mean() - prev["fold_accs"].mean()
            print(f"\n  vs {mname}: {delta*100:+.2f}%")
    print("\n  Band attention weights (mean across all folds):")
    for name, w in zip(BAND_NAMES, mean_band_weights):
        print(f"    {name:<7}: {w:.4f}  {'X'*int(w*40)}")
    print("\n  Top-5 channel attention weights:")
    for ch, w in sorted(zip(CHANNEL_NAMES, mean_chan_weights), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {ch:<5}: {w:.4f}  {'X'*int(w*40)}")
    print("\n  Per-class report:")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))


def main():
    print(f"Device: {DEVICE}\n")
    print("Loading dataset...")
    X, y_class, y_cont, subject_ids, subjects, games = load_dataset()
    print(f"  X shape : {X.shape}  (77 = {N_PSD} PSD + {N_ENG} eng + {N_FTI} FTI_z)")
    print(f"  Subjects: {len(np.unique(subject_ids))}")
    print(f"  Feature structure: {N_BANDS} bands x {N_CHANNELS} channels = {N_PSD} PSD features")
    print(f"  Seq len : {SEQ_LEN} windows = {SEQ_LEN*2}s of EEG context\n")

    (fold_accs, fold_f1s, fold_kappas,
     y_true, y_pred, y_prob,
     train_losses, mean_band_weights, mean_chan_weights) = run_loso(X, y_class, subject_ids, subjects)

    print_summary(fold_accs, fold_f1s, fold_kappas, y_true, y_pred, mean_band_weights, mean_chan_weights)
    save_results(fold_accs, fold_f1s, fold_kappas, y_true, y_pred, mean_band_weights, mean_chan_weights)
    plot_all(fold_accs, fold_f1s, fold_kappas, y_true, y_pred, y_prob,
             subjects, subject_ids, train_losses, mean_band_weights, mean_chan_weights)
    print("\n  Done. All 6 models complete. Run compare_results.py for final table.")


if __name__ == "__main__":
    main()
