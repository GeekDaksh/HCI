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

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS     = ["Low", "Medium", "High"]
COLORS     = ["#4C72B0", "#DD8452", "#55A868"]
SEQ_COLORS = ["#534AB7", "#1D9E75", "#D85A30"]

# Sequence parameters
SEQ_LEN    = 15     # 15 consecutive windows = 30 seconds of EEG context
STRIDE     = 1      # predict every window (dense prediction)

# Model hyperparameters
INPUT_DIM  = 77     # feature vector size
HIDDEN_DIM = 128    # BiLSTM hidden size per direction (256 total)
NUM_LAYERS = 2      # stacked BiLSTM layers
DROPOUT    = 0.3    # dropout between layers
N_CLASSES  = 3

# Training
EPOCHS     = 30
BATCH_SIZE = 256
LR         = 1e-3
PATIENCE   = 7      # early stopping patience


# ─────────────────────────────────────────────
#  SEQUENCE BUILDER
# ─────────────────────────────────────────────

def build_sequences(X, y, seq_len=SEQ_LEN, stride=STRIDE):
    """
    Convert flat window array into overlapping sequences.

    X : (n_windows, n_features)
    y : (n_windows,)

    Returns
    -------
    X_seq : (n_sequences, seq_len, n_features)
    y_seq : (n_sequences,)  — label of the LAST window in each sequence
    """
    n = len(X)
    X_seq, y_seq = [], []
    for i in range(0, n - seq_len + 1, stride):
        X_seq.append(X[i: i + seq_len])
        y_seq.append(y[i + seq_len - 1])   # label of last window
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


def build_sequences_by_subject(X, y, subject_ids, seq_len=SEQ_LEN):
    """
    Build sequences WITHOUT crossing subject boundaries.
    Sequences are built independently per subject so the model
    never sees a sequence that starts with one subject and ends
    with another.
    """
    X_all, y_all = [], []
    for subj in np.unique(subject_ids):
        mask   = subject_ids == subj
        X_s, y_s = build_sequences(X[mask], y[mask], seq_len)
        X_all.append(X_s)
        y_all.append(y_s)
    return np.vstack(X_all), np.concatenate(y_all)


# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """
    Bahdanau Additive Attention — Bahdanau et al. 2015.

    Learns which hidden states in the BiLSTM sequence are most
    informative for the current workload prediction.

    Given hidden states H = [h1, h2, ..., h_T] from BiLSTM:
      score_t = v^T tanh(W h_t)          score each state
      alpha_t = softmax(scores)           attention weights
      context = sum(alpha_t * h_t)        weighted context vector

    The attention weights are interpretable — they show which
    windows in the 30-second sequence drove the prediction.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H):
        """
        H : (batch, seq_len, hidden_dim)
        Returns context (batch, hidden_dim) and weights (batch, seq_len)
        """
        scores  = self.v(torch.tanh(self.W(H)))   # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)     # (batch, seq_len, 1)
        context = (weights * H).sum(dim=1)         # (batch, hidden_dim)
        return context, weights.squeeze(-1)        # weights for visualisation


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Bahdanau Attention.

    Architecture
    ------------
    Input (batch, seq_len, 77)
      → LayerNorm
      → BiLSTM × NUM_LAYERS  (hidden_dim × 2 output per step)
      → Dropout
      → BahdanauAttention    (context vector = weighted sum of hidden states)
      → LayerNorm
      → Linear(hidden_dim×2, 128) + ReLU + Dropout
      → Linear(128, 3)
      → output logits (batch, 3)

    Why bidirectional:
      Forward LSTM sees windows 1→15 (past context).
      Backward LSTM sees windows 15→1 (future context within the sequence).
      Both directions concatenated give richer hidden states.

    Why attention on top:
      The final hidden state of LSTM only summarises the sequence from
      the last step's perspective. Attention lets the classifier use
      information from any point in the 30-second window, weighted by
      how informative each moment was.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, n_classes):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)

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
        self.norm      = nn.LayerNorm(hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, return_attention=False):
        """
        x : (batch, seq_len, input_dim)
        """
        x = self.input_norm(x)
        H, _ = self.lstm(x)           # (batch, seq_len, hidden_dim*2)
        H = self.dropout(H)
        context, attn_weights = self.attention(H)
        context = self.norm(context)
        logits  = self.classifier(context)   # (batch, n_classes)

        if return_attention:
            return logits, attn_weights
        return logits


# ─────────────────────────────────────────────
#  TRAINING HELPERS
# ─────────────────────────────────────────────

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
def evaluate(model, loader):
    model.eval()
    preds, probs, labels = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        logits  = model(X_batch)
        p       = torch.softmax(logits, dim=1).cpu().numpy()
        preds.extend(logits.argmax(1).cpu().numpy())
        probs.extend(p)
        labels.extend(y_batch.numpy())
    return np.array(labels), np.array(preds), np.array(probs)


# ─────────────────────────────────────────────
#  LOSO TRAINING LOOP
# ─────────────────────────────────────────────

def run_loso(X, y_class, subject_ids, subjects):
    n_subjects = len(np.unique(subject_ids))
    fold_accs, fold_f1s, fold_kappas = [], [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    train_losses_all, val_accs_all = [], []

    print(f"BiLSTM + Attention — LOSO cross-validation ({n_subjects} folds)")
    print(f"  seq_len={SEQ_LEN}  hidden={HIDDEN_DIM}×2  layers={NUM_LAYERS}  "
          f"dropout={DROPOUT}  epochs={EPOCHS}  device={DEVICE}")
    print(f"{'='*65}")

    for fold, test_subj, X_train, y_train, X_test, y_test in loso_splits(
        X, y_class, subject_ids
    ):
        subj_str = np.unique(subjects[subject_ids == test_subj])[0]

        # ── Standardize ──
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # ── Build sequences ──
        # Get subject_ids for training subjects only
        train_mask    = ~(subject_ids == test_subj)
        train_sub_ids = subject_ids[train_mask]

        X_tr_seq, y_tr_seq = build_sequences_by_subject(
            X_train, y_train, train_sub_ids, SEQ_LEN
        )
        X_te_seq, y_te_seq = build_sequences(X_test, y_test, SEQ_LEN)

        if len(X_tr_seq) == 0 or len(X_te_seq) == 0:
            print(f"  [SKIP] Fold {fold+1} — insufficient sequences")
            continue

        # ── DataLoaders ──
        train_ds = TensorDataset(
            torch.from_numpy(X_tr_seq),
            torch.from_numpy(y_tr_seq)
        )
        test_ds  = TensorDataset(
            torch.from_numpy(X_te_seq),
            torch.from_numpy(y_te_seq)
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  drop_last=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                  shuffle=False, drop_last=False)

        # ── Model ──
        model = BiLSTMAttention(
            input_dim  = INPUT_DIM,
            hidden_dim = HIDDEN_DIM,
            num_layers = NUM_LAYERS,
            dropout    = DROPOUT,
            n_classes  = N_CLASSES,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )

        # ── Training with early stopping ──
        best_val_acc  = 0.0
        best_state    = None
        patience_ctr  = 0
        fold_tr_losses = []

        for epoch in range(EPOCHS):
            tr_loss, tr_acc = train_epoch(model, train_loader,
                                          optimizer, criterion)
            val_labels, val_preds, _ = evaluate(model, test_loader)
            val_acc = accuracy_score(val_labels, val_preds)

            scheduler.step(val_acc)
            fold_tr_losses.append(tr_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

        # ── Evaluate best model ──
        model.load_state_dict(best_state)
        y_true_f, y_pred_f, y_prob_f = evaluate(model, test_loader)

        acc   = accuracy_score(y_true_f, y_pred_f)
        f1    = f1_score(y_true_f, y_pred_f, average="macro")
        kappa = cohen_kappa_score(y_true_f, y_pred_f)

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_kappas.append(kappa)
        all_y_true.extend(y_true_f)
        all_y_pred.extend(y_pred_f)
        all_y_prob.extend(y_prob_f)
        train_losses_all.append(fold_tr_losses)
        val_accs_all.append(best_val_acc)

        print(f"  Fold {fold+1:>2} | {subj_str} | "
              f"seq={len(X_te_seq):>4} | "
              f"acc={acc:.4f} | f1={f1:.4f} | kappa={kappa:.4f} | "
              f"epochs={len(fold_tr_losses)}")

    return (
        np.array(fold_accs), np.array(fold_f1s), np.array(fold_kappas),
        np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob),
        train_losses_all
    )


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────

def plot_all(fold_accs, fold_f1s, fold_kappas,
             y_true, y_pred, y_prob,
             subjects, subject_ids, train_losses_all):

    fig = plt.figure(figsize=(20, 22))
    fig.suptitle("BiLSTM + Bahdanau Attention — EEG Cognitive Workload",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    unique_subjs = [np.unique(subjects[subject_ids == s])[0]
                    for s in np.unique(subject_ids)]

    # ── 1. Per-fold accuracy bar ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x    = np.arange(len(fold_accs))
    bars = ax1.bar(x, fold_accs * 100, color=SEQ_COLORS[0],
                   alpha=0.8, edgecolor="white", linewidth=0.5)
    ax1.axhline(fold_accs.mean() * 100, color="red", linestyle="--",
                linewidth=1.5, label=f"Mean {fold_accs.mean()*100:.2f}%")
    ax1.set_xticks(x)
    ax1.set_xticklabels(unique_subjs[:len(fold_accs)],
                        rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Per-subject accuracy — BiLSTM + Attention (LOSO)")
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
    metrics_list = [
        ("Mean Accuracy",  f"{fold_accs.mean()*100:.2f}%"),
        ("Std",            f"±{fold_accs.std()*100:.2f}%"),
        ("Best Fold",      f"{fold_accs.max()*100:.2f}%"),
        ("Worst Fold",     f"{fold_accs.min()*100:.2f}%"),
        ("Mean F1 (macro)",f"{fold_f1s.mean():.4f}"),
        ("Mean Kappa",     f"{fold_kappas.mean():.4f}"),
        ("Seq length",     f"{SEQ_LEN} windows ({SEQ_LEN*2}s)"),
    ]
    y_pos = 0.95
    ax2.text(0.5, 1.0, "BiLSTM + Attention",
             ha="center", va="top", fontsize=11, fontweight="bold",
             transform=ax2.transAxes)
    for label, val in metrics_list:
        ax2.text(0.05, y_pos, label, ha="left", fontsize=9,
                 color="gray", transform=ax2.transAxes)
        ax2.text(0.95, y_pos, val, ha="right", fontsize=9,
                 fontweight="bold", transform=ax2.transAxes)
        y_pos -= 0.11
    rect = plt.Rectangle((0,0), 1, 1, fill=False,
                          edgecolor=SEQ_COLORS[0], linewidth=2,
                          transform=ax2.transAxes)
    ax2.add_patch(rect)

    # ── 3. Confusion matrix ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cm     = confusion_matrix(y_true, y_pred)
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
    report = classification_report(y_true, y_pred,
                                   target_names=LABELS, output_dict=True)
    metric_names = ["precision", "recall", "f1-score"]
    x_cls = np.arange(len(LABELS))
    width = 0.25
    for i, m in enumerate(metric_names):
        vals = [report[l][m] for l in LABELS]
        ax4.bar(x_cls + i*width, vals, width,
                label=m.replace("-score",""), color=COLORS[i],
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
    y_true_bin = label_binarize(y_true, classes=[0,1,2])
    y_prob_arr = np.array(y_prob)
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_prob_arr[:,i])
            auc = roc_auc_score(y_true_bin[:,i], y_prob_arr[:,i])
            ax5.plot(fpr, tpr, color=color, lw=1.5,
                     label=f"{label} (AUC={auc:.3f})")
        except Exception:
            pass
    ax5.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
    ax5.set_xlabel("False positive rate")
    ax5.set_ylabel("True positive rate")
    ax5.set_title("ROC curves (one-vs-rest)")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # ── 6. Comparison with XGBoost ───────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    xgb_path = os.path.join(RESULTS_DIR, "xgboost_results.npz")
    if os.path.exists(xgb_path):
        xgb_data = np.load(xgb_path, allow_pickle=True)
        xgb_accs = xgb_data["fold_accs"]
        n = min(len(fold_accs), len(xgb_accs))
        x_ = np.arange(n)
        w  = 0.35
        ax6.bar(x_ - w/2, xgb_accs[:n]*100, w, label="XGBoost",
                color="#888780", alpha=0.8, edgecolor="white")
        ax6.bar(x_ + w/2, fold_accs[:n]*100, w, label="BiLSTM+Attention",
                color=SEQ_COLORS[0], alpha=0.8, edgecolor="white")
        ax6.axhline(xgb_accs[:n].mean()*100, color="#888780",
                    linestyle="--", linewidth=1, alpha=0.7)
        ax6.axhline(fold_accs[:n].mean()*100, color=SEQ_COLORS[0],
                    linestyle="--", linewidth=1.5)
        ax6.set_xticks(x_)
        ax6.set_xticklabels(unique_subjs[:n], rotation=45,
                            ha="right", fontsize=8)
        ax6.set_ylabel("Accuracy (%)")
        ax6.set_title(
            f"Per-fold: XGBoost ({xgb_accs[:n].mean()*100:.2f}%) "
            f"vs BiLSTM+Attn ({fold_accs[:n].mean()*100:.2f}%)"
        )
        ax6.legend(fontsize=9)
        ax6.grid(axis="y", alpha=0.3)
        ax6.set_ylim(0, 100)
    else:
        ax6.text(0.5, 0.5, "Run xgboost_model.py first for comparison",
                 ha="center", va="center", transform=ax6.transAxes,
                 fontsize=11, color="gray")
        ax6.axis("off")

    # ── 7. Training loss curves (sample of folds) ────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    for i, losses in enumerate(train_losses_all[:5]):
        ax7.plot(losses, alpha=0.7, linewidth=1,
                 label=f"Fold {i+1}")
    ax7.set_xlabel("Epoch")
    ax7.set_ylabel("Cross-entropy loss")
    ax7.set_title("Training loss (first 5 folds)")
    ax7.legend(fontsize=7)
    ax7.grid(alpha=0.3)

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

    out_path = os.path.join(RESULTS_DIR, "bilstm_charts.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Charts saved → {out_path}")


# ─────────────────────────────────────────────
#  SAVE + SUMMARY
# ─────────────────────────────────────────────

def save_results(fold_accs, fold_f1s, fold_kappas, y_true, y_pred):
    out = os.path.join(RESULTS_DIR, "bilstm_results.npz")
    np.savez(out,
             fold_accs   = fold_accs,
             fold_f1s    = fold_f1s,
             fold_kappas = fold_kappas,
             y_true      = y_true,
             y_pred      = y_pred,
             model_name  = np.array("BiLSTM+Attention"))
    print(f"  Results saved → {out}")


def print_summary(fold_accs, fold_f1s, fold_kappas, y_true, y_pred):
    print(f"\n{'='*65}")
    print(f"  BiLSTM + Attention — Final Results")
    print(f"{'='*65}")
    print(f"  Mean Accuracy  : {fold_accs.mean()*100:.2f}%  "
          f"±{fold_accs.std()*100:.2f}%")
    print(f"  Best Fold      : {fold_accs.max()*100:.2f}%")
    print(f"  Worst Fold     : {fold_accs.min()*100:.2f}%")
    print(f"  Mean F1 (macro): {fold_f1s.mean():.4f}")
    print(f"  Mean Kappa     : {fold_kappas.mean():.4f}")

    xgb_path = os.path.join(RESULTS_DIR, "xgboost_results.npz")
    if os.path.exists(xgb_path):
        xgb = np.load(xgb_path, allow_pickle=True)
        delta = fold_accs.mean() - xgb["fold_accs"].mean()
        print(f"\n  vs XGBoost baseline:")
        print(f"  Accuracy gain  : {delta*100:+.2f}%")
        print(f"  XGBoost mean   : {xgb['fold_accs'].mean()*100:.2f}%")
        print(f"  BiLSTM mean    : {fold_accs.mean()*100:.2f}%")

    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred,
                                target_names=LABELS, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix:")
    print(f"  {'':>10}  " + "  ".join(f"{l:>10}" for l in LABELS))
    for i, row in enumerate(cm):
        print(f"  {LABELS[i]:>10}  " + "  ".join(f"{v:>10}" for v in row))


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}\n")
    print("Loading dataset...")
    X, y_class, y_cont, subject_ids, subjects, games = load_dataset()
    print(f"  X shape : {X.shape}")
    print(f"  Subjects: {len(np.unique(subject_ids))}")
    print(f"  Seq len : {SEQ_LEN} windows = {SEQ_LEN*2}s of EEG context\n")

    fold_accs, fold_f1s, fold_kappas, \
    y_true, y_pred, y_prob, \
    train_losses = run_loso(X, y_class, subject_ids, subjects)

    print_summary(fold_accs, fold_f1s, fold_kappas, y_true, y_pred)
    save_results(fold_accs, fold_f1s, fold_kappas, y_true, y_pred)
    plot_all(fold_accs, fold_f1s, fold_kappas,
             y_true, y_pred, y_prob,
             subjects, subject_ids, train_losses)

    print(f"\n  Done. Next: run models/tcn_model.py")


if __name__ == "__main__":
    main()
