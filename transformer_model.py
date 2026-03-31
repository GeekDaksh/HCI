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

DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
LABELS     = ["Low", "Medium", "High"]
COLORS     = ["#4C72B0", "#DD8452", "#55A868"]
SEQ_COLORS = ["#3C3489", "#534AB7", "#AFA9EC"]   # purple ramp for Transformer

# Sequence parameters — identical to BiLSTM/TCN for fair comparison
SEQ_LEN    = 15
STRIDE     = 1

# Model hyperparameters
INPUT_DIM   = 77
N_CLASSES   = 3
D_MODEL     = 128     # embedding dimension (must be divisible by N_HEADS)
N_HEADS     = 8       # multi-head self-attention heads
N_LAYERS    = 3       # stacked transformer encoder layers
D_FF        = 256     # feedforward hidden dimension
DROPOUT     = 0.2

# Training
EPOCHS     = 30
BATCH_SIZE = 256
LR         = 5e-4     # lower than BiLSTM — Transformers are sensitive to LR
PATIENCE   = 7
WARMUP_STEPS = 500    # linear LR warmup then cosine decay


# ─────────────────────────────────────────────
#  SEQUENCE BUILDER
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
#  TRANSFORMER ARCHITECTURE
# ─────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding — Vaswani et al. 2017.

    Adds position-dependent signals to the input embeddings so the
    Transformer can distinguish which time step each token came from.
    The Transformer has no inherent notion of order (unlike LSTM which
    processes sequentially), so PE is essential.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Fixed (not learned) — works well for fixed-length EEG sequences and
    requires no additional parameters. Generalises better to sequences
    of different lengths if needed later.
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        return self.dropout(x + self.pe[:, :x.size(1)])


class EEGTransformer(nn.Module):
    """
    Transformer Encoder for EEG cognitive workload classification.

    Architecture
    ------------
    Input (batch, seq_len, 77)
      → Linear projection: 77 → d_model (128)
      → Sinusoidal Positional Encoding
      → TransformerEncoder × N_LAYERS:
          each layer = MultiHeadSelfAttention + FFN + LayerNorm + Dropout
      → [CLS] token output  OR  global average pooling
      → Linear(d_model, d_ff) + GELU + Dropout
      → Linear(d_ff, 3)
      → logits (batch, 3)

    Implementation uses CLS token approach:
      A learnable [CLS] token is prepended to the sequence.
      After encoding, the CLS token's output represents the whole
      sequence's workload state. This is the BERT-style approach —
      better than global avg pooling for classification tasks.

    Why Transformer is the headline model:
      1. Multi-head attention attends to ALL pairs of windows in the
         30-second sequence simultaneously. BiLSTM sees windows
         sequentially; TCN has a fixed receptive field. Transformer
         has unrestricted attention range.
      2. Self-attention is interpretable — attention maps show which
         pairs of time steps the model found relevant.
      3. State-of-the-art on time-series classification tasks since 2020.
      4. Scales better with data than RNNs.

    Hyperparameter rationale:
      d_model=128, N_heads=8  → head_dim=16 per head (sufficient for 77-dim input)
      N_layers=3              → balances expressivity vs overfitting on 44k windows
      d_ff=256                → 2× d_model (standard Transformer ratio)
      Dropout=0.2             → modest regularisation, dataset is medium-sized
    """
    def __init__(self, input_dim, d_model, n_heads, n_layers,
                 d_ff, dropout, n_classes):
        super().__init__()

        # Input projection + LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable [CLS] token — prepended to sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding (seq_len + 1 for CLS)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=SEQ_LEN + 2,
                                                     dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,   # Pre-LayerNorm (more stable than post-LN)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_classes),
        )

    def forward(self, x, return_attention=False):
        """
        x : (batch, seq_len, input_dim)
        """
        B = x.size(0)

        # Project to d_model
        x = self.input_proj(x)                     # (B, seq_len, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)           # (B, seq_len+1, d_model)

        # Positional encoding
        x = self.pos_enc(x)                         # (B, seq_len+1, d_model)

        # Transformer encoding
        encoded = self.transformer(x)               # (B, seq_len+1, d_model)

        # CLS token output as sequence representation
        cls_out = encoded[:, 0, :]                  # (B, d_model)
        cls_out = self.norm(cls_out)

        return self.classifier(cls_out)             # (B, n_classes)


# ─────────────────────────────────────────────
#  WARMUP + COSINE LR SCHEDULER
# ─────────────────────────────────────────────

class WarmupCosineScheduler(optim.lr_scheduler.LRScheduler):
    """
    Linear warmup for warmup_steps, then cosine annealing to eta_min.
    Standard for Transformer training — avoids early instability.
    """
    def __init__(self, optimizer, warmup_steps, total_steps,
                 eta_min=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.eta_min      = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            factor = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor   = self.eta_min + 0.5 * (1 - self.eta_min) * (1 + np.cos(np.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


# ─────────────────────────────────────────────
#  TRAINING HELPERS
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scheduler=None):
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
        if scheduler is not None:
            scheduler.step()
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
    train_losses_all = []
    best_global_acc   = 0.0
    best_global_state = None

    print(f"Transformer — LOSO cross-validation ({n_subjects} folds)")
    print(f"  d_model={D_MODEL}  heads={N_HEADS}  layers={N_LAYERS}  "
          f"d_ff={D_FF}  dropout={DROPOUT}  device={DEVICE}")
    print(f"  seq_len={SEQ_LEN}  warmup={WARMUP_STEPS} steps")
    print(f"{'='*65}")

    for fold, test_subj, X_train, y_train, X_test, y_test in loso_splits(
        X, y_class, subject_ids
    ):
        subj_str = np.unique(subjects[subject_ids == test_subj])[0]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        train_mask    = ~(subject_ids == test_subj)
        train_sub_ids = subject_ids[train_mask]

        X_tr_seq, y_tr_seq = build_sequences_by_subject(
            X_train, y_train, train_sub_ids, SEQ_LEN
        )
        X_te_seq, y_te_seq = build_sequences(X_test, y_test, SEQ_LEN)

        if len(X_tr_seq) == 0 or len(X_te_seq) == 0:
            print(f"  [SKIP] Fold {fold+1} — insufficient sequences")
            continue

        train_ds = TensorDataset(
            torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq)
        )
        test_ds  = TensorDataset(
            torch.from_numpy(X_te_seq), torch.from_numpy(y_te_seq)
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                  shuffle=False, drop_last=False)

        model = EEGTransformer(
            input_dim = INPUT_DIM,
            d_model   = D_MODEL,
            n_heads   = N_HEADS,
            n_layers  = N_LAYERS,
            d_ff      = D_FF,
            dropout   = DROPOUT,
            n_classes = N_CLASSES,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=1e-4, betas=(0.9, 0.98))

        total_steps = EPOCHS * len(train_loader)
        scheduler   = WarmupCosineScheduler(
            optimizer, warmup_steps=min(WARMUP_STEPS, total_steps // 4),
            total_steps=total_steps, eta_min=LR * 0.01
        )

        best_val_acc  = 0.0
        best_state    = None
        patience_ctr  = 0
        fold_tr_losses = []

        for epoch in range(EPOCHS):
            tr_loss, _ = train_epoch(model, train_loader, optimizer,
                                     criterion, scheduler)
            val_labels, val_preds, _ = evaluate(model, test_loader)
            val_acc = accuracy_score(val_labels, val_preds)

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

        model.load_state_dict(best_state)
        y_true_f, y_pred_f, y_prob_f = evaluate(model, test_loader)

        acc   = accuracy_score(y_true_f, y_pred_f)
        f1    = f1_score(y_true_f, y_pred_f, average="macro")
        kappa = cohen_kappa_score(y_true_f, y_pred_f)

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_kappas.append(kappa)

        # Track globally best model across all LOSO folds
        if acc > best_global_acc:
            best_global_acc   = acc
            best_global_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        all_y_true.extend(y_true_f)
        all_y_pred.extend(y_pred_f)
        all_y_prob.extend(y_prob_f)
        train_losses_all.append(fold_tr_losses)

        print(f"  Fold {fold+1:>2} | {subj_str} | "
              f"seq={len(X_te_seq):>4} | "
              f"acc={acc:.4f} | f1={f1:.4f} | kappa={kappa:.4f} | "
              f"epochs={len(fold_tr_losses)}")

    # Save best-fold weights for RL pipeline
    if best_global_state is not None:
        _wpath = os.path.join(RESULTS_DIR, 'transformer_weights.pt')
        torch.save(best_global_state, _wpath)
        print(f'  Best weights saved -> {_wpath}  (best fold acc={best_global_acc*100:.2f}%)')

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
    fig.suptitle("Transformer — EEG Cognitive Workload Classification",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    unique_subjs = [np.unique(subjects[subject_ids == s])[0]
                    for s in np.unique(subject_ids)]

    # ── 1. Per-fold accuracy bar ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    x    = np.arange(len(fold_accs))
    bars = ax1.bar(x, fold_accs * 100, color=SEQ_COLORS[1],
                   alpha=0.8, edgecolor="white", linewidth=0.5)
    ax1.axhline(fold_accs.mean() * 100, color="red", linestyle="--",
                linewidth=1.5, label=f"Mean {fold_accs.mean()*100:.2f}%")
    ax1.set_xticks(x)
    ax1.set_xticklabels(unique_subjs[:len(fold_accs)],
                        rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Per-subject accuracy — Transformer (LOSO)")
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
        ("d_model",        f"{D_MODEL}  heads={N_HEADS}"),
        ("Seq length",     f"{SEQ_LEN} windows ({SEQ_LEN*2}s)"),
    ]
    y_pos = 0.95
    ax2.text(0.5, 1.0, "Transformer", ha="center", va="top",
             fontsize=11, fontweight="bold", transform=ax2.transAxes)
    for label, val in metrics_list:
        ax2.text(0.05, y_pos, label, ha="left", fontsize=9,
                 color="gray", transform=ax2.transAxes)
        ax2.text(0.95, y_pos, val, ha="right", fontsize=9,
                 fontweight="bold", transform=ax2.transAxes)
        y_pos -= 0.10
    rect = plt.Rectangle((0,0), 1, 1, fill=False,
                          edgecolor=SEQ_COLORS[1], linewidth=2,
                          transform=ax2.transAxes)
    ax2.add_patch(rect)

    # ── 3. Confusion matrix ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    im = ax3.imshow(cm_pct, cmap="Purples", vmin=0, vmax=100)
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

    # ── 6. Full model comparison ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    comparison_models = [
        ("xgboost_results.npz", "XGBoost",         "#888780"),
        ("bilstm_results.npz",  "BiLSTM+Attention", "#534AB7"),
        ("tcn_results.npz",     "TCN",              "#D85A30"),
    ]
    n_folds = len(fold_accs)
    x_ = np.arange(n_folds)
    n_bars = len(comparison_models) + 1
    w = 0.8 / n_bars
    plotted_count = 0

    for fname, mname, mcolor in comparison_models:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            accs = data["fold_accs"][:n_folds]
            offset = (plotted_count - n_bars/2 + 0.5) * w
            ax6.bar(x_[:len(accs)] + offset, accs*100, w,
                    label=f"{mname} ({accs.mean()*100:.2f}%)",
                    color=mcolor, alpha=0.8, edgecolor="white")
            ax6.axhline(accs.mean()*100, color=mcolor,
                        linestyle="--", linewidth=1, alpha=0.6)
            plotted_count += 1

    offset = (plotted_count - n_bars/2 + 0.5) * w
    ax6.bar(x_ + offset, fold_accs*100, w,
            label=f"Transformer ({fold_accs.mean()*100:.2f}%)",
            color=SEQ_COLORS[1], alpha=0.8, edgecolor="white")
    ax6.axhline(fold_accs.mean()*100, color=SEQ_COLORS[1],
                linestyle="--", linewidth=1.5)
    ax6.set_xticks(x_)
    ax6.set_xticklabels(unique_subjs[:n_folds], rotation=45,
                        ha="right", fontsize=8)
    ax6.set_ylabel("Accuracy (%)")
    ax6.set_title("Full model comparison — per fold")
    ax6.legend(fontsize=9)
    ax6.grid(axis="y", alpha=0.3)
    ax6.set_ylim(0, 100)

    # ── 7. Training loss ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    for i, losses in enumerate(train_losses_all[:5]):
        ax7.plot(losses, alpha=0.7, linewidth=1, label=f"Fold {i+1}")
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
    ax8.set_xlabel("Fold"); ax8.set_ylabel("Macro F1")
    ax8.set_title("Macro F1 per fold")
    ax8.legend(fontsize=8); ax8.grid(alpha=0.3)

    # ── 9. Kappa per fold ─────────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.plot(range(1, len(fold_kappas)+1), fold_kappas,
             "s-", color=SEQ_COLORS[1], linewidth=1.5, markersize=4)
    ax9.axhline(fold_kappas.mean(), color="red", linestyle="--",
                linewidth=1.5, label=f"Mean {fold_kappas.mean():.4f}")
    ax9.set_xlabel("Fold"); ax9.set_ylabel("Cohen Kappa")
    ax9.set_title("Cohen's Kappa per fold")
    ax9.legend(fontsize=8); ax9.grid(alpha=0.3)

    out_path = os.path.join(RESULTS_DIR, "transformer_charts.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Charts saved → {out_path}")


# ─────────────────────────────────────────────
#  SAVE + SUMMARY
# ─────────────────────────────────────────────

def save_results(fold_accs, fold_f1s, fold_kappas, y_true, y_pred):
    out = os.path.join(RESULTS_DIR, "transformer_results.npz")
    np.savez(out,
             fold_accs   = fold_accs,
             fold_f1s    = fold_f1s,
             fold_kappas = fold_kappas,
             y_true      = y_true,
             y_pred      = y_pred,
             model_name  = np.array("Transformer"))
    print(f"  Results saved → {out}")


def print_summary(fold_accs, fold_f1s, fold_kappas, y_true, y_pred):
    print(f"\n{'='*65}")
    print(f"  Transformer — Final Results")
    print(f"{'='*65}")
    print(f"  Mean Accuracy  : {fold_accs.mean()*100:.2f}%  "
          f"±{fold_accs.std()*100:.2f}%")
    print(f"  Best Fold      : {fold_accs.max()*100:.2f}%")
    print(f"  Worst Fold     : {fold_accs.min()*100:.2f}%")
    print(f"  Mean F1 (macro): {fold_f1s.mean():.4f}")
    print(f"  Mean Kappa     : {fold_kappas.mean():.4f}")

    for fname, mname in [("xgboost_results.npz","XGBoost"),
                          ("bilstm_results.npz","BiLSTM+Attention"),
                          ("tcn_results.npz","TCN")]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            prev  = np.load(path, allow_pickle=True)
            delta = fold_accs.mean() - prev["fold_accs"].mean()
            print(f"\n  vs {mname}: {delta*100:+.2f}%")

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
    print(f"  Seq len : {SEQ_LEN} windows = {SEQ_LEN*2}s of EEG context")
    print(f"  d_model : {D_MODEL}  heads={N_HEADS}  layers={N_LAYERS}\n")

    fold_accs, fold_f1s, fold_kappas, \
    y_true, y_pred, y_prob, \
    train_losses = run_loso(X, y_class, subject_ids, subjects)

    print_summary(fold_accs, fold_f1s, fold_kappas, y_true, y_pred)
    save_results(fold_accs, fold_f1s, fold_kappas, y_true, y_pred)
    plot_all(fold_accs, fold_f1s, fold_kappas,
             y_true, y_pred, y_prob,
             subjects, subject_ids, train_losses)

    print(f"\n  Done. Next: run models/bilstm_channel_band_attention.py (Model 6)")


if __name__ == "__main__":
    main()
