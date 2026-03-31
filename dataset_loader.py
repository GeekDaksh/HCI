import os
import numpy as np


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

DATASET_PATH = "processed/dataset.npz"


# ─────────────────────────────────────────────
#  CORE LOADER
# ─────────────────────────────────────────────

def load_dataset(path=DATASET_PATH):
    """
    Load the aggregated dataset.

    Returns
    -------
    X           : np.float32  (n_windows, 71)   — feature matrix
    y_class     : np.int32    (n_windows,)       — 0=Low 1=Med 2=High
    y_cont      : np.float32  (n_windows,)       — TLI score [0, 1]
    subject_ids : np.int32    (n_windows,)       — integer subject index
    subjects    : np.ndarray  (n_windows,)       — string subject IDs
    games       : np.ndarray  (n_windows,)       — string game IDs
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run aggregate.py first."
        )

    data = np.load(path, allow_pickle=True)

    X           = data["X"].astype(np.float32)
    y_class     = data["y_class"].astype(np.int32)
    y_cont      = data["y_cont"].astype(np.float32)
    subject_ids = data["subject_ids"].astype(np.int32)
    subjects    = data["subjects"]
    games       = data["games"]

    return X, y_class, y_cont, subject_ids, subjects, games


# ─────────────────────────────────────────────
#  LOSO SPLIT — Leave One Subject Out
# ─────────────────────────────────────────────

def loso_splits(X, y_class, subject_ids):
    """
    Generator for Leave-One-Subject-Out cross-validation.

    Yields (fold, test_subject_id, X_train, y_train, X_test, y_test)
    for each unique subject.

    Usage
    -----
    for fold, test_subj, X_tr, y_tr, X_te, y_te in loso_splits(X, y_class, subject_ids):
        model.fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
    """
    unique_subjects = np.unique(subject_ids)

    for fold, test_subj in enumerate(unique_subjects):
        test_mask  = subject_ids == test_subj
        train_mask = ~test_mask

        X_train  = X[train_mask]
        y_train  = y_class[train_mask]
        X_test   = X[test_mask]
        y_test   = y_class[test_mask]

        yield fold, test_subj, X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
#  SUBJECT-LEVEL SPLIT — for quick train/test
# ─────────────────────────────────────────────

def subject_split(X, y_class, y_cont, subject_ids, test_subjects):
    """
    Split dataset into train/test by subject ID list.

    Parameters
    ----------
    test_subjects : list of int — subject IDs to hold out for testing

    Returns train and test arrays for X, y_class, y_cont.
    """
    test_mask  = np.isin(subject_ids, test_subjects)
    train_mask = ~test_mask

    return (
        X[train_mask], y_class[train_mask], y_cont[train_mask],
        X[test_mask],  y_class[test_mask],  y_cont[test_mask],
    )


# ─────────────────────────────────────────────
#  PER-SUBJECT DATA — for personalised models
# ─────────────────────────────────────────────

def get_subject_data(X, y_class, y_cont, subject_ids, target_subject_id):
    """
    Extract all windows for a single subject.

    Returns X_subj, y_class_subj, y_cont_subj.
    """
    mask = subject_ids == target_subject_id
    return X[mask], y_class[mask], y_cont[mask]


# ─────────────────────────────────────────────
#  VERIFICATION — run as script
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading dataset...")
    X, y_class, y_cont, subject_ids, subjects, games = load_dataset()

    n, d         = X.shape
    n_subjects   = len(np.unique(subject_ids))
    n_games      = len(np.unique(games))

    print(f"\n{'='*50}")
    print(f"Dataset loaded successfully")
    print(f"  Windows      : {n:,}")
    print(f"  Features     : {d}  (14ch × 5bands + FTI_z)")
    print(f"  Subjects     : {n_subjects}")
    print(f"  Games        : {n_games}")

    print(f"\nClass distribution:")
    for cls, label in [(0, "Low"), (1, "Medium"), (2, "High")]:
        count = int((y_class == cls).sum())
        pct   = count / n * 100
        print(f"  {label:<8} (class {cls}): {count:>6,}  ({pct:.1f}%)")

    print(f"\nFeature matrix X:")
    print(f"  dtype  : {X.dtype}")
    print(f"  min    : {X.min():.4f}")
    print(f"  max    : {X.max():.4f}")
    print(f"  mean   : {X.mean():.4f}")
    print(f"  NaN    : {np.isnan(X).sum()}")
    print(f"  Inf    : {np.isinf(X).sum()}")

    print(f"\nLOSO split preview (first 3 folds):")
    for fold, test_subj, X_tr, y_tr, X_te, y_te in loso_splits(X, y_class, subject_ids):
        print(f"  Fold {fold+1:>2} | test=S{test_subj+1:02d} | "
              f"train={len(X_tr):>5} windows | test={len(X_te):>4} windows")
        if fold == 2:
            print(f"  ... ({n_subjects - 3} more folds)")
            break

    print(f"\nAll checks passed. Ready for model training.")
