import os
import numpy as np

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

WINDOWS_DIR = "windows"
OUTPUT_DIR  = "processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.npz")

# Minimum windows a session must have to be included
MIN_WINDOWS = 30


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted([f for f in os.listdir(WINDOWS_DIR) if f.endswith(".npz")])
    if not files:
        print(f"[ERROR] No .npz files found in {WINDOWS_DIR}/")
        print("  Run preprocess.py first.")
        return

    print(f"Found {len(files)} session files in {WINDOWS_DIR}/\n")

    X_list       = []
    y_cont_list  = []
    y_class_list = []
    subject_list = []
    game_list    = []
    skipped      = []

    for fname in files:
        path = os.path.join(WINDOWS_DIR, fname)
        data = np.load(path, allow_pickle=True)

        X       = data["X"]
        y_cont  = data["y_cont"]
        y_class = data["y_class"]
        subject = str(data["subject"])
        game    = str(data["game"])
        n       = len(X)

        # Skip sessions with too few windows
        if n < MIN_WINDOWS:
            print(f"  [SKIP] {fname:<35} only {n} windows (< {MIN_WINDOWS})")
            skipped.append(fname)
            continue

        X_list.append(X)
        y_cont_list.append(y_cont)
        y_class_list.append(y_class)
        subject_list.extend([subject] * n)
        game_list.extend([game] * n)

        dist = {c: int((y_class == c).sum()) for c in [0, 1, 2]}
        print(f"  {fname:<35} {n:>4} windows  "
              f"L:{dist[0]:>3} M:{dist[1]:>3} H:{dist[2]:>3}  "
              f"subject={subject}  game={game}")

    if not X_list:
        print("\n[ERROR] No sessions passed the minimum window threshold.")
        return

    # ── Stack all sessions ──
    X_all       = np.vstack(X_list).astype(np.float32)
    y_cont_all  = np.concatenate(y_cont_list).astype(np.float32)
    y_class_all = np.concatenate(y_class_list).astype(np.int32)
    subjects    = np.array(subject_list)
    games       = np.array(game_list)

    # ── Build integer subject ID map ──
    unique_subjects = sorted(set(subject_list))
    subject_to_int  = {s: i for i, s in enumerate(unique_subjects)}
    subject_ids     = np.array(
        [subject_to_int[s] for s in subject_list], dtype=np.int32
    )

    # ── Sanity check ──
    nan_count = int(np.isnan(X_all).sum())
    inf_count = int(np.isinf(X_all).sum())
    if nan_count > 0 or inf_count > 0:
        print(f"\n[WARN] {nan_count} NaN + {inf_count} Inf values in X — replacing with 0")
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Save ──
    np.savez(
        OUTPUT_FILE,
        X           = X_all,
        y_cont      = y_cont_all,
        y_class     = y_class_all,
        subjects    = subjects,
        subject_ids = subject_ids,
        games       = games,
    )

    # ── Summary ──
    total = len(X_all)
    print(f"\n{'='*60}")
    print(f"Saved → {OUTPUT_FILE}")
    print(f"  Total windows : {total:,}")
    print(f"  Feature dim   : {X_all.shape[1]}  (14ch × 5bands + FTI_z)")
    print(f"  Subjects      : {len(unique_subjects)}")
    print(f"  Sessions kept : {len(X_list)}  |  skipped: {len(skipped)}")

    print(f"\nClass distribution (global):")
    for cls, label in [(0, "Low"), (1, "Medium"), (2, "High")]:
        count = int((y_class_all == cls).sum())
        pct   = count / total * 100
        bar   = "█" * int(pct / 2)
        print(f"  {label:<8} (class {cls}): {count:>6,}  ({pct:.1f}%)  {bar}")

    print(f"\nPer-subject window counts:")
    for s in unique_subjects:
        mask = subjects == s
        n    = int(mask.sum())
        dist = {c: int((y_class_all[mask] == c).sum()) for c in [0, 1, 2]}
        print(f"  {s}: {n:>5} windows  "
              f"L:{dist[0]:>4} M:{dist[1]:>4} H:{dist[2]:>4}")

    print(f"\nWorkload score (y_cont) stats:")
    print(f"  min={y_cont_all.min():.4f}  max={y_cont_all.max():.4f}  "
          f"mean={y_cont_all.mean():.4f}  std={y_cont_all.std():.4f}")

    if skipped:
        print(f"\nSkipped sessions ({len(skipped)}):")
        for f in skipped:
            print(f"  {f}")

    print(f"\nNext step: run dataset_loader.py to verify, then model training.")


if __name__ == "__main__":
    main()
