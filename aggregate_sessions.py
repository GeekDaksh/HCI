import os
import numpy as np

WINDOWS_DIR = "windows"
OUTPUT_DIR  = "processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.npz")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(WINDOWS_DIR) if f.endswith(".npz")])

    if not files:
        print(f"[ERROR] No .npz files in {WINDOWS_DIR}/  — run preprocess.py first")
        return

    print(f"Found {len(files)} session files\n")

    X_list, y_cont_list, y_class_list = [], [], []
    subject_list, game_list = [], []

    for fname in files:
        data    = np.load(os.path.join(WINDOWS_DIR, fname), allow_pickle=True)
        X       = data["X"]
        y_cont  = data["y_cont"]
        y_class = data["y_class"]
        subject = str(data["subject"])
        game    = str(data["game"])
        n       = len(X)

        X_list.append(X);        y_cont_list.append(y_cont)
        y_class_list.append(y_class)
        subject_list.extend([subject] * n)
        game_list.extend([game] * n)

        print(f"  {fname:<32} {n:>5} windows | "
              f"workload={y_cont[0]:.2f} (class {y_class[0]})")

    X_all       = np.vstack(X_list)
    y_cont_all  = np.concatenate(y_cont_list)
    y_class_all = np.concatenate(y_class_list).astype(np.int32)
    subjects    = np.array(subject_list)
    games       = np.array(game_list)

    unique_subjects = sorted(set(subject_list))
    subject_to_int  = {s: i for i, s in enumerate(unique_subjects)}
    subject_ids     = np.array([subject_to_int[s] for s in subject_list], dtype=np.int32)

    nan_count = np.isnan(X_all).sum()
    inf_count = np.isinf(X_all).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"\n[WARN] {nan_count} NaN + {inf_count} Inf — replacing with 0")
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    np.savez(OUTPUT_FILE, X=X_all, y_cont=y_cont_all, y_class=y_class_all,
             subjects=subjects, subject_ids=subject_ids, games=games)

    print(f"\n{'='*60}")
    print(f"Saved → {OUTPUT_FILE}  |  Shape: {X_all.shape}")
    print(f"\nClass distribution:")
    for cls, label in [(0,"Low"),(1,"Medium"),(2,"High")]:
        count = (y_class_all == cls).sum()
        pct   = count / len(y_class_all) * 100
        print(f"  {label:<8} (class {cls}): {count:>5} ({pct:.1f}%)  {'█'*int(pct/2)}")

    print(f"\nPer-subject breakdown:")
    for s in unique_subjects:
        mask = subjects == s
        dist = {c: (y_class_all[mask]==c).sum() for c in [0,1,2]}
        print(f"  {s}: {mask.sum():>5} windows | L={dist[0]} M={dist[1]} H={dist[2]}")

    print(f"\nWorkload: [{y_cont_all.min():.2f}, {y_cont_all.max():.2f}] "
          f"mean={y_cont_all.mean():.2f} std={y_cont_all.std():.2f}")


if __name__ == "__main__":
    main()