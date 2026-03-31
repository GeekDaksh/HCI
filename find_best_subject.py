# find_best_subject.py  — run from your project root
import os
import numpy as np

windows_dir = "windows"
results = []

for fname in sorted(os.listdir(windows_dir)):
    if not fname.endswith(".npz"):
        continue
    data   = np.load(os.path.join(windows_dir, fname), allow_pickle=True)
    y_cont = data["y_cont"].astype(np.float32)
    mean   = float(np.mean(y_cont))
    std    = float(np.std(y_cont))
    # score = how close mean is to 0.6, and how much it varies (higher std = more interesting)
    score  = -abs(mean - 0.6) + 0.3 * std
    results.append((score, fname, mean, std))

results.sort(reverse=True)
print(f"{'File':<30} {'Mean':>6} {'Std':>6} {'Score':>7}")
print("-" * 55)
for score, fname, mean, std in results[:10]:
    print(f"{fname:<30} {mean:>6.3f} {std:>6.3f} {score:>7.3f}")