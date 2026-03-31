# check_ycont_distribution.py
import os
import numpy as np

windows_dir = "windows"
all_y = []

for fname in os.listdir(windows_dir):
    if not fname.endswith(".npz"):
        continue
    data = np.load(os.path.join(windows_dir, fname), allow_pickle=True)
    all_y.extend(data["y_cont"].astype(np.float32).tolist())

all_y = np.array(all_y)
print(f"Global y_cont stats across all sessions:")
print(f"  Mean   : {all_y.mean():.3f}")
print(f"  Median : {np.median(all_y):.3f}")
print(f"  Std    : {all_y.std():.3f}")
print(f"  25th % : {np.percentile(all_y, 25):.3f}")
print(f"  75th % : {np.percentile(all_y, 75):.3f}")
print(f"  Min    : {all_y.min():.3f}")
print(f"  Max    : {all_y.max():.3f}")