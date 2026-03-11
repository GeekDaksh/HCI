import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import entropy

# =========================================================
# PARAMETERS
# =========================================================

FS = 128
TARGET_CHANNELS = 14
REMOVE_FIRST_SEC = 30
WINDOW_SEC = 4
OVERLAP = 0.5

FRONTAL_IDX = [2, 3, 6, 7]     # F3, F4, FC5, FC6
PARIETAL_IDX = [10, 11, 8, 9]  # P7, P8, O1, O2

# =========================================================
# SAFE Z-SCORE
# =========================================================

def safe_zscore(series):
    mean = series.mean()
    std = series.std()
    if std == 0:
        return np.zeros(len(series))
    return (series - mean) / std

# =========================================================
# FILTER
# =========================================================

def bandpass_filter(eeg, low=0.5, high=45):
    b, a = butter(4, [low/(FS/2), high/(FS/2)], btype='band')
    return filtfilt(b, a, eeg, axis=0)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def spectral_entropy(psd):
    psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-8)
    return entropy(psd_norm, axis=0)

def extract_features(window):

    freqs, psd = welch(window, fs=FS, axis=0)

    def band(f1, f2):
        idx = np.logical_and(freqs >= f1, freqs <= f2)
        return psd[idx].mean(axis=0)

    delta = band(0.5, 4)
    theta = band(4, 8)
    alpha = band(8, 13)
    beta  = band(13, 30)

    # Ratios
    theta_alpha_ratio = theta / (alpha + 1e-8)
    beta_alpha_ratio  = beta / (alpha + 1e-8)

    # Spatial Features
    frontal_theta = theta[FRONTAL_IDX].mean()
    parietal_alpha = alpha[PARIETAL_IDX].mean()

    # Spectral Entropy
    ent = spectral_entropy(psd)

    return np.hstack([
        delta, theta, alpha, beta,
        theta_alpha_ratio,
        beta_alpha_ratio,
        ent,
        [frontal_theta, parietal_alpha]
    ])

def window_and_extract(eeg):
    win_size = int(WINDOW_SEC * FS)
    step = int(win_size * (1 - OVERLAP))

    features = []
    for start in range(0, eeg.shape[0] - win_size, step):
        window = eeg[start:start + win_size]
        features.append(extract_features(window))

    return np.array(features)

# =========================================================
# WORKLOAD CONSTRUCTION
# =========================================================

def construct_workload(sam_df):

    cols = sam_df.columns

    # Case 1: Arousal + Valence
    if "arousal" in cols and "valence" in cols:
        arousal_z = safe_zscore(sam_df["arousal"].astype(float))
        valence_z = safe_zscore(sam_df["valence"].astype(float))
        workload = arousal_z - valence_z

    # Case 2: workload + horrible + calm
    elif all(col in cols for col in ["workload", "horrible", "calm"]):
        workload_z = safe_zscore(sam_df["workload"].astype(float))
        horrible_z = safe_zscore(sam_df["horrible"].astype(float))
        calm_z     = safe_zscore(sam_df["calm"].astype(float))
        workload = workload_z + horrible_z - calm_z

    else:
        raise ValueError("Unsupported SAM format")

    return workload.values

# =========================================================
# MAIN
# =========================================================

os.makedirs("processed", exist_ok=True)

subjects = [s for s in os.listdir() if s.startswith("(") and s.endswith(")")]

for sub in subjects:

    print(f"\nProcessing {sub}...")

    sam_file = os.path.join(sub, "sam_workload.csv")
    raw_base = os.path.join(sub, "Raw EEG Data", ".csv format")

    if not os.path.exists(sam_file):
        print("⚠️ sam_workload.csv missing")
        continue

    if not os.path.exists(raw_base):
        print("⚠️ EEG folder missing")
        continue

    sam_df = pd.read_csv(sam_file)

    try:
        workload_scores = construct_workload(sam_df)
    except Exception as e:
        print("⚠️ SAM format error:", e)
        continue

    X_all = []
    y_all = []

    for file in sorted(os.listdir(raw_base)):

        if not file.endswith(".csv"):
            continue

        match = re.search(r"G\d", file)
        if not match:
            continue

        game_name = match.group()

        row_idx = sam_df[sam_df["game"] == game_name].index
        if len(row_idx) == 0:
            continue

        workload = workload_scores[row_idx[0]]

        df = pd.read_csv(os.path.join(raw_base, file))
        numeric_df = df.select_dtypes(include=[np.number])
        eeg = numeric_df.iloc[:, :TARGET_CHANNELS].values

        if eeg.shape[1] != TARGET_CHANNELS:
            continue

        # Remove first 30 seconds
        eeg = eeg[int(REMOVE_FIRST_SEC * FS):]

        eeg = bandpass_filter(eeg)
        X = window_and_extract(eeg)

        if len(X) == 0:
            continue

        X_all.append(X)
        y_all.extend([workload] * len(X))

    if len(X_all) == 0:
        print("⚠️ No valid data")
        continue

    X_all = np.vstack(X_all)
    y_all = np.array(y_all)

    np.savez(
        f"processed/{sub}_features.npz",
        X=X_all,
        workload=y_all
    )

    print(f"✅ Saved {sub}")

print("\nALL SUBJECTS PROCESSED SUCCESSFULLY")
