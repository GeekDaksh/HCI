import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import entropy

# ─── Constants ────────────────────────────────────────────────────────────────

FS              = 128          # Emotiv EPOC sampling rate (Hz)
TARGET_CHANNELS = 14           # Emotiv EPOC channel count
REMOVE_FIRST_SEC = 30          # Drop first 30s (baseline/artifact-prone)
WINDOW_SEC      = 4            # 4-second analysis windows
OVERLAP         = 0.5          # 50% overlap between windows

# ─── Emotiv EPOC 14-channel order (GAMEEMO) ───────────────────────────────────
# AF3=0, F7=1, F3=2, FC5=3, T7=4, P7=5, O1=6, O2=7,
# P8=8,  T8=9, FC6=10, F4=11, F8=12, AF4=13
EMOTIV_CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7',  'P7',
    'O1',  'O2', 'P8', 'T8',  'FC6', 'F4', 'F8', 'AF4'
]

# Correct region indices based on actual Emotiv EPOC layout
FRONTAL_IDX   = [0, 1, 2, 11, 12, 13]   # AF3, F7, F3, F4, F8, AF4
TEMPORAL_IDX  = [4, 9]                   # T7, T8
PARIETAL_IDX  = [5, 6, 7, 8]            # P7, O1, O2, P8
CENTRAL_IDX   = [3, 10]                 # FC5, FC6

# ─── Frequency bands (Hz) ─────────────────────────────────────────────────────
BANDS = {
    'delta': (0.5, 4),
    'theta': (4,   8),
    'alpha': (8,  13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}


# ─── Signal processing ────────────────────────────────────────────────────────

def bandpass_filter(eeg: np.ndarray,
                    low: float = 0.5,
                    high: float = 45.0) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.
    eeg shape: (samples, channels)
    """
    nyq = FS / 2
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, eeg, axis=0)


def reject_artifacts(eeg: np.ndarray,
                     amplitude_thresh: float = 100.0) -> np.ndarray:
    """
    Simple amplitude-based artifact rejection.
    Clips samples exceeding ±100 µV (common EEG threshold).
    Does NOT remove the sample — clips to prevent feature explosion.
    """
    return np.clip(eeg, -amplitude_thresh, amplitude_thresh)


# ─── Feature extraction ───────────────────────────────────────────────────────

def spectral_entropy(psd: np.ndarray) -> np.ndarray:
    """
    Normalized spectral entropy per channel.
    psd shape: (freqs, channels)
    Returns shape: (channels,)
    """
    psd_norm = psd / (psd.sum(axis=0, keepdims=True) + 1e-8)
    return entropy(psd_norm, axis=0)


def compute_band_power(psd: np.ndarray,
                       freqs: np.ndarray,
                       f_low: float,
                       f_high: float) -> np.ndarray:
    """
    Mean PSD within a frequency band, per channel.
    Returns shape: (channels,)
    """
    idx = np.logical_and(freqs >= f_low, freqs <= f_high)
    return psd[idx].mean(axis=0)


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract spectral + spatial features from a single EEG window.

    window shape: (samples, 14_channels)

    Feature vector layout (per-channel bands + spatial summaries):
    ┌─────────────────────────────────────────────────────┐
    │ Band powers:     delta, theta, alpha, beta, gamma   │  14 ch × 5 = 70
    │ Band ratios:     theta/alpha, beta/alpha             │  14 ch × 2 = 28
    │ Spectral entropy                                     │  14 ch × 1 = 14
    │ Frontal theta   (mean across frontal channels)       │  1
    │ Parietal alpha  (mean across parietal channels)      │  1
    │ Frontal asymmetry alpha  (F4-F3 / F4+F3)            │  1
    │ Engagement index beta/(alpha+theta) frontal mean     │  1
    └─────────────────────────────────────────────────────┘
    Total: 116 features
    """
    freqs, psd = welch(window, fs=FS, axis=0, nperseg=FS)
    # psd shape: (freq_bins, 14)

    # ── Per-channel band powers ────────────────────────────────────────────
    delta = compute_band_power(psd, freqs, *BANDS['delta'])
    theta = compute_band_power(psd, freqs, *BANDS['theta'])
    alpha = compute_band_power(psd, freqs, *BANDS['alpha'])
    beta  = compute_band_power(psd, freqs, *BANDS['beta'])
    gamma = compute_band_power(psd, freqs, *BANDS['gamma'])

    # ── Band ratios (workload-sensitive) ──────────────────────────────────
    # Clip alpha to avoid division instability
    alpha_safe   = np.clip(alpha, 1e-8, None)
    theta_alpha  = np.clip(theta / alpha_safe, 0, 20)   # cognitive load marker
    beta_alpha   = np.clip(beta  / alpha_safe, 0, 20)   # alertness marker

    # ── Spectral entropy per channel ──────────────────────────────────────
    ent = spectral_entropy(psd)

    # ── Spatial summary features ──────────────────────────────────────────
    frontal_theta   = theta[FRONTAL_IDX].mean()         # task engagement
    parietal_alpha  = alpha[PARIETAL_IDX].mean()        # relaxation/attention

    # Frontal alpha asymmetry: (right - left) / (right + left)
    # F4=idx11 (right), F3=idx2 (left) — positive = approach motivation
    f4_alpha = alpha[11]
    f3_alpha = alpha[2]
    frontal_asymmetry = (f4_alpha - f3_alpha) / (f4_alpha + f3_alpha + 1e-8)

    # Engagement index: beta / (alpha + theta) on frontal channels
    frontal_alpha = alpha[FRONTAL_IDX].mean()
    frontal_beta  = beta[FRONTAL_IDX].mean()
    frontal_theta_mean = theta[FRONTAL_IDX].mean()
    engagement_idx = frontal_beta / (frontal_alpha + frontal_theta_mean + 1e-8)
    engagement_idx = np.clip(engagement_idx, 0, 20)

    return np.hstack([
        delta, theta, alpha, beta, gamma,    # 70 features
        theta_alpha, beta_alpha,             # 28 features
        ent,                                 # 14 features
        [frontal_theta,                      #  1 feature
         parietal_alpha,                     #  1 feature
         frontal_asymmetry,                  #  1 feature
         engagement_idx]                     #  1 feature
    ])   # total: 116


def window_and_extract(eeg: np.ndarray) -> np.ndarray:
    """
    Slide window over EEG recording and extract features per window.

    eeg shape:    (samples, 14)
    returns shape: (n_windows, 116)
    """
    win  = int(WINDOW_SEC * FS)           # 512 samples
    step = int(win * (1 - OVERLAP))       # 256 samples

    features = []

    for start in range(0, eeg.shape[0] - win + 1, step):
        window = eeg[start : start + win]
        feat   = extract_features(window)
        features.append(feat)

    return np.array(features) if features else np.empty((0, 116))


def normalize_subject(X: np.ndarray) -> np.ndarray:
    """
    Z-score normalization per feature across all windows for ONE subject.

    Critical for EEG: raw amplitudes vary significantly between subjects
    due to skull thickness, electrode impedance, and individual differences.
    Without this, a model learns 'who' the subject is, not their workload state.

    X shape: (n_windows, n_features)
    """
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    return (X - mean) / (std + 1e-8)


# ─── CSV loading ──────────────────────────────────────────────────────────────

def load_eeg_csv(filepath: str) -> np.ndarray | None:
    """
    Load EEG CSV and return numpy array of shape (samples, 14).

    Robust loader that handles:
    - Mixed dtype columns (string markers, timestamps)
    - Embedded second header rows
    - Extra metadata columns
    - Emotiv software export quirks
    """
    # Read with dtype=str first — let us inspect before converting
    df = pd.read_csv(filepath, dtype=str)

    # Strip whitespace from all column names
    df.columns = [c.strip() for c in df.columns]

    # ── Detect and drop embedded repeated header rows ─────────────────────
    # Emotiv exports sometimes repeat the header mid-file
    # These rows have the column name as the cell value
    first_col = df.columns[0]
    repeated_header_mask = df[first_col] == first_col
    if repeated_header_mask.any():
        print(f"  [INFO] Dropping {repeated_header_mask.sum()} repeated header rows")
        df = df[~repeated_header_mask].reset_index(drop=True)

    # ── Convert all columns to numeric, coerce bad values to NaN ──────────
    df = df.apply(pd.to_numeric, errors='coerce')

    # ── Strategy 1: exact Emotiv channel name match ───────────────────────
    cols_upper = {c.upper(): c for c in df.columns}
    emotiv_upper = [c.upper() for c in EMOTIV_CHANNELS]
    matched = [cols_upper[c] for c in emotiv_upper if c in cols_upper]

    if len(matched) == TARGET_CHANNELS:
        eeg = df[matched].values.astype(np.float64)

    # ── Strategy 2: first 14 numeric-looking columns ──────────────────────
    else:
        # Drop columns that are >50% NaN (likely string/metadata cols)
        valid_cols = [c for c in df.columns
                      if df[c].notna().mean() > 0.5]

        if len(valid_cols) < TARGET_CHANNELS:
            print(f"  [ERROR] Only {len(valid_cols)} usable columns in {os.path.basename(filepath)}")
            print(f"          All columns: {df.columns.tolist()}")
            return None

        print(f"  [WARN] Emotiv channel names not found — using first {TARGET_CHANNELS} "
              f"valid numeric cols from: {valid_cols[:TARGET_CHANNELS]}")
        eeg = df[valid_cols[:TARGET_CHANNELS]].values.astype(np.float64)

    # ── Drop rows with any NaN (from coerced bad values) ──────────────────
    nan_rows = np.isnan(eeg).any(axis=1)
    if nan_rows.sum() > 0:
        print(f"  [INFO] Dropping {nan_rows.sum()} rows with NaN values")
        eeg = eeg[~nan_rows]

    if len(eeg) == 0:
        print(f"  [ERROR] No valid rows remain in {os.path.basename(filepath)}")
        return None

    return eeg


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main(base_dir: str = ".") -> None:

    os.makedirs("windows", exist_ok=True)

    subjects = sorted([
        s for s in os.listdir(base_dir)
        if s.startswith("(") and s.endswith(")")
        and os.path.isdir(os.path.join(base_dir, s))
    ])

    print(f"Found {len(subjects)} subjects\n")

    total_windows  = 0
    skipped_games  = 0
    processed_games = 0

    for sub in subjects:

        # ── Load SAM workload labels ───────────────────────────────────────
        sam_file = os.path.join(base_dir, sub, "sam_workload.csv")
        if not os.path.exists(sam_file):
            print(f"[SKIP] {sub} — sam_workload.csv not found (run sam_workload.py first)")
            continue

        sam_df = pd.read_csv(sam_file)

        # Validate required columns exist
        required_cols = {"game", "workload_continuous", "workload_class"}
        if not required_cols.issubset(sam_df.columns):
            print(f"[SKIP] {sub} — sam_workload.csv missing columns: "
                  f"{required_cols - set(sam_df.columns)}")
            continue

        raw_base = os.path.join(base_dir, sub, "Raw EEG Data", ".csv format")
        if not os.path.exists(raw_base):
            print(f"[SKIP] {sub} — Raw EEG folder not found at {raw_base}")
            continue

        print(f"Processing {sub}...")

        subject_windows = []   # collect all windows for this subject (for normalization)
        subject_meta    = []   # game ID and label per window batch

        # ── Pass 1: load + filter all games for this subject ──────────────
        for filename in sorted(os.listdir(raw_base)):

            if not filename.lower().endswith(".csv"):
                continue

            match = re.search(r'(G\d)', filename, re.IGNORECASE)
            if not match:
                continue

            game_id = match.group(1).upper()

            # Get labels for this game
            row = sam_df[sam_df["game"] == game_id]
            if row.empty:
                print(f"  [SKIP] {game_id} — no SAM label found")
                skipped_games += 1
                continue

            workload_cont  = row["workload_continuous"].values[0]
            workload_class = row["workload_class"].values[0]

            if pd.isna(workload_cont) or pd.isna(workload_class):
                print(f"  [SKIP] {game_id} — NaN workload label")
                skipped_games += 1
                continue

            # Load EEG
            filepath = os.path.join(raw_base, filename)
            eeg = load_eeg_csv(filepath)
            if eeg is None:
                skipped_games += 1
                continue

            # ── Preprocessing pipeline ────────────────────────────────────
            # 1. Remove first 30s (settling period)
            eeg = eeg[int(REMOVE_FIRST_SEC * FS):]

            if eeg.shape[0] < WINDOW_SEC * FS:
                print(f"  [SKIP] {game_id} — too short after trimming: "
                      f"{eeg.shape[0]} samples")
                skipped_games += 1
                continue

            # 2. Amplitude artifact rejection (clip to ±100 µV)
            eeg = reject_artifacts(eeg, amplitude_thresh=100.0)

            # 3. Bandpass filter (0.5–45 Hz)
            eeg = bandpass_filter(eeg, low=0.5, high=45.0)

            # 4. Windowed feature extraction
            X = window_and_extract(eeg)

            if len(X) == 0:
                print(f"  [SKIP] {game_id} — no windows extracted")
                skipped_games += 1
                continue

            subject_windows.append(X)
            subject_meta.append({
                "game":                game_id,
                "n_windows":           len(X),
                "workload_continuous": workload_cont,
                "workload_class":      int(workload_class),
            })

            print(f"  {game_id}: {eeg.shape[0]} samples → "
                  f"{len(X)} windows | "
                  f"workload={workload_cont:.2f} (class {int(workload_class)})")

        if not subject_windows:
            print(f"  [WARN] No valid games for {sub}\n")
            continue

        # ── Pass 2: per-subject normalization ─────────────────────────────
        # Stack ALL windows for this subject, normalize together,
        # then split back out per game
        all_X = np.vstack(subject_windows)      # (total_windows, 116)
        all_X_norm = normalize_subject(all_X)   # Z-score per feature

        # Split normalized array back into per-game chunks
        cursor = 0
        for meta, X_raw in zip(subject_meta, subject_windows):

            n    = meta["n_windows"]
            X    = all_X_norm[cursor : cursor + n]
            cursor += n

            y_cont  = np.full(n, meta["workload_continuous"])
            y_class = np.full(n, meta["workload_class"], dtype=np.int32)

            # Save windowed features + both label types
            out_path = f"windows/{sub}_{meta['game']}.npz"
            np.savez(
                out_path,
                X       = X,             # (n_windows, 116) — normalized features
                y_cont  = y_cont,        # (n_windows,)     — continuous workload 1–9
                y_class = y_class,       # (n_windows,)     — class 0/1/2
                subject = sub,
                game    = meta["game"],
            )

            total_windows  += n
            processed_games += 1

        print(f"  Subject total: {sum(m['n_windows'] for m in subject_meta)} windows "
              f"from {len(subject_meta)} games\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Preprocessing complete")
    print(f"  Processed games : {processed_games}")
    print(f"  Skipped games   : {skipped_games}")
    print(f"  Total windows   : {total_windows}")
    print(f"  Feature dim     : 116")
    print(f"  Saved to        : windows/")


if __name__ == "__main__":
    main(base_dir=".")