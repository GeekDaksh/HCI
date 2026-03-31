import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import entropy
from scipy.linalg import sqrtm

# ─── Constants ────────────────────────────────────────────────────────────────

FS               = 128
TARGET_CHANNELS  = 14
REMOVE_FIRST_SEC = 30
WINDOW_SEC       = 4
OVERLAP          = 0.5

# ─── Emotiv EPOC 14-channel order (GAMEEMO) ───────────────────────────────────
# AF3=0, F7=1, F3=2, FC5=3, T7=4, P7=5, O1=6, O2=7,
# P8=8,  T8=9, FC6=10, F4=11, F8=12, AF4=13
EMOTIV_CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7',  'P7',
    'O1',  'O2', 'P8', 'T8',  'FC6', 'F4', 'F8', 'AF4'
]

FRONTAL_IDX  = [0, 1, 2, 11, 12, 13]   # AF3, F7, F3, F4, F8, AF4
TEMPORAL_IDX = [4, 9]                   # T7, T8
PARIETAL_IDX = [5, 6, 7, 8]            # P7, O1, O2, P8
CENTRAL_IDX  = [3, 10]                 # FC5, FC6

# Symmetric electrode pairs (left_idx, right_idx) for asymmetry features
ASYM_PAIRS = [
    (0, 13),   # AF3 ↔ AF4
    (1, 12),   # F7  ↔ F8
    (2, 11),   # F3  ↔ F4
    (3, 10),   # FC5 ↔ FC6
    (4, 9),    # T7  ↔ T8
    (5, 8),    # P7  ↔ P8
    (6, 7),    # O1  ↔ O2
]

BANDS = {
    'delta': (0.5, 4),
    'theta': (4,   8),
    'alpha': (8,  13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

# Occipital artifact feature indices in the 116-feature vector
# delta_O1=6, delta_O2=7, entropy_O1=104, entropy_O2=105
ARTIFACT_FEATURES = [6, 7, 104, 105]


# ─── Signal processing ────────────────────────────────────────────────────────

def bandpass_filter(eeg, low=0.5, high=45.0):
    """Zero-phase Butterworth bandpass. eeg: (samples, channels)"""
    nyq  = FS / 2
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, eeg, axis=0)


def reject_artifacts(eeg, amplitude_thresh=100.0):
    """Clip ±100 µV amplitude artifacts."""
    return np.clip(eeg, -amplitude_thresh, amplitude_thresh)


# ── NEW: Euclidean Alignment ─────────────────────────────────────────────────

def euclidean_align(windows):
    """
    Euclidean Alignment (EA) — He & Wu 2020.

    Whitens each subject's EEG covariance matrix to identity before
    feature extraction. This is the single most effective preprocessing
    step for cross-subject EEG generalisation.

    What it removes:
      - Individual differences in electrode amplitude (skull thickness,
        electrode contact quality, hair density)
      - Subject-specific channel correlation structure (brain anatomy)

    Result: two subjects experiencing the same workload will have more
    similar EEG distributions after alignment.

    windows: list of (samples, channels) arrays — ALL games for ONE subject
    Returns: list of aligned (samples, channels) arrays
    """
    # Stack all samples to estimate mean covariance
    all_samples = np.vstack(windows)                    # (total_samples, 14)
    n_ch        = all_samples.shape[1]

    # Compute covariance matrix (channels × channels)
    C    = np.cov(all_samples.T)                        # (14, 14)
    C   += np.eye(n_ch) * 1e-6                          # regularise

    # Whitening matrix R^{-1/2}
    try:
        C_sqrt     = sqrtm(C).real
        C_inv_sqrt = np.linalg.inv(C_sqrt)
    except Exception:
        diag       = np.sqrt(np.diag(C))
        C_inv_sqrt = np.diag(1.0 / (diag + 1e-8))

    # Apply to each window: X_aligned = X @ C_inv_sqrt.T
    aligned = [w @ C_inv_sqrt.T for w in windows]
    return aligned


# ─── Spectral feature extraction (original 116 features) ─────────────────────

def spectral_entropy_feat(psd):
    """Normalized spectral entropy per channel. psd: (freqs, channels)"""
    psd_norm = psd / (psd.sum(axis=0, keepdims=True) + 1e-8)
    return entropy(psd_norm, axis=0)


def compute_band_power(psd, freqs, f_low, f_high):
    """Mean PSD within band, per channel. Returns (channels,)"""
    idx = np.logical_and(freqs >= f_low, freqs <= f_high)
    return psd[idx].mean(axis=0)


def extract_spectral_features(window):
    """
    Extract 116 spectral features from (samples, 14) window.

    Band powers:     delta/theta/alpha/beta/gamma × 14 = 70
    Band ratios:     theta/alpha, beta/alpha × 14       = 28
    Spectral entropy × 14                               = 14
    Spatial:         frontal_theta, parietal_alpha,
                     frontal_asymmetry, engagement_idx  =  4
    Total: 116
    """
    freqs, psd = welch(window, fs=FS, axis=0, nperseg=FS)

    delta = compute_band_power(psd, freqs, *BANDS['delta'])
    theta = compute_band_power(psd, freqs, *BANDS['theta'])
    alpha = compute_band_power(psd, freqs, *BANDS['alpha'])
    beta  = compute_band_power(psd, freqs, *BANDS['beta'])
    gamma = compute_band_power(psd, freqs, *BANDS['gamma'])

    alpha_safe  = np.clip(alpha, 1e-8, None)
    theta_alpha = np.clip(theta / alpha_safe, 0, 20)
    beta_alpha  = np.clip(beta  / alpha_safe, 0, 20)

    ent = spectral_entropy_feat(psd)

    frontal_theta  = theta[FRONTAL_IDX].mean()
    parietal_alpha = alpha[PARIETAL_IDX].mean()
    f4_alpha       = alpha[11];  f3_alpha = alpha[2]
    frontal_asym   = (f4_alpha - f3_alpha) / (f4_alpha + f3_alpha + 1e-8)
    eng_idx        = np.clip(
        beta[FRONTAL_IDX].mean() /
        (alpha[FRONTAL_IDX].mean() + theta[FRONTAL_IDX].mean() + 1e-8),
        0, 20
    )

    return np.hstack([
        delta, theta, alpha, beta, gamma,
        theta_alpha, beta_alpha,
        ent,
        [frontal_theta, parietal_alpha, frontal_asym, eng_idx]
    ])   # 116 features


# ── NEW: Connectivity feature extraction (92 features) ───────────────────────

def extract_connectivity_features(window):
    """
    Extract 92 connectivity features from (samples, 14) window.

    Hemisphere log-asymmetry per band: 5 bands × 7 pairs = 35
      Captures left/right hemisphere power differences per frequency.
      Well-validated workload marker — high beta right-asymmetry
      correlates with cognitive engagement.

    Frontal-parietal correlation per band: 5 bands = 5
      Captures executive control ↔ working memory coordination.
      Increases significantly under cognitive load.

    Cross-channel correlations (subgraph):
      Frontal intra-group:      6C2 = 15
      Parietal intra-group:     4C2 =  6
      Frontal × Parietal cross: 6×4 = 24
      Temporal × Frontal mean:      =  2
      Temporal pair, Central pair,
      Central × Parietal, Temporal × Parietal = 5
    Cross-channel total: 52

    Grand total: 35 + 5 + 52 = 92 features
    """
    feats = []

    # Transpose to (channels, samples) for per-channel operations
    w = window.T   # (14, samples)

    # Pre-compute band-filtered signals per channel
    nyq = FS / 2
    band_sigs = {}
    for band, (lo, hi) in BANDS.items():
        b, a = butter(4, [lo/nyq, hi/nyq], btype='band')
        band_sigs[band] = filtfilt(b, a, w, axis=1)   # (14, samples)

    # ── Hemisphere log-asymmetry per band: 5 × 7 = 35 ───────────────────
    for band in BANDS:
        sig = band_sigs[band]
        for l_idx, r_idx in ASYM_PAIRS:
            lp = np.mean(sig[l_idx] ** 2) + 1e-12
            rp = np.mean(sig[r_idx] ** 2) + 1e-12
            feats.append(float(np.log(rp) - np.log(lp)))

    # ── Frontal-parietal correlation per band: 5 ─────────────────────────
    for band in BANDS:
        sig  = band_sigs[band]
        fp   = sig[FRONTAL_IDX].mean(axis=0)    # mean frontal signal
        pp   = sig[PARIETAL_IDX].mean(axis=0)   # mean parietal signal
        corr = np.corrcoef(fp, pp)[0, 1]
        feats.append(float(np.nan_to_num(corr)))

    # ── Cross-channel correlations (52) ──────────────────────────────────
    # Use broadband (unfiltered) channel signals for correlation
    fi = FRONTAL_IDX
    pi = PARIETAL_IDX

    # Frontal intra-group: 6C2 = 15
    for i in range(len(fi)):
        for j in range(i+1, len(fi)):
            c = np.corrcoef(w[fi[i]], w[fi[j]])[0, 1]
            feats.append(float(np.nan_to_num(c)))

    # Parietal intra-group: 4C2 = 6
    for i in range(len(pi)):
        for j in range(i+1, len(pi)):
            c = np.corrcoef(w[pi[i]], w[pi[j]])[0, 1]
            feats.append(float(np.nan_to_num(c)))

    # Frontal × Parietal cross: 6×4 = 24
    for f in fi:
        for p in pi:
            c = np.corrcoef(w[f], w[p])[0, 1]
            feats.append(float(np.nan_to_num(c)))

    # Temporal × Frontal mean: 2
    fm = w[fi].mean(axis=0)
    for t in TEMPORAL_IDX:
        c = np.corrcoef(w[t], fm)[0, 1]
        feats.append(float(np.nan_to_num(c)))

    # Temporal pair (1), Central pair (1), Central×Parietal (2),
    # Temporal×Parietal mean (1) = 5
    pm = w[pi].mean(axis=0)
    feats.append(float(np.nan_to_num(
        np.corrcoef(w[TEMPORAL_IDX[0]], w[TEMPORAL_IDX[1]])[0, 1])))
    feats.append(float(np.nan_to_num(
        np.corrcoef(w[CENTRAL_IDX[0]], w[CENTRAL_IDX[1]])[0, 1])))
    for ci in CENTRAL_IDX:
        feats.append(float(np.nan_to_num(np.corrcoef(w[ci], pm)[0, 1])))
    feats.append(float(np.nan_to_num(
        np.corrcoef(w[TEMPORAL_IDX[0]], pm)[0, 1])))

    return np.array(feats[:92], dtype=np.float32)


def extract_features(window):
    """
    Full feature extraction: 116 spectral + 92 connectivity = 208 features.
    window: (samples, 14_channels)
    """
    spec = extract_spectral_features(window)   # 116
    conn = extract_connectivity_features(window)  # 92
    return np.hstack([spec, conn]).astype(np.float32)   # 208


def window_and_extract(eeg):
    """
    Slide window over EEG and extract features per window.
    eeg: (samples, 14)  →  returns (n_windows, 208)
    """
    win  = int(WINDOW_SEC * FS)
    step = int(win * (1 - OVERLAP))
    features = []
    for start in range(0, eeg.shape[0] - win + 1, step):
        feat = extract_features(eeg[start: start + win])
        features.append(feat)
    return np.array(features) if features else np.empty((0, 208))


def normalize_subject(X):
    """Z-score per feature across all windows for one subject."""
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    return (X - mean) / (std + 1e-8)


# ─── CSV loading (unchanged) ──────────────────────────────────────────────────

def load_eeg_csv(filepath):
    """Load Emotiv EPOC CSV → (samples, 14) float64 or None."""
    df = pd.read_csv(filepath, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    first_col = df.columns[0]
    repeated  = df[first_col] == first_col
    if repeated.any():
        df = df[~repeated].reset_index(drop=True)

    df = df.apply(pd.to_numeric, errors='coerce')

    cols_upper  = {c.upper(): c for c in df.columns}
    emotiv_upper = [c.upper() for c in EMOTIV_CHANNELS]
    matched     = [cols_upper[c] for c in emotiv_upper if c in cols_upper]

    if len(matched) == TARGET_CHANNELS:
        eeg = df[matched].values.astype(np.float64)
    else:
        valid_cols = [c for c in df.columns if df[c].notna().mean() > 0.5]
        if len(valid_cols) < TARGET_CHANNELS:
            print(f"  [ERROR] Only {len(valid_cols)} usable columns in "
                  f"{os.path.basename(filepath)}")
            return None
        print(f"  [WARN] Channel names not found — using first {TARGET_CHANNELS} "
              f"valid cols")
        eeg = df[valid_cols[:TARGET_CHANNELS]].values.astype(np.float64)

    nan_rows = np.isnan(eeg).any(axis=1)
    if nan_rows.sum() > 0:
        eeg = eeg[~nan_rows]

    return eeg if len(eeg) > 0 else None


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main(base_dir="."):

    os.makedirs("windows", exist_ok=True)

    subjects = sorted([
        s for s in os.listdir(base_dir)
        if s.startswith("(") and s.endswith(")")
        and os.path.isdir(os.path.join(base_dir, s))
    ])

    print(f"Found {len(subjects)} subjects")
    print(f"NEW: Euclidean Alignment per subject")
    print(f"NEW: 208 features (116 spectral + 92 connectivity)\n")

    total_windows   = 0
    skipped_games   = 0
    processed_games = 0

    for sub in subjects:

        sam_file = os.path.join(base_dir, sub, "sam_workload.csv")
        if not os.path.exists(sam_file):
            print(f"[SKIP] {sub} — sam_workload.csv not found")
            continue

        sam_df = pd.read_csv(sam_file)
        required = {"game", "workload_continuous", "workload_class"}
        if not required.issubset(sam_df.columns):
            print(f"[SKIP] {sub} — missing columns: {required - set(sam_df.columns)}")
            continue

        raw_base = os.path.join(base_dir, sub, "Raw EEG Data", ".csv format")
        if not os.path.exists(raw_base):
            print(f"[SKIP] {sub} — Raw EEG folder not found")
            continue

        print(f"Processing {sub}...")

        # ── Pass 1: load all EEG windows for this subject ─────────────────
        subject_windows = []   # list of (samples, 14) per game
        subject_meta    = []

        for filename in sorted(os.listdir(raw_base)):
            if not filename.lower().endswith(".csv"):
                continue
            match = re.search(r'(G\d)', filename, re.IGNORECASE)
            if not match:
                continue
            game_id = match.group(1).upper()

            row = sam_df[sam_df["game"] == game_id]
            if row.empty:
                print(f"  [SKIP] {game_id} — no SAM label")
                skipped_games += 1
                continue

            workload_cont  = row["workload_continuous"].values[0]
            workload_class = row["workload_class"].values[0]

            if pd.isna(workload_cont) or pd.isna(workload_class):
                print(f"  [SKIP] {game_id} — NaN label")
                skipped_games += 1
                continue

            eeg = load_eeg_csv(os.path.join(raw_base, filename))
            if eeg is None:
                skipped_games += 1
                continue

            eeg = eeg[int(REMOVE_FIRST_SEC * FS):]
            if eeg.shape[0] < WINDOW_SEC * FS:
                print(f"  [SKIP] {game_id} — too short")
                skipped_games += 1
                continue

            eeg = reject_artifacts(eeg, 100.0)
            eeg = bandpass_filter(eeg, 0.5, 45.0)

            subject_windows.append(eeg)
            subject_meta.append({
                "game":                game_id,
                "workload_continuous": workload_cont,
                "workload_class":      int(workload_class),
                "eeg":                 eeg,
            })

        if not subject_windows:
            print(f"  [WARN] No valid games for {sub}\n")
            continue

        # ── Pass 2: Euclidean Alignment across all games ──────────────────
        aligned_windows = euclidean_align(subject_windows)

        # ── Pass 3: feature extraction per game ──────────────────────────
        all_X    = []
        all_meta = []

        for meta, eeg_aligned in zip(subject_meta, aligned_windows):
            X = window_and_extract(eeg_aligned)
            if len(X) == 0:
                skipped_games += 1
                continue

            # Suppress occipital artifact features
            X[:, ARTIFACT_FEATURES] = 0.0
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            all_X.append(X)
            all_meta.append({**meta, "n_windows": len(X)})

            print(f"  {meta['game']}: {eeg_aligned.shape[0]} samples → "
                  f"{len(X)} windows | "
                  f"workload={meta['workload_continuous']:.2f} "
                  f"(class {meta['workload_class']})")

        if not all_X:
            continue

        # ── Pass 4: per-subject Z-score normalisation ─────────────────────
        all_X_stacked = np.vstack(all_X)
        all_X_norm    = normalize_subject(all_X_stacked)

        # Split back per game and save
        cursor = 0
        for meta in all_meta:
            n = meta["n_windows"]
            X = all_X_norm[cursor: cursor + n]
            cursor += n

            y_cont  = np.full(n, meta["workload_continuous"])
            y_class = np.full(n, meta["workload_class"], dtype=np.int32)

            out_path = f"windows/{sub}_{meta['game']}.npz"
            np.savez(
                out_path,
                X       = X.astype(np.float32),
                y_cont  = y_cont,
                y_class = y_class,
                subject = sub,
                game    = meta["game"],
            )
            total_windows  += n
            processed_games += 1

        print(f"  EA applied + Z-scored | feat_dim=208 | "
              f"windows={sum(m['n_windows'] for m in all_meta)}\n")

    print("=" * 60)
    print(f"Preprocessing complete")
    print(f"  Processed games : {processed_games}")
    print(f"  Skipped games   : {skipped_games}")
    print(f"  Total windows   : {total_windows}")
    print(f"  Feature dim     : 208  (116 spectral + 92 connectivity)  [NEW]")
    print(f"  EA applied      : yes — per-subject covariance whitening [NEW]")
    print(f"  Saved to        : windows/")


if __name__ == "__main__":
    main()
