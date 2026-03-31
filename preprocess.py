import os
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import zscore
from scipy.signal import welch
from scipy.linalg import sqrtm

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

RAW_DIR     = "."                 # run from HCI/ — subject folders are (S01), (S02) etc.
OUTPUT_DIR  = "windows"           # created automatically inside HCI/
SFREQ       = 128                 # Emotiv EPOC sampling rate (Hz)
WINDOW_SEC  = 2                   # window length in seconds
OVERLAP     = 0.75                # 75% overlap
WIN_SAMPLES = int(WINDOW_SEC * SFREQ)          # 256 samples per window
STEP        = int(WIN_SAMPLES * (1 - OVERLAP)) # 64 samples step
AMP_THRESH  = 150                 # epoch rejection threshold (microvolts)
BASELINE_SEC = 120                # first 2 minutes = resting baseline

# Emotiv EPOC 14-channel layout
ALL_CHANNELS = [
    "AF3","AF4","F3","F4","F7","F8",
    "FC5","FC6","O1","O2","P7","P8","T7","T8"
]

# TLI channel clusters (Gevins & Smith 2003)
FRONTAL_CH  = ["F3", "F4"]          # closest to Fz available on EPOC
PARIETAL_CH = ["P7", "P8"]          # closest to Pz available on EPOC

# Frequency bands
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  12),
    "beta":  (13, 30),
    "gamma": (31, 45),
}

# ─────────────────────────────────────────────
#  SIGNAL PROCESSING HELPERS
# ─────────────────────────────────────────────

def bandpass(data, lo, hi, fs, order=4):
    """Apply zero-phase Butterworth bandpass filter. data: (n_channels, n_samples)"""
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def notch(data, freq, fs, quality=30):
    """Apply zero-phase notch filter at freq Hz."""
    nyq = fs / 2.0
    b, a = iirnotch(freq / nyq, quality)
    return filtfilt(b, a, data, axis=1)


def load_csv(path):
    """
    Load one session CSV.
    Returns DataFrame with columns = channel names, rows = samples.
    Drops any non-channel columns silently.
    Cleans stray non-numeric characters before casting.
    """
    df = pd.read_csv(path, header=0)

    # Strip whitespace from column names
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Keep only known EEG channels
    present = [c for c in ALL_CHANNELS if c in df.columns]
    missing = [c for c in ALL_CHANNELS if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing channels in {path}: {missing}")
    df = df[present].copy()

    # Remove stray non-numeric characters (e.g. '12.3m' → '12.3')
    df = df.apply(lambda col: col.astype(str)
                               .str.replace(r"[^0-9.\-eE]", "", regex=True))
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    return df, present


def compute_psd(epoch, fs, bands):
    """
    epoch: (n_channels, n_samples)
    Returns flat feature vector: mean band power per channel per band.
    Shape: (n_channels * n_bands,)
    Order: ch0_delta, ch0_theta, ..., ch0_gamma, ch1_delta, ...
    """
    n_ch = epoch.shape[0]
    features = []
    for ch in range(n_ch):
        freqs, psd = welch(epoch[ch], fs=fs, nperseg=fs)  # 1Hz resolution
        for band, (lo, hi) in bands.items():
            idx = np.logical_and(freqs >= lo, freqs <= hi)
            features.append(np.mean(psd[idx]))
    return np.array(features, dtype=np.float32)


def reject_epoch(epoch, threshold):
    """Return True if epoch should be REJECTED (peak-to-peak > threshold)."""
    for ch in range(epoch.shape[0]):
        if (epoch[ch].max() - epoch[ch].min()) > threshold:
            return True
    return False





def compute_engineered_features(epoch, fs, ch_names, frontal_ch, parietal_ch):
    """
    6 engineered workload features per window — all validated in literature.

    1. TLI ratio         theta_frontal / alpha_parietal  (Gevins 2003)
    2. Beta/Alpha ratio  beta_frontal  / alpha_frontal   (engagement)
    3. Theta/Beta ratio  theta_frontal / beta_frontal    (inversely ~ effort)
    4. Frontal FAA       log(F4_alpha) - log(F3_alpha)   (arousal asymmetry)
    5. Engagement index  beta / (alpha + theta) frontal  (Pope 1995 NASA EI)
    6. Parietal alpha    mean alpha P7/P8                (suppression marker)

    Returns np.array of shape (6,)
    """
    freqs, psd = welch(epoch, fs=fs, nperseg=fs, axis=1)   # (n_ch, n_freqs)

    def band_mean(ch_list, lo, hi):
        idx = np.logical_and(freqs >= lo, freqs <= hi)
        cidx = [ch_names.index(c) for c in ch_list if c in ch_names]
        if not cidx:
            return 1e-8
        return float(np.mean(psd[cidx][:, idx]))

    def ch_band(ch, lo, hi):
        idx = np.logical_and(freqs >= lo, freqs <= hi)
        if ch not in ch_names:
            return 1e-8
        ci = ch_names.index(ch)
        return float(np.mean(psd[ci, idx]))

    th_f  = band_mean(frontal_ch,  4,  8)   # frontal theta
    al_f  = band_mean(frontal_ch,  8, 12)   # frontal alpha
    be_f  = band_mean(frontal_ch, 13, 30)   # frontal beta
    al_p  = band_mean(parietal_ch, 8, 12)   # parietal alpha

    # 1. TLI
    tli = th_f / max(al_p, 1e-8)

    # 2. Beta/Alpha (frontal engagement)
    beta_alpha = be_f / max(al_f, 1e-8)

    # 3. Theta/Beta (inversely related to effort)
    theta_beta = th_f / max(be_f, 1e-8)

    # 4. Frontal Alpha Asymmetry (FAA)
    f4_al = ch_band("F4", 8, 12)
    f3_al = ch_band("F3", 8, 12)
    faa   = np.log(max(f4_al, 1e-8)) - np.log(max(f3_al, 1e-8))

    # 5. Engagement Index (Pope 1995 NASA formula)
    ei = be_f / max(al_f + th_f, 1e-8)

    # 6. Parietal Alpha Suppression (raw mean — high = low workload)
    pal = al_p

    feats = np.array([tli, beta_alpha, theta_beta, faa, ei, pal],
                     dtype=np.float32)
    # Clip to prevent extreme outliers
    feats = np.clip(feats, -20, 20)
    return feats

# ─────────────────────────────────────────────
#  EUCLIDEAN ALIGNMENT (He & Wu 2020)
# ─────────────────────────────────────────────

def euclidean_align(data):
    """
    Euclidean Alignment — He & Wu 2020.

    Whitens the EEG covariance matrix to identity so that
    individual differences in electrode amplitude, skull thickness,
    and channel correlation structure are removed before feature
    extraction. The single most effective step for cross-subject
    EEG generalisation.

    data : (n_channels, n_samples) — full session after filtering
    Returns: (n_channels, n_samples) — aligned signal
    """
    n_ch = data.shape[0]
    # Covariance matrix across channels
    C = np.cov(data)                         # (n_ch, n_ch)
    C += np.eye(n_ch) * 1e-6                 # regularise

    try:
        C_sqrt     = sqrtm(C).real
        C_inv_sqrt = np.linalg.inv(C_sqrt)
    except Exception:
        # Fallback: diagonal whitening
        diag       = np.sqrt(np.diag(C))
        C_inv_sqrt = np.diag(1.0 / (diag + 1e-8))

    # Apply: X_aligned = C_inv_sqrt @ X
    return C_inv_sqrt @ data

# ─────────────────────────────────────────────
#  TLI COMPUTATION
# ─────────────────────────────────────────────

def get_band_power(epoch, fs, lo, hi):
    """Mean PSD power across all channels in epoch for one frequency band."""
    freqs, psd = welch(epoch, fs=fs, nperseg=fs, axis=1)
    idx = np.logical_and(freqs >= lo, freqs <= hi)
    return np.mean(psd[:, idx], axis=1)  # (n_channels,)


def compute_tli_per_window(epoch, fs, ch_names, frontal_ch, parietal_ch):
    """
    Compute raw TLI for one window.
    TLI = mean_theta(frontal) / mean_alpha(parietal)
    epoch: (n_channels, n_samples)
    """
    theta_lo, theta_hi = BANDS["theta"]
    alpha_lo, alpha_hi = BANDS["alpha"]

    freqs, psd = welch(epoch, fs=fs, nperseg=fs, axis=1)  # psd: (n_ch, n_freqs)

    theta_idx = np.logical_and(freqs >= theta_lo, freqs <= theta_hi)
    alpha_idx = np.logical_and(freqs >= alpha_lo, freqs <= alpha_hi)

    # Frontal theta — mean across frontal channels
    f_idx = [ch_names.index(c) for c in frontal_ch if c in ch_names]
    if not f_idx:
        raise ValueError(f"No frontal channels found in {ch_names}")
    theta_frontal = np.mean(psd[f_idx][:, theta_idx])

    # Parietal alpha — mean across parietal channels
    p_idx = [ch_names.index(c) for c in parietal_ch if c in ch_names]
    if not p_idx:
        raise ValueError(f"No parietal channels found in {ch_names}")
    alpha_parietal = np.mean(psd[p_idx][:, alpha_idx])

    if alpha_parietal == 0:
        return np.nan

    return theta_frontal / alpha_parietal


def compute_fti_per_window(epoch, fs, ch_names, frontal_ch):
    """
    Frontal Theta Index — mean frontal theta PSD.
    Raw value; caller z-scores across all windows.
    """
    theta_lo, theta_hi = BANDS["theta"]
    freqs, psd = welch(epoch, fs=fs, nperseg=fs, axis=1)
    theta_idx = np.logical_and(freqs >= theta_lo, freqs <= theta_hi)
    f_idx = [ch_names.index(c) for c in frontal_ch if c in ch_names]
    return np.mean(psd[f_idx][:, theta_idx])


# ─────────────────────────────────────────────
#  MAIN SESSION PROCESSOR
# ─────────────────────────────────────────────

def process_session(csv_path, subject, game):
    """
    Full pipeline for one CSV file.
    Returns dict with X, y_cont, y_class, subject, game
    or None if processing fails.
    """
    print(f"\n  Processing: {os.path.basename(csv_path)}")

    # ── 1. Load ──
    df, ch_names = load_csv(csv_path)
    if df.shape[0] < SFREQ * 10:
        print(f"  [SKIP] Too short: {df.shape[0]} samples")
        return None

    # data shape: (n_channels, n_samples)
    data = df.values.T.astype(np.float64)
    n_ch, n_samples = data.shape
    print(f"  Channels: {ch_names}  |  Samples: {n_samples}  ({n_samples/SFREQ:.1f}s)")

    # ── 2. Bandpass 1–45 Hz ──
    data = bandpass(data, lo=1, hi=45, fs=SFREQ)

    # ── 3. Notch 50 Hz ──
    data = notch(data, freq=50, fs=SFREQ)

    # ── 3b. Euclidean Alignment (He & Wu 2020) ──
    # Whitens subject-specific covariance to identity — the most
    # effective published technique for cross-subject EEG generalisation.
    data = euclidean_align(data)

    # ── 4. Baseline split ──
    baseline_samples = min(BASELINE_SEC * SFREQ, n_samples // 3)
    baseline_data = data[:, :baseline_samples]
    game_data     = data[:, baseline_samples:]

    if game_data.shape[1] < WIN_SAMPLES * 2:
        print(f"  [SKIP] Insufficient game data after baseline split")
        return None

    # ── 5. Compute baseline TLI statistics ──
    bl_tli_values = []
    bl_starts = range(0, baseline_data.shape[1] - WIN_SAMPLES + 1, STEP)
    for start in bl_starts:
        epoch = baseline_data[:, start:start + WIN_SAMPLES]
        if reject_epoch(epoch, AMP_THRESH):
            continue
        tli = compute_tli_per_window(epoch, SFREQ, ch_names, FRONTAL_CH, PARIETAL_CH)
        if not np.isnan(tli):
            bl_tli_values.append(tli)

    if len(bl_tli_values) < 5:
        print(f"  [WARN] Very few clean baseline windows ({len(bl_tli_values)}). Using fallback normalization.")
        mu_baseline    = 0.0
        sigma_baseline = 1.0
    else:
        mu_baseline    = np.mean(bl_tli_values)
        sigma_baseline = np.std(bl_tli_values)
        if sigma_baseline == 0:
            sigma_baseline = 1.0
        print(f"  Baseline TLI: mu={mu_baseline:.4f}  sigma={sigma_baseline:.4f}  ({len(bl_tli_values)} windows)")

    # ── 6. Window game data & extract features ──
    X_list    = []
    tli_raw   = []
    fti_raw   = []
    rejected  = 0

    starts = range(0, game_data.shape[1] - WIN_SAMPLES + 1, STEP)
    for start in starts:
        epoch = game_data[:, start:start + WIN_SAMPLES]

        if reject_epoch(epoch, AMP_THRESH):
            rejected += 1
            continue

        # PSD feature vector: (n_channels * n_bands,)
        psd_features = compute_psd(epoch, SFREQ, BANDS)

        # TLI and FTI raw values
        tli = compute_tli_per_window(epoch, SFREQ, ch_names, FRONTAL_CH, PARIETAL_CH)
        fti = compute_fti_per_window(epoch, SFREQ, ch_names, FRONTAL_CH)

        if np.isnan(tli):
            rejected += 1
            continue

        # Engineered features (6 literature-validated workload markers)
        eng_features = compute_engineered_features(
            epoch, SFREQ, ch_names, FRONTAL_CH, PARIETAL_CH
        )

        X_list.append(np.concatenate([psd_features, eng_features]))
        tli_raw.append(tli)
        fti_raw.append(fti)

    n_accepted = len(X_list)
    print(f"  Windows: {n_accepted} accepted, {rejected} rejected")

    if n_accepted < 10:
        print(f"  [SKIP] Too few clean windows: {n_accepted}")
        return None

    X       = np.stack(X_list, axis=0)          # (n_windows, n_ch * n_bands)
    tli_arr = np.array(tli_raw, dtype=np.float64)
    fti_arr = np.array(fti_raw, dtype=np.float64)

    # ── 7. Z-score TLI against baseline ──
    tli_z = (tli_arr - mu_baseline) / sigma_baseline

    # ── 8. Normalize TLI_z to [0, 1] → y_cont ──
    tli_min = tli_z.min()
    tli_max = tli_z.max()
    if tli_max - tli_min == 0:
        y_cont = np.full(n_accepted, 0.5, dtype=np.float32)
    else:
        y_cont = ((tli_z - tli_min) / (tli_max - tli_min)).astype(np.float32)

    # ── 9. Percentile-based thresholds → y_class ──
    # Computed per session — guarantees balanced classes regardless of
    # each session's TLI distribution.
    p33 = float(np.percentile(y_cont, 33))
    p66 = float(np.percentile(y_cont, 66))
    y_class = np.where(y_cont <= p33, 0,
              np.where(y_cont <= p66, 1, 2)).astype(np.int32)

    # ── 10. Z-score FTI across session → append to X ──
    if fti_arr.std() == 0:
        fti_z = np.zeros(n_accepted, dtype=np.float32)
    else:
        fti_z = zscore(fti_arr).astype(np.float32)

    X = np.hstack([X, fti_z.reshape(-1, 1)])  # (n_windows, n_ch*n_bands + 6_eng + 1_fti = 77)

    # ── 11. Sanity check ──
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  [WARN] {nan_count} NaN + {inf_count} Inf in X — replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  X shape: {X.shape}  |  y_cont: [{y_cont.min():.3f}, {y_cont.max():.3f}]")
    dist = {0: (y_class==0).sum(), 1: (y_class==1).sum(), 2: (y_class==2).sum()}
    print(f"  Class dist → Low:{dist[0]}  Med:{dist[1]}  High:{dist[2]}")

    return {
        "X":              X,
        "y_cont":         y_cont,
        "y_class":        y_class,
        "subject":        subject,
        "game":           game,
        "ch_names":       np.array(ch_names),
        "mu_baseline":    np.float64(mu_baseline),
        "sigma_baseline": np.float64(sigma_baseline),
        "tli_min":        np.float64(tli_min),
        "tli_max":        np.float64(tli_max),
        "sfreq":          np.int32(SFREQ),
    }


# ─────────────────────────────────────────────
#  DISCOVERY & BATCH RUNNER
# ─────────────────────────────────────────────

def discover_sessions(raw_dir):
    """
    Discovers all EEG CSV sessions matching the GAMEEMO dataset structure.

    Actual layout on disk:
        HCI/
          (S01)/
            Raw EEG Data/
              .csv format/
                S01G1AllRawChannels.csv   -> subject=S01, game=G1
                S01G2AllRawChannels.csv
                S01G3AllRawChannels.csv
                S01G4AllRawChannels.csv
          (S02)/
            ...

    Subject folders are wrapped in parentheses: (S01), (S02) etc.
    CSV files follow the pattern: S{nn}G{n}AllRawChannels.csv

    Returns list of (csv_path, subject_id, game_id).
    """
    sessions = []

    for entry in sorted(os.listdir(raw_dir)):
        # Match parenthesised subject folders: (S01), (S02) ...
        if not (entry.startswith("(") and entry.endswith(")")):
            continue
        subj_dir = os.path.join(raw_dir, entry)
        if not os.path.isdir(subj_dir):
            continue

        # Navigate into Raw EEG Data/.csv format/
        csv_dir = os.path.join(subj_dir, "Raw EEG Data", ".csv format")
        if not os.path.isdir(csv_dir):
            print(f"  [WARN] CSV folder not found for {entry}: {csv_dir}")
            continue

        for fname in sorted(os.listdir(csv_dir)):
            if not fname.lower().endswith(".csv"):
                continue

            # Parse S01G1AllRawChannels.csv -> subject=S01, game=G1
            match = re.match(r"(S\d+)(G\d+)", fname, re.IGNORECASE)
            if not match:
                print(f"  [WARN] Skipping unrecognised filename: {fname}")
                continue

            subj_id = match.group(1).upper()  # S01
            game_id = match.group(2).upper()  # G1
            sessions.append((os.path.join(csv_dir, fname), subj_id, game_id))

    return sessions


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sessions = discover_sessions(RAW_DIR)
    if not sessions:
        print(f"[ERROR] No CSV files found under {RAW_DIR}/")
        print("  Expected layout:")
        print("    HCI/(S01)/Raw EEG Data/.csv format/S01G1AllRawChannels.csv")
        return

    print(f"Found {len(sessions)} session(s)\n{'='*60}")

    saved   = 0
    skipped = 0

    for csv_path, subject, game in sessions:
        out_name = f"{subject}_{game}.npz"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if os.path.exists(out_path):
            print(f"  [SKIP] Already exists: {out_name}")
            continue

        result = process_session(csv_path, subject, game)

        if result is None:
            skipped += 1
            continue

        np.savez(out_path, **result)
        print(f"  Saved → {out_path}")
        saved += 1

    print(f"\n{'='*60}")
    print(f"Done.  Saved: {saved}  |  Skipped: {skipped}")
    print(f"Output folder: {OUTPUT_DIR}/")
    print(f"\nFeature vector size per window: {14 * 5 + 6 + 1} = 77 features")
    print(f"  (14 channels x 5 bands + 6 engineered + 1 FTI_z column)")
    print(f"\nNext step: run aggregate.py to combine all sessions into dataset.npz")


if __name__ == "__main__":
    main()
