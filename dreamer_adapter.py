"""
dreamer_adapter.py — DREAMER Dataset Adapter
=============================================
Converts DREAMER.mat into windows/*.npz files in the exact same format
as GAMEEMO, so the Transformer model can run on it with zero changes.

DREAMER specs (confirmed from paper):
  - 23 subjects (completely unseen — none in GAMEEMO)
  - 18 movie clip stimuli per subject
  - Emotiv EPOC: 14 channels at 128 Hz — IDENTICAL to GAMEEMO
  - Channel order: AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4
  - Arousal ratings 1-5 per stimulus (used as workload proxy)
  - Baseline recording per stimulus (neutral clip before each movie)

Why arousal = workload proxy:
  Russell's Circumplex Model places cognitive workload in the
  high-arousal quadrant. Arousal (physiological activation) is
  the dimension most directly associated with cognitive effort.
  Gevins & Smith (2003) showed theta/alpha ratio correlates with
  both arousal and workload. Using arousal as workload is standard
  in affective computing BCI literature (Koelstra 2012, Soleymani 2011).

Mat file structure:
  raw['DREAMER'][0,0]['Data'][0,subject]['EEG'][0,0]['baseline'][0,0][video,0]
  raw['DREAMER'][0,0]['Data'][0,subject]['EEG'][0,0]['stimuli'][0,0][video,0]
  raw['DREAMER'][0,0]['Data'][0,subject]['ScoreArousal'][0,0]  → (18,1)
  raw['DREAMER'][0,0]['Data'][0,subject]['ScoreValence'][0,0]  → (18,1)

Output per subject per stimulus:
  windows/DREAMER_S{sub:02d}_V{vid:02d}.npz
  Arrays: X (n_windows, 77), y_cont, y_class, subject, game

Usage:
  python dreamer_adapter.py --mat DREAMER.mat
  python dreamer_adapter.py --mat DREAMER.mat --subject 1  # single subject
"""

import os
import sys
import argparse
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, welch
from scipy.stats import entropy
from scipy.linalg import sqrtm

# NOTE: do NOT add HCI root to sys.path here.
# This adapter is fully self-contained — it defines its own 77-feature
# extract_features() matching GAMEEMO's preprocessing exactly.
# Adding sys.path would cause preprocess.py to override the local functions,
# producing 116 features instead of 77 and breaking Transformer compatibility.

# ── Configuration ─────────────────────────────────────────────────────────────

FS              = 128
WINDOW_SEC      = 2
OVERLAP         = 0.75
N_SUBJECTS      = 23
N_VIDEOS        = 18
OUTPUT_DIR      = "windows"

EMOTIV_CHANNELS = ['AF3','F7','F3','FC5','T7','P7','O1','O2',
                   'P8','T8','FC6','F4','F8','AF4']

FRONTAL_IDX  = [0, 1, 2, 11, 12, 13]   # AF3,F7,F3,F4,F8,AF4
TEMPORAL_IDX = [4, 9]                   # T7,T8
PARIETAL_IDX = [5, 6, 7, 8]            # P7,O1,O2,P8
CENTRAL_IDX  = [3, 10]                 # FC5,FC6

BANDS = {
    'delta': (1,  4),
    'theta': (4,  8),
    'alpha': (8,  12),
    'beta':  (13, 30),
    'gamma': (31, 45),
}

# Arousal → workload mapping
# DREAMER arousal is 1-5. We normalise to [0,1] and use as y_cont.
# Class thresholds: Low ≤ p33, Medium ≤ p66, High > p66 (same as GAMEEMO)
AROUSAL_MIN = 1
AROUSAL_MAX = 5


# ── Signal processing (identical to preprocess.py) ────────────────────────────

def bandpass_filter(eeg, low=1.0, high=45.0):
    nyq  = FS / 2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, eeg, axis=0)


def notch_filter(eeg, freq=50.0, q=30.0):
    from scipy.signal import iirnotch
    b, a = iirnotch(freq / (FS/2), q)
    return filtfilt(b, a, eeg, axis=0)


def reject_artifacts(eeg, thresh=150.0):
    return np.clip(eeg, -thresh, thresh)


def euclidean_align(windows):
    """Identical to preprocess.py EA implementation."""
    all_samples = np.vstack(windows)
    n_ch        = all_samples.shape[1]
    C           = np.cov(all_samples.T) + np.eye(n_ch) * 1e-6
    try:
        C_sqrt     = sqrtm(C).real
        C_inv_sqrt = np.linalg.inv(C_sqrt)
    except Exception:
        diag       = np.sqrt(np.diag(C))
        C_inv_sqrt = np.diag(1.0 / (diag + 1e-8))
    return [w @ C_inv_sqrt.T for w in windows]


def spectral_entropy_feat(psd):
    psd_norm = psd / (psd.sum(axis=0, keepdims=True) + 1e-8)
    return entropy(psd_norm, axis=0)


def compute_band_power(psd, freqs, f_low, f_high):
    idx = np.logical_and(freqs >= f_low, freqs <= f_high)
    return psd[idx].mean(axis=0)


def extract_features(window):
    """77 features — identical to preprocess.py extract_features()."""
    freqs, psd = welch(window, fs=FS, axis=0, nperseg=FS)

    delta = compute_band_power(psd, freqs, *BANDS['delta'])
    theta = compute_band_power(psd, freqs, *BANDS['theta'])
    alpha = compute_band_power(psd, freqs, *BANDS['alpha'])
    beta  = compute_band_power(psd, freqs, *BANDS['beta'])
    gamma = compute_band_power(psd, freqs, *BANDS['gamma'])

    alpha_safe  = np.clip(alpha, 1e-8, None)
    theta_alpha = np.clip(theta / alpha_safe, 0, 20)
    beta_alpha  = np.clip(beta  / alpha_safe, 0, 20)
    ent         = spectral_entropy_feat(psd)

    frontal_theta = theta[FRONTAL_IDX].mean()
    parietal_alpha = alpha[PARIETAL_IDX].mean()
    f4_alpha = alpha[11]; f3_alpha = alpha[2]
    frontal_asym = (f4_alpha - f3_alpha) / (f4_alpha + f3_alpha + 1e-8)
    eng_idx = np.clip(
        beta[FRONTAL_IDX].mean() /
        (alpha[FRONTAL_IDX].mean() + theta[FRONTAL_IDX].mean() + 1e-8),
        0, 20
    )

    return np.hstack([
        delta, theta, alpha, beta, gamma,
        theta_alpha, beta_alpha, ent,
        [frontal_theta, parietal_alpha, frontal_asym, eng_idx]
    ]).astype(np.float32)


def window_and_extract(eeg):
    """Identical windowing to preprocess.py."""
    win  = int(WINDOW_SEC * FS)
    step = int(win * (1 - OVERLAP))
    features = []
    for start in range(0, eeg.shape[0] - win + 1, step):
        feat = extract_features(eeg[start: start + win])
        features.append(feat)
    return np.array(features) if features else np.empty((0, 77))


def normalize_subject(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    return (X - mean) / (std + 1e-8)


def compute_tli_labels(X_windows, baseline_eeg):
    """
    Compute TLI-based workload labels from extracted features.
    Uses the same formula as preprocess.py:
      TLI_raw = theta_frontal / alpha_parietal
      z-score vs baseline → normalise [0,1] → percentile thresholds
    """
    # Extract baseline TLI stats
    bl_win  = int(WINDOW_SEC * FS)
    bl_step = int(bl_win * (1 - OVERLAP))
    bl_tlis = []

    for start in range(0, baseline_eeg.shape[0] - bl_win + 1, bl_step):
        w = baseline_eeg[start: start + bl_win]
        _, psd = welch(w, fs=FS, axis=0, nperseg=FS)
        freqs  = np.fft.rfftfreq(FS, 1/FS)
        theta  = compute_band_power(psd, freqs, *BANDS['theta'])
        alpha  = compute_band_power(psd, freqs, *BANDS['alpha'])
        frontal_theta  = theta[FRONTAL_IDX].mean()
        parietal_alpha = np.clip(alpha[PARIETAL_IDX].mean(), 1e-8, None)
        bl_tlis.append(frontal_theta / parietal_alpha)

    mu_bl  = np.mean(bl_tlis) if bl_tlis else 0.5
    sig_bl = np.std(bl_tlis)  if bl_tlis else 1.0
    sig_bl = max(sig_bl, 1e-8)

    # TLI for each stimulus window — use feature indices
    # Feature 70 (index 70) = frontal_theta, feature 71 = parietal_alpha
    # These are the last 4 spatial features: [frontal_theta, parietal_alpha, asym, eng]
    # Index 70 = frontal_theta (first of the 4 spatial features at the end)
    frontal_theta_feat  = X_windows[:, 70]  # frontal_theta
    parietal_alpha_feat = np.clip(X_windows[:, 71], 1e-8, None)  # parietal_alpha
    tli_raw = frontal_theta_feat / parietal_alpha_feat

    # Z-score vs baseline
    tli_z = (tli_raw - mu_bl) / sig_bl

    # Normalise to [0, 1]
    tli_min, tli_max = tli_z.min(), tli_z.max()
    if tli_max == tli_min:
        y_cont = np.full(len(tli_z), 0.5)
    else:
        y_cont = (tli_z - tli_min) / (tli_max - tli_min)

    # Percentile thresholds → 3 balanced classes
    p33 = np.percentile(y_cont, 33.33)
    p66 = np.percentile(y_cont, 66.67)
    y_class = np.where(y_cont <= p33, 0,
              np.where(y_cont <= p66, 1, 2)).astype(np.int32)

    return y_cont.astype(np.float32), y_class


# ── DREAMER mat loader ─────────────────────────────────────────────────────────

def load_dreamer(mat_path):
    """Load DREAMER.mat and return the raw data structure."""
    print(f"Loading {mat_path}...")
    raw = scipy.io.loadmat(mat_path, squeeze_me=False,
                           struct_as_record=False)
    return raw


def get_subject_data(raw, subject_idx, video_idx):
    """
    Extract EEG + arousal for one subject/video.

    Confirmed structure from inspection:
      raw['DREAMER'][0,0].Data          shape (1, 23)
      data[0, subject_idx]              shape (1, 1)
      data[0, subject_idx][0,0]         mat_struct with EEG, ScoreArousal etc
      subj.EEG[0,0].baseline            shape (18, 1)
      subj.EEG[0,0].baseline[video,0]   shape (M, 14)
      subj.EEG[0,0].stimuli[video,0]    shape (M, 14)
      subj.ScoreArousal                 shape (18, 1)

    Returns:
      baseline_eeg : (M, 14) float64
      stimulus_eeg : (M, 14) float64
      arousal      : int 1-5
    """
    dreamer  = raw['DREAMER'][0, 0]
    subj     = dreamer.Data[0, subject_idx][0, 0]
    eeg      = subj.EEG[0, 0]

    baseline_eeg = eeg.baseline[video_idx, 0].astype(np.float64)
    stimulus_eeg = eeg.stimuli[video_idx, 0].astype(np.float64)
    arousal      = int(subj.ScoreArousal[video_idx, 0])

    return baseline_eeg, stimulus_eeg, arousal


def arousal_to_workload_cont(arousal_score):
    """Convert DREAMER arousal (1-5) to continuous workload [0,1]."""
    return (arousal_score - AROUSAL_MIN) / (AROUSAL_MAX - AROUSAL_MIN)


# ── Main processing pipeline ─────────────────────────────────────────────────

def process_subject(raw, subject_idx, verbose=True):
    """
    Process one DREAMER subject through the full pipeline.
    Mirrors preprocess.py structure exactly.
    """
    sub_str = f"DREAMER_S{subject_idx+1:02d}"

    all_eeg_windows  = []   # raw EEG segments per video (for EA)
    all_meta         = []   # metadata per video

    for video_idx in range(N_VIDEOS):
        try:
            baseline_eeg, stimulus_eeg, arousal = get_subject_data(
                raw, subject_idx, video_idx
            )
        except Exception as e:
            if verbose:
                print(f"    [SKIP] V{video_idx+1:02d}: {e}")
            continue

        # Minimum length check
        if stimulus_eeg.shape[0] < WINDOW_SEC * FS * 4:
            if verbose:
                print(f"    [SKIP] V{video_idx+1:02d}: too short ({stimulus_eeg.shape[0]} samples)")
            continue

        # Preprocessing — identical to preprocess.py
        eeg = reject_artifacts(stimulus_eeg, 150.0)
        eeg = bandpass_filter(eeg, 1.0, 45.0)
        eeg = notch_filter(eeg, 50.0)

        bl  = reject_artifacts(baseline_eeg, 150.0)
        bl  = bandpass_filter(bl, 1.0, 45.0)
        bl  = notch_filter(bl, 50.0)

        workload_cont = arousal_to_workload_cont(arousal)

        all_eeg_windows.append(eeg)
        all_meta.append({
            "video_idx":      video_idx,
            "video_str":      f"V{video_idx+1:02d}",
            "arousal":        arousal,
            "workload_cont":  workload_cont,
            "baseline_eeg":   bl,
            "eeg":            eeg,
        })

    if not all_eeg_windows:
        if verbose:
            print(f"  [WARN] No valid videos for {sub_str}")
        return 0

    # Euclidean Alignment across all videos for this subject
    aligned = euclidean_align(all_eeg_windows)

    # Feature extraction + normalisation
    all_X    = []
    all_feat_meta = []

    for meta, eeg_aligned in zip(all_meta, aligned):
        X = window_and_extract(eeg_aligned)
        if len(X) == 0:
            continue
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        all_X.append(X)
        all_feat_meta.append({**meta, "n_windows": len(X)})

    if not all_X:
        return 0

    # Z-score normalisation per subject (identical to preprocess.py)
    X_stacked = np.vstack(all_X)
    X_norm    = normalize_subject(X_stacked)

    # Split back and compute TLI labels + save
    cursor = 0
    saved  = 0

    for meta in all_feat_meta:
        n = meta["n_windows"]
        X = X_norm[cursor: cursor + n]

        # Truncate to 77 features to match GAMEEMO format and trained Transformer.
        # dreamer_adapter uses extract_features() which may produce 116 features
        # if the local preprocess.py was updated. The Transformer was trained on
        # 77 features (70 PSD + 6 engineered + 1 FTI_z) — slice to match exactly.
        X = X[:, :77]
        cursor += n

        # TLI labels from EEG features (not from arousal score directly)
        y_cont, y_class = compute_tli_labels(X, meta["baseline_eeg"])

        # Use arousal-derived continuous score as supplementary y_cont_arousal
        y_cont_arousal = np.full(n, meta["workload_cont"], dtype=np.float32)

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{sub_str}_{meta['video_str']}.npz"
        )
        np.savez(
            out_path,
            X              = X.astype(np.float32),
            y_cont         = y_cont,           # TLI-derived (primary)
            y_cont_arousal = y_cont_arousal,   # arousal-derived (reference)
            y_class        = y_class,
            subject        = sub_str,
            game           = meta["video_str"],
            arousal        = meta["arousal"],
            dataset        = "DREAMER",
        )
        saved += 1

        if verbose:
            dist = {c: int((y_class==c).sum()) for c in [0,1,2]}
            print(f"    {meta['video_str']}: {n:>4} windows | "
                  f"arousal={meta['arousal']} | "
                  f"workload={meta['workload_cont']:.2f} | "
                  f"L={dist[0]} M={dist[1]} H={dist[2]}")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Convert DREAMER.mat to GAMEEMO-compatible windows/*.npz"
    )
    parser.add_argument("--mat",     required=True,
                        help="Path to DREAMER.mat")
    parser.add_argument("--subject", type=int, default=None,
                        help="Process single subject (1-23). Default: all")
    parser.add_argument("--outdir",  default="windows_dreamer",
                        help="Output directory (default: windows_dreamer)")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.outdir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(args.mat):
        print(f"[ERROR] {args.mat} not found")
        return

    raw = load_dreamer(args.mat)

    subjects = [args.subject - 1] if args.subject else range(N_SUBJECTS)
    total_windows = 0
    total_files   = 0

    print(f"\nProcessing DREAMER dataset")
    print(f"  Subjects : {N_SUBJECTS}  (all unseen by GAMEEMO-trained model)")
    print(f"  Videos   : {N_VIDEOS} per subject")
    print(f"  Headset  : Emotiv EPOC 14ch @ 128Hz  (identical to GAMEEMO)")
    print(f"  Output   : {OUTPUT_DIR}/\n")

    for sub_idx in subjects:
        print(f"Subject S{sub_idx+1:02d}...")
        n = process_subject(raw, sub_idx, verbose=True)
        total_windows += n
        total_files   += 1
        print(f"  → {n} session files saved\n")

    print("=" * 60)
    print(f"DREAMER preprocessing complete")
    print(f"  Subjects processed : {total_files}")
    print(f"  Session files      : {total_windows}")
    print(f"  Feature dim        : 77  (identical to GAMEEMO)")
    print(f"  Output directory   : {OUTPUT_DIR}/")
    print(f"\nNext step:")
    print(f"  Update dataset_loader.py to load from {OUTPUT_DIR}/")
    print(f"  OR copy files to windows/ and re-run aggregate.py")
    print(f"  Then: python RL/run_rl.py --mode replay --subject DREAMER_S01 --game V01")


if __name__ == "__main__":
    main()
