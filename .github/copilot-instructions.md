# AI Coding Agent Instructions — EEG Workload Classification

## Project Overview
**HCI Workload Study**: Multi-subject EEG-based cognitive workload classification using Emotiv EPOC headset. Predicts workload intensity (Low/Medium/High) from preprocessed 4-second EEG windows paired with Self-Assessment Manikin (SAM) ratings.

**Data Flow**: Raw EEG CSV → Preprocess (feature extraction) → Windowed datasets → Aggregate → Train (Leave-One-Group-Out CV) → Model evaluation

---

## Critical Architecture & Data Structures

### Subject Organization
- **28 subjects** (S01–S28) in folders like `(S01)/`, `(S02)/`, etc.
- Each has: `Raw EEG Data/`, `Preprocessed EEG Data/`, `SAM Ratings/` subdirectories
- SAM (workload) labels extracted from PDF files in `SAM Ratings/` and stored in `sam_workload.csv` per subject

### Processing Pipeline Dependencies
1. **`sam_workload.py`** → Extract SAM PDFs → `sam_workload.csv` per subject
2. **`preprocess.py`** → Read EEG CSVs + SAM labels → sliding windows (4-sec, 50% overlap) → `windows/` dir
3. **`aggregate_sessions.py`** → Merge all `windows/*.npz` → `processed/dataset.npz` (single dataset file)
4. **`train_model.py`** → Load `processed/dataset.npz` → Leave-One-Group-Out CV → train XGBoost

### Core Data Formats
- **Dataset**: `processed/dataset.npz` contains:
  - `X`: Feature matrix (N_samples, 70 features) — band power + spectral entropy across 5 bands × 14 channels
  - `y_class`: Integer labels [0=Low, 1=Medium, 2=High]
  - `y_cont`: Continuous workload scores (1–9 from SAM)
  - `subject_ids`, `subjects`, `games`: Metadata for LOGO stratification

---

## Key Technical Patterns

### EEG Signal Processing ([../preprocess.py](../preprocess.py))
- **Emotiv EPOC**: 14 channels at 128 Hz (constants at top of file)
- **Channel mapping**: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 (indices hardcoded)
- **Frequency bands**: Delta (0.5–4 Hz), Theta, Alpha, Beta, Gamma (30–45 Hz) — defined in `BANDS` dict
- **Feature extraction per window**: PSD via Welch, band power, spectral entropy, regional averages (frontal/temporal/parietal)
- **Artifact handling**: Butterworth 4th-order bandpass (0.5–45 Hz), amplitude clipping at ±100 µV

### SAM PDF Extraction ([../sam_workload.py](../sam_workload.py))
- GAMEEMO PDFs have two 1–9 rating scales (valence & arousal)
- **Extraction strategy** (priority order):
  1. Find standalone digits near anchor words (e.g., "excited", "calm")
  2. Fall back to first two standalone digits in document
  3. Return `None` if ambiguous
- Valence → ignored; Arousal → **workload score** (1–9)

### Model Architecture ([../train_model.py](../train_model.py))
- **CV strategy**: Leave-One-Group-Out (LOGO) with `group=subject_ids` → no subject data leak
- **Model**: XGBoost classifier + regressor (parallel)
  - Classification: 3 classes, balanced weights, `objective="multi:softmax"`
  - Regression: continuous workload prediction (MAE loss)
- **Preprocessing**: StandardScaler per fold (fit on train only), feature clipping to [-10, 10] for stability
- **Metrics**: Accuracy, Cohen's kappa, MAE, R² per fold + confusion matrix

---

## Developer Workflows

### Run Complete Pipeline
```bash
# 1. Extract SAM labels from PDFs
python sam_workload.py

# 2. Preprocess EEG → create windows
python preprocess.py

# 3. Merge windows into single dataset
python aggregate_sessions.py

# 4. Train models with LOGO CV
python train_model.py

# 5. Analyze dataset quality (run before training)
python eda.py
```

### Key Files Reference
- **Config constants**: All sampling rate, channel names, frequency bands, window size (4 sec, 50% overlap) defined at file tops
- **Subject enumeration**: `sorted([s for s in os.listdir(".") if s.startswith("(") and s.endswith(")")])` (see [preprocess details](../preprocess.py))
- **Output locations**:
  - Processed data: `windows/` → `(S01)_0.npz`, `(S01)_1.npz`, etc. (multiple games per subject)
  - Final dataset: `processed/dataset.npz` (aggregated)
  - EDA plots: `processed/eda/` (created by eda.py)
  - Trained models: `models/` (checkpoint format varies)

---

## Common Pitfalls & Defensive Patterns

### Emotiv CSV Parsing
- Emotiv exports may have **repeated header rows** mid-file — `preprocess.py` detects and drops these
- Fallback column detection if channel names don't match exactly (Strategy 2 in load_eeg_csv)
- Always check for NaN/Inf after numeric conversion

### Subject-Wise Isolation
- **Always use LOGO with `group=subject_ids`** to prevent cross-subject data leakage in CV
- `subject_ids` is integer-encoded in dataset; mapping is `{S01→0, S02→1, ...}` from aggregate_sessions.py

### Label Class Imbalance
- Check distribution in `eda.py` output (common: High class underrepresented)
- Use `compute_sample_weight("balanced", y_train)` for training (already applied in train_model.py)

### Feature Stability
- Clipping to [-10, 10] in both `preprocess.py` (artifact rejection) and `train_model.py` (stabilize_features)
- Helps XGBoost handle outliers from corrupted EEG segments

---

## When Adding Features or Modifying Pipelines
1. **Add to preprocess.py feature extraction** → automatically propagates to all windows
2. **Update BANDS or CHANNELS** → ensure region indices (FRONTAL_IDX, etc.) stay consistent
3. **Change SAM extraction logic** → test on PDF sample first (likely corruption edge cases)
4. **Modify CV strategy** → always validate with LOGO + subject grouping
5. **Add model hyperparameters** → increment n_estimators or max_depth conservatively (already tuned)
