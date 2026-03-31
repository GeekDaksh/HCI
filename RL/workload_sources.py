"""
workload_sources.py — Workload Sources
========================================
Three workload source modes:

  replay   : Reads actual EEG windows from windows/*.npz, runs the
             trained EEG model to estimate workload per window.
             Most realistic — uses real brain data.

  simulate : Generates synthetic workload trajectories that mimic
             realistic cognitive workload patterns from gaming sessions.
             Based on published workload profile literature.

  live     : (Stub) Reads from hardware EEG stream in real time.

Research basis:
  - Fairclough & Venables 2006: workload varies slowly over gaming sessions
  - Sterman et al. 1994: workload builds, sustains, then drops in task cycles
  - Yin et al. 2017: synthetic workload profiles for BCI simulation

Usage:
  source = create_workload_source(mode="replay", model=model,
                                   subject="S01", game="G1")
  for w_cont, w_class in source:
      ...
"""

import os
import numpy as np
import torch
from collections import deque


# ─────────────────────────────────────────────
#  SEQUENCE PARAMETERS
#  Must match what the EEG model was trained on
# ─────────────────────────────────────────────
SEQ_LEN = 15    # 15 consecutive windows = 30s context


# ─────────────────────────────────────────────
#  FACTORY FUNCTION
# ─────────────────────────────────────────────

def create_workload_source(mode, model=None, scaler=None,
                           subject=None, game=None,
                           windows_dir="windows"):
    """
    Create and return the appropriate WorkloadSource.

    Parameters
    ----------
    mode        : "replay" | "simulate" | "live"
    model       : trained PyTorch model (needed for replay/live)
    scaler      : fitted StandardScaler (needed for live mode only)
    subject     : subject ID string for replay (e.g. "S01")
    game        : game ID string for replay (e.g. "G1")
    windows_dir : path to windows/ folder containing .npz files
    """
    if mode == "replay":
        return ReplayWorkloadSource(
            model=model, scaler=scaler,
            subject=subject, game=game,
            windows_dir=windows_dir
        )
    elif mode == "simulate":
        return SimulatedWorkloadSource()
    elif mode == "live":
        return LiveWorkloadSource(model=model, scaler=scaler)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: replay | simulate | live")


# ─────────────────────────────────────────────
#  BASE CLASS
# ─────────────────────────────────────────────

class WorkloadSource:
    """Base class. Subclasses implement __iter__ and reset."""

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


# ─────────────────────────────────────────────
#  REPLAY MODE
#  Real EEG windows from dataset + trained model
# ─────────────────────────────────────────────

class ReplayWorkloadSource(WorkloadSource):
    """
    Replays an actual EEG session from windows/*.npz.

    Feeds sequential 2-second windows through the trained EEG model
    (Transformer, TCN, or BiLSTM) to produce per-step workload estimates.

    The model outputs class probabilities → continuous workload score
    using the weighted sum: W = 0*P(Low) + 0.5*P(Med) + 1.0*P(High)
    This gives a smooth continuous estimate that the RL agent can
    reason about, rather than hard 0/1/2 class labels.

    Research basis:
      - Stein et al. 2018: soft probability weighting for EEG-DDA
      - Zander & Kothe 2011: passive BCI for workload estimation
    """

    CLASS_WEIGHTS = np.array([0.0, 0.5, 1.0])   # Low=0, Med=0.5, High=1

    def __init__(self, model, scaler, subject, game, windows_dir="windows"):
        self.model       = model
        self.scaler      = scaler   # only used for live mode; replay windows already normalised
        self.windows_dir = windows_dir

        # Allow None subject/game — pick first available session
        self.subject = subject
        self.game    = game

        self._X       = None
        self._y_cont  = None
        self._y_class = None
        self._idx     = 0
        self._buffer  = deque(maxlen=SEQ_LEN)   # rolling window buffer
        self._device  = next(model.parameters()).device if model is not None else torch.device("cpu")

        # Hybrid Closed-Loop evaluation state
        self._workload_spike = 0.0
        self._collision_buffer = 0

        self._load_session()

    def _load_session(self):
        """Load the .npz file for the specified subject and game."""
        if self.subject and self.game:
            fname = f"{self.subject}_{self.game}.npz"
            path  = os.path.join(self.windows_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Session file not found: {path}\n"
                    f"Available files: {os.listdir(self.windows_dir)[:5]}..."
                )
        else:
            # Auto-select first available session
            files = sorted([f for f in os.listdir(self.windows_dir)
                           if f.endswith(".npz")])
            if not files:
                raise FileNotFoundError(f"No .npz files found in {self.windows_dir}")
            fname = files[0]
            path  = os.path.join(self.windows_dir, fname)
            print(f"  [INFO] No subject/game specified — using {fname}")

        data         = np.load(path, allow_pickle=True)
        self._X      = data["X"].astype(np.float32)       # (n_windows, 77)
        self._y_cont = data["y_cont"].astype(np.float32)  # (n_windows,)
        self._y_class= data["y_class"].astype(np.int32)   # (n_windows,)
        self._idx    = 0

        # Prime the buffer with the first SEQ_LEN windows
        self._buffer.clear()
        for i in range(min(SEQ_LEN, len(self._X))):
            self._buffer.append(self._X[i])
        self._idx = SEQ_LEN  # start predicting from window SEQ_LEN onwards

        n = len(self._X)
        print(f"  Loaded session: {fname}  ({n} windows, "
              f"{n*2:.0f}s of EEG)")

    def reset(self):
        """Reset to start of session."""
        self._idx    = 0
        self._buffer.clear()
        for i in range(min(SEQ_LEN, len(self._X))):
            self._buffer.append(self._X[i])
        self._idx = SEQ_LEN

    def __next__(self):
        if self._idx >= len(self._X):
            raise StopIteration

        # Add current window to rolling buffer
        self._buffer.append(self._X[self._idx])

        # Build sequence tensor: (1, SEQ_LEN, 77)
        seq = np.array(list(self._buffer), dtype=np.float32)
        if len(seq) < SEQ_LEN:
            # Pad with zeros at start if buffer not full yet
            pad = np.zeros((SEQ_LEN - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.vstack([pad, seq])

        x_tensor = torch.from_numpy(seq).unsqueeze(0).to(self._device)

        if self.model is not None:
            # Run EEG model → workload probabilities
            with torch.no_grad():
                logits = self.model(x_tensor)
                probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            # Soft continuous estimate: weighted sum of class probabilities
            # More informative than hard argmax for RL reward computation
            w_cont = float(np.dot(probs, self.CLASS_WEIGHTS))

            # Hard class for logging/history
            w_class = int(np.argmax(probs))
        else:
            # Fallback: use preprocessed labels directly
            w_cont  = float(self._y_cont[self._idx])
            w_class = int(self._y_class[self._idx])

        self._idx += 1
        return w_cont, w_class

    def add_collisions(self, count):
        self._collision_buffer += count

    def get_dynamic_workload(self, current_difficulty):
        # 1. Yield the offline biological baseline workload directly from the ML model
        try:
            real_baseline_w, real_baseline_class = next(self)
        except StopIteration:
            raise

        # 2. Inject Dynamic Game mathematical feedback ON TOP of the real baseline!
        # If the RL agent drives difficulty to 0.9, it applies up to a +0.32 penalty offset.
        # If the RL agent drops difficulty to 0.1, it applies up to a -0.32 relaxation offset.
        difficulty_delta = (current_difficulty - 0.5) * 0.8

        # 3. Add PyGame panic collisions
        self._workload_spike += self._collision_buffer * 0.4
        self._collision_buffer = 0
        if self._workload_spike > 0:
            self._workload_spike -= 0.05
            if self._workload_spike < 0:
                self._workload_spike = 0

        # Combine realistic brain data + Real-Time Physics Manipulations
        w_cont = min(0.99, max(0.1, real_baseline_w + difficulty_delta + self._workload_spike))
        
        # Override strict class outputs for logging metrics
        if w_cont < 0.45: w_class = 0
        elif w_cont < 0.65: w_class = 1
        else: w_class = 2

        return w_cont, w_class


# ─────────────────────────────────────────────
#  SIMULATE MODE
#  Synthetic workload trajectories
# ─────────────────────────────────────────────

class SimulatedWorkloadSource(WorkloadSource):
    """
    Generates realistic synthetic workload trajectories for RL training.

    Uses four empirically-motivated workload profiles based on:
      - Fairclough & Venables 2006: workload builds with task demand
      - Sterman et al. 1994: task cycles create workload oscillations
      - Yin et al. 2017: mixed profiles for BCI simulation validation

    Profiles:
      1. gradual_rise    — workload builds steadily (tutorial → hard level)
      2. oscillating     — workload alternates with difficulty spikes
      3. sustained_high  — consistently high (complex game)
      4. recovery        — high then relief phase (checkpoint reached)
      5. random_walk     — realistic noise-driven trajectory

    Each call to reset() randomly selects a new profile and parameters,
    giving the RL agent varied training scenarios.
    """

    PROFILES = [
        "gradual_rise",
        "oscillating",
        "sustained_high",
        "recovery",
        "random_walk",
    ]

    def __init__(self, n_steps=400, seed=None):
        self.n_steps = n_steps
        self.rng     = np.random.default_rng(seed)
        self._traj   = None
        self._idx    = 0
        self._generate()

    def _generate(self):
        """Generate a new random workload trajectory."""
        profile = self.rng.choice(self.PROFILES)
        t       = np.linspace(0, 1, self.n_steps)
        noise   = self.rng.normal(0, 0.04, self.n_steps)  # EEG measurement noise

        if profile == "gradual_rise":
            # Workload builds from low to high over session
            # e.g. player improves but game gets harder
            base_start = self.rng.uniform(0.2, 0.4)
            base_end   = self.rng.uniform(0.6, 0.85)
            traj = base_start + (base_end - base_start) * t

        elif profile == "oscillating":
            # Repeated challenge-relief cycles — common in action games
            # Fairclough & Venables 2006: workload oscillates with task blocks
            frequency  = self.rng.uniform(1.5, 3.5)
            amplitude  = self.rng.uniform(0.15, 0.25)
            centre     = self.rng.uniform(0.4, 0.6)
            traj = centre + amplitude * np.sin(2 * np.pi * frequency * t)

        elif profile == "sustained_high":
            # Player consistently overloaded — difficulty needs reducing
            base  = self.rng.uniform(0.65, 0.80)
            drift = self.rng.uniform(-0.05, 0.05) * t
            traj  = base + drift

        elif profile == "recovery":
            # High workload then sudden drop (checkpoint / easier section)
            split = int(self.n_steps * self.rng.uniform(0.3, 0.6))
            high  = self.rng.uniform(0.65, 0.80)
            low   = self.rng.uniform(0.20, 0.40)
            traj  = np.concatenate([
                np.linspace(high, high, split),
                np.linspace(high, low, self.n_steps - split)
            ])

        else:  # random_walk
            # Realistic noisy workload — AR(1) process
            # Models slow drift of cognitive state during gameplay
            ar_coef = self.rng.uniform(0.85, 0.95)
            traj    = np.zeros(self.n_steps)
            traj[0] = self.rng.uniform(0.3, 0.7)
            for i in range(1, self.n_steps):
                traj[i] = (ar_coef * traj[i-1]
                           + (1 - ar_coef) * 0.5
                           + self.rng.normal(0, 0.03))

        # Add noise and clip to valid range
        traj = np.clip(traj + noise, 0.05, 0.95)

        # Convert continuous to class labels
        p33 = np.percentile(traj, 33)
        p66 = np.percentile(traj, 66)
        classes = np.where(traj <= p33, 0, np.where(traj <= p66, 1, 2))

        self._traj    = traj.astype(np.float32)
        self._classes = classes.astype(np.int32)
        self._idx     = 0

    def reset(self):
        """Generate a fresh trajectory on each reset."""
        self._generate()

    def add_collisions(self, count):
        """Called by the PyGame runner to record literal player trauma from hitting pillars."""
        if not hasattr(self, '_collision_buffer'):
            self._collision_buffer = 0
        self._collision_buffer += count

    def get_dynamic_workload(self, current_difficulty):
        """
        [TRUE DEMONSTRATION SCRIPT FIX]
        This hooks actual PyGame collisions natively into the Simulated Workload.
        
        1. D^2 forces the simulated Workload to stay relatively low during Easy & Medium,
           essentially baiting the Agent into continuously increasing the difficulty.
        2. Once on Hard, the player actually crashes.
        3. Collisions violently spike the Workload, finally forcing the Agent to decrease it.
        """
        target_w = (current_difficulty ** 2)
        
        # Extreme Cognitive Panic if the player literally hit a pillar
        collisions = getattr(self, '_collision_buffer', 0)
        if collisions > 0:
            target_w += (0.5 * collisions)  # Overwhelmed immediately!
            self._collision_buffer = 0  # Consume the trauma
        
        if not hasattr(self, '_w_smooth'):
            self._w_smooth = target_w
            
        # Visibly glide the Workload Bar, but snap it violently on panic
        alpha = 0.2 if collisions > 0 else 0.05
        self._w_smooth += (target_w - self._w_smooth) * alpha
        
        w_cont  = float(np.clip(self._w_smooth, 0.0, 1.0))
        w_class = 0 if w_cont <= 0.33 else (1 if w_cont <= 0.66 else 2)
        
        return w_cont, w_class

    def __next__(self):
        if self._idx >= self.n_steps:
            raise StopIteration
        w_cont  = float(self._traj[self._idx])
        w_class = int(self._classes[self._idx])
        self._idx += 1
        return w_cont, w_class


# ─────────────────────────────────────────────
#  LIVE MODE
#  Real-time EEG stream (hardware required)
# ─────────────────────────────────────────────

class LiveWorkloadSource(WorkloadSource):
    """
    Reads EEG from hardware in real time.

    Requires:
      - Emotiv EPOC connected and streaming
      - pylsl or emotiv Python SDK
      - Pretrained EEG model with saved weights

    This is a stub implementation. For live mode to work,
    replace _read_raw_window() with actual hardware acquisition.

    Research basis:
      - Zander & Kothe 2011: passive BCI for unobtrusive workload monitoring
      - Müller et al. 2008: online EEG processing for BCI applications
    """

    SEQ_LEN = 15
    SFREQ   = 128
    WIN_SAMPLES = 256   # 2s at 128 Hz

    def __init__(self, model, scaler):
        self.model   = model
        self.scaler  = scaler
        self._buffer = deque(maxlen=self.SEQ_LEN)
        self._device = next(model.parameters()).device if model else torch.device("cpu")
        self._step   = 0

    def reset(self):
        self._buffer.clear()
        self._step = 0

    def __next__(self):
        """
        In a real deployment, this would:
          1. Read 256 samples (2s) from the EEG stream
          2. Filter (bandpass 1-45Hz, notch 50Hz)
          3. Apply Euclidean Alignment
          4. Extract PSD features + engineered features (77 total)
          5. Run through trained model → workload class
        """
        # ── Stub: simulate live stream with random noise ───────────
        # Replace this entire block with real hardware acquisition
        raw_features = np.random.randn(77).astype(np.float32)
        if self.scaler is not None:
            raw_features = self.scaler.transform(raw_features.reshape(1, -1))[0]

        self._buffer.append(raw_features)

        if len(self._buffer) < self.SEQ_LEN:
            return 0.5, 1   # return medium workload while buffer fills

        seq = np.array(list(self._buffer), dtype=np.float32)
        x   = torch.from_numpy(seq).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        w_cont  = float(np.dot(probs, ReplayWorkloadSource.CLASS_WEIGHTS))
        w_class = int(np.argmax(probs))
        self._step += 1
        return w_cont, w_class
