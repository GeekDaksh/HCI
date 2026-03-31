"""
game_env.py — Adaptive Game Environment
========================================
Implements the closed-loop EEG → RL → difficulty adjustment system.

Research basis:
  - Flow Theory: Csikszentmihalyi 1990 — optimal experience at skill-challenge balance
  - GameFlow model: Sweetser & Wyeth 2005 — flow in video games
  - EEG-DDA: Stein et al. 2018 — EEG-triggered DDA for games
  - Reward shaping: Ng et al. 1999 — potential-based reward shaping

State space  : [workload_smooth, difficulty]  — both in [0, 1]
Action space : {0: Decrease, 1: Maintain, 2: Increase}
Reward       : Flow-theory shaped reward (see _compute_reward)
"""

import numpy as np
from collections import deque

# ─────────────────────────────────────────────
#  FLOW THEORY PARAMETERS
#  Csikszentmihalyi 1990 — optimal experience
#  when challenge matches skill level
# ─────────────────────────────────────────────

# Target workload for flow state
# 0.5 = medium workload — neither bored (too low) nor overwhelmed (too high)
# Literature: Stein et al. 2018 used W* = 0.5 for EEG-DDA systems
TARGET_WORKLOAD = 0.5

# Flow zone half-width — workload within ±FLOW_TOLERANCE of target
# is considered "in flow". Wider = more forgiving.
# Csikszentmihalyi (1990): flow channel is narrow around optimal
FLOW_TOLERANCE  = 0.10

# Difficulty adjustment step per action
# Small step prevents jarring transitions — gradual as per HCI guidelines
# Hendrix et al. 2018: gradual DDA outperforms abrupt changes for engagement
DIFF_STEP       = 0.05

# Difficulty bounds
DIFF_MIN        = 0.05   # never trivially easy
DIFF_MAX        = 0.95   # never impossibly hard

# Workload smoothing window (exponential moving average alpha)
# Gerjets et al. 2014: 5-window EMA smooths EEG noise without lag
EMA_ALPHA       = 0.25   # lower = smoother, higher = more responsive

# Score parameters
SCORE_BASE_PER_STEP   = 10     # base score earned each step
SCORE_FLOW_BONUS      = 5      # extra per step when in flow
SCORE_DIFFICULTY_MULT = True   # score scales with difficulty (harder = more points)

# Penalty zone boundaries
BOREDOM_THRESHOLD     = 0.40   # increased boredom threshold to trigger increases earlier
OVERLOAD_THRESHOLD    = 0.65   # above this = overwhelmed (too hard)


class AdaptiveGameEnv:
    """
    Simulates a game session where difficulty adjusts in response to
    the player's estimated cognitive workload from EEG.

    At each step:
      1. A new EEG window is processed → workload estimate W_t ∈ {0,1,2}
         converted to continuous score ∈ [0, 1]
      2. The RL agent observes state = [W_smooth, difficulty]
      3. Agent selects action: decrease / maintain / increase difficulty
      4. Environment updates difficulty, computes reward, updates score

    The reward function implements Flow Theory:
      - Maximum reward when workload ≈ target (0.5)
      - Penalty when bored (workload < 0.35) or overloaded (workload > 0.65)
      - Shaped bonus for moving toward flow from outside it
      - Penalty for oscillation (unnecessary difficulty changes)
    """

    def __init__(self, workload_source):
        """
        workload_source : WorkloadSource instance from workload_sources.py
                          Must implement __iter__ yielding (w_cont, w_class)
        """
        self.source = workload_source

        # Episode state
        self.difficulty     = 0.5    # start at medium difficulty
        self.workload_raw   = 0.5
        self.workload_smooth= 0.5
        self.step_count     = 0
        self.done           = False
        self.score          = 0.0

        # EMA smoothing buffer
        self._ema           = 0.5
        self.max_steps      = getattr(workload_source, "n_steps", 400)

        # Oscillation penalty tracking
        self._last_action   = 1      # 1 = maintain
        self._oscillation_count = 0

        # History for plotting
        self.history = {
            "workload_raw":    [],
            "workload_smooth": [],
            "difficulty":      [],
            "action":          [],
            "reward":          [],
            "score":           [],
            "flow_zone":       [],
            "workload_class":  [],
        }

    def reset(self):
        """Reset environment for a new episode."""
        self.difficulty      = 0.5
        self.workload_raw    = 0.5
        self.workload_smooth = 0.5
        self._ema            = 0.5
        self.step_count      = 0
        self.done            = False
        self.score           = 0.0
        self._last_action    = 1
        self._oscillation_count = 0

        self.source.reset()
        self.history = {k: [] for k in self.history}

        return self._get_state()

    def step(self, action):
        """
        Execute one step of the environment.

        Parameters
        ----------
        action : int  0=Decrease, 1=Maintain, 2=Increase

        Returns
        -------
        next_state : np.array (2,)
        reward     : float
        done       : bool
        info       : dict
        """
        # ── 1. Get next workload estimate ──────────────────────────
        try:
            # Inject game difficulty natively if the source mathematically supports Closed-Loop interaction
            if hasattr(self.source, 'get_dynamic_workload'):
                w_cont, w_class = self.source.get_dynamic_workload(self.difficulty)
            else:
                w_cont, w_class = next(self.source)
        except StopIteration:
            self.done = True
            return self._get_state(), 0.0, True, self._make_info()

        # Map discrete class → continuous if needed
        # Classes: 0=Low→0.2, 1=Medium→0.5, 2=High→0.8
        # w_cont already in [0,1] from preprocess.py
        self.workload_raw = float(np.clip(w_cont, 0.0, 1.0))

        # ── 2. EMA smoothing ───────────────────────────────────────
        # Gerjets et al. 2014: EMA smoothing reduces EEG noise
        # while preserving genuine workload trends
        self._ema = EMA_ALPHA * self.workload_raw + (1 - EMA_ALPHA) * self._ema
        self.workload_smooth = self._ema

        # ── 3. Apply action → update difficulty ───────────────────
        prev_difficulty = self.difficulty
        if action == 0:
            self.difficulty = max(DIFF_MIN, self.difficulty - DIFF_STEP)
        elif action == 2:
            self.difficulty = min(DIFF_MAX, self.difficulty + DIFF_STEP)
        # action == 1: maintain

        # ── 4. Compute reward ──────────────────────────────────────
        reward = self._compute_reward(
            w_smooth    = self.workload_smooth,
            w_raw       = self.workload_raw,
            action      = action,
            prev_diff   = prev_difficulty,
        )

        # ── 5. Update score ────────────────────────────────────────
        step_score = SCORE_BASE_PER_STEP
        if SCORE_DIFFICULTY_MULT:
            step_score *= (0.5 + self.difficulty)   # harder = more points
        in_flow = abs(self.workload_smooth - TARGET_WORKLOAD) <= FLOW_TOLERANCE
        if in_flow:
            step_score += SCORE_FLOW_BONUS
        self.score += step_score

        # ── 6. Oscillation tracking ────────────────────────────────
        if action != 1 and self._last_action != 1 and action != self._last_action:
            self._oscillation_count += 1
        self._last_action = action

        # ── 7. Record history ──────────────────────────────────────
        self.history["workload_raw"].append(self.workload_raw)
        self.history["workload_smooth"].append(self.workload_smooth)
        self.history["difficulty"].append(self.difficulty)
        self.history["action"].append(action)
        self.history["reward"].append(reward)
        self.history["score"].append(self.score)
        self.history["flow_zone"].append(in_flow)
        self.history["workload_class"].append(int(w_class))

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
            
        next_state = self._get_state()

        return next_state, reward, self.done, self._make_info()

    def _compute_reward(self, w_smooth, w_raw, action, prev_diff):
        """
        Flow-theory shaped reward function.

        Research basis:
          - Csikszentmihalyi 1990: flow occurs when challenge ≈ skill
          - Ng et al. 1999: potential-based reward shaping preserves
            optimal policy while accelerating convergence
          - Sweetser & Wyeth 2005: GameFlow model — reward engagement
            not just performance

        Components:
          1. Core flow reward: -|W_smooth - W*|
             Maximum 0 when exactly at target, minimum -0.5
          2. Asymmetric zone penalties:
             - Boredom (W < 0.35): linear penalty, easier to escape
             - Overload (W > 0.65): steeper penalty — overload more harmful
          3. Directional shaping:
             - Bonus for moving difficulty toward flow-inducing level
             - Penalty for moving away from target
          4. Oscillation penalty:
             - Small penalty for rapid direction reversals
             - Prevents oscillating between increase/decrease
          5. Stability bonus:
             - Small reward for maintaining when already in flow
        """
        error = abs(w_smooth - TARGET_WORKLOAD)

        # ── Component 1: Core proximity reward ────────────────────
        # Gaussian reward centred on target — smooth gradient
        # σ = FLOW_TOLERANCE so reward decays naturally from centre
        r_core = np.exp(-(error ** 2) / (2 * FLOW_TOLERANCE ** 2)) - 1.0
        # r_core ∈ [-1, 0], peaks at 0 when w_smooth = TARGET_WORKLOAD

        # ── Component 2: Asymmetric zone penalties ─────────────────
        r_zone = 0.0
        if w_smooth < BOREDOM_THRESHOLD:
            # Boredom — linear penalty scaling with distance from boundary
            r_zone = -0.3 * (BOREDOM_THRESHOLD - w_smooth) / BOREDOM_THRESHOLD
        elif w_smooth > OVERLOAD_THRESHOLD:
            # Overload — steeper penalty (overload is worse than boredom)
            # Babiloni 2019: cognitive overload causes rapid disengagement
            r_zone = -0.5 * (w_smooth - OVERLOAD_THRESHOLD) / (1 - OVERLOAD_THRESHOLD)

        # ── Component 3: Directional shaping ──────────────────────
        # Reward actions that should move workload toward target
        r_direction = 0.0
        w_error_signed = w_smooth - TARGET_WORKLOAD   # +ve = overloaded, -ve = bored

        if w_smooth > TARGET_WORKLOAD + FLOW_TOLERANCE:
            # Overloaded — reducing difficulty is an absolute emergency
            if action == 0:    r_direction = +0.35  # MASSIVE reward for saving the panicked player
            elif action == 1:  r_direction = -0.20  # Penalty for standing by while player panics
            elif action == 2:  r_direction = -0.50  # Nuke for increasing difficulty during panic
        elif w_smooth < TARGET_WORKLOAD - FLOW_TOLERANCE:
            # Bored — increasing difficulty is extremely correct
            if action == 2:    r_direction = +0.25  # aggressively reward increasing
            elif action == 1:  r_direction = -0.10  # Penalty for standing by while player is bored
            elif action == 0:  r_direction = -0.20  # penalize decreasing

        # ── Component 4: Oscillation penalty ──────────────────────
        r_oscillation = 0.0
        if (action != 1 and
                self._last_action != 1 and
                action != self._last_action):
            r_oscillation = -0.05  # small penalty for reversal

        # ── Component 5: Stability bonus ──────────────────────────
        r_stability = 0.0
        in_flow = error <= FLOW_TOLERANCE
        if in_flow and action == 1:
            r_stability = +0.05   # maintain when in flow = good

        # ── Combine ────────────────────────────────────────────────
        reward = r_core + r_zone + r_direction + r_oscillation + r_stability

        # Clip to prevent extreme values from destabilising training
        return float(np.clip(reward, -1.5, 0.2))

    def _get_state(self):
        """
        State vector for the RL agent.

        State: [workload_smooth, difficulty]
          Both in [0, 1]. Simple 2D state space is intentional —
          DQN for DDA does not need high-dimensional state to converge.
          Mnih et al. 2015: DQN works well on compact state representations.
        """
        return np.array([self.workload_smooth, self.difficulty],
                        dtype=np.float32)

    def _make_info(self):
        return {
            "workload_raw":    self.workload_raw,
            "workload_smooth": self.workload_smooth,
            "difficulty":      self.difficulty,
            "score":           self.score,
            "flow_zone":       abs(self.workload_smooth - TARGET_WORKLOAD) <= FLOW_TOLERANCE,
            "oscillations":    self._oscillation_count,
            "step":            self.step_count,
        }

    # ── Properties for external access ────────────────────────────

    @property
    def flow_percentage(self):
        """% of steps spent in the flow zone."""
        if not self.history["flow_zone"]:
            return 0.0
        return sum(self.history["flow_zone"]) / len(self.history["flow_zone"]) * 100

    @property
    def mean_reward(self):
        if not self.history["reward"]:
            return 0.0
        return float(np.mean(self.history["reward"]))

    @property
    def final_score(self):
        return self.score

    def summary(self):
        """Print a summary of the episode."""
        print(f"\n  Episode Summary:")
        print(f"  Steps         : {self.step_count}")
        print(f"  Flow zone     : {self.flow_percentage:.1f}%  "
              f"(target W*={TARGET_WORKLOAD}, tolerance ±{FLOW_TOLERANCE})")
        print(f"  Mean reward   : {self.mean_reward:.4f}")
        print(f"  Final score   : {self.final_score:.0f}")
        print(f"  Final diff    : {self.difficulty:.3f}")
        print(f"  Oscillations  : {self._oscillation_count}")

        if self.history["workload_smooth"]:
            w = np.array(self.history["workload_smooth"])
            print(f"  Workload mean : {w.mean():.3f}  std={w.std():.3f}  "
                  f"[{w.min():.3f}, {w.max():.3f}]")
