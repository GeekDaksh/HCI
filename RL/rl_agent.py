"""
rl_agent.py — DQN Agent for Adaptive Difficulty
=================================================
Implements Deep Q-Network (DQN) for dynamic difficulty adjustment.

Research basis:
  - Mnih et al. 2015 (DQN Nature paper): experience replay + target network
  - Stein et al. 2018: DQN for EEG-triggered DDA
  - Van Hasselt et al. 2016: Double DQN to reduce overestimation bias
  - Prioritised Replay: Schaul et al. 2016 (simplified version used here)

Architecture:
  State  : [workload_smooth, difficulty]  — 2D continuous
  Action : {0: Decrease, 1: Maintain, 2: Increase}
  Network: MLP with 2 hidden layers (64 units each)

Key DQN improvements over naive Q-learning:
  1. Experience replay: decorrelates consecutive experiences
  2. Target network: stabilises Q-value targets during training
  3. ε-greedy with decay: exploration → exploitation over time
  4. Double DQN: separate networks for action selection and evaluation
  5. Gradient clipping: prevents destructive parameter updates
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# ─────────────────────────────────────────────
#  HYPERPARAMETERS
#  Tuned for the 2D state, 3-action DDA problem
# ─────────────────────────────────────────────

# Network architecture
HIDDEN_DIM     = 64       # hidden layer size — small network for small state space

# Training
LR             = 1e-3     # Adam learning rate
GAMMA          = 0.95     # discount factor — slightly less than 1 for finite episodes
BATCH_SIZE     = 64       # minibatch size from replay buffer
REPLAY_CAPACITY= 10_000   # replay buffer size
MIN_REPLAY     = 256      # minimum experiences before training starts

# Target network
TARGET_UPDATE  = 50       # update target network every N steps

# Exploration (ε-greedy)
EPS_START      = 1.0      # start fully random
EPS_END        = 0.05     # end with 5% random — always keep some exploration
EPS_DECAY      = 0.997    # multiply epsilon by this each episode
                           # 0.997^300 ≈ 0.41 — still exploring at end of training

# Gradient clipping — prevents exploding gradients
GRAD_CLIP      = 1.0


# ─────────────────────────────────────────────
#  Q-NETWORK
# ─────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Simple MLP Q-network.

    Maps state (2D) → Q-values for each action (3D).
    Small network is appropriate because:
      - State space is 2D (workload, difficulty)
      - Action space is 3 (decrease, maintain, increase)
      - We don't need deep feature extraction — the workload
        estimate already contains rich information
    """

    def __init__(self, n_states, n_actions, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        # Xavier initialisation — better convergence than default
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
#  EXPERIENCE REPLAY BUFFER
# ─────────────────────────────────────────────

class ReplayBuffer:
    """
    Experience replay buffer — Mnih et al. 2015.

    Stores (state, action, reward, next_state, done) tuples.
    Random sampling decorrelates consecutive experiences,
    which is critical for stable Q-learning.

    Without replay: consecutive (s,a,r,s') are highly correlated
    → gradients point in similar directions → unstable training.
    With replay: random batch breaks temporal correlation.
    """

    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer   = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
#  DQN AGENT
# ─────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN agent with experience replay and target network.

    Double DQN (Van Hasselt et al. 2016):
      - Online network selects the best action
      - Target network evaluates that action's Q-value
      - Reduces overestimation bias vs standard DQN

    Training step:
      1. Sample random batch from replay buffer
      2. Compute target Q using Double DQN formula
      3. Compute MSE loss between predicted and target Q
      4. Backprop with gradient clipping
      5. Periodically copy online → target network
    """

    def __init__(self, n_states=2, n_actions=3, device=None):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.device    = device or torch.device("cpu")

        # Online network — trained every step
        self.online_net = QNetwork(n_states, n_actions).to(self.device)

        # Target network — updated periodically, provides stable targets
        self.target_net = QNetwork(n_states, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net never in training mode

        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=LR)
        self.buffer     = ReplayBuffer(REPLAY_CAPACITY)

        self.epsilon    = EPS_START
        self.steps_done = 0
        self.episodes   = 0

        # Logging
        self._losses    = []
        self._q_values  = []

    def select_action(self, state, training=True):
        """
        ε-greedy action selection.

        During training: random action with probability ε, else greedy.
        During evaluation: always greedy (ε=0).

        Parameters
        ----------
        state    : array (2,) — [workload_smooth, difficulty]
        training : bool — whether to apply exploration

        Returns
        -------
        action : int  0=Decrease, 1=Maintain, 2=Increase
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        # Greedy action from online network
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals  = self.online_net(state_t)
            self._q_values.append(q_vals.max().item())
            return int(q_vals.argmax().item())

    def push_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        self.steps_done += 1

    def train_step(self):
        """
        One gradient update step.

        Returns loss value (float) or None if buffer not ready.
        """
        if len(self.buffer) < MIN_REPLAY:
            return None

        # Sample minibatch
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Convert to tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # ── Current Q-values ──────────────────────────────────────
        # Q(s, a) for the actions that were actually taken
        q_current = self.online_net(states_t)
        q_taken   = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # ── Double DQN target ────────────────────────────────────
        # Van Hasselt et al. 2016:
        #   action* = argmax_a Q_online(s', a)   ← online net selects action
        #   target  = r + γ * Q_target(s', a*)   ← target net evaluates it
        # This separates action selection from evaluation to reduce overestimation.
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(1)
            next_q_vals  = self.target_net(next_states_t)
            next_q_target= next_q_vals.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target     = rewards_t + GAMMA * next_q_target * (1 - dones_t)

        # ── Huber loss (smooth L1) ───────────────────────────────
        # More robust than MSE to outlier Q-value estimates
        loss = nn.functional.smooth_l1_loss(q_taken, q_target)

        # ── Backpropagation ──────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        # ── Update target network ────────────────────────────────
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        loss_val = loss.item()
        self._losses.append(loss_val)
        return loss_val

    def end_episode(self, episode_reward=None):
        """
        Called at end of each episode.
        Decays epsilon for exploration schedule.
        """
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        self.episodes += 1

    def save(self, path):
        """Save agent state to disk."""
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "steps_done":  self.steps_done,
            "episodes":    self.episodes,
        }, path)
        print(f"  Agent saved → {path}")

    def load(self, path):
        """Load agent state from disk."""
        if not os.path.exists(path):
            print(f"  [WARN] Agent checkpoint not found: {path}")
            return
        checkpoint = torch.load(path, map_location=self.device)
        
        # Backward compatibility for old checkpoints that used 'q_net'
        if "online_net" in checkpoint:
            self.online_net.load_state_dict(checkpoint["online_net"])
        elif "q_net" in checkpoint:
            self.online_net.load_state_dict(checkpoint["q_net"])
            
        if "target_net" in checkpoint:
            self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon   = checkpoint.get("epsilon",    EPS_END)
        self.steps_done= checkpoint.get("steps_done", 0)
        self.episodes  = checkpoint.get("episodes",   0)
        print(f"  Agent loaded from {path}  "
              f"(episode={self.episodes}, ε={self.epsilon:.3f})")

    @property
    def mean_loss(self):
        if not self._losses:
            return 0.0
        return float(np.mean(self._losses[-100:]))

    @property
    def mean_q(self):
        if not self._q_values:
            return 0.0
        return float(np.mean(self._q_values[-100:]))
